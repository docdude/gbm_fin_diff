"""
Financial time series data pipeline for GBM diffusion model.

Data construction per paper Section 3.1.2:
  1. S&P 500 constituents, stocks with >40 years of daily history
  2. Log-returns: r_t = log(p_t / p_{t-1})
  3. Sliding window: length 2048, stride 400
  4. For GBM SDE: cumulative log-returns (= log-prices relative to first price)
  5. For VE/VP SDE: raw log-returns

Supports loading from:
  - Local CSV files (OHLCV format, e.g. data/sp500.csv)
  - yfinance downloads (as fallback)
  - Synthetic GBM data (for testing)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# S&P 500 tickers (representative subset — full list fetched at runtime if possible)
# ---------------------------------------------------------------------------

# Core long-history tickers known to have >40 years of data
LONG_HISTORY_TICKERS = [
    "AAPL", "MSFT", "JNJ", "PG", "KO", "PEP", "MRK", "ABT", "MMM", "GE",
    "IBM", "XOM", "CVX", "CL", "EMR", "SYY", "ADP", "DOV", "ITW", "PH",
    "SWK", "GPC", "LOW", "TGT", "BDX", "SHW", "ECL", "AFL", "ED", "ATO",
    "MDT", "CAT", "DE", "HON", "WMT", "HD", "MCD", "DIS", "NKE", "TXN",
    "INTC", "CSCO", "ORCL", "UNP", "FDX", "UPS", "LMT", "RTX", "BA", "GD",
    "NOC", "GIS", "K", "HRL", "SJM", "CPB", "MKC", "HSY", "CLX", "CHD",
    "BEN", "TROW", "CINF", "CB", "MMC", "AIG", "TRV", "ALL", "PFG", "LNC",
    "WFC", "JPM", "BAC", "C", "GS", "MS", "USB", "PNC", "BK", "STT",
    "SO", "DUK", "NEE", "D", "AEE", "WEC", "CMS", "ES", "EXC", "AEP",
    "PPL", "FE", "ETR", "XEL", "DTE", "SRE", "NI", "PNW", "OKE", "KMI",
    "ADM", "APD", "LIN", "DD", "PPG", "NUE", "FCX", "IP", "PKG", "WRK",
    "AVY", "SEE", "BLL", "AMCR", "RPM", "FAST", "GWW", "WST", "ROP", "IFF",
    "WAT", "A", "TMO", "DHR", "STE", "HOLX", "BAX", "ZBH", "EW", "BSX",
    "AMGN", "GILD", "BIIB", "REGN", "VRTX", "LLY", "PFE", "BMY",
    "CVS", "UNH", "CI", "HUM", "CNC", "ANTM",
    "SPY"  # S&P 500 ETF as benchmark
]


def load_csv_data(csv_path="data/sp500.csv"):
    """Load stock price data from a local OHLCV CSV file.

    Supports single-ticker CSVs (like ^GSPC) and multi-ticker CSVs.
    The CSV is expected to have a header row with Date,Close,High,Low,Open,Volume
    and optionally a second row with ticker names.

    Args:
        csv_path: path to the CSV file
    Returns:
        dict mapping ticker → numpy array of close prices
    """
    if not os.path.exists(csv_path):
        return {}

    print("Loading data from {}".format(csv_path))
    # Read the CSV; skip the ticker-name row if present
    df = pd.read_csv(csv_path, header=0)

    # Detect the yfinance multi-header format: second row has ticker names like ^GSPC
    # Check if the first data row has non-numeric values in the Close column
    first_val = df.iloc[0]["Close"] if "Close" in df.columns else None
    skip_row = False
    if first_val is not None:
        try:
            float(first_val)
        except (ValueError, TypeError):
            skip_row = True

    if skip_row:
        # Re-read skipping the ticker-name row
        df = pd.read_csv(csv_path, header=0, skiprows=[1])

    # Parse dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # Extract close prices
    if "Close" not in df.columns:
        # Try case-insensitive lookup
        close_col = [c for c in df.columns if c.lower() == "close"]
        if close_col:
            df = df.rename(columns={close_col[0]: "Close"})
        else:
            raise ValueError("No 'Close' column found in CSV. Columns: {}".format(list(df.columns)))

    prices = df["Close"].dropna().values.astype(np.float64)

    # Determine ticker name from the file
    ticker = os.path.splitext(os.path.basename(csv_path))[0].upper()
    stock_data = {ticker: prices}

    print("  {}: {} trading days ({} to {})".format(
        ticker, len(prices),
        df["Date"].iloc[0].date() if "Date" in df.columns else "?",
        df["Date"].iloc[-1].date() if "Date" in df.columns else "?",
    ))

    return stock_data


def download_stock_data(tickers, min_years=40, cache_dir="data/financial"):
    """Download and cache historical stock data.

    Args:
        tickers: list of ticker symbols
        min_years: minimum years of history required
        cache_dir: directory to cache downloaded data
    Returns:
        dict mapping ticker → numpy array of adjusted close prices
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "sp500_prices.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    cutoff_date = datetime.now() - timedelta(days=min_years * 365)
    # Filter out problematic tickers
    clean_tickers = [t for t in tickers if "." not in t and "-" not in t]

    stock_data = {}
    print(f"Downloading data for {len(clean_tickers)} tickers...")

    for ticker in clean_tickers:
        try:
            df = yf.download(ticker, start="1970-01-01", end=datetime.now().strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                continue
            # Check if history extends far enough
            first_date = df.index[0].to_pydatetime()
            if hasattr(first_date, 'tz') and first_date.tz:
                first_date = first_date.replace(tzinfo=None)
            if first_date <= cutoff_date:
                prices = df["Close"].dropna().values
                if len(prices) > 2048:
                    stock_data[ticker] = prices.astype(np.float64)
                    print(f"  {ticker}: {len(prices)} days from {df.index[0].date()}")
        except Exception as e:
            print(f"  {ticker}: failed ({e})")
            continue

    print(f"Downloaded {len(stock_data)} stocks with >{min_years} years of data")

    with open(cache_file, "wb") as f:
        pickle.dump(stock_data, f)

    return stock_data


def create_subsequences(stock_data, window_len=2048, stride=400, mode="log_price"):
    """Extract overlapping subsequences from stock price data.

    Args:
        stock_data: dict of ticker → price array
        window_len: subsequence length (default 2048)
        stride: sliding window stride (default 400)
        mode: 'log_price' (for GBM SDE) or 'log_return' (for VE/VP SDE)
    Returns:
        numpy array of shape (N, window_len)
    """
    sequences = []

    for ticker, prices in stock_data.items():
        if mode == "log_price":
            # Log-prices (normalized to start at 0 within each window)
            log_prices = np.log(prices)
            for start in range(0, len(log_prices) - window_len + 1, stride):
                window = log_prices[start: start + window_len]
                # Normalize: subtract first value so each window starts at 0
                window = window - window[0]
                sequences.append(window)
        elif mode == "log_return":
            # Log-returns
            log_returns = np.diff(np.log(prices))
            for start in range(0, len(log_returns) - window_len + 1, stride):
                window = log_returns[start: start + window_len]
                sequences.append(window)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    sequences = np.array(sequences, dtype=np.float32)
    print(f"Created {len(sequences)} subsequences of length {window_len} (mode={mode})")
    return sequences


def generate_synthetic_gbm_data(n_sequences=500, seq_len=2048, mu=0.0005, sigma=0.02):
    """Generate synthetic GBM data for testing (no yfinance dependency).

    This creates realistic-looking stock price paths following GBM:
        dS = μ S dt + σ S dW
    with occasional volatility regime changes to simulate real markets.

    Returns:
        dict mapping synthetic ticker → price array
    """
    stock_data = {}
    np.random.seed(42)

    for i in range(n_sequences):
        n_days = seq_len + 500  # extra for stride
        # Random initial price
        S0 = np.random.uniform(10, 500)
        # Random drift and volatility
        mu_i = np.random.normal(mu, 0.0003)
        sigma_i = np.random.uniform(0.01, 0.04)

        # Generate GBM path with volatility clustering (GARCH-like)
        log_prices = np.zeros(n_days)
        log_prices[0] = np.log(S0)
        vol = sigma_i
        for t in range(1, n_days):
            # Simple volatility clustering: vol partially reverts to mean
            vol = 0.95 * vol + 0.05 * sigma_i + 0.1 * abs(np.random.normal(0, sigma_i))
            log_prices[t] = log_prices[t - 1] + (mu_i - 0.5 * vol ** 2) + vol * np.random.normal()

        prices = np.exp(log_prices)
        stock_data[f"SYN_{i:04d}"] = prices

    return stock_data


class FinancialTimeSeriesDataset(Dataset):
    """PyTorch dataset for financial time series subsequences."""

    def __init__(self, sequences):
        """
        Args:
            sequences: numpy array of shape (N, L)
        """
        self.data = torch.from_numpy(sequences).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloaders(sde_type="gbm", window_len=2048, stride=400, batch_size=64,
                    use_synthetic=False, min_years=40, cache_dir="data/financial",
                    csv_path="data/sp500.csv",
                    train_ratio=0.9, num_workers=0):
    """Create train and validation dataloaders.

    Args:
        sde_type: 'gbm' uses log-prices; 've' and 'vp' use log-returns
        window_len: sliding window length
        stride: sliding window stride
        batch_size: training batch size
        use_synthetic: if True, use synthetic GBM data (no download)
        min_years: minimum years of stock history
        cache_dir: data cache directory
        csv_path: path to local OHLCV CSV (used first if it exists)
        train_ratio: fraction of data for training
        num_workers: dataloader workers
    Returns:
        train_loader, val_loader, data_info dict
    """
    # Get raw price data — priority: CSV > yfinance download > synthetic
    if use_synthetic:
        stock_data = generate_synthetic_gbm_data(n_sequences=100, seq_len=window_len + 500)
    elif os.path.exists(csv_path):
        stock_data = load_csv_data(csv_path)
        if len(stock_data) == 0:
            print("CSV load failed, falling back to yfinance download")
            stock_data = download_stock_data(LONG_HISTORY_TICKERS, min_years=min_years,
                                             cache_dir=cache_dir)
    else:
        stock_data = download_stock_data(LONG_HISTORY_TICKERS, min_years=min_years,
                                         cache_dir=cache_dir)

    if len(stock_data) == 0:
        print("No real data available, falling back to synthetic data")
        stock_data = generate_synthetic_gbm_data(n_sequences=100, seq_len=window_len + 500)

    # Create subsequences
    mode = "log_price" if sde_type == "gbm" else "log_return"

    # Validate data/SDE mode consistency
    if sde_type == "gbm" and mode != "log_price":
        raise ValueError(
            f"GBM SDE requires log-price data (cumulative log-returns), "
            f"but mode={mode}. Set sde_type='gbm' to use log-prices."
        )
    if sde_type in ("ve", "vp") and mode == "log_price":
        raise ValueError(
            f"{sde_type.upper()} SDE expects log-return data, but mode={mode}. "
            f"Log-price data with VE/VP SDE will produce incorrect results."
        )

    sequences = create_subsequences(stock_data, window_len=window_len, stride=stride, mode=mode)

    if len(sequences) == 0:
        raise RuntimeError("No subsequences created. Check data availability.")

    # Train/val split
    n_train = int(len(sequences) * train_ratio)
    np.random.seed(42)
    perm = np.random.permutation(len(sequences))
    train_seqs = sequences[perm[:n_train]]
    val_seqs = sequences[perm[n_train:]]

    train_dataset = FinancialTimeSeriesDataset(train_seqs)
    val_dataset = FinancialTimeSeriesDataset(val_seqs)

    # Only drop last batch if we have more samples than batch_size;
    # with a single stock (e.g., one CSV), drop_last=True can yield 0 batches.
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_seqs)),
                              shuffle=True, num_workers=num_workers,
                              drop_last=(len(train_seqs) > batch_size))
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, max(len(val_seqs), 1)),
                            shuffle=False, num_workers=num_workers, drop_last=False)

    if len(train_loader) == 0:
        raise RuntimeError(
            f"Training loader is empty ({n_train} samples, batch_size={batch_size}). "
            f"Reduce stride or batch_size, or add more data."
        )

    data_info = {
        "n_train": n_train,
        "n_val": len(sequences) - n_train,
        "n_total": len(sequences),
        "seq_len": window_len,
        "mode": mode,
        "n_stocks": len(stock_data),
    }

    return train_loader, val_loader, data_info
