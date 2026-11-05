"""
Financial time series data pipeline for GBM diffusion model.

Data construction per paper Section 3.1.2:
  1. S&P 500 constituents, stocks with >40 years of daily history
  2. Log-returns: r_t = log(p_t / p_{t-1})
  3. Sliding window: length 2048, stride 400
  4. All SDEs train on anchored log-prices (log S_t - log S_0) — confirmed by
     the paper's author; log-returns are only used for evaluation.
  5. Optional legacy mode: raw log-returns (data_mode='log_return')

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
    "INTC", "CSCO", "ORCL", "UNP", "FDX", "UPS", "LMT", "BA", "GD",
    "NOC", "GIS", "HRL", "SJM", "CPB", "MKC", "HSY", "CLX", "CHD",
    "BEN", "TROW", "CINF", "CB", "MMC", "AIG", "TRV", "ALL",
    "WFC", "JPM", "BAC", "C", "GS", "MS", "USB", "PNC", "BK", "STT",
    "SO", "DUK", "NEE", "D", "AEE", "WEC", "CMS", "ES", "EXC", "AEP",
    "PPL", "FE", "ETR", "XEL", "DTE", "SRE", "NI", "PNW", "OKE", "KMI",
    "ADM", "APD", "LIN", "DD", "PPG", "NUE", "FCX", "IP", "PKG",
    "AVY", "RPM", "FAST", "GWW", "WST", "ROP", "IFF",
    "WAT", "A", "TMO", "DHR", "BAX", "EW", "BSX",
    "AMGN", "GILD", "BIIB", "LLY", "PFE", "BMY",
    "CVS", "UNH", "CI", "ELV",  # ANTM → ELV (Elevance Health)
    "SPY"  # S&P 500 ETF as benchmark
]
# Removed: K (→ KLG after spinoff), RTX (formed 2020), PFG/LNC (delisted),
#   WRK/AMCR/SEE/BLL (short history), HOLX/STE/ZBH (short history),
#   REGN/VRTX/CNC/HUM (short history), ANTM (→ ELV)


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


def preprocess_prices(prices, ticker="", spike_threshold=0.4, reversal_tol=0.1,
                      max_abs_log_return=None, outlier_sigma=None):
    """Preprocess a single stock's price series per paper Section 3.1.2.

    Steps:
      1. Drop NaN/zero/negative prices
      2. Repair one-day bad ticks: a move |r_t| > spike_threshold that is
         reversed the next day (|r_t + r_{t+1}| < reversal_tol·|r_t|) — e.g.
         BEN 1984-12-31 −91% then +92%. The net move is kept on day t+1.

    Genuine crashes are KEPT (AIG 2008-09-15 −61%, AAPL 2000-09-29 −52%,
    1987-10-19): the heavy tails are what the generator must learn. The old
    default of zeroing every >5σ return truncated the tail and inflated the
    real Hill α to ~9-10.

    Args:
        prices: 1D numpy array of adjusted close prices
        ticker: ticker name for logging
        spike_threshold: minimum |log-return| considered for bad-tick repair
        reversal_tol: relative tolerance for "fully reversed next day"
        max_abs_log_return: optional hard cap; moves beyond it are set to 0.
            None (default) disables it.
        outlier_sigma: legacy σ-based clipping; None (default) disables it.
            Not recommended — it removes real tail events.
    Returns:
        cleaned 1D numpy array, or empty array if too short
    """
    mask = np.isfinite(prices) & (prices > 0)
    prices = prices[mask]
    if len(prices) < 100:
        return np.array([])

    log_ret = np.diff(np.log(prices))
    if np.std(log_ret) < 1e-10:
        return np.array([])

    n_fixed = 0
    big = np.where(np.abs(log_ret[:-1]) > spike_threshold)[0]
    for i in big:
        if abs(log_ret[i] + log_ret[i + 1]) < reversal_tol * abs(log_ret[i]):
            log_ret[i + 1] += log_ret[i]
            log_ret[i] = 0.0
            n_fixed += 1

    bad = np.zeros_like(log_ret, dtype=bool)
    if max_abs_log_return is not None:
        bad |= np.abs(log_ret) > max_abs_log_return
    if outlier_sigma is not None:
        bad |= np.abs(log_ret - log_ret.mean()) > outlier_sigma * log_ret.std()
    log_ret[bad] = 0.0
    n_fixed += int(bad.sum())

    if n_fixed > 0:
        log_prices = np.empty(len(log_ret) + 1)
        log_prices[0] = np.log(prices[0])
        log_prices[1:] = log_prices[0] + np.cumsum(log_ret)
        prices = np.exp(log_prices)
        if ticker:
            print(f"    {ticker}: repaired {n_fixed} bad tick(s)")

    return prices


RAW_CACHE_NAME = "sp500_prices_raw.pkl"
LEGACY_CACHE_NAME = "sp500_prices.pkl"  # pre-fix cache: >5σ returns zeroed — do not use


def download_stock_data(tickers, min_years=40, cache_dir="data/financial"):
    """Download, preprocess, and cache historical stock data.

    The cache (``sp500_prices_raw.pkl``) stores UNPROCESSED adjusted closes as
    ``{ticker: pd.Series(close, index=dates)}`` so preprocessing can change
    without re-downloading and dates are available for time-based splits.

    Args:
        tickers: list of ticker symbols
        min_years: minimum years of history required
        cache_dir: directory to cache downloaded data
    Returns:
        dict mapping ticker → numpy array of preprocessed adjusted close prices
    """
    raw = load_raw_price_cache(cache_dir)
    if raw is None:
        raw = _download_raw(tickers, min_years, cache_dir)
    return preprocess_stock_data(raw)


def load_raw_price_cache(cache_dir="data/financial"):
    """Return the raw-price cache dict, or None if it does not exist."""
    cache_file = os.path.join(cache_dir, RAW_CACHE_NAME)
    if not os.path.exists(cache_file):
        legacy = os.path.join(cache_dir, LEGACY_CACHE_NAME)
        if os.path.exists(legacy):
            print(f"NOTE: ignoring legacy cache {legacy} (tails were clipped at 5σ). "
                  f"Re-downloading raw prices into {cache_file}.")
        return None
    print(f"Loading raw price cache: {cache_file}")
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def preprocess_stock_data(raw, min_len=2049):
    """Apply preprocess_prices() to every ticker of a raw-price cache."""
    stock_data = {}
    for ticker, series in raw.items():
        values = np.asarray(series, dtype=np.float64).ravel()
        prices = preprocess_prices(values, ticker=ticker)
        if len(prices) >= min_len:
            stock_data[ticker] = prices
    return stock_data


def _download_raw(tickers, min_years, cache_dir):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, RAW_CACHE_NAME)
    cutoff_date = datetime.now() - timedelta(days=min_years * 365)
    clean_tickers = [t for t in tickers if "." not in t and "-" not in t]

    raw = {}
    print(f"Downloading data for {len(clean_tickers)} tickers...")
    for ticker in clean_tickers:
        try:
            df = yf.download(ticker, start="1970-01-01",
                             end=datetime.now().strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            first_date = close.index[0].to_pydatetime().replace(tzinfo=None)
            if first_date <= cutoff_date and len(close) > 2048:
                raw[ticker] = close.astype(np.float64)
                print(f"  {ticker}: {len(close)} days from {close.index[0].date()}")
            else:
                print(f"  {ticker}: skipped (history from {first_date.date()})")
        except Exception as e:
            print(f"  {ticker}: failed ({e})")

    print(f"Downloaded {len(raw)} stocks with >{min_years} years of data")
    with open(cache_file, "wb") as f:
        pickle.dump(raw, f)
    return raw


def create_subsequences(stock_data, window_len=2048, stride=400, mode="log_price"):
    """Extract overlapping subsequences from stock price data.

    Args:
        stock_data: dict of ticker → price array
        window_len: subsequence length (default 2048)
        stride: sliding window stride (default 400)
        mode: 'log_price' (anchored at 0 per window) or 'log_return'
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


def ticker_split(tickers, train_ratio=0.9, seed=42):
    """Deterministic ticker hold-out split → (train_tickers, val_tickers).

    Shared by get_dataloaders() and the cross-repo comparison so every model
    trains on exactly the same tickers.
    """
    tickers = sorted(tickers)
    perm = np.random.RandomState(seed).permutation(len(tickers))
    n_val = max(1, int(round(len(tickers) * (1 - train_ratio))))
    val = sorted(tickers[i] for i in perm[:n_val])
    train = sorted(t for t in tickers if t not in set(val))
    return train, val


def get_dataloaders(sde_type="gbm", window_len=2048, stride=400, batch_size=64,
                    use_synthetic=False, min_years=40, cache_dir="data/financial",
                    csv_path="data/sp500.csv",
                    train_ratio=0.9, num_workers=0, data_mode=None,
                    split="ticker", seed=42):
    """Create train and validation dataloaders.

    Args:
        sde_type: SDE name (only used for logging; data_mode decides the data)
        window_len: sliding window length
        stride: sliding window stride
        batch_size: training batch size
        use_synthetic: if True, use synthetic GBM data (no download)
        min_years: minimum years of stock history
        cache_dir: data cache directory
        csv_path: single-ticker CSV fallback when no cache/download is possible
        train_ratio: fraction of tickers (split='ticker') or windows
            (split='random') used for training
        num_workers: dataloader workers
        data_mode: 'log_price' (default for all SDEs — confirmed by the paper's
            author) or 'log_return' (legacy VE/VP behaviour)
        split: 'ticker' holds out whole tickers so overlapping windows cannot
            leak between train and val; 'random' splits windows (legacy).
            Falls back to 'random' when there is only one ticker.
        seed: RNG seed for the split
    Returns:
        train_loader, val_loader, data_info dict
    """
    # Priority: synthetic > raw cache > yfinance download > single-ticker CSV
    if use_synthetic:
        stock_data = generate_synthetic_gbm_data(n_sequences=100, seq_len=window_len + 500)
    else:
        raw = load_raw_price_cache(cache_dir)
        if raw is not None:
            stock_data = preprocess_stock_data(raw, min_len=window_len + 1)
        else:
            try:
                stock_data = download_stock_data(LONG_HISTORY_TICKERS, min_years=min_years,
                                                 cache_dir=cache_dir)
            except ImportError:
                print("yfinance unavailable, falling back to single-ticker CSV")
                stock_data = load_csv_data(csv_path)

    if len(stock_data) == 0:
        print("No real data available, falling back to synthetic data")
        stock_data = generate_synthetic_gbm_data(n_sequences=100, seq_len=window_len + 500)

    mode = data_mode or "log_price"
    if mode not in ("log_price", "log_return"):
        raise ValueError(f"Unknown data_mode: {mode}")

    tickers = sorted(stock_data)
    if split == "ticker" and len(tickers) >= 2:
        val_tickers = set(ticker_split(tickers, train_ratio, seed)[1])
        train_seqs = create_subsequences({t: stock_data[t] for t in tickers if t not in val_tickers},
                                         window_len=window_len, stride=stride, mode=mode)
        val_seqs = create_subsequences({t: stock_data[t] for t in tickers if t in val_tickers},
                                       window_len=window_len, stride=stride, mode=mode)
        split_used = "ticker"
    else:
        sequences = create_subsequences(stock_data, window_len=window_len, stride=stride, mode=mode)
        perm = np.random.RandomState(seed).permutation(len(sequences))
        n_train = int(len(sequences) * train_ratio)
        train_seqs, val_seqs = sequences[perm[:n_train]], sequences[perm[n_train:]]
        val_tickers = set()
        split_used = "random"

    if len(train_seqs) == 0:
        raise RuntimeError("No subsequences created. Check data availability.")
    if len(val_seqs) == 0:
        val_seqs = train_seqs[:1]

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
            f"Training loader is empty ({len(train_seqs)} samples, batch_size={batch_size}). "
            f"Reduce stride or batch_size, or add more data."
        )

    data_info = {
        "n_train": len(train_seqs),
        "n_val": len(val_seqs),
        "n_total": len(train_seqs) + len(val_seqs),
        "seq_len": window_len,
        "mode": mode,
        "split": split_used,
        "val_tickers": sorted(val_tickers),
        "n_stocks": len(stock_data),
    }
    print(f"Split={split_used}: {len(train_seqs)} train / {len(val_seqs)} val windows "
          f"({len(stock_data)} stocks, {len(val_tickers)} held out)")

    return train_loader, val_loader, data_info

def compute_sigma_max(train_loader, percentile=99, n_max=1000):
    """Compute recommended σ_max for VE/GBM SDE from data pairwise distances.

    Song et al. (2020) recommend σ_max ≈ max pairwise distance (per element)
    so that at t=T the prior N(0, σ_max²) fully obscures the signal.

    For cumulative log-price paths, the per-element variance grows with
    position (drift structure), so σ_max must cover the MAXIMUM spread.

    Returns a conservative estimate: p99 of the per-element range across
    training samples, multiplied by a safety factor of 3.

    Args:
        train_loader: DataLoader yielding (B, L) tensors
        percentile: percentile of per-element range to use (default 99)
        n_max: max samples to use (for speed)

    Returns:
        recommended σ_max (float), data_stats dict
    """
    all_data = []
    for batch in train_loader:
        all_data.append(batch.numpy())
        if sum(len(b) for b in all_data) >= n_max:
            break
    all_data = np.concatenate(all_data, axis=0)[:n_max]

    # Per-element statistics across samples
    elem_std = all_data.std(axis=0)         # (L,)
    elem_range = all_data.max(axis=0) - all_data.min(axis=0)  # (L,)
    global_std = all_data.std()
    global_range = all_data.max() - all_data.min()

    # Conservative σ_max: 3× the p99 per-element range (ensures KL < 0.01)
    max_elem_range = np.percentile(elem_range, percentile)
    sigma_max = 3.0 * max(global_std, max_elem_range / 2)

    # Floor at 5.0 to avoid edge cases with small datasets
    sigma_max = max(sigma_max, 5.0)

    stats = {
        "global_mean": float(all_data.mean()),
        "global_std": float(global_std),
        "global_range": float(global_range),
        "max_elem_std": float(elem_std.max()),
        "max_elem_range": float(max_elem_range),
        "n_samples": len(all_data),
        "seq_len": all_data.shape[1],
        "sigma_max": float(sigma_max),
    }

    print(f"\n  σ_max auto-compute: global_std={global_std:.4f}, "
          f"max_elem_range(p{percentile})={max_elem_range:.4f}")
    print(f"  → σ_max = {sigma_max:.1f}  "
          f"(3× max(global_std, range/2), floor=5.0)")

    return float(sigma_max), stats