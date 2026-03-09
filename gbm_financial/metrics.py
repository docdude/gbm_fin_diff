"""
Stylized facts evaluation metrics for generated financial time series.

Implements the three key metrics from Section 4 of arXiv:2507.19003,
following the methodology of Takahashi et al. (2019) [ref 7 in paper]:

  (i)   Heavy-tail exponent α via Hill estimator
        - P(r) ~ |r|^{-α}, empirical S&P 500: α ≈ 4.35
        - α is dataset-specific: the goal is generated α ≈ real α

  (ii)  Volatility clustering via ACF of absolute returns
        - Corr(|r_t|, |r_{t+k}|) ~ k^{-β}, β ∈ [0.1, 0.5]

  (iii) Leverage effect via lead-lag correlation (Bouchaud et al. 2001)
        - L(k) = E[r_t |r_{t+k}|² - r_t |r_t|²] / E[|r_t|²]²
        - Should be negative at small lags (negative returns → higher future vol)

Reference: Takahashi, Chen, Tanaka-Ishii (2019), "Modeling financial time-series
with generative adversarial networks", Physica A, vol. 527.
"""

import numpy as np
from scipy.stats import linregress
from collections import Counter
import warnings


# ============================================================================
# Data preparation
# ============================================================================

def compute_log_returns(data, mode="log_return"):
    """Convert generated data to log-returns for evaluation.

    Args:
        data: numpy array of shape (N, L) — generated sequences
        mode: 'log_return' → data IS returns; 'log_price' → compute returns
    Returns:
        returns: numpy array of shape (N, L-1) if log_price, else (N, L)
    """
    if mode == "log_price":
        return np.diff(data, axis=-1)
    return data


# ============================================================================
# (i) Heavy-tail exponent — Hill estimator
# ============================================================================

def hill_estimator(abs_returns_sorted_desc, k):
    """Compute Hill estimator for a sorted (descending) array.

    Hill (1975): α̂ = k / Σ_{i=1}^{k} [log(X_{(i)}) - log(X_{(k+1)})]
    where X_{(1)} ≥ X_{(2)} ≥ ... are order statistics.

    Args:
        abs_returns_sorted_desc: sorted |returns| in descending order
        k: number of tail observations to use
    Returns:
        alpha estimate (float)
    """
    if k < 2 or k >= len(abs_returns_sorted_desc):
        return np.nan
    log_x = np.log(abs_returns_sorted_desc[:k + 1])
    # Hill inverse: (1/k) * Σ [log(X_i) - log(X_{k+1})]
    hill_inv = np.mean(log_x[:k] - log_x[k])
    if hill_inv <= 0:
        return np.nan
    return 1.0 / hill_inv


def compute_heavy_tail_exponent(returns, method="hill", n_bins=100,
                                tail_fraction=0.15):
    """Estimate the tail exponent α of the return distribution.

    Two methods available:
      - 'hill': Hill estimator (standard for financial data)
      - 'loglog': Log-log density slope fitting

    The Hill estimator reports α for EACH tail (positive & negative)
    and the combined two-tail estimate. The key metric is the combined α.

    Empirical S&P 500 benchmark: α ≈ 4.35, but this is DATASET-SPECIFIC.
    The goal is: generated α ≈ real data α (closeness, not a fixed target).

    Args:
        returns: 1D array of all returns (flattened)
        method: 'hill' or 'loglog'
        n_bins: bins for loglog method
        tail_fraction: fraction of data for loglog method
    Returns:
        dict with 'alpha', 'alpha_pos', 'alpha_neg', and plotting data
    """
    returns = returns.flatten()
    abs_returns = np.abs(returns)
    abs_returns = abs_returns[abs_returns > 0]

    if len(abs_returns) < 100:
        return {"alpha": np.nan, "alpha_pos": np.nan, "alpha_neg": np.nan,
                "log_x": [], "log_density": []}

    result = {}

    if method == "hill":
        # Hill estimator — combined (both tails)
        sorted_abs = np.sort(abs_returns)[::-1]
        n = len(sorted_abs)
        k_default = int(np.sqrt(n))
        k_default = max(10, min(k_default, n // 4))

        # Compute for a range of k to find stable region
        k_range = np.arange(max(10, k_default // 3),
                            min(k_default * 3, n // 2), max(1, k_default // 10))
        alphas = [hill_estimator(sorted_abs, k) for k in k_range]
        valid = [(k, a) for k, a in zip(k_range, alphas) if not np.isnan(a)]

        if valid:
            alpha_values = [a for _, a in valid]
            result["alpha"] = np.median(alpha_values)
            result["hill_k_used"] = k_default
        else:
            result["alpha"] = np.nan

        # Separate tails
        pos_returns = returns[returns > 0]
        neg_returns = -returns[returns < 0]
        for label, tail_data in [("pos", pos_returns), ("neg", neg_returns)]:
            if len(tail_data) > 50:
                sorted_tail = np.sort(tail_data)[::-1]
                k_tail = max(10, int(np.sqrt(len(sorted_tail))))
                k_tail = min(k_tail, len(sorted_tail) // 4)
                result[f"alpha_{label}"] = hill_estimator(sorted_tail, k_tail)
            else:
                result[f"alpha_{label}"] = np.nan
    else:
        result["alpha_pos"] = np.nan
        result["alpha_neg"] = np.nan

    # Always compute log-log density for plotting
    abs_normed = abs_returns / np.std(abs_returns)
    x_min = np.percentile(abs_normed, 100 * (1 - tail_fraction))
    x_max = np.percentile(abs_normed, 99.9)

    if x_min > 0 and x_max > x_min:
        bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins)
        hist, bin_edges = np.histogram(abs_normed, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = hist > 0

        if mask.sum() >= 5:
            log_x = np.log10(bin_centers[mask])
            log_density = np.log10(hist[mask])
            result["log_x"] = log_x.tolist()
            result["log_density"] = log_density.tolist()

            if method == "loglog":
                try:
                    slope, _, r_value, _, _ = linregress(log_x, log_density)
                    result["alpha"] = -(slope + 1)
                    result["r_squared"] = r_value ** 2
                except Exception:
                    result["alpha"] = result.get("alpha", np.nan)
        else:
            result["log_x"] = []
            result["log_density"] = []
    else:
        result["log_x"] = []
        result["log_density"] = []

    for key in ["alpha", "alpha_pos", "alpha_neg", "log_x", "log_density"]:
        result.setdefault(key, np.nan if "alpha" in key else [])

    return result


# ============================================================================
# (ii) Volatility clustering — ACF of absolute returns
# ============================================================================

def compute_volatility_clustering(returns, max_lag=100):
    """Compute autocorrelation of absolute returns (volatility clustering).

    Paper Section 2.2 (ii): Corr(|r_t|, |r_{t+k}|) ~ k^{-β}, β ∈ [0.1, 0.5]
    Paper Section 4.3: "autocorrelation of absolute log-returns"

    Returns are pooled across all sequences for robust estimation.

    Args:
        returns: 2D array (N, L) of return sequences
        max_lag: maximum lag to compute (paper uses 100)
    Returns:
        dict with 'lags', 'autocorrelation', 'beta'
    """
    abs_returns = np.abs(returns)
    autocorr = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for seq in abs_returns:
        n = len(seq)
        if n < max_lag + 1:
            continue
        m = seq.mean()
        s = seq.std()
        if s < 1e-12:
            continue
        centered = (seq - m) / s

        for lag in range(1, min(max_lag + 1, n)):
            c = np.mean(centered[:n - lag] * centered[lag:])
            autocorr[lag - 1] += c
            counts[lag - 1] += 1

    valid_counts = counts > 0
    autocorr[valid_counts] /= counts[valid_counts]

    lags = list(range(1, max_lag + 1))

    acf_array = autocorr
    positive = acf_array > 0
    if positive.sum() > 5:
        log_lags = np.log10(np.array(lags)[positive])
        log_acf = np.log10(acf_array[positive])
        try:
            slope, _, _, _, _ = linregress(log_lags, log_acf)
            beta = -slope
        except Exception:
            beta = np.nan
    else:
        beta = np.nan

    return {
        "lags": lags,
        "autocorrelation": acf_array.tolist(),
        "beta": beta,
    }


# ============================================================================
# (iii) Leverage effect — lead-lag correlation (Bouchaud et al. 2001)
# ============================================================================

def compute_leverage_effect(returns, max_lag=100):
    """Compute leverage effect following the paper's formula (Section 2.2 iii).

    L(k) = E[r_t * r_{t+k}² - r_t * r_t²] / E[r_t²]²

    Paper reference: Bouchaud, Matacz, Potters (2001), "Leverage effect in
    financial markets: The retarded volatility model", PRL 87(22).

    Values can be outside [-1, 1] — this is NOT Pearson correlation.
    Empirical markets show L(k) < 0 at small lags.

    Args:
        returns: 2D array (N, L) of return sequences
        max_lag: maximum lag (paper uses 100)
    Returns:
        dict with 'lags', 'leverage_correlation'
    """
    leverage = []

    for lag in range(0, max_lag + 1):
        numerators = []
        denominators = []

        for seq in returns:
            n = len(seq)
            if n <= lag:
                continue

            if lag > 0:
                r = seq[:n - lag]
                r_future = seq[lag:]
            else:
                r = seq
                r_future = seq

            min_len = min(len(r), len(r_future))
            r = r[:min_len]
            r_f = r_future[:min_len]

            r_sq = r ** 2
            r_f_sq = r_f ** 2
            e_r_sq = np.mean(r_sq)

            if e_r_sq > 1e-20:
                num = np.mean(r * r_f_sq - r * r_sq)
                denom = e_r_sq ** 2
                numerators.append(num)
                denominators.append(denom)

        if numerators:
            leverage.append(
                np.mean(np.array(numerators) / np.array(denominators))
            )
        else:
            leverage.append(0.0)

    return {
        "lags": list(range(0, max_lag + 1)),
        "leverage_correlation": leverage,
    }


# ============================================================================
# Pathwise diagnostics — Audit D expansion
# ============================================================================
# These metrics probe local path dynamics that marginal stylized facts miss.
# A generator can match tail exponent, vol clustering, and leverage while
# still producing unrealistically smooth or bursty paths.  The tests below
# target that gap.
# ============================================================================

def compute_rolling_volatility(returns, window=20):
    """Compute rolling realized volatility (std of returns in a sliding window).

    Args:
        returns: 2D array (N, L)
        window: rolling window size
    Returns:
        dict with 'real_vol_traces' (list of arrays) and summary stats
    """
    traces = []
    for seq in returns:
        if len(seq) < window:
            continue
        # Rolling std via convolution — fast
        sq = seq ** 2
        # cumsum trick for rolling mean of squared returns
        cs = np.cumsum(sq)
        cs = np.insert(cs, 0, 0)
        mean_sq = (cs[window:] - cs[:-window]) / window
        cs_r = np.cumsum(seq)
        cs_r = np.insert(cs_r, 0, 0)
        mean_r = (cs_r[window:] - cs_r[:-window]) / window
        vol = np.sqrt(np.maximum(mean_sq - mean_r ** 2, 0))
        traces.append(vol)

    if not traces:
        return {"traces": [], "mean_vol": np.nan, "std_vol": np.nan,
                "vol_of_vol": np.nan}

    all_vols = np.concatenate(traces)
    return {
        "traces": traces,
        "mean_vol": float(np.mean(all_vols)),
        "std_vol": float(np.std(all_vols)),
        "vol_of_vol": float(np.std(all_vols) / max(np.mean(all_vols), 1e-12)),
    }


def compute_burst_duration(returns, window=20, quantile=0.9):
    """Measure duration of high-volatility bursts.

    Threshold rolling vol at the given quantile → binary high/low.
    Report run-lengths of consecutive high-vol states.

    This is the auditor's "burst persistence test": many generators match
    overall clustering but fail to sustain bursts for realistic durations.

    Args:
        returns: 2D array (N, L)
        window: rolling vol window
        quantile: threshold quantile for "high vol" (default: top decile)
    Returns:
        dict with 'run_lengths' (array), 'mean', 'median', 'max', 'p90'
    """
    rv = compute_rolling_volatility(returns, window=window)
    if not rv["traces"]:
        return {"run_lengths": [], "mean": np.nan, "median": np.nan,
                "max": np.nan, "p90": np.nan}

    all_vols = np.concatenate(rv["traces"])
    threshold = np.quantile(all_vols, quantile)

    run_lengths = []
    for trace in rv["traces"]:
        high = trace >= threshold
        # Run-length encoding
        runs = np.diff(np.concatenate([[0], high.astype(int), [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        run_lengths.extend((ends - starts).tolist())

    run_lengths = np.array(run_lengths) if run_lengths else np.array([0])
    return {
        "run_lengths": run_lengths.tolist(),
        "mean": float(np.mean(run_lengths)),
        "median": float(np.median(run_lengths)),
        "max": int(np.max(run_lengths)),
        "p90": float(np.percentile(run_lengths, 90)) if len(run_lengths) > 1 else 0.0,
    }


def compute_drawdown_stats(log_prices):
    """Compute drawdown depth and duration distributions.

    Drawdown = decline from running maximum of the log-price path.

    Args:
        log_prices: 2D array (N, L)
    Returns:
        dict with 'depths' (max drawdown per path), 'durations' (time in drawdown)
    """
    depths = []
    durations = []

    for path in log_prices:
        running_max = np.maximum.accumulate(path)
        dd = running_max - path  # drawdown at each point (≥ 0)

        # Max drawdown depth for this path
        depths.append(float(np.max(dd)))

        # Duration: lengths of contiguous drawdown episodes
        in_dd = dd > 1e-10
        runs = np.diff(np.concatenate([[0], in_dd.astype(int), [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        durations.extend((ends - starts).tolist())

    depths = np.array(depths)
    durations = np.array(durations) if durations else np.array([0])
    return {
        "depths": depths.tolist(),
        "durations": durations.tolist(),
        "mean_depth": float(np.mean(depths)),
        "mean_duration": float(np.mean(durations)),
        "max_depth": float(np.max(depths)) if len(depths) > 0 else np.nan,
        "max_duration": int(np.max(durations)) if len(durations) > 0 else 0,
    }


def compute_sign_runs(returns):
    """Compute run-length distribution of return signs.

    Counts consecutive runs of positive or negative returns.
    Real markets show characteristic run-length distributions that
    purely i.i.d. models cannot reproduce.

    Args:
        returns: 2D array (N, L)
    Returns:
        dict with 'pos_runs', 'neg_runs' distributions
    """
    pos_runs = []
    neg_runs = []

    for seq in returns:
        sign = np.sign(seq)
        # Remove zeros (treat as continuation)
        sign[sign == 0] = 1

        runs = np.diff(np.concatenate([[0], sign, [0]]))
        change_points = np.where(runs != 0)[0]

        for i in range(len(change_points) - 1):
            run_len = change_points[i + 1] - change_points[i]
            if sign[change_points[i]] > 0:
                pos_runs.append(run_len)
            else:
                neg_runs.append(run_len)

    return {
        "pos_runs": pos_runs,
        "neg_runs": neg_runs,
        "pos_mean": float(np.mean(pos_runs)) if pos_runs else np.nan,
        "neg_mean": float(np.mean(neg_runs)) if neg_runs else np.nan,
        "pos_counter": dict(Counter(pos_runs)),
        "neg_counter": dict(Counter(neg_runs)),
    }


def compute_turning_points(log_prices):
    """Count local extrema (turning points) per unit path length.

    A turning point is where sign(Δ) changes. Higher density → rougher path.

    Args:
        log_prices: 2D array (N, L)
    Returns:
        dict with per-path turning point density
    """
    densities = []

    for path in log_prices:
        if len(path) < 3:
            continue
        diffs = np.diff(path)
        sign_changes = np.abs(np.diff(np.sign(diffs)))
        n_turns = np.sum(sign_changes > 0)
        density = n_turns / (len(path) - 2)  # normalized by path length
        densities.append(float(density))

    return {
        "densities": densities,
        "mean_density": float(np.mean(densities)) if densities else np.nan,
        "std_density": float(np.std(densities)) if densities else np.nan,
    }


def compute_path_roughness(returns):
    """Compute total variation and quadratic variation of return paths.

    The auditor's key roughness test:
        TV  = Σ |r_t - r_{t-1}|           (total variation of returns)
        QV  = Σ r_t²                       (quadratic variation)
        TV₀ = Σ |r_t|                      (total variation of prices via returns)

    Flat synthetic paths show immediately lower TV than real paths.

    Args:
        returns: 2D array (N, L)
    Returns:
        dict with per-path TV, QV statistics
    """
    tv_returns = []   # Σ |r_t - r_{t-1}| — roughness of the return series itself
    qv = []           # Σ r_t²
    tv_prices = []    # Σ |r_t| — total variation of the log-price path

    for seq in returns:
        if len(seq) < 2:
            continue
        tv_returns.append(float(np.sum(np.abs(np.diff(seq)))))
        qv.append(float(np.sum(seq ** 2)))
        tv_prices.append(float(np.sum(np.abs(seq))))

    return {
        "tv_returns": tv_returns,
        "qv": qv,
        "tv_prices": tv_prices,
        "mean_tv_returns": float(np.mean(tv_returns)) if tv_returns else np.nan,
        "mean_qv": float(np.mean(qv)) if qv else np.nan,
        "mean_tv_prices": float(np.mean(tv_prices)) if tv_prices else np.nan,
        "std_tv_returns": float(np.std(tv_returns)) if tv_returns else np.nan,
    }


def compute_squared_return_acf(returns, max_lag=100):
    """Compute ACF of squared returns with confidence bands.

    Squared-return ACF is a more direct probe of vol clustering than
    absolute-return ACF and is sensitive to how the model sequences shocks.

    Args:
        returns: 2D array (N, L)
        max_lag: maximum lag
    Returns:
        dict with 'lags', 'acf', 'ci_upper', 'ci_lower'
    """
    sq_returns = returns ** 2
    acf = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for seq in sq_returns:
        n = len(seq)
        if n < max_lag + 1:
            continue
        m = seq.mean()
        s = seq.std()
        if s < 1e-12:
            continue
        centered = (seq - m) / s
        for lag in range(1, min(max_lag + 1, n)):
            c = np.mean(centered[:n - lag] * centered[lag:])
            acf[lag - 1] += c
            counts[lag - 1] += 1

    valid = counts > 0
    acf[valid] /= counts[valid]

    # Bartlett 95% confidence band for white noise: ±1.96/√n
    n_eff = int(np.median(counts[valid])) if valid.any() else 1
    ci = 1.96 / np.sqrt(max(n_eff, 1))

    return {
        "lags": list(range(1, max_lag + 1)),
        "acf": acf.tolist(),
        "ci_upper": float(ci),
        "ci_lower": float(-ci),
    }


def compute_regime_persistence(returns, window=20, quantile=0.9):
    """Measure persistence of high-vol and low-vol regimes.

    Threshold rolling vol into top-quantile (high) and bottom-(1-quantile)
    (low) states. Report run-length distributions for each regime.

    Args:
        returns: 2D array (N, L)
        window: rolling vol window
        quantile: threshold (default 0.9 = top decile)
    Returns:
        dict with high/low regime run-length stats
    """
    rv = compute_rolling_volatility(returns, window=window)
    if not rv["traces"]:
        return {"high_runs": [], "low_runs": [],
                "high_mean": np.nan, "low_mean": np.nan}

    all_vols = np.concatenate(rv["traces"])
    hi_thresh = np.quantile(all_vols, quantile)
    lo_thresh = np.quantile(all_vols, 1 - quantile)

    high_runs = []
    low_runs = []

    for trace in rv["traces"]:
        # High-vol runs
        hi = trace >= hi_thresh
        runs = np.diff(np.concatenate([[0], hi.astype(int), [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        high_runs.extend((ends - starts).tolist())

        # Low-vol runs
        lo = trace <= lo_thresh
        runs = np.diff(np.concatenate([[0], lo.astype(int), [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        low_runs.extend((ends - starts).tolist())

    return {
        "high_runs": high_runs,
        "low_runs": low_runs,
        "high_mean": float(np.mean(high_runs)) if high_runs else np.nan,
        "low_mean": float(np.mean(low_runs)) if low_runs else np.nan,
        "high_max": int(np.max(high_runs)) if high_runs else 0,
        "low_max": int(np.max(low_runs)) if low_runs else 0,
    }


def compute_pathwise_diagnostics(data, mode="log_return", vol_window=20):
    """Run all pathwise diagnostic computations.

    Args:
        data: (N, L) numpy array
        mode: 'log_return' or 'log_price'
        vol_window: rolling vol window size
    Returns:
        dict with all pathwise diagnostics
    """
    returns = compute_log_returns(data, mode=mode)
    log_prices = data if mode == "log_price" else np.cumsum(data, axis=-1)

    return {
        "rolling_vol": compute_rolling_volatility(returns, window=vol_window),
        "burst_duration": compute_burst_duration(returns, window=vol_window),
        "drawdown": compute_drawdown_stats(log_prices),
        "sign_runs": compute_sign_runs(returns),
        "turning_points": compute_turning_points(log_prices),
        "roughness": compute_path_roughness(returns),
        "squared_return_acf": compute_squared_return_acf(returns),
        "regime_persistence": compute_regime_persistence(returns, window=vol_window),
    }


def print_pathwise_summary(gen_pw, real_pw):
    """Print a compact comparison table of pathwise diagnostics."""
    print("\n" + "=" * 70)
    print("PATHWISE DIAGNOSTICS — Real vs Generated")
    print("=" * 70)

    rows = [
        ("Rolling vol (mean)",
         real_pw["rolling_vol"]["mean_vol"],
         gen_pw["rolling_vol"]["mean_vol"]),
        ("Vol-of-vol ratio",
         real_pw["rolling_vol"]["vol_of_vol"],
         gen_pw["rolling_vol"]["vol_of_vol"]),
        ("Burst duration (mean)",
         real_pw["burst_duration"]["mean"],
         gen_pw["burst_duration"]["mean"]),
        ("Burst duration (p90)",
         real_pw["burst_duration"]["p90"],
         gen_pw["burst_duration"]["p90"]),
        ("Max drawdown (mean)",
         real_pw["drawdown"]["mean_depth"],
         gen_pw["drawdown"]["mean_depth"]),
        ("Drawdown duration (mean)",
         real_pw["drawdown"]["mean_duration"],
         gen_pw["drawdown"]["mean_duration"]),
        ("Sign-run length (+, mean)",
         real_pw["sign_runs"]["pos_mean"],
         gen_pw["sign_runs"]["pos_mean"]),
        ("Sign-run length (-, mean)",
         real_pw["sign_runs"]["neg_mean"],
         gen_pw["sign_runs"]["neg_mean"]),
        ("Turning-point density",
         real_pw["turning_points"]["mean_density"],
         gen_pw["turning_points"]["mean_density"]),
        ("TV of returns (mean)",
         real_pw["roughness"]["mean_tv_returns"],
         gen_pw["roughness"]["mean_tv_returns"]),
        ("Quadratic variation (mean)",
         real_pw["roughness"]["mean_qv"],
         gen_pw["roughness"]["mean_qv"]),
        ("High-vol regime (mean run)",
         real_pw["regime_persistence"]["high_mean"],
         gen_pw["regime_persistence"]["high_mean"]),
        ("Low-vol regime (mean run)",
         real_pw["regime_persistence"]["low_mean"],
         gen_pw["regime_persistence"]["low_mean"]),
    ]

    print(f"  {'Metric':<30s} {'Real':>10s} {'Gen':>10s} {'Ratio':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    for name, real_val, gen_val in rows:
        if np.isnan(real_val) or np.isnan(gen_val):
            ratio_str = "  N/A"
        elif abs(real_val) < 1e-12:
            ratio_str = "  N/A"
        else:
            ratio_str = f"{gen_val / real_val:8.2f}"
        print(f"  {name:<30s} {real_val:10.4f} {gen_val:10.4f} {ratio_str}")

    print("=" * 70)


# ============================================================================
# Combined evaluation
# ============================================================================

def evaluate_stylized_facts(generated_data, mode="log_return", display=True):
    """Comprehensive evaluation of all three stylized facts.

    Args:
        generated_data: numpy array (N, L) of generated sequences
        mode: 'log_return' or 'log_price'
        display: whether to print results
    Returns:
        dict with all metrics
    """
    returns = compute_log_returns(generated_data, mode=mode)

    heavy_tail = compute_heavy_tail_exponent(returns)
    vol_cluster = compute_volatility_clustering(returns)
    leverage = compute_leverage_effect(returns)

    results = {
        "heavy_tail": heavy_tail,
        "volatility_clustering": vol_cluster,
        "leverage_effect": leverage,
    }

    if display:
        print("=" * 60)
        print("STYLIZED FACTS EVALUATION")
        print("=" * 60)
        print(f"\n(i) Heavy-tail exponent (Hill estimator):")
        print(f"    α (combined)  = {heavy_tail['alpha']:.2f}")
        print(f"    α (pos tail)  = {heavy_tail.get('alpha_pos', float('nan')):.2f}")
        print(f"    α (neg tail)  = {heavy_tail.get('alpha_neg', float('nan')):.2f}")
        print(f"    (Goal: generated α ≈ real data α)")

        print(f"\n(ii) Volatility clustering decay β: {vol_cluster['beta']:.3f}")
        print(f"     (empirical range: 0.1 ≤ β ≤ 0.5)")
        if vol_cluster['autocorrelation']:
            print(f"     ACF at lag 1:  {vol_cluster['autocorrelation'][0]:.4f}")
            if len(vol_cluster['autocorrelation']) >= 10:
                print(f"     ACF at lag 10: {vol_cluster['autocorrelation'][9]:.4f}")

        print(f"\n(iii) Leverage effect:")
        lev = leverage['leverage_correlation']
        print(f"      L(0) = {lev[0]:.4f}")
        if len(lev) > 1:
            print(f"      L(1) = {lev[1]:.4f}")
        neg_count = sum(1 for c in lev[:20] if c < 0)
        print(f"      Negative in first 20 lags: {neg_count}/20")
        print(f"      (Should be negative at small lags)")
        print("=" * 60)

    return results


# ============================================================================
# Plotting — matches paper figures (Figs 3, 5, 6, 7, 8)
# ============================================================================

def plot_stylized_facts(results, real_results=None, save_path=None):
    """Plot stylized facts comparison following paper figures.

    Layout: 3 panels (a) heavy-tail, (b) vol clustering, (c) leverage
    Matches the format in Figures 5-8 of arXiv:2507.19003.

    Args:
        results: dict from evaluate_stylized_facts (generated data)
        real_results: optional dict from evaluate_stylized_facts (real data)
        save_path: if provided, save figure to this path
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ----- (a) Heavy-tail distribution -----
    ax = axes[0]
    ht = results["heavy_tail"]
    if len(ht.get("log_x", [])) > 0:
        x_vals = 10 ** np.array(ht["log_x"])
        y_vals = 10 ** np.array(ht["log_density"])
        ax.plot(x_vals, y_vals, "b.",
                label=f"Generated (α={ht['alpha']:.2f})", markersize=4)
    if real_results and len(real_results["heavy_tail"].get("log_x", [])) > 0:
        rht = real_results["heavy_tail"]
        rx_vals = 10 ** np.array(rht["log_x"])
        ry_vals = 10 ** np.array(rht["log_density"])
        ax.plot(rx_vals, ry_vals, "r.",
                label=f"Real (α={rht['alpha']:.2f})", markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|r / σ|")
    ax.set_ylabel("Density")
    ax.set_title("(a) Heavy-tail distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ----- (b) Volatility clustering -----
    ax = axes[1]
    vc = results["volatility_clustering"]
    ax.plot(vc["lags"], vc["autocorrelation"], "b.",
            label=f"Generated (β={vc['beta']:.3f})", markersize=4)
    if real_results:
        rvc = real_results["volatility_clustering"]
        ax.plot(rvc["lags"], rvc["autocorrelation"], "r.",
                label=f"Real (β={rvc['beta']:.3f})", markersize=4)
    ax.set_xlabel("Lag k")
    ax.set_ylabel("Corr(|rₜ|, |rₜ₊ₖ|)")
    ax.set_title("(b) Volatility clustering")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ----- (c) Leverage effect -----
    ax = axes[2]
    le = results["leverage_effect"]
    ax.plot(le["lags"], le["leverage_correlation"], "b-",
            label="Generated", linewidth=2)
    if real_results:
        rle = real_results["leverage_effect"]
        ax.plot(rle["lags"], rle["leverage_correlation"], "r--",
                label="Real", linewidth=2)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("Lag k")
    ax.set_ylabel("L(k)")
    ax.set_title("(c) Leverage effect")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.close()


def plot_diagnostics(gen_data, real_data, mode="log_return", save_path=None):
    """Comprehensive 6-panel diagnostic figure.

    Row 1: Return distribution (bell curve) | QQ plot vs Normal | Sample paths
    Row 2: Heavy-tail (log-log) | Volatility clustering | Leverage effect

    Args:
        gen_data: (N, L) generated data
        real_data: (N, L) real data
        mode: 'log_return' or 'log_price'
        save_path: path to save figure
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("matplotlib/scipy not available")
        return

    gen_returns = compute_log_returns(gen_data, mode=mode)
    real_returns = compute_log_returns(real_data, mode=mode)

    gen_results = evaluate_stylized_facts(gen_data, mode=mode, display=False)
    real_results = evaluate_stylized_facts(real_data, mode=mode, display=False)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ---- Row 1, Col 1: Return distribution (bell curve) ----
    ax = axes[0, 0]
    gen_flat = gen_returns.flatten()
    real_flat = real_returns.flatten()

    # Use real data scale for histogram range
    clip = max(np.percentile(np.abs(real_flat), 99.5) * 3, 0.01)
    bins = np.linspace(-clip, clip, 100)

    ax.hist(real_flat, bins=bins, density=True, alpha=0.5, color="red",
            label=f"Real (std={real_flat.std():.4f})")
    ax.hist(gen_flat, bins=bins, density=True, alpha=0.5, color="blue",
            label=f"Generated (std={gen_flat.std():.4f})")

    # Overlay fitted normal for reference
    x_norm = np.linspace(-clip, clip, 200)
    ax.plot(x_norm, stats.norm.pdf(x_norm, 0, real_flat.std()),
            "r-", linewidth=1.5, alpha=0.7, label="Normal fit (real)")

    ax.set_xlabel("Log-return")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(-clip, clip)

    # ---- Row 1, Col 2: QQ plot vs theoretical Normal ----
    ax = axes[0, 1]
    gen_standardized = (gen_flat - gen_flat.mean()) / max(gen_flat.std(), 1e-10)
    real_standardized = (real_flat - real_flat.mean()) / max(real_flat.std(), 1e-10)

    n_quantiles = min(500, len(gen_standardized))
    probs = np.linspace(0.001, 0.999, n_quantiles)
    theoretical_q = stats.norm.ppf(probs)
    gen_q = np.quantile(gen_standardized, probs)
    real_q = np.quantile(real_standardized, probs)

    ax.plot(theoretical_q, real_q, "r.", markersize=2, alpha=0.5, label="Real")
    ax.plot(theoretical_q, gen_q, "b.", markersize=2, alpha=0.5, label="Generated")
    lim = max(abs(theoretical_q).max(), abs(gen_q).max(), abs(real_q).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.5,
            label="Perfect Normal")
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("QQ Plot vs Normal Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # ---- Row 1, Col 3: Sample paths ----
    ax = axes[0, 2]
    n_show = min(5, len(gen_data), len(real_data))
    for i in range(n_show):
        if mode == "log_price":
            ax.plot(gen_data[i], color="blue", alpha=0.5,
                    linewidth=0.8, label="Generated" if i == 0 else None)
            ax.plot(real_data[i], color="red", alpha=0.5,
                    linewidth=0.8, label="Real" if i == 0 else None)
        else:
            ax.plot(np.cumsum(gen_returns[i]), color="blue", alpha=0.5,
                    linewidth=0.8, label="Generated" if i == 0 else None)
            ax.plot(np.cumsum(real_returns[i]), color="red", alpha=0.5,
                    linewidth=0.8, label="Real" if i == 0 else None)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Log-price" if mode == "log_price" else "Cumulative return")
    ax.set_title("Sample Paths")
    ax.legend(fontsize=8)

    # ---- Row 2, Col 1: Heavy-tail (log-log) ----
    ax = axes[1, 0]
    ht_gen = gen_results["heavy_tail"]
    ht_real = real_results["heavy_tail"]
    if len(ht_gen.get("log_x", [])) > 0:
        x_vals = 10 ** np.array(ht_gen["log_x"])
        y_vals = 10 ** np.array(ht_gen["log_density"])
        ax.plot(x_vals, y_vals, "b.",
                label=f"Generated (α={ht_gen['alpha']:.2f})", markersize=4)
    if len(ht_real.get("log_x", [])) > 0:
        rx_vals = 10 ** np.array(ht_real["log_x"])
        ry_vals = 10 ** np.array(ht_real["log_density"])
        ax.plot(rx_vals, ry_vals, "r.",
                label=f"Real (α={ht_real['alpha']:.2f})", markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|r / σ|")
    ax.set_ylabel("Density")
    ax.set_title("(a) Heavy-tail distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Row 2, Col 2: Volatility clustering ----
    ax = axes[1, 1]
    vc_gen = gen_results["volatility_clustering"]
    vc_real = real_results["volatility_clustering"]
    ax.plot(vc_gen["lags"], vc_gen["autocorrelation"], "b.",
            label=f"Generated (β={vc_gen['beta']:.3f})", markersize=4)
    ax.plot(vc_real["lags"], vc_real["autocorrelation"], "r.",
            label=f"Real (β={vc_real['beta']:.3f})", markersize=4)
    ax.set_xlabel("Lag k")
    ax.set_ylabel("Corr(|rₜ|, |rₜ₊ₖ|)")
    ax.set_title("(b) Volatility clustering")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Row 2, Col 3: Leverage effect ----
    ax = axes[1, 2]
    le_gen = gen_results["leverage_effect"]
    le_real = real_results["leverage_effect"]
    ax.plot(le_gen["lags"], le_gen["leverage_correlation"], "b-",
            label="Generated", linewidth=2)
    ax.plot(le_real["lags"], le_real["leverage_correlation"], "r--",
            label="Real", linewidth=2)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("Lag k")
    ax.set_ylabel("L(k)")
    ax.set_title("(c) Leverage effect")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Diagnostics — Gen α={ht_gen['alpha']:.2f} vs Real α={ht_real['alpha']:.2f}  |  "
        f"Gen β={vc_gen['beta']:.3f} vs Real β={vc_real['beta']:.3f}",
        fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Diagnostics saved to {save_path}")
    plt.close()

    return gen_results, real_results


def plot_pathwise_diagnostics(gen_data, real_data, mode="log_return",
                              save_path=None, vol_window=20):
    """Plot pathwise diagnostics: 4×2 panel comparing real vs generated.

    Panels:
        (a) Rolling vol traces overlay (sample paths)
        (b) Rolling vol distribution (histogram)
        (c) Burst duration distribution (high-vol run lengths)
        (d) Drawdown depth distribution
        (e) Roughness: TV of returns (histogram)
        (f) Squared-return ACF with confidence bands
        (g) Sign-run length distribution
        (h) Regime persistence (high/low vol run lengths)

    Args:
        gen_data: (N, L) generated data
        real_data: (N, L) real data
        mode: 'log_return' or 'log_price'
        save_path: path to save figure
        vol_window: rolling vol window size
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return None, None

    gen_pw = compute_pathwise_diagnostics(gen_data, mode=mode, vol_window=vol_window)
    real_pw = compute_pathwise_diagnostics(real_data, mode=mode, vol_window=vol_window)

    # Print summary table
    print_pathwise_summary(gen_pw, real_pw)

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))

    # ---- (a) Rolling vol traces ----
    ax = axes[0, 0]
    n_show = min(5, len(real_pw["rolling_vol"]["traces"]),
                 len(gen_pw["rolling_vol"]["traces"]))
    for i in range(n_show):
        ax.plot(real_pw["rolling_vol"]["traces"][i], color="red",
                alpha=0.4, linewidth=0.8, label="Real" if i == 0 else None)
        ax.plot(gen_pw["rolling_vol"]["traces"][i], color="blue",
                alpha=0.4, linewidth=0.8, label="Gen" if i == 0 else None)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Rolling σ")
    ax.set_title(f"(a) Rolling Realized Volatility (window={vol_window})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (b) Rolling vol distribution ----
    ax = axes[0, 1]
    if real_pw["rolling_vol"]["traces"]:
        real_vols = np.concatenate(real_pw["rolling_vol"]["traces"])
        ax.hist(real_vols, bins=80, density=True, alpha=0.5, color="red", label="Real")
    if gen_pw["rolling_vol"]["traces"]:
        gen_vols = np.concatenate(gen_pw["rolling_vol"]["traces"])
        ax.hist(gen_vols, bins=80, density=True, alpha=0.5, color="blue", label="Gen")
    ax.set_xlabel("Rolling σ")
    ax.set_ylabel("Density")
    ax.set_title("(b) Rolling Vol Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (c) Burst duration ----
    ax = axes[1, 0]
    max_burst = 0
    if real_pw["burst_duration"]["run_lengths"]:
        rl_real = np.array(real_pw["burst_duration"]["run_lengths"])
        max_burst = max(max_burst, int(np.percentile(rl_real, 99)))
    if gen_pw["burst_duration"]["run_lengths"]:
        rl_gen = np.array(gen_pw["burst_duration"]["run_lengths"])
        max_burst = max(max_burst, int(np.percentile(rl_gen, 99)))
    max_burst = max(max_burst, 5)
    bins_burst = np.arange(0.5, max_burst + 1.5, 1)
    if real_pw["burst_duration"]["run_lengths"]:
        ax.hist(rl_real, bins=bins_burst, density=True, alpha=0.5, color="red",
                label=f"Real (μ={real_pw['burst_duration']['mean']:.1f})")
    if gen_pw["burst_duration"]["run_lengths"]:
        ax.hist(rl_gen, bins=bins_burst, density=True, alpha=0.5, color="blue",
                label=f"Gen (μ={gen_pw['burst_duration']['mean']:.1f})")
    ax.set_xlabel("Burst duration (steps)")
    ax.set_ylabel("Density")
    ax.set_title("(c) High-Vol Burst Duration (top decile)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (d) Drawdown depth ----
    ax = axes[1, 1]
    if real_pw["drawdown"]["depths"]:
        ax.hist(real_pw["drawdown"]["depths"], bins=40, density=True,
                alpha=0.5, color="red",
                label=f"Real (μ={real_pw['drawdown']['mean_depth']:.4f})")
    if gen_pw["drawdown"]["depths"]:
        ax.hist(gen_pw["drawdown"]["depths"], bins=40, density=True,
                alpha=0.5, color="blue",
                label=f"Gen (μ={gen_pw['drawdown']['mean_depth']:.4f})")
    ax.set_xlabel("Max drawdown (log-price)")
    ax.set_ylabel("Density")
    ax.set_title("(d) Drawdown Depth Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (e) Roughness: TV of returns ----
    ax = axes[2, 0]
    if real_pw["roughness"]["tv_returns"]:
        ax.hist(real_pw["roughness"]["tv_returns"], bins=40, density=True,
                alpha=0.5, color="red",
                label=f"Real (μ={real_pw['roughness']['mean_tv_returns']:.2f})")
    if gen_pw["roughness"]["tv_returns"]:
        ax.hist(gen_pw["roughness"]["tv_returns"], bins=40, density=True,
                alpha=0.5, color="blue",
                label=f"Gen (μ={gen_pw['roughness']['mean_tv_returns']:.2f})")
    ax.set_xlabel("Σ|rₜ - rₜ₋₁|  (total variation of returns)")
    ax.set_ylabel("Density")
    ax.set_title("(e) Path Roughness — Total Variation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (f) Squared-return ACF ----
    ax = axes[2, 1]
    sq_real = real_pw["squared_return_acf"]
    sq_gen = gen_pw["squared_return_acf"]
    ax.plot(sq_real["lags"], sq_real["acf"], "r.", markersize=3,
            label="Real r² ACF")
    ax.plot(sq_gen["lags"], sq_gen["acf"], "b.", markersize=3,
            label="Gen r² ACF")
    ax.axhline(sq_real["ci_upper"], color="red", ls="--", alpha=0.4,
               label="Real 95% CI")
    ax.axhline(sq_real["ci_lower"], color="red", ls="--", alpha=0.4)
    ax.axhline(sq_gen["ci_upper"], color="blue", ls="--", alpha=0.4,
               label="Gen 95% CI")
    ax.axhline(sq_gen["ci_lower"], color="blue", ls="--", alpha=0.4)
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xlabel("Lag k")
    ax.set_ylabel("ACF(r²)")
    ax.set_title("(f) Squared-Return Autocorrelation")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---- (g) Sign-run lengths ----
    ax = axes[3, 0]
    max_run = 15
    bins_sr = np.arange(0.5, max_run + 1.5, 1)
    if real_pw["sign_runs"]["pos_runs"]:
        all_real = real_pw["sign_runs"]["pos_runs"] + real_pw["sign_runs"]["neg_runs"]
        ax.hist(all_real, bins=bins_sr, density=True, alpha=0.5, color="red",
                label=f"Real (μ+={real_pw['sign_runs']['pos_mean']:.2f})")
    if gen_pw["sign_runs"]["pos_runs"]:
        all_gen = gen_pw["sign_runs"]["pos_runs"] + gen_pw["sign_runs"]["neg_runs"]
        ax.hist(all_gen, bins=bins_sr, density=True, alpha=0.5, color="blue",
                label=f"Gen (μ+={gen_pw['sign_runs']['pos_mean']:.2f})")
    ax.set_xlabel("Run length")
    ax.set_ylabel("Density")
    ax.set_title("(g) Sign-Run Length Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max_run + 0.5)

    # ---- (h) Regime persistence ----
    ax = axes[3, 1]
    max_reg = 30
    bins_reg = np.arange(0.5, max_reg + 1.5, 1)
    if real_pw["regime_persistence"]["high_runs"]:
        ax.hist(real_pw["regime_persistence"]["high_runs"], bins=bins_reg,
                density=True, alpha=0.5, color="red",
                label=f"Real hi-vol (μ={real_pw['regime_persistence']['high_mean']:.1f})")
    if gen_pw["regime_persistence"]["high_runs"]:
        ax.hist(gen_pw["regime_persistence"]["high_runs"], bins=bins_reg,
                density=True, alpha=0.5, color="blue",
                label=f"Gen hi-vol (μ={gen_pw['regime_persistence']['high_mean']:.1f})")
    ax.set_xlabel("Regime duration (steps)")
    ax.set_ylabel("Density")
    ax.set_title("(h) High-Vol Regime Persistence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Pathwise Diagnostics — Audit D Expansion", fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Pathwise diagnostics saved to {save_path}")
    plt.close()

    return gen_pw, real_pw

# ============================================================================
# Cross-sectional mean path diagnostic (z-score diagnosis tool)
# ============================================================================

def compute_cross_sectional_stats(data):
    """Compute cross-sectional mean path and percentile envelopes.

    For a batch of (N, L) paths, compute at each time step t:
      - mean(t) = (1/N) Σᵢ xᵢ(t)
      - std(t)
      - 10th/90th percentile envelope

    If paths are mean-reverting (z-score artefact), the cross-sectional std
    will be much smaller than expected from independent random walks.

    Args:
        data: (N, L) array of paths (log-price or cumulative log-returns)

    Returns:
        dict with keys: mean_path, std_path, p10, p90, terminal_values
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    mean_path = np.mean(data, axis=0)
    std_path = np.std(data, axis=0)
    p10 = np.percentile(data, 10, axis=0)
    p90 = np.percentile(data, 90, axis=0)
    terminal_values = data[:, -1]

    return {
        "mean_path": mean_path,
        "std_path": std_path,
        "p10": p10,
        "p90": p90,
        "terminal_values": terminal_values,
        "n_paths": data.shape[0],
        "path_length": data.shape[1],
    }


def plot_mean_path_diagnostic(gen_data, real_data, mode="log_price",
                              save_path=None, title_suffix=""):
    """Plot cross-sectional mean path diagnostic — the key z-score test.

    Two panels:
      Left:  Mean path ± 1σ envelope for real vs generated
      Right: Terminal value distribution (histogram)

    If the model learned a mean-reverting score (z-score artefact):
      - Generated mean path will hug the real mean path too closely
      - Generated terminal value distribution will be much narrower
      - Generated σ envelope will be compressed

    After fix (no z-score): generated paths should fan out similarly to real.

    Args:
        gen_data: (N, L) generated paths
        real_data: (N, L) real paths
        mode: 'log_price' or 'log_return'
        save_path: path to save figure (optional)
        title_suffix: extra text for title (e.g., "z-scored model" vs "raw model")
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return None

    gen_stats = compute_cross_sectional_stats(gen_data)
    real_stats = compute_cross_sectional_stats(real_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t = np.arange(gen_stats["path_length"])

    # ---- Left: Mean path ± σ envelope ----
    ax = axes[0]
    ax.plot(t, real_stats["mean_path"], color="red", linewidth=1.5,
            label=f"Real mean (N={real_stats['n_paths']})")
    ax.fill_between(t,
                    real_stats["mean_path"] - real_stats["std_path"],
                    real_stats["mean_path"] + real_stats["std_path"],
                    color="red", alpha=0.15, label="Real ±1σ")
    ax.fill_between(t, real_stats["p10"], real_stats["p90"],
                    color="red", alpha=0.08, label="Real 10–90%")

    ax.plot(t, gen_stats["mean_path"], color="blue", linewidth=1.5,
            label=f"Gen mean (N={gen_stats['n_paths']})")
    ax.fill_between(t,
                    gen_stats["mean_path"] - gen_stats["std_path"],
                    gen_stats["mean_path"] + gen_stats["std_path"],
                    color="blue", alpha=0.15, label="Gen ±1σ")
    ax.fill_between(t, gen_stats["p10"], gen_stats["p90"],
                    color="blue", alpha=0.08, label="Gen 10–90%")

    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative log-price" if mode == "log_price" else "Value")
    ax.set_title("Cross-Sectional Mean Path ± Spread")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Report path std ratio at terminal step
    real_term_std = real_stats["std_path"][-1]
    gen_term_std = gen_stats["std_path"][-1]
    ratio = gen_term_std / real_term_std if real_term_std > 0 else float("nan")
    ax.text(0.02, 0.02,
            f"Terminal σ ratio: {ratio:.2f}  (gen/real)\n"
            f"Real σ(T)={real_term_std:.4f}  Gen σ(T)={gen_term_std:.4f}",
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # ---- Right: Terminal value distribution ----
    ax = axes[1]
    bins = np.linspace(
        min(real_stats["terminal_values"].min(), gen_stats["terminal_values"].min()),
        max(real_stats["terminal_values"].max(), gen_stats["terminal_values"].max()),
        50,
    )
    ax.hist(real_stats["terminal_values"], bins=bins, density=True,
            alpha=0.5, color="red", label=f"Real (μ={np.mean(real_stats['terminal_values']):.3f})")
    ax.hist(gen_stats["terminal_values"], bins=bins, density=True,
            alpha=0.5, color="blue", label=f"Gen (μ={np.mean(gen_stats['terminal_values']):.3f})")
    ax.set_xlabel("Terminal value x(T)")
    ax.set_ylabel("Density")
    ax.set_title("Terminal Value Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    suptitle = "Cross-Sectional Mean Path Diagnostic"
    if title_suffix:
        suptitle += f" — {title_suffix}"
    plt.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Mean path diagnostic saved to {save_path}")
    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print("Cross-Sectional Mean Path Summary")
    print(f"{'='*60}")
    print(f"  Real: N={real_stats['n_paths']}, L={real_stats['path_length']}")
    print(f"  Gen:  N={gen_stats['n_paths']}, L={gen_stats['path_length']}")
    print(f"  Terminal σ — Real: {real_term_std:.4f}  Gen: {gen_term_std:.4f}  "
          f"Ratio: {ratio:.2f}")
    print(f"  Terminal μ — Real: {np.mean(real_stats['terminal_values']):.4f}  "
          f"Gen: {np.mean(gen_stats['terminal_values']):.4f}")
    if ratio < 0.5:
        print("  ⚠ WARNING: Generated path spread is <50% of real → "
              "likely mean-reverting (z-score artefact)")
    elif ratio > 1.5:
        print("  ⚠ WARNING: Generated path spread is >150% of real → "
              "possible instability")
    else:
        print("  ✓ Terminal spread ratio near 1.0 — paths fan out correctly")
    print(f"{'='*60}\n")

    return gen_stats, real_stats