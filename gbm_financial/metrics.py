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
