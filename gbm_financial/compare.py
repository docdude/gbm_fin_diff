"""Model-agnostic comparison harness for synthetic financial return windows.

Scores any generator (this repo, vnhtg01/gbm-diffusion-finance-ts, GANs,
bootstrap/GARCH baselines) against the same real reference with identical,
within-window metrics. Generated windows are independent, so every statistic
is computed inside a window — never across concatenated window boundaries.

Inputs are return windows of shape (N, L) in the SAME units as the reference
(use `to_returns` for anchored log-price paths).

CLI:
    python -m gbm_financial.compare \\
        --real save/cmp/real_train_logprice.npy --holdout save/cmp/real_val_logprice.npy \\
        --input log_price \\
        --gen ours=run/seed0.npy,run/seed1.npy,run/seed2.npy \\
        --baselines --out save/cmp/report
"""

import argparse
import json
import os

import numpy as np
from scipy import stats

from .metrics import compute_heavy_tail_exponent, compute_volatility_clustering


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def to_returns(x, mode):
    """(N, L) log-price paths → (N, L-1) returns; log-returns pass through."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(-1)
    return np.diff(x, axis=-1) if mode == "log_price" else x


def _acf_windows(x, lags):
    """Mean within-window autocorrelation (each window demeaned/scaled)."""
    xc = x - x.mean(axis=1, keepdims=True)
    var = (xc ** 2).mean(axis=1)
    ok = var > 1e-20
    xc, var = xc[ok], var[ok]
    return np.array([np.mean((xc[:, :-k] * xc[:, k:]).mean(axis=1) / var) for k in lags])


def _leverage(r, lags):
    """Bouchaud et al. (2001) L(k) = E[r_t r_{t+k}²] / E[r²]², averaged per window."""
    m2 = (r ** 2).mean(axis=1)
    ok = m2 > 1e-20
    r, m2 = r[ok], m2[ok]
    return np.array([np.mean((r[:, :-k] * r[:, k:] ** 2).mean(axis=1) / m2 ** 2) for k in lags])


def _max_drawdown(r):
    p = np.cumsum(r, axis=1)
    p = np.concatenate([np.zeros((len(p), 1)), p], axis=1)
    return (p - np.maximum.accumulate(p, axis=1)).min(axis=1)


def _window_features(r):
    """Per-window summary features for the discriminative test."""
    a = np.abs(r)
    feats = [r.std(1), stats.skew(r, axis=1), stats.kurtosis(r, axis=1),
             np.abs(r).max(1) / (r.std(1) + 1e-12), _max_drawdown(r), r.sum(1)]
    for k in (1, 5, 20):
        if r.shape[1] > k + 2:
            ac = (lambda z: ((z[:, :-k] - z.mean(1, keepdims=True)) *
                             (z[:, k:] - z.mean(1, keepdims=True))).mean(1) /
                  (z.var(1) + 1e-20))
            feats += [ac(a), ac(r)]
    half = r.shape[1] // 2
    feats.append(np.log((r[:, half:].std(1) + 1e-12) / (r[:, :half].std(1) + 1e-12)))
    return np.nan_to_num(np.column_stack(feats))


def _nn_distance(query, ref, max_query=200, rng=None):
    """Mean Euclidean nearest-neighbour distance from query paths to ref paths."""
    rng = rng or np.random.default_rng(0)
    if len(query) > max_query:
        query = query[rng.choice(len(query), max_query, replace=False)]
    ref_sq = (ref ** 2).sum(1)
    d = []
    for q in query:
        dist2 = ref_sq - 2 * ref @ q + (q ** 2).sum()
        d.append(np.sqrt(max(dist2.min(), 0.0)))
    return float(np.mean(d))


def shift_nn_ratio(gen_r, history_r, holdout_r, w=126, stride=21, max_query=200, seed=0):
    """Shift-invariant memorisation test on return sequences.

    Every length-w segment (step ``stride``) of a generated or held-out path is
    compared with EVERY length-w segment (any offset) of the training return
    history ``history_r`` (1-D or list of 1-D series), after z-scoring each
    segment, so a replay of real history shifted by a few days or rescaled is
    caught. Returns (median gen→history distance, median holdout→history
    distance, ratio); ratio ≪ 1 suggests memorisation."""
    rng = np.random.default_rng(seed)

    def segs(paths):
        out = [p[i:i + w] for p in paths for i in range(0, len(p) - w + 1, stride)]
        return np.asarray(out, np.float64)

    def z(a):
        a = a - a.mean(1, keepdims=True)
        return a / np.maximum(a.std(1, keepdims=True), 1e-12)

    hist = [np.asarray(h, np.float64) for h in (history_r if isinstance(history_r, (list, tuple))
                                                  else [history_r])]
    H = z(np.concatenate([np.lib.stride_tricks.sliding_window_view(h, w) for h in hist if len(h) >= w]))

    def med(paths):
        Q = z(segs(np.asarray(paths, np.float64)))
        if len(Q) > max_query:
            Q = Q[rng.choice(len(Q), max_query, replace=False)]
        # ‖q − h‖² = 2w(1 − corr) for z-scored segments
        corr = Q @ H.T / w
        return float(np.median(np.sqrt(np.maximum(2 * w * (1 - corr.max(1)), 0.0))))

    d_gen, d_hold = med(gen_r), med(holdout_r)
    return d_gen, d_hold, (d_gen / d_hold if d_hold > 0 else float("nan"))


def discriminative_auc(real_r, gen_r, seed=0):
    """5-fold CV ROC-AUC of a gradient-boosted classifier on window features.

    0.5 = indistinguishable; 1.0 = trivially separable. Returns NaN if
    scikit-learn is unavailable or there are fewer than 40 windows per class
    (too few to judge; a classifier that cannot split would report 0.5).
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = min(len(real_r), len(gen_r))
    if n < 40:
        return float("nan")
    xr = _window_features(real_r[rng.choice(len(real_r), n, replace=False)])
    xg = _window_features(gen_r[rng.choice(len(gen_r), n, replace=False)])
    X = np.vstack([xr, xg])
    y = np.r_[np.ones(n), np.zeros(n)]
    clf = HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=max(5, n // 20),
                                         random_state=seed)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compare_returns(gen_r, real_r, holdout_r=None, seed=0):
    """Score generated return windows against a real reference.

    Args:
        gen_r, real_r: (N, L) return windows (same units, same L)
        holdout_r: optional held-out real windows (e.g. val tickers). Used as
            the reference for the memorisation ratio and as a "real vs real"
            sanity row.
    Returns:
        dict of scalar metrics (gen value, plus *_real reference values)
    """
    gen_r = np.asarray(gen_r, np.float64)
    real_r = np.asarray(real_r, np.float64)
    L = min(gen_r.shape[1], real_r.shape[1])
    gen_r, real_r = gen_r[:, :L], real_r[:, :L]
    max_lag = max(2, min(100, L // 4))
    short = range(1, min(11, L - 1))
    long_ = range(1, max_lag + 1)

    g, rr = gen_r.ravel(), real_r.ravel()
    out = {"n_windows": int(len(gen_r)), "seq_len_returns": int(L)}

    # Marginal distribution
    out["std_ratio"] = float(g.std() / rr.std())
    out["mean_daily"] = float(g.mean())
    out["mean_daily_real"] = float(rr.mean())
    out["kurtosis"] = float(stats.kurtosis(g))
    out["kurtosis_real"] = float(stats.kurtosis(rr))
    out["skew"] = float(stats.skew(g))
    out["skew_real"] = float(stats.skew(rr))
    out["alpha_top5"] = float(compute_heavy_tail_exponent(g)["alpha"])
    out["alpha_top5_real"] = float(compute_heavy_tail_exponent(rr)["alpha"])
    out["ks_stat"] = float(stats.ks_2samp(g, rr).statistic)
    out["ks_stat_standardized"] = float(stats.ks_2samp(g / g.std(), rr / rr.std()).statistic)
    out["wasserstein_over_std"] = float(stats.wasserstein_distance(g, rr) / rr.std())

    # Temporal structure (within window)
    acf_r_g, acf_r_r = _acf_windows(gen_r, short), _acf_windows(real_r, short)
    acf_a_g, acf_a_r = _acf_windows(np.abs(gen_r), long_), _acf_windows(np.abs(real_r), long_)
    out["acf_r_lag1"] = float(acf_r_g[0])
    out["acf_r_lag1_real"] = float(acf_r_r[0])
    out["acf_r_rmse_1_10"] = float(np.sqrt(np.mean((acf_r_g - acf_r_r) ** 2)))
    out["acf_abs_1_10"] = float(acf_a_g[:10].mean())
    out["acf_abs_1_10_real"] = float(acf_a_r[:10].mean())
    out["acf_abs_rmse_1_%d" % max_lag] = float(np.sqrt(np.mean((acf_a_g - acf_a_r) ** 2)))
    out["vol_cluster_beta"] = float(compute_volatility_clustering(gen_r, max_lag)["beta"])
    out["vol_cluster_beta_real"] = float(compute_volatility_clustering(real_r, max_lag)["beta"])
    lev_g, lev_r = _leverage(gen_r, short), _leverage(real_r, short)
    out["leverage_1_10"] = float(lev_g.mean())
    out["leverage_1_10_real"] = float(lev_r.mean())

    # Regime / path structure
    vol_g, vol_r = gen_r.std(1), real_r.std(1)
    out["vol_dispersion"] = float(vol_g.std() / vol_g.mean())
    out["vol_dispersion_real"] = float(vol_r.std() / vol_r.mean())
    out["terminal_std_ratio"] = float(gen_r.sum(1).std() / real_r.sum(1).std())
    out["terminal_mean"] = float(gen_r.sum(1).mean())
    out["terminal_mean_real"] = float(real_r.sum(1).mean())
    out["max_drawdown_ratio"] = float(_max_drawdown(gen_r).mean() / _max_drawdown(real_r).mean())

    # Fidelity / memorisation
    out["disc_auc"] = discriminative_auc(real_r, gen_r, seed=seed)
    gp, rp = np.cumsum(gen_r, 1), np.cumsum(real_r, 1)
    rng = np.random.default_rng(seed)
    nn_gen = _nn_distance(gp, rp, rng=rng)
    out["nn_dist_gen_to_train"] = nn_gen
    if holdout_r is not None:
        hp = np.cumsum(np.asarray(holdout_r, np.float64)[:, :L], 1)
        nn_hold = _nn_distance(hp, rp, rng=rng)
        out["nn_dist_holdout_to_train"] = nn_hold
        # <1 → generated paths sit closer to training paths than unseen real
        # paths do (memorisation); ≈1 healthy; >1 off-manifold.
        out["memorization_ratio"] = float(nn_gen / nn_hold) if nn_hold > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def block_bootstrap(real_r, n, L, block=20, seed=0):
    """Concatenate random length-`block` blocks drawn from real windows."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, L))
    n_blocks = int(np.ceil(L / block))
    for i in range(n):
        rows = rng.integers(0, len(real_r), n_blocks)
        starts = rng.integers(0, real_r.shape[1] - block + 1, n_blocks)
        out[i] = np.concatenate([real_r[r, s:s + block] for r, s in zip(rows, starts)])[:L]
    return out


def iid_shuffle(real_r, n, L, seed=0):
    """i.i.d. draws from the pooled real returns (right marginal, no dynamics)."""
    rng = np.random.default_rng(seed)
    return rng.choice(real_r.ravel(), size=(n, L))


def garch_t(real_r, n, L, seed=0, burn=500):
    """GARCH(1,1) with Student-t innovations fitted on real windows (needs `arch`)."""
    try:
        from arch import arch_model
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    sample = real_r[rng.choice(len(real_r), min(len(real_r), 200), replace=False)]
    scale = 100.0
    res = arch_model(sample.ravel() * scale, mean="Constant", vol="GARCH",
                     p=1, q=1, dist="t").fit(disp="off")
    sims = np.empty((n, L))
    for i in range(n):
        sim = res.model.simulate(res.params, nobs=L, burn=burn,
                                 initial_value=None)
        sims[i] = sim["data"].values / scale
    return sims


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def aggregate(runs):
    """Mean and std across seeds for each scalar metric."""
    keys = runs[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in runs if isinstance(r.get(k), (int, float))], float)
        if len(vals):
            agg[k] = {"mean": float(np.nanmean(vals)),
                      "std": float(np.nanstd(vals)) if len(vals) > 1 else 0.0,
                      "n": int(len(vals))}
    return agg


REPORT_COLUMNS = [
    ("std_ratio", "std×", 1.0),
    ("kurtosis", "kurt", "kurtosis_real"),
    ("alpha_top5", "α5%", "alpha_top5_real"),
    ("ks_stat_standardized", "KS(z)", 0.0),
    ("acf_r_lag1", "ACF r₁", "acf_r_lag1_real"),
    ("acf_abs_1_10", "ACF|r|₁₋₁₀", "acf_abs_1_10_real"),
    ("vol_cluster_beta", "β", "vol_cluster_beta_real"),
    ("leverage_1_10", "lev₁₋₁₀", "leverage_1_10_real"),
    ("vol_dispersion", "volDisp", "vol_dispersion_real"),
    ("terminal_std_ratio", "termσ×", 1.0),
    ("max_drawdown_ratio", "MDD×", 1.0),
    ("disc_auc", "discAUC", 0.5),
    ("memorization_ratio", "NN×", 1.0),
]


def format_report(results):
    """Markdown table: one row per model, mean±std over seeds, target row first."""
    first = next(iter(results.values()))
    header = "| model | " + " | ".join(c[1] for c in REPORT_COLUMNS) + " |"
    sep = "|---|" + "---|" * len(REPORT_COLUMNS)
    target = []
    for key, _, ref in REPORT_COLUMNS:
        if isinstance(ref, str):
            target.append(f"{first[ref]['mean']:.3g}" if ref in first else "–")
        else:
            target.append(f"{ref:g}")
    lines = [header, sep, "| **target (real)** | " + " | ".join(target) + " |"]
    for name, agg in results.items():
        cells = []
        for key, _, _ in REPORT_COLUMNS:
            if key in agg:
                m, s = agg[key]["mean"], agg[key]["std"]
                cells.append(f"{m:.3g}" + (f"±{s:.2g}" if agg[key]["n"] > 1 else ""))
            else:
                cells.append("–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_comparison(real_r, gen_sets, holdout_r=None, baselines=True,
                   n_baseline=120, n_baseline_seeds=3):
    """Evaluate {name: [returns_seed0, returns_seed1, ...]} plus baselines."""
    L = real_r.shape[1]
    sets = dict(gen_sets)
    if holdout_r is not None:
        sets = {"real held-out tickers": [holdout_r[:, :L]], **sets}
    if baselines:
        seeds = range(n_baseline_seeds)
        sets["baseline: iid shuffle"] = [iid_shuffle(real_r, n_baseline, L, s) for s in seeds]
        sets["baseline: block bootstrap(20)"] = [block_bootstrap(real_r, n_baseline, L, 20, s)
                                                 for s in seeds]
        g = [garch_t(real_r, n_baseline, L, s) for s in seeds]
        if g[0] is not None:
            sets["baseline: GARCH(1,1)-t"] = g
    results = {}
    for name, runs in sets.items():
        per_seed = [compare_returns(r, real_r, holdout_r, seed=i) for i, r in enumerate(runs)]
        results[name] = aggregate(per_seed)
        print(f"  scored {name}: {len(runs)} run(s)")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help=".npy of real windows (training reference)")
    ap.add_argument("--holdout", default=None, help=".npy of held-out real windows")
    ap.add_argument("--input", default="log_price", choices=["log_price", "log_return"],
                    help="representation of --real/--holdout and default for --gen")
    ap.add_argument("--gen", nargs="*", default=[],
                    help="name=path1,path2[:mode] — one .npy per seed; optional "
                         ":log_price/:log_return suffix overrides --input")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--n-baseline", type=int, default=120)
    ap.add_argument("--out", default="save/comparison/report")
    args = ap.parse_args()

    real_r = to_returns(np.load(args.real), args.input)
    hold_r = to_returns(np.load(args.holdout), args.input) if args.holdout else None
    gen_sets = {}
    for spec in args.gen:
        name, rest = spec.split("=", 1)
        mode = args.input
        if rest.endswith(":log_price") or rest.endswith(":log_return"):
            rest, mode = rest.rsplit(":", 1)
        gen_sets[name] = [to_returns(np.load(p), mode) for p in rest.split(",")]

    results = run_comparison(real_r, gen_sets, hold_r, args.baselines, args.n_baseline)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(results, f, indent=2)
    table = format_report(results)
    with open(args.out + ".md", "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nSaved {args.out}.json and {args.out}.md")


if __name__ == "__main__":
    main()
