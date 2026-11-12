"""Panel-level (cross-sectional) evaluation of synthetic multi-asset OHLCV scenarios.

Compares scenario panels with real 512-day panels of the same tickers drawn from
the train period, plus two baselines:
  * shuffled-date panels   each ticker's window from an independent random date
                           (right marginals, no cross-sectional dependence)
  * block bootstrap         20-day blocks of real *days* (keeps the cross-section)

Metrics (averaged over panels):
  mean pairwise return correlation, PC1 variance share, dispersion of betas to
  the equal-weight market, cross-sectional dispersion level and its ACF (dispersion
  clustering), turbulence (Mahalanobis vs train covariance) median / p99,
  pooled per-ticker kurtosis and ACF(|r|) lags 1-10, corr(|r|, log range),
  corr(|r|, log volume), market drawdown.

    python -m gbm_financial.panel_eval --scenarios save/panel/scenarios
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from .panel_data import PanelData

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _acf(x, lags):
    x = x - x.mean()
    v = (x ** 2).mean()
    return np.array([(x[:-k] * x[k:]).mean() / v for k in lags]) if v > 0 else np.zeros(len(lags))


def panel_stats(R, HL, V, inv_cov):
    """R, HL, V: (T, N) returns, log(H/L), log volume (NaN allowed for missing tickers)."""
    keep = np.isfinite(R).all(0) & np.isfinite(HL).all(0) & np.isfinite(V).all(0)
    R, HL, V = R[:, keep], HL[:, keep], V[:, keep]
    if R.shape[1] < 5:
        return None
    C = np.corrcoef(R.T)
    iu = np.triu_indices_from(C, 1)
    ev = np.linalg.eigvalsh(C)
    mkt = R.mean(1)
    beta = (R - R.mean(0)).T @ (mkt - mkt.mean()) / ((mkt - mkt.mean()) ** 2).sum()
    disp = R.std(1)
    turb = np.einsum("ti,ij,tj->t", R - R.mean(0), inv_cov[np.ix_(keep, keep)], R - R.mean(0))
    absr = np.abs(R)
    corr_rng = np.mean([np.corrcoef(absr[:, j], HL[:, j])[0, 1] for j in range(R.shape[1])])
    dv = V - V.mean(0)
    corr_vol = np.mean([np.corrcoef(absr[:, j], dv[:, j])[0, 1] for j in range(R.shape[1])])
    z = (R - R.mean(0)) / R.std(0)
    acf_abs = np.mean([_acf(absr[:, j], range(1, 11)).mean() for j in range(R.shape[1])])
    cum = np.cumsum(mkt)
    return {
        "mean_pair_corr": float(C[iu].mean()),
        "pc1_share": float(ev[-1] / ev.sum()),
        "beta_dispersion": float(beta.std()),
        "dispersion": float(disp.mean()),
        "dispersion_acf1_10": float(_acf(disp, range(1, 11)).mean()),
        "turbulence_median": float(np.median(turb)),
        "turbulence_p99": float(np.percentile(turb, 99)),
        "kurtosis_pooled": float(stats.kurtosis(z.ravel())),
        "acf_absret_1_10": float(acf_abs),
        "corr_absret_range": float(corr_rng),
        "corr_absret_logvol": float(corr_vol),
        "market_max_drawdown": float((cum - np.maximum.accumulate(cum)).min()),
        "market_vol": float(mkt.std()),
        "n_tickers": int(R.shape[1]),
    }


def arrays_from_df(df, tickers):
    w = {k: df.pivot(index="date", columns="tic", values=k).reindex(columns=tickers)
         for k in ("close", "high", "low", "volume")}
    c = w["close"].to_numpy(float)
    R = np.log(c[1:] / c[:-1])
    HL = np.log(w["high"].to_numpy(float) / w["low"].to_numpy(float))[1:]
    V = np.log(np.maximum(w["volume"].to_numpy(float), 1.0))[1:]
    return R, HL, V


def real_panel(P, tickers, s, L):
    J = [P.tickers.index(t) for t in tickers]
    sl = slice(s + 1, s + L)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (P.ret[sl][:, J], np.log(P.high[sl][:, J] / P.low[sl][:, J]),
                np.log(np.maximum(P.volume[sl][:, J], 1.0)))


def summarize(rows):
    rows = [r for r in rows if r]
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--n_real", type=int, default=40)
    ap.add_argument("--real_start", default="2010-03-30")
    ap.add_argument("--ohlcv", default=os.path.join(ROOT, "data/nas100/tiingo_ohlcv.csv.gz"))
    ap.add_argument("--vix", default=os.path.join(ROOT, "data/nas100/vix.csv"))
    ap.add_argument("--end_date", default="2024-05-31")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = PanelData(args.ohlcv, args.vix, end_date=args.end_date)
    files = sorted(glob.glob(os.path.join(args.scenarios, "scenario_*[0-9].csv.gz")))
    first = pd.read_csv(files[0])
    tickers = sorted(first.tic.unique())
    L = first.date.nunique()
    J = [P.tickers.index(t) for t in tickers]
    rng = np.random.default_rng(args.seed)

    lo = int(np.searchsorted(P.dates, args.real_start))
    full = P.ret[lo:][:, J]
    okc = np.isfinite(full).all(0)
    cov = np.cov(np.nan_to_num(full).T)
    inv_cov = np.linalg.pinv(cov)

    res = {"synthetic": [], "real": [], "shuffled-date": [], "block bootstrap(20d)": []}
    for f in files:
        R, HL, V = arrays_from_df(pd.read_csv(f), tickers)
        res["synthetic"].append(panel_stats(R, HL, V, inv_cov))
    starts = rng.integers(lo, len(P.dates) - L, args.n_real)
    for s in starts:
        res["real"].append(panel_stats(*real_panel(P, tickers, s, L), inv_cov))
        # shuffled dates: each ticker from its own random window
        parts = [real_panel(P, [t], rng.integers(lo, len(P.dates) - L), L) for t in tickers]
        res["shuffled-date"].append(panel_stats(*[np.concatenate([p[k] for p in parts], 1)
                                                  for k in range(3)], inv_cov))
        # block bootstrap of whole days
        idx = np.concatenate([np.arange(b, b + 20) for b in rng.integers(lo + 1, len(P.dates) - 20,
                                                                          L // 20 + 1)])[:L - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            res["block bootstrap(20d)"].append(panel_stats(
                P.ret[idx][:, J], np.log(P.high[idx][:, J] / P.low[idx][:, J]),
                np.log(np.maximum(P.volume[idx][:, J], 1.0)), inv_cov))
    out = {k: summarize(v) for k, v in res.items()}
    out["per_scenario"] = {os.path.basename(f): r for f, r in zip(files, res["synthetic"])}
    with open(os.path.join(args.scenarios, "panel_eval.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    keys = list(out["real"].keys())
    print(f"{'metric':<22}" + "".join(f"{k[:20]:>22}" for k in res))
    for m in keys:
        print(f"{m:<22}" + "".join(f"{out[k][m]['mean']:>14.4g}±{out[k][m]['std']:<7.2g}" for k in res))
    print(f"\n{len(files)} synthetic panels, {args.n_real} real/baseline panels, L={L}, "
          f"{len(tickers)} tickers ({int(okc.sum())} with complete data since {args.real_start})")


if __name__ == "__main__":
    main()
