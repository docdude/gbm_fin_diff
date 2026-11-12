#!/usr/bin/env python3
"""Characterise synthetic scenarios by market regime (bull / bear / sideways / crash).

Works on the env configs (``scenario_*.env.pkl``), i.e. exactly the price paths the
DRL agent trades, and compares the bank with every real window of the same length
from the training data.

Per episode (equal-weight index of the scenario's tickers):
  ann_return, ann_vol, max_drawdown, worst_20d (worst 20-day return), mean VIX
Label (whole episode):
  crash     worst_20d <= -15%  (fast drop; may recover, e.g. 2020)
  bear      ann_return <= -10%
  bull      ann_return >= +15%
  sideways  otherwise
Day shares: rolling 63-day return > +5% bull, < -5% bear, else sideways.

    python scripts/characterize_scenarios.py --scenarios save/panel/scenarios [--plot]
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CRASH_20D, BEAR_ANN, BULL_ANN, STATE_63D = -0.15, -0.10, 0.15, 0.05


def characterize(index, vix=None):
    r = np.diff(np.log(index))
    n = len(r)
    cum = np.cumsum(r)
    c0 = np.r_[0.0, cum]
    mdd = float(np.exp((c0 - np.maximum.accumulate(c0)).min()) - 1)
    w20 = float(np.exp(min(cum[i + 20] - cum[i] for i in range(n - 20))) - 1) if n > 21 else np.nan
    ann = float(np.exp(cum[-1] * 252 / n) - 1)
    lab = ("crash" if w20 <= CRASH_20D else "bear" if ann <= BEAR_ANN
           else "bull" if ann >= BULL_ANN else "sideways")
    r63 = np.exp(cum[63:] - cum[:-63]) - 1 if n > 63 else np.array([0.0])
    return {"label": lab, "ann_return": ann, "ann_vol": float(r.std() * np.sqrt(252)),
            "max_drawdown": mdd, "worst_20d": w20,
            "vix_mean": float(np.nanmean(vix)) if vix is not None else np.nan,
            "days_bull": float((r63 > STATE_63D).mean()), "days_bear": float((r63 < -STATE_63D).mean()),
            "days_sideways": float((np.abs(r63) <= STATE_63D).mean())}


def ew_index(price):
    p = np.asarray(price, float)
    r = np.nanmean(np.diff(np.log(p), axis=0), axis=1)
    return np.exp(np.r_[0.0, np.cumsum(r)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--ohlcv", default=os.path.join(ROOT, "data/nas100/tiingo_ohlcv.csv.gz"))
    ap.add_argument("--vix", default=os.path.join(ROOT, "data/nas100/vix.csv"))
    ap.add_argument("--real_start", default="2010-03-30")
    ap.add_argument("--end_date", default="2024-05-31")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.scenarios, "scenario_*.env.pkl")))
    rows, paths = [], []
    for f in files:
        with open(f, "rb") as fh:
            c = pickle.load(fh)
        idx = ew_index(c["price_array"])
        mk = f.replace(".env.pkl", "_market.csv")
        vix = None
        if os.path.exists(mk):
            m = pd.read_csv(mk).set_index("date").reindex(np.asarray(c["date_list"]).astype(str))
            vix = m["vix"].to_numpy(float)
        rows.append({"scenario": os.path.basename(f)[:12], **characterize(idx, vix)})
        paths.append(idx)
    syn = pd.DataFrame(rows).set_index("scenario")
    L = len(paths[0])

    from gbm_financial.panel_data import PanelData
    P = PanelData(a.ohlcv, a.vix, end_date=a.end_date)
    lo = int(np.searchsorted(P.dates, a.real_start))
    real, rpaths = [], []
    for s in range(lo, len(P.dates) - L + 1, 5):
        idx = np.exp(np.r_[0.0, np.cumsum(P.mret[s + 1:s + L])])
        real.append({"start": P.dates[s], **characterize(idx, P.vix[s:s + L])})
        rpaths.append(idx)
    real = pd.DataFrame(real)

    pd.set_option("display.width", 200)
    num = ["ann_return", "ann_vol", "max_drawdown", "worst_20d", "vix_mean", "days_bull", "days_bear", "days_sideways"]
    print(f"Synthetic scenarios ({len(syn)}, {L - 1}-day episodes):")
    print(syn.round(3).to_string())
    mix = pd.DataFrame({"synthetic": syn.label.value_counts(normalize=True),
                        f"real {a.real_start[:4]}+ windows": real.label.value_counts(normalize=True)}).fillna(0)
    print("\nRegime mix:"); print(mix.round(2).to_string())
    print("\nMeans:"); print(pd.DataFrame({"synthetic": syn[num].mean(), "real": real[num].mean()}).round(3).to_string())
    out = {"synthetic": syn.reset_index().to_dict("records"), "mix": mix.to_dict(),
           "means": {"synthetic": syn[num].mean().to_dict(), "real": real[num].mean().to_dict()}}
    with open(os.path.join(a.scenarios, "regimes.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"bull": "tab:green", "bear": "tab:red", "sideways": "tab:gray", "crash": "tab:purple"}
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        R = np.array(rpaths)
        for q, al in ((5, .15), (25, .25)):
            ax[0].fill_between(range(L), np.percentile(R, q, 0), np.percentile(R, 100 - q, 0),
                               color="tab:blue", alpha=al, lw=0)
        ax[0].plot(np.median(R, 0), color="tab:blue", lw=1, label="real windows (median, 50/90% band)")
        for (name, r), p in zip(syn.iterrows(), paths):
            ax[0].plot(p, color=col[r.label], lw=1.1, alpha=.85)
            ax[0].annotate(name[-3:], (L - 1, p[-1]), fontsize=7, color=col[r.label])
        for k, c in col.items():
            ax[0].plot([], [], color=c, label=k)
        ax[0].set(yscale="log", title="Equal-weight index per scenario", xlabel="episode day")
        ax[0].legend(fontsize=8)
        for k, c in col.items():
            sr, rr = syn[syn.label == k], real[real.label == k]
            ax[1].scatter(rr.ann_vol, rr.ann_return, s=6, color=c, alpha=.15)
            ax[1].scatter(sr.ann_vol, sr.ann_return, s=45, color=c, edgecolor="k", label=f"{k} ({len(sr)})")
        ax[1].axhline(0, color="k", lw=.5)
        ax[1].set(xlabel="annualised vol", ylabel="annualised return",
                  title="Scenarios (outlined) vs real windows (faint)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        f = os.path.join(a.scenarios, "regimes.png")
        fig.savefig(f, dpi=130)
        print(f"\nfigure: {f}")


if __name__ == "__main__":
    main()
