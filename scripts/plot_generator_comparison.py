#!/usr/bin/env python3
"""Visual comparison of generators against real data + the hybrid sampling schedule.

Each --model directory must contain generated_seed*.npy (anchored log-price
paths) and may contain comparison.json (from train_l4 / --eval_only) for the
discriminative-AUC panel. All statistics are computed within windows.

Example:
    python scripts/plot_generator_comparison.py \\
        --real save/fixed/gbm_exponential/real_train.npy \\
        --holdout save/fixed/gbm_exponential/real_val.npy \\
        --model "ε-model=save/fixed/gbm_exponential/eval_pc_filter" \\
        --model "EDM=save/edm_scalar/gbm_exponential/eval_hybrid_0.5_filter" \\
        --model "EDM→ε splice=save/edm_scalar/gbm_exponential/eval_hybrid_0.5_filter_splice0.03" \\
        --out save/comparison/figures/hybrid_comparison.png
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gbm_financial.compare import _acf_windows, _leverage  # noqa: E402

COLORS = ["tab:orange", "tab:purple", "tab:blue", "tab:green", "tab:brown"]


def load_model(path):
    files = sorted(glob.glob(os.path.join(path, "generated_seed*.npy")))
    if not files:
        sys.exit(f"No generated_seed*.npy in {path}")
    paths = np.concatenate([np.load(f) for f in files])
    auc = None
    cj = os.path.join(path, "comparison.json")
    if os.path.exists(cj):
        res = json.load(open(cj))
        ours = next(v for k, v in res.items() if k.startswith("ours"))
        auc = (ours["disc_auc"]["mean"], ours["disc_auc"]["std"])
        auc_ref = {k: v["disc_auc"]["mean"] for k, v in res.items() if not k.startswith("ours")}
    else:
        auc_ref = {}
    return paths, auc, auc_ref


def schedule_panel(ax, sigma_max, sigma_min, switch, fine_sigma, ode_steps, sde_steps, rho=7.0,
                   eps=1e-5, daily_std=None):
    k = np.arange(ode_steps)
    sig_ode = (sigma_max ** (1 / rho) + k / (ode_steps - 1) *
               (switch ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    s_end = sigma_min * (sigma_max / sigma_min) ** eps
    sig_sde = np.exp(np.linspace(np.log(switch), np.log(s_end), sde_steps))
    x_ode = np.arange(ode_steps)
    x_sde = ode_steps + np.arange(sde_steps)
    coarse = sig_sde >= fine_sigma
    ax.plot(x_ode, sig_ode, color="tab:purple", lw=3, label="EDM model · deterministic Heun")
    ax.plot(x_sde[coarse], sig_sde[coarse], color="tab:purple", lw=3, ls=(0, (1, 1)),
            label="EDM model · reverse SDE")
    ax.plot(x_sde[~coarse], sig_sde[~coarse], color="tab:orange", lw=3,
            label="ε-model · reverse SDE")
    for y, txt in ((switch, f"σ_switch={switch:g}"), (fine_sigma, f"σ_fine={fine_sigma:g}")):
        ax.axhline(y, color="gray", lw=0.8, ls="--")
        ax.text(x_sde[-1], y * 1.15, txt, ha="right", va="bottom", fontsize=8, color="gray")
    if daily_std:
        ax.axhline(daily_std, color="black", lw=0.8, ls=":")
        ax.text(2, daily_std * 0.75, f"daily return std ≈ {daily_std:.3f}", fontsize=8, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("sampler step (Heun steps, then SDE steps)")
    ax.set_ylabel("noise level σ")
    ax.set_title("Hybrid sampler: who denoises what")
    ax.text(0.03, 0.06, "high σ → coarse shape\n(trend, spread, regimes)", transform=ax.transAxes,
            fontsize=8, color="tab:purple")
    ax.text(0.97, 0.98, "low σ → fine structure\n(daily returns, ACF)", transform=ax.transAxes,
            fontsize=8, color="tab:orange", ha="right", va="top")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.89))


def paths_panel(ax, paths, title, color, n=8, seed=0, ylim=None):
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(paths), n, replace=False):
        ax.plot(paths[i] - paths[i, 0], color=color, lw=0.7, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("day")
    ax.set_ylabel("log-price (anchored)")
    if ylim:
        ax.set_ylim(*ylim)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--model", action="append", required=True, help="label=dir (repeatable)")
    ap.add_argument("--paths_model", default=None,
                    help="label of the model shown in the sample-path panel (default: last)")
    ap.add_argument("--sigma_max", type=float, default=12.561595916748047)
    ap.add_argument("--sigma_min", type=float, default=1e-3)
    ap.add_argument("--switch", type=float, default=0.5)
    ap.add_argument("--fine_sigma", type=float, default=0.03)
    ap.add_argument("--ode_steps", type=int, default=100)
    ap.add_argument("--sde_steps", type=int, default=1000)
    ap.add_argument("--out", default="save/comparison/figures/hybrid_comparison.png")
    args = ap.parse_args()

    real_p = np.load(args.real)
    real = np.diff(real_p, axis=1)
    hold_p = np.load(args.holdout) if args.holdout else None
    models = []
    for spec in args.model:
        label, path = spec.split("=", 1)
        p, auc, auc_ref = load_model(path)
        models.append((label, np.diff(p, axis=1), p, auc, auc_ref))
    show = next((m for m in models if m[0] == args.paths_model), models[-1])

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    a = axes.ravel()

    schedule_panel(a[0], args.sigma_max, args.sigma_min, args.switch, args.fine_sigma,
                   args.ode_steps, args.sde_steps, daily_std=float(real.std()))

    ref_paths = hold_p if hold_p is not None else real_p
    lo = min(np.percentile(ref_paths[:, -1] - ref_paths[:, 0], 1), np.percentile(show[2][:, -1], 1))
    hi = max(np.percentile(ref_paths[:, -1] - ref_paths[:, 0], 99), np.percentile(show[2][:, -1], 99))
    ylim = (lo - 0.5, hi + 0.5)
    paths_panel(a[1], ref_paths, "Real paths" + (" (held-out tickers)" if hold_p is not None else ""),
                "black", ylim=ylim)
    paths_panel(a[2], show[2], f"Generated paths — {show[0]}", COLORS[models.index(show)], ylim=ylim)

    curves = [("real", "black", real, 2.2)] + [(m[0], COLORS[i], m[1], 1.5) for i, m in enumerate(models)]

    ax = a[3]
    for lab, c, r, lw in curves:
        z = np.sort(np.abs((r - r.mean()) / r.std()).ravel())[::-1]
        surv = np.arange(1, len(z) + 1) / len(z)
        keep = z > 1
        ax.loglog(z[keep], surv[keep], color=c, lw=lw, label=lab)
    ax.set_xlabel("|r| / std  (each standardized)")
    ax.set_ylabel("P(|r|/std > x)")
    ax.set_title("Tail shape (heavy tails)")
    ax.legend(fontsize=8)

    ax = a[4]
    lags = np.arange(1, 101)
    for lab, c, r, lw in curves:
        ax.loglog(lags, np.clip(_acf_windows(np.abs(r), lags), 1e-3, None), color=c, lw=lw, label=lab)
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("ACF of |r|")
    ax.set_title("Volatility clustering")
    ax.legend(fontsize=8)

    ax = a[5]
    lags = np.arange(1, 51)
    for lab, c, r, lw in curves:
        ax.plot(lags, _leverage(r, lags), color=c, lw=lw, label=lab)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("lag k (days)")
    ax.set_ylabel("E[r_t r²_{t+k}] / E[r²]²")
    ax.set_title("Leverage effect (negative = real)")
    ax.legend(fontsize=8)

    ax = a[6]
    bins = np.logspace(np.log10(0.004), np.log10(0.12), 50)
    for lab, c, r, lw in curves:
        ax.hist(r.std(1), bins=bins, density=True, histtype="step", color=c, lw=lw, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("per-window daily volatility")
    ax.set_ylabel("density")
    ax.set_title("Volatility regimes across windows")
    ax.legend(fontsize=8)

    ax = a[7]
    bins = np.linspace(-3, 4, 60)
    for lab, c, r, lw in curves:
        t = r.sum(1)
        ax.hist(t, bins=bins, density=True, histtype="step", color=c, lw=lw,
                label=f"{lab} (mean {t.mean():+.2f})")
    ax.set_xlabel("2048-day log-return")
    ax.set_ylabel("density")
    ax.set_title("Terminal return (trend + spread)")
    ax.legend(fontsize=8)

    ax = a[8]
    names, vals, errs, cols = [], [], [], []
    for i, m in enumerate(models):
        if m[3]:
            names.append(m[0]); vals.append(m[3][0]); errs.append(m[3][1]); cols.append(COLORS[i])
    ref = next((m[4] for m in models if m[4]), {})
    for k in ("baseline: block bootstrap(20)", "baseline: GARCH(1,1)-t", "baseline: iid shuffle"):
        if k in ref:
            names.append(k.replace("baseline: ", "")); vals.append(ref[k]); errs.append(0); cols.append("lightgray")
    if names:
        y = np.arange(len(names))
        ax.barh(y, vals, xerr=errs, color=cols)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        floor = ref.get("real held-out tickers")
        if floor:
            ax.axvline(floor, color="black", ls="--", lw=1)
            ax.text(floor, len(names) - 0.4, f" real-vs-real floor {floor:.3f}", fontsize=8)
        ax.set_xlim(0.8, 1.0)
        ax.set_xlabel("discriminative ROC-AUC (lower = harder to tell from real)")
        ax.set_title("Can a classifier spot the fakes?")
    else:
        ax.axis("off")

    fig.suptitle("Synthetic S&P 500 paths (2048 days): generators vs real data", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
