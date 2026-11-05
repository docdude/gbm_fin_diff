#!/usr/bin/env python3
"""Head-to-head: this repo vs vnhtg01/gbm-diffusion-finance-ts on identical data.

Both models train on the same tickers (this repo's ticker split) built from the
same raw price cache (data/financial/sp500_prices_raw.pkl), and are scored by the
same harness (gbm_financial.compare). Each model is scored against the real
windows in its OWN training representation:
  - ours:     anchored log-price windows → returns (raw log-return units)
  - vnhtg01:  per-ticker z-scored log-returns (their pipeline, as published)
Scale-free metrics (kurtosis, α, ACFs, β, vol dispersion, disc. AUC, NN ratio)
are directly comparable across the two tables.

Stages (each is resumable — finished outputs are skipped):
    prepare   clone vnhtg01 at the pinned commit; write its raw parquet
              restricted to our training tickers; build their real windows
    train     train vnhtg01 (paper.yaml, GBM + exponential)
    generate  sample n_seeds × n_samples from their checkpoint (seeded)
    report    score ours + theirs + baselines → save/comparison/REPORT.md

Usage (single GPU):
    python scripts/compare_vnhtg01.py all --ours save/fixed/gbm_exponential
    python scripts/compare_vnhtg01.py report --ours save/fixed/gbm_exponential \\
        --ours-extra "ours σ_min=0.01=save/ablation_sigmamin/gbm_exponential"
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gbm_financial.compare import format_report, run_comparison, to_returns  # noqa: E402
from gbm_financial.data import (load_raw_price_cache, preprocess_prices,  # noqa: E402
                                ticker_split)

REPO_URL = "https://github.com/vnhtg01/gbm-diffusion-finance-ts.git"
PINNED_COMMIT = "1433c31c39ab9fe0546d12f2607592186fd47c2a"  # audited 2026-11-04
EXT_DIR = os.path.join(ROOT, "external", "gbm-diffusion-finance-ts")
OUT_DIR = os.path.join(ROOT, "save", "comparison")
SEQ_LEN, STRIDE = 2048, 400


def sh(cmd, cwd=None):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _price_frame(tickers):
    """Preprocessed close prices (same cleaning as this repo) as a DataFrame."""
    import pandas as pd
    raw = load_raw_price_cache(os.path.join(ROOT, "data", "financial"))
    if raw is None:
        sys.exit("Raw price cache missing: run `python -m gbm_financial.train_l4` once "
                 "or copy data/financial/sp500_prices_raw.pkl")
    cols = {}
    for t in tickers:
        s = raw[t].dropna()
        s = s[s > 0]
        cols[t] = pd.Series(preprocess_prices(s.values.astype(np.float64)), index=s.index)
    return pd.DataFrame(cols).sort_index()


def _their_windows(df):
    """Windows built exactly as vnhtg01 does (per-ticker z-scored log-returns)."""
    sys.path.insert(0, EXT_DIR)
    from src.data import build_dataset  # type: ignore
    return build_dataset(df, SEQ_LEN, STRIDE).astype(np.float32)


def stage_prepare(args):
    if not os.path.isdir(EXT_DIR):
        os.makedirs(os.path.dirname(EXT_DIR), exist_ok=True)
        sh(["git", "clone", REPO_URL, EXT_DIR])
    sh(["git", "checkout", "-q", PINNED_COMMIT], cwd=EXT_DIR)

    raw = load_raw_price_cache(os.path.join(ROOT, "data", "financial"))
    train_t, val_t = ticker_split(sorted(raw), args.train_ratio, args.split_seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "split.json"), "w") as f:
        json.dump({"train": train_t, "val": val_t, "seed": args.split_seed}, f, indent=2)

    raw_parquet = os.path.join(EXT_DIR, "data", "raw", "sp500.parquet")
    proc = os.path.join(EXT_DIR, "data", "processed", f"sp500_L{SEQ_LEN}_S{STRIDE}.npz")
    os.makedirs(os.path.dirname(raw_parquet), exist_ok=True)
    for stale in (proc,):
        if os.path.exists(stale):
            os.remove(stale)
    train_df = _price_frame(train_t)
    train_df.to_parquet(raw_parquet)
    np.save(os.path.join(OUT_DIR, "theirs_real_train.npy"), _their_windows(train_df))
    np.save(os.path.join(OUT_DIR, "theirs_real_val.npy"), _their_windows(_price_frame(val_t)))
    print(f"Prepared vnhtg01 data: {len(train_t)} train / {len(val_t)} held-out tickers")


def _ckpt_path():
    return os.path.join(EXT_DIR, "experiments", "checkpoints", "cmp_gbm_exponential.pt")


def stage_train(args):
    if os.path.exists(_ckpt_path()) and not args.force:
        print(f"Skip train: {_ckpt_path()} exists")
        return
    sh([sys.executable, "scripts/train.py", "--config", "configs/paper.yaml",
        "--sde", "gbm", "--schedule", "exponential",
        "--epochs", str(args.their_epochs), "--tag", "cmp_gbm_exponential"], cwd=EXT_DIR)


def stage_generate(args):
    for k in range(args.n_seeds):
        out = os.path.join(OUT_DIR, f"theirs_gen_seed{k}.npy")
        if os.path.exists(out) and not args.force:
            print(f"Skip generate seed {k}: exists")
            continue
        # Their generate.py has no seed flag; seed torch before running it.
        code = (
            "import sys, runpy, torch, numpy as np\n"
            f"torch.manual_seed({k}); np.random.seed({k})\n"
            "sys.argv = ['generate.py', '--config', 'configs/paper.yaml', "
            f"'--ckpt', {_ckpt_path()!r}, '--n-samples', '{args.n_samples}', "
            f"'--out', {out!r}]\n"
            "runpy.run_path('scripts/generate.py', run_name='__main__')\n"
        )
        sh([sys.executable, "-c", code], cwd=EXT_DIR)


def _load_ours(exp_dir):
    gens = sorted(glob.glob(os.path.join(exp_dir, "generated_seed*.npy")))
    if not gens:
        sys.exit(f"No generated_seed*.npy in {exp_dir} — run train_l4 first")
    cfg = json.load(open(os.path.join(exp_dir, "results.json"))).get("config", {})
    mode = cfg.get("data_mode", "log_price")
    real = to_returns(np.load(os.path.join(exp_dir, "real_train.npy")), mode)
    val = to_returns(np.load(os.path.join(exp_dir, "real_val.npy")), mode)
    return real, val, [to_returns(np.load(g), mode) for g in gens]


def stage_report(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    sections = []

    real, val, gens = _load_ours(args.ours)
    ours_sets = {"ours (fixed): gbm+exponential": gens}
    for spec in args.ours_extra:
        name, path = spec.rsplit("=", 1)
        _, _, g = _load_ours(path)
        ours_sets[name] = g
    res_ours = run_comparison(real, ours_sets, holdout_r=val, baselines=True,
                              n_baseline=args.n_samples)
    sections.append(("This repo — reference: raw log-returns of training tickers",
                     res_ours))

    their_gen = sorted(glob.glob(os.path.join(OUT_DIR, "theirs_gen_seed*.npy")))
    if their_gen:
        t_real = np.load(os.path.join(OUT_DIR, "theirs_real_train.npy"))
        t_val = np.load(os.path.join(OUT_DIR, "theirs_real_val.npy"))
        res_theirs = run_comparison(
            to_returns(t_real, "log_return"),
            {"vnhtg01 (as published): gbm+exponential":
             [to_returns(np.load(g), "log_return") for g in their_gen]},
            holdout_r=to_returns(t_val, "log_return"), baselines=True,
            n_baseline=args.n_samples)
        sections.append(("vnhtg01 — reference: per-ticker z-scored log-returns "
                         "(their representation), same training tickers", res_theirs))

    lines = ["# Generator comparison", "",
             "Metrics are within-window, mean±std over sampling seeds. "
             "`target (real)` is the real reference for that table; ratios (×) target 1, "
             "NN× (gen→train vs held-out→train distance) targets 1 (<1 suggests "
             "memorisation). discAUC is ideally 0.5, but read it against the "
             "`real held-out tickers` row: with long windows a classifier can tell "
             "different real tickers apart too, so that row is the practical floor.", ""]
    for title, res in sections:
        lines += [f"## {title}", "", format_report(res), ""]
    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as f:
        f.write(report)
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump({t: r for t, r in sections}, f, indent=2)
    print(report)
    print(f"\nSaved {OUT_DIR}/REPORT.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "train", "generate", "report", "all"])
    ap.add_argument("--ours", default="save/fixed/gbm_exponential",
                    help="experiment dir produced by gbm_financial.train_l4")
    ap.add_argument("--ours-extra", nargs="*", default=[],
                    help="extra runs of this repo as 'label=exp_dir'")
    ap.add_argument("--their-epochs", type=int, default=1000,
                    help="vnhtg01 epochs (their paper.yaml uses 100; 1000 matches ours)")
    ap.add_argument("--n-samples", type=int, default=120)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="must equal the `seed` used by train_l4 (default 0)")
    ap.add_argument("--force", action="store_true", help="redo finished stages")
    args = ap.parse_args()
    args.ours = os.path.join(ROOT, args.ours) if not os.path.isabs(args.ours) else args.ours

    stages = ["prepare", "train", "generate", "report"] if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"\n=== {st} ===")
        globals()[f"stage_{st}"](args)


if __name__ == "__main__":
    main()
