#!/usr/bin/env python3
"""
L4 GPU training script for GBM Financial Diffusion Model.

Runs on any single ~24-48 GB GPU (L4, 4090, L40S).
Uses paper's full hyperparameters: 128/256/64, 4 layers, 8 heads,
seq_len=2048, 1000 epochs, N=2000 (σ_min lowered to 1e-3, see PAPER_CONFIG).

Multi-stock data: downloads S&P 500 constituents with >40 years history
to match the paper's data collection methodology (Section 3.1.2). Whole
tickers are held out for validation; evaluation samples n_seeds × n_generate
paths and scores them with gbm_financial.compare (writes comparison.md).

Usage:
    # Full paper reproduction (GBM + exponential, ~8 hours on L4):
    python -m gbm_financial.train_l4

    # Quick test (smaller config, ~30 min):
    python -m gbm_financial.train_l4 --quick

    # Full 3x3 experiment grid:
    python -m gbm_financial.train_l4 --grid

    # Resume from checkpoint:
    python -m gbm_financial.train_l4 --resume save/gbm_financial/gbm_exponential/checkpoint_epoch500.pth
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch

from gbm_financial.train import GBMFinancialDiffusion
from gbm_financial.data import (get_dataloaders, download_stock_data,
                               load_raw_price_cache, LONG_HISTORY_TICKERS,
                               compute_sigma_max)
from gbm_financial.compare import (to_returns, run_comparison, format_report)


# Paper's full configuration (Section 4)
PAPER_CONFIG = {
    # SDE
    "sde_type": "gbm",
    "schedule": "exponential",
    # Paper: σ_min=0.01. Lowered to 1e-3: on log-price paths 0.01 is the size
    # of a daily return (std≈0.017), so the network never learned sub-daily
    # structure and the samplers' last step decided the daily stylized facts.
    "sigma_min": 1e-3,
    "sigma_max": "auto",   # Auto-computed from data (≈ 3× data range ≈ 8.35)
    "n_reverse_steps": 2000,
    "vp_beta_min": 0.1,    # VP: β(t) = σ_VP(t)², σ_VP ∈ [√β_min, √β_max]
    "vp_beta_max": 20.0,

    # Architecture (Section 3.1.1)
    "channels": 128,
    "diff_emb_dim": 256,
    "feat_emb_dim": 64,
    "time_emb_dim": 128,
    "n_layers": 4,
    "n_heads": 8,

    # WaveNet temporal branch (off by default — original Transformer-only)
    "wavenet_branch": False,

    # Training (Section 4)
    "seq_len": 2048,
    "epochs": 1000,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "ema_decay": 0.999,
    # loss_weighting: unset → "uniform" (ε-prediction) or "edm" when precond="edm"

    # Data (Section 3.1.2)
    "window_len": 2048,
    "stride": 400,
    "min_years": 40,
    "use_synthetic": False,
    "data_mode": "log_price",  # all SDEs (author's clarification)
    "split": "ticker",         # hold out whole tickers → no window-overlap leakage

    # Generation / Sampling
    "n_generate": 120,
    "n_seeds": 3,              # independent sampling seeds for evaluation
    "seed": 0,
    "sampling_eps": 1e-5,      # samplers end with a Tweedie denoise, eps is no longer critical
    "sampler_noise_scale": 1.0,  # >1 counters score under-dispersion (PC/EM only)

    # Network parameterisation: "none" (ε-prediction, as in the paper) or
    # "edm" (Karras et al. 2022 preconditioning + loss weighting)
    "precond": "none",
    "edm_sigma_data_mode": "scalar",   # "scalar" or "per_position"
    "pc_corrector_steps": 0,

    # No z-score normalization for GBM/log-price (paper faithful).
    # Global z-score of cumulative paths induces mean-reversion.
    "normalize_data": False,
}

# Reduced config for quick validation on L4
# Paper's data params (seq_len=2048, stride=400) but smaller model → faster
# With ~40 stocks: ~800 windows, ~12 batches/epoch, 500 epochs = ~6000 steps
QUICK_CONFIG = {
    **PAPER_CONFIG,
    "channels": 64,
    "diff_emb_dim": 128,
    "feat_emb_dim": 32,
    "n_layers": 2,
    "n_heads": 4,
    "seq_len": 2048,
    "window_len": 2048,
    "epochs": 500,
    "n_reverse_steps": 1000,
    "stride": 200,        # denser than paper (400) for more data
    "n_generate": 60,
}

# Minimal config for single-stock CSV (no yfinance)
# Proven on CPU: window=256, stride=25, 1000 epochs → ~3500 steps
MINIMAL_CONFIG = {
    **PAPER_CONFIG,
    "channels": 64,
    "diff_emb_dim": 128,
    "feat_emb_dim": 32,
    "n_layers": 2,
    "n_heads": 4,
    "seq_len": 256,
    "window_len": 256,
    "epochs": 1000,
    "n_reverse_steps": 500,
    "stride": 25,
    "n_generate": 60,
}


def ensure_data(config, cache_dir="data/financial"):
    """Make sure the raw multi-stock price cache exists (downloads if needed).

    The cache (data/financial/sp500_prices_raw.pkl) stores unprocessed adjusted
    closes; copy it to another machine to train on identical data.
    """
    if config.get("use_synthetic"):
        return
    if load_raw_price_cache(cache_dir) is not None:
        return
    print("No raw price cache found. Downloading with yfinance...")
    try:
        import yfinance  # noqa: F401
    except ImportError:
        print("ERROR: yfinance not installed (pip install yfinance), or copy "
              f"{cache_dir}/sp500_prices_raw.pkl from another machine.")
        sys.exit(1)
    download_stock_data(LONG_HISTORY_TICKERS, min_years=config.get("min_years", 40),
                        cache_dir=cache_dir)


def run_experiment(config, save_dir, resume_path=None, warmstart_path=None,
                   warmstart_ema=False):
    """Run a single experiment with the given config."""
    sde_type = config["sde_type"]
    schedule = config["schedule"]
    exp_dir = os.path.join(save_dir, f"{sde_type}_{schedule}")

    print(f"\n{'=' * 70}")
    print(f"  {sde_type.upper()} + {schedule}")
    print(f"  Channels={config['channels']}, Layers={config['n_layers']}, "
          f"Heads={config['n_heads']}")
    print(f"  seq_len={config['seq_len']}, epochs={config['epochs']}, "
          f"batch_size={config['batch_size']}")
    print(f"  N_reverse={config['n_reverse_steps']}")
    print(f"{'=' * 70}")

    torch.manual_seed(config.get("seed", 0))
    np.random.seed(config.get("seed", 0))

    # Data
    train_loader, val_loader, data_info = get_dataloaders(
        sde_type=sde_type,
        window_len=config["seq_len"],
        stride=config.get("stride", 400),
        batch_size=config["batch_size"],
        use_synthetic=config.get("use_synthetic", False),
        min_years=config.get("min_years", 40),
        num_workers=0,
        data_mode=config.get("data_mode", "log_price"),
        split=config.get("split", "ticker"),
        seed=config.get("seed", 0),
    )
    n_batches = len(train_loader)
    total_steps = config["epochs"] * n_batches
    print(f"Data: {data_info['n_train']} train, {data_info['n_val']} val "
          f"({data_info['n_stocks']} stocks)")
    print(f"  {n_batches} batches/epoch → {total_steps} total gradient steps")
    if total_steps < 5000:
        print(f"  WARNING: Only {total_steps} steps — may be too few for convergence.")
        print(f"  Consider: pip install yfinance (more stocks), or reduce stride, or increase epochs.")

    # Auto-compute σ_max from data if set to "auto"
    # Song et al. (2020): σ_max should be large enough that the prior N(0, σ_max²)
    # approximates the noised marginal at t=T.  With σ_max=1.0 on raw log-price
    # paths (range [-1, 3], std≈0.6), KL(marginal||prior)=0.16 → severe mismatch.
    # Auto-compute picks σ_max ≈ 3× data std (typically 5-10), giving KL < 0.01.
    if sde_type == "vp":
        pass  # VP prior is N(0, 1); σ bounds come from vp_beta_min/max
    elif config.get("sigma_max") == "auto":
        sigma_max, sigma_stats = compute_sigma_max(train_loader)
        config["sigma_max"] = sigma_max
        print(f"  σ_max auto-set to {sigma_max:.1f}")
    elif isinstance(config.get("sigma_max"), str):
        raise ValueError(f"Unknown sigma_max value: {config['sigma_max']}")

    if config.get("precond") == "edm" and config.get("edm_sigma_data") is None:
        windows = train_loader.dataset.data
        if config.get("edm_sigma_data_mode", "scalar") == "per_position":
            config["edm_sigma_data"] = windows.std(dim=0).tolist()
        else:
            config["edm_sigma_data"] = float(windows.std())
        sd = np.asarray(config["edm_sigma_data"])
        print(f"  EDM σ_data ({config.get('edm_sigma_data_mode', 'scalar')}): "
              f"{sd.min():.4f} … {sd.max():.4f}")

    # Model
    model = GBMFinancialDiffusion(config)

    if resume_path:
        model.load(resume_path)
        print(f"Resumed from {resume_path}")

    if warmstart_path:
        model.load_weights_only(warmstart_path, use_ema_weights=warmstart_ema)
        print(f"Warm-started model weights from {warmstart_path}")

    # Train
    start = time.time()
    model.train(train_loader, val_loader, save_dir=exp_dir)
    train_time = time.time() - start
    print(f"Training time: {train_time / 3600:.1f} hours")

    # Reload best model for evaluation (training ends at final epoch, not best)
    best_path = os.path.join(exp_dir, "best_model.pth")
    if os.path.exists(best_path):
        model.load(best_path)
        print(f"Loaded best model from {best_path}")
    else:
        print("WARNING: No best_model.pth found, evaluating final epoch model")

    results = evaluate_model(model, config, train_loader, val_loader, exp_dir)
    results.update({
        "sde_type": sde_type,
        "schedule": schedule,
        "n_train": data_info["n_train"],
        "n_val": data_info["n_val"],
        "val_tickers": data_info["val_tickers"],
        "split": data_info["split"],
        "n_stocks": data_info["n_stocks"],
        "train_time_hours": train_time / 3600,
        "config": {k: v for k, v in config.items()
                   if isinstance(v, (int, float, str, bool))},
        "training_params": {
            "data_mode": model.data_mode,
            "mask_anchor": model.mask_anchor,
            "normalize_mode": model.normalize_mode,
            "data_mean": model.data_mean,
            "data_std": model.data_std,
            "use_ema": model.use_ema,
            "loss_weighting": model.loss_weighting,
            "n_model_params": sum(p.numel() for p in model.model.parameters()),
            "total_gradient_steps": config["epochs"] * n_batches,
            "n_batches_per_epoch": n_batches,
            "device": str(model.device),
        },
    })

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def _sample(model, config, n, seed):
    return model.sample(n_samples=n, seq_len=config["seq_len"], seed=seed)


def set_reject_threshold(config, train_loader, data_mode):
    """reject_failed → reject_ac1_below = 0.1th pct of real per-path lag-1 ACF."""
    if config.get("reject_failed") and config.get("reject_ac1_below") is None:
        ac1 = GBMFinancialDiffusion.path_lag1_autocorr(
            train_loader.dataset.data.numpy(), data_mode)
        config["reject_ac1_below"] = float(np.percentile(ac1, 0.1))
        print(f"  Failed-sample filter: reject paths with lag-1 ACF < "
              f"{config['reject_ac1_below']:.3f} (real 0.1th pct)")


def evaluate_model(model, config, train_loader, val_loader, exp_dir):
    """Multi-seed generation + shared comparison harness + diagnostic plots.

    Saves generated_seed{k}.npy, real_train.npy / real_val.npy (the reference
    windows, reusable for cross-repo comparisons), comparison.{json,md}.
    """
    mode = model.data_mode
    n_gen = config.get("n_generate", 120)
    seeds = [config.get("seed", 0) + k for k in range(config.get("n_seeds", 3))]
    set_reject_threshold(config, train_loader, mode)
    model.config.update({k: config[k] for k in ("reject_ac1_below",) if k in config})
    os.makedirs(exp_dir, exist_ok=True)

    real_train = np.concatenate([b.numpy() for b in train_loader.dataset.data.split(512)], 0)
    real_val = np.concatenate([b.numpy() for b in val_loader.dataset.data.split(512)], 0)
    np.save(os.path.join(exp_dir, "real_train.npy"), real_train)
    np.save(os.path.join(exp_dir, "real_val.npy"), real_val)

    gens, rejection = [], []
    for sd in seeds:
        g = _sample(model, config, n_gen, sd)
        rejection.append(getattr(model, "last_rejection_rate", 0.0))
        np.save(os.path.join(exp_dir, f"generated_seed{sd}.npy"), g)
        gens.append(g)
    np.save(os.path.join(exp_dir, "generated_data.npy"), gens[0])

    real_r = to_returns(real_train, mode)
    val_r = to_returns(real_val, mode)
    label = f"ours: {config['sde_type']}+{config['schedule']}"
    if config.get("precond", "none") != "none":
        label += f" {config['precond']}"
    if config.get("sampler", "pc") != "pc":
        label += f" {config['sampler']}"
    if config.get("reject_ac1_below") is not None:
        label += " +filter"
    cmp = run_comparison(real_r, {label: [to_returns(g, mode) for g in gens]},
                         holdout_r=val_r, baselines=True, n_baseline=n_gen)
    table = format_report(cmp)
    with open(os.path.join(exp_dir, "comparison.json"), "w") as f:
        json.dump(cmp, f, indent=2)
    with open(os.path.join(exp_dir, "comparison.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)

    # Stylized-fact, diagnostic, pathwise and mean-path plots (seed 0)
    model.evaluate(gens[0], real_train, save_dir=exp_dir)

    ours = cmp[label]
    return {
        "seeds": seeds,
        "sampler": config.get("sampler", "pc"),
        "rejection_rate": float(np.mean(rejection)),
        "metrics": {k: v["mean"] for k, v in ours.items()},
        "metrics_std": {k: v["std"] for k, v in ours.items()},
        "alpha_gen": ours["alpha_top5"]["mean"],
        "alpha_real": ours["alpha_top5_real"]["mean"],
        "beta_gen": ours["vol_cluster_beta"]["mean"],
        "beta_real": ours["vol_cluster_beta_real"]["mean"],
    }


def run_eval_only(cli_config, save_dir, tag=None):
    """Re-score an existing experiment with new sampler settings (no training).

    Loads <save_dir>/<sde>_<schedule>/results.json config + best_model.pth,
    applies sampler-related overrides from the CLI and writes all outputs to
    <exp_dir>/eval_<tag>/.
    """
    exp_dir = os.path.join(save_dir, f"{cli_config['sde_type']}_{cli_config['schedule']}")
    with open(os.path.join(exp_dir, "results.json")) as f:
        config = json.load(f)["config"]
    for k in ("sampler", "hybrid_sigma_switch", "hybrid_ode_steps", "hybrid_sde_steps",
              "karras_rho", "n_reverse_steps", "sampling_eps", "pc_corrector_steps",
              "sampler_noise_scale", "sampler_noise_sigma_min", "sampler_noise_sigma_max",
              "reject_failed", "reject_ac1_below", "n_generate", "n_seeds", "seed"):
        if k in cli_config and cli_config[k] is not None:
            config[k] = cli_config[k]
    tag = tag or f"{config.get('sampler', 'pc')}" + (
        f"_{config['hybrid_sigma_switch']:g}" if config.get("sampler") == "hybrid" else "") + (
        "_filter" if config.get("reject_failed") else "")
    out_dir = os.path.join(exp_dir, f"eval_{tag}")

    train_loader, val_loader, _ = get_dataloaders(
        sde_type=config["sde_type"], window_len=config["seq_len"],
        stride=config.get("stride", 400), batch_size=config["batch_size"],
        min_years=config.get("min_years", 40), num_workers=0,
        data_mode=config.get("data_mode", "log_price"),
        split=config.get("split", "ticker"), seed=config.get("seed", 0))
    model = GBMFinancialDiffusion(config)
    ckpt = os.path.join(exp_dir, "best_model.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(exp_dir, "final_model.pth")
    model.load(ckpt)
    results = evaluate_model(model, config, train_loader, val_loader, out_dir)
    results["checkpoint"] = ckpt
    results["config"] = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_grid(base_config, save_dir):
    """Run all 9 SDE × schedule combinations from the paper."""
    sde_types = ["ve", "vp", "gbm"]
    schedules = ["linear", "exponential", "cosine"]

    all_results = []

    for sde_type in sde_types:
        for schedule in schedules:
            config = base_config.copy()
            config["sde_type"] = sde_type
            config["schedule"] = schedule

            try:
                result = run_experiment(config, save_dir)
                all_results.append(result)
            except Exception as e:
                print(f"\nERROR in {sde_type}/{schedule}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "sde_type": sde_type,
                    "schedule": schedule,
                    "error": str(e),
                })

    # Print results table (matches paper Table 1)
    print(f"\n{'=' * 80}")
    print("RESULTS (compare with paper Table 1)")
    print(f"{'=' * 80}")
    print(f"{'SDE':<6} {'Schedule':<14} {'α (gen)':<10} {'α (real)':<10} "
          f"{'β (gen)':<10} {'β (real)':<10}")
    print("-" * 64)
    for r in all_results:
        if "error" in r:
            print(f"{r['sde_type']:<6} {r['schedule']:<14} ERROR")
        else:
            print(f"{r['sde_type']:<6} {r['schedule']:<14} "
                  f"{r.get('alpha_gen', float('nan')):<10.2f} "
                  f"{r.get('alpha_real', float('nan')):<10.2f} "
                  f"{r.get('beta_gen', float('nan')):<10.3f} "
                  f"{r.get('beta_real', float('nan')):<10.3f}")
    print(f"\nPaper reference (GBM+exp): α ≈ 4.62, β reported in Fig 5-8")
    print(f"{'=' * 80}")

    with open(os.path.join(save_dir, "grid_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="L4 GPU training for GBM Financial Diffusion"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Reduced model, paper data params (multi-stock, ~1-2h on L4)")
    parser.add_argument("--minimal", action="store_true",
                        help="Minimal config for single-stock CSV (window=256, ~5 min on L4)")
    parser.add_argument("--grid", action="store_true",
                        help="Run full 3×3 experiment grid")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--sde_type", type=str, default=None,
                        choices=["ve", "vp", "gbm"])
    parser.add_argument("--schedule", type=str, default=None,
                        choices=["linear", "exponential", "cosine"])
    parser.add_argument("--save_dir", type=str, default="save/gbm_financial")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override sequence length")
    parser.add_argument("--stride", type=int, default=None,
                        help="Override sliding window stride")
    parser.add_argument("--zscore", type=str, default=None,
                        choices=["none", "global", "per_path"],
                        help="Normalization mode: none, global, per_path")
    parser.add_argument("--sigma_max", type=str, default=None,
                        help="Override sigma_max (number or 'auto')")
    parser.add_argument("--sigma_min", type=float, default=None,
                        help="Override sigma_min (default 1e-3; paper used 0.01)")
    parser.add_argument("--data_mode", type=str, default=None,
                        choices=["log_price", "log_return"],
                        help="Training representation (default log_price for all SDEs)")
    parser.add_argument("--split", type=str, default=None, choices=["ticker", "random"],
                        help="Train/val split (default: ticker hold-out)")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed")
    parser.add_argument("--noise_scale", type=float, default=None,
                        help="Stochastic-sampler noise multiplier S_noise (PC/EM)")
    parser.add_argument("--noise_sigma_range", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="σ band where --noise_scale applies")
    parser.add_argument("--n_seeds", type=int, default=None,
                        help="Number of sampling seeds for evaluation (default 3)")
    parser.add_argument("--n_generate", type=int, default=None,
                        help="Samples per seed (default 120, as in the paper)")
    parser.add_argument("--no-anchor", action="store_true",
                        help="Disable anchor masking (include pos 0 in loss)")
    parser.add_argument("--n-reverse", type=int, default=None,
                        help="Override number of reverse SDE steps for generation")
    parser.add_argument("--loss-weighting", type=str, default=None,
                        choices=["uniform", "min_snr_5", "min_snr_3", "min_snr_1", "likelihood", "edm"],
                        help="Loss weighting strategy (default: uniform)")
    parser.add_argument("--precond", type=str, default=None, choices=["none", "edm"],
                        help="Network parameterisation (default none = ε-prediction)")
    parser.add_argument("--edm_sigma_data", type=str, default=None,
                        choices=["scalar", "per_position"],
                        help="EDM σ_data: global std or per-position std of training windows")
    parser.add_argument("--lr", type=float, default=None,
                        help="Initial learning rate (default 1e-3)")
    parser.add_argument("--warmstart_ema", action="store_true",
                        help="With --warmstart: load the checkpoint's EMA weights")
    parser.add_argument("--lr-schedule", type=str, default=None,
                        choices=["multistep", "cosine"],
                        help="LR schedule: multistep (default) or cosine annealing")
    parser.add_argument("--lr-min", type=float, default=None,
                        help="Minimum LR for cosine schedule (default: 0)")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N epochs (default: 100)")
    parser.add_argument("--wavenet-branch", action="store_true",
                        help="Enable parallel WaveNet dilated conv branch in residual blocks")
    parser.add_argument("--film-conditioning", action="store_true",
                        help="Enable FiLM noise-level conditioning in residual blocks")
    parser.add_argument("--spectral-loss", type=float, default=None, metavar="WEIGHT",
                        help="Enable spectral auxiliary loss with given weight λ (e.g. 0.1)")
    parser.add_argument("--sampler", type=str, default=None,
                        choices=["pc", "karras", "em", "ode", "hybrid"],
                        help="Sampling method for generation (default: pc)")
    parser.add_argument("--hybrid_sigma_switch", type=float, default=None,
                        help="Hybrid sampler: σ below which the stochastic SDE takes over")
    parser.add_argument("--reject_failed", action="store_true",
                        help="Reject failed samples (lag-1 return ACF below the real "
                             "0.1th percentile) and resample")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training: re-score <save_dir>/<sde>_<schedule>/best_model.pth "
                             "with the given sampler settings (outputs in eval_<tag>/)")
    parser.add_argument("--eval_tag", type=str, default=None,
                        help="Output subdirectory suffix for --eval_only")
    parser.add_argument("--karras-rho", type=float, default=None,
                        help="Karras sigma schedule exponent ρ (default: 7)")
    parser.add_argument("--warmstart", type=str, default=None,
                        help="Load model weights only (no optimizer/epoch/scheduler). "
                             "Use for warm-starting a new architecture from pretrained weights.")
    args = parser.parse_args()

    if args.minimal:
        config = MINIMAL_CONFIG.copy()
    elif args.quick:
        config = QUICK_CONFIG.copy()
    else:
        config = PAPER_CONFIG.copy()
    if args.sde_type:
        config["sde_type"] = args.sde_type
    if args.schedule:
        config["schedule"] = args.schedule
    if args.use_synthetic:
        config["use_synthetic"] = True
    if args.epochs:
        config["epochs"] = args.epochs
    if args.seq_len:
        config["seq_len"] = args.seq_len
        config["window_len"] = args.seq_len
    if args.stride:
        config["stride"] = args.stride
    if args.zscore:
        config["normalize_mode"] = args.zscore
    if args.no_anchor:
        config["mask_anchor"] = False
    if args.n_reverse:
        config["n_reverse_steps"] = args.n_reverse
    if args.loss_weighting:
        config["loss_weighting"] = args.loss_weighting
    if args.precond:
        config["precond"] = args.precond
    if args.edm_sigma_data:
        config["edm_sigma_data_mode"] = args.edm_sigma_data
    if args.lr is not None:
        config["lr"] = args.lr
    if args.lr_schedule:
        config["lr_schedule"] = args.lr_schedule
    if args.lr_min is not None:
        config["lr_min"] = args.lr_min
    if args.checkpoint_every:
        config["checkpoint_every"] = args.checkpoint_every
    if args.wavenet_branch:
        config["wavenet_branch"] = True
    if args.film_conditioning:
        config["film_conditioning"] = True
    if args.spectral_loss is not None:
        config["spectral_loss_weight"] = args.spectral_loss
    if args.sampler:
        config["sampler"] = args.sampler
    if args.karras_rho is not None:
        config["karras_rho"] = args.karras_rho
    if args.hybrid_sigma_switch is not None:
        config["hybrid_sigma_switch"] = args.hybrid_sigma_switch
    if args.reject_failed:
        config["reject_failed"] = True
    if args.sigma_min is not None:
        config["sigma_min"] = args.sigma_min
    if args.data_mode:
        config["data_mode"] = args.data_mode
    if args.split:
        config["split"] = args.split
    if args.seed is not None:
        config["seed"] = args.seed
    if args.noise_scale is not None:
        config["sampler_noise_scale"] = args.noise_scale
    if args.noise_sigma_range is not None:
        config["sampler_noise_sigma_min"], config["sampler_noise_sigma_max"] = args.noise_sigma_range
    if args.n_seeds is not None:
        config["n_seeds"] = args.n_seeds
    if args.n_generate is not None:
        config["n_generate"] = args.n_generate
    if args.sigma_max:
        try:
            config["sigma_max"] = float(args.sigma_max)
        except ValueError:
            config["sigma_max"] = args.sigma_max  # "auto"

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")
        if not args.quick:
            print("Consider using --quick for CPU-only runs.")

    # Ensure data
    ensure_data(config)

    if args.eval_only:
        cli = {k: v for k, v in config.items()}
        if not args.sampler:
            cli.pop("sampler", None)
        for k in ("n_reverse_steps", "sampling_eps", "pc_corrector_steps", "sampler_noise_scale",
                  "n_generate", "n_seeds", "seed"):
            if getattr(args, {"n_reverse_steps": "n_reverse", "sampler_noise_scale": "noise_scale"}
                       .get(k, k), None) is None:
                cli.pop(k, None)
        run_eval_only(cli, args.save_dir, args.eval_tag)
        return

    # Run
    if args.grid:
        run_grid(config, args.save_dir)
    else:
        run_experiment(config, args.save_dir, resume_path=args.resume,
                       warmstart_path=args.warmstart, warmstart_ema=args.warmstart_ema)


if __name__ == "__main__":
    main()
