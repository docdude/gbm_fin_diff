#!/usr/bin/env python3
"""
L4 GPU training script for GBM Financial Diffusion Model.

Designed for Lightning AI Studio with L4 GPU (24GB VRAM).
Uses paper's full hyperparameters: 128/256/64, 4 layers, 8 heads,
seq_len=2048, 1000 epochs, N=2000.

Multi-stock data: downloads S&P 500 constituents with >40 years history
to match the paper's data collection methodology (Section 3.1.2).

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
                               LONG_HISTORY_TICKERS, compute_sigma_max)
from gbm_financial.metrics import (evaluate_stylized_facts, plot_stylized_facts,
                                   plot_diagnostics, plot_pathwise_diagnostics,
                                   plot_mean_path_diagnostic)


# Paper's full configuration (Section 4)
PAPER_CONFIG = {
    # SDE
    "sde_type": "gbm",
    "schedule": "exponential",
    "sigma_min": 0.01,
    "sigma_max": "auto",   # Auto-computed from data (≈ 3× data range ≈ 8.35)
    "n_reverse_steps": 2000,

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
    "loss_weighting": "uniform",

    # Data (Section 3.1.2)
    "window_len": 2048,
    "stride": 400,
    "min_years": 40,
    "use_synthetic": False,

    # Generation / Sampling
    "n_generate": 120,
    "sampling_eps": 4e-4,
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


def ensure_data(config, csv_path="data/sp500.csv"):
    """Ensure financial data is available.

    Priority:
      1. Local CSV (data/sp500.csv) — single stock, limited windows
      2. yfinance download — multiple stocks, paper's methodology

    For paper reproduction, yfinance with multiple stocks is needed.
    """
    if os.path.exists(csv_path):
        print(f"Found local CSV: {csv_path}")
        # Check if we also have multi-stock cache
        cache_file = "data/financial/sp500_prices.pkl"
        if os.path.exists(cache_file):
            print(f"Also found multi-stock cache: {cache_file}")
        else:
            print("TIP: For paper reproduction, install yfinance to download multiple stocks:")
            print("     pip install yfinance")
            print("     This gives 10-100x more training data.")
    else:
        print("No local CSV found. Attempting yfinance download...")
        try:
            import yfinance
            download_stock_data(LONG_HISTORY_TICKERS, min_years=config.get("min_years", 40))
        except ImportError:
            print("ERROR: yfinance not installed. Install it or provide data/sp500.csv")
            print("       pip install yfinance")
            sys.exit(1)


def run_experiment(config, save_dir, resume_path=None, warmstart_path=None):
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

    # Data
    train_loader, val_loader, data_info = get_dataloaders(
        sde_type=sde_type,
        window_len=config["seq_len"],
        stride=config.get("stride", 400),
        batch_size=config["batch_size"],
        use_synthetic=config.get("use_synthetic", False),
        min_years=config.get("min_years", 40),
        num_workers=2,
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
    if config.get("sigma_max") == "auto":
        sigma_max, sigma_stats = compute_sigma_max(train_loader)
        config["sigma_max"] = sigma_max
        print(f"  σ_max auto-set to {sigma_max:.1f}")
    elif isinstance(config.get("sigma_max"), str):
        raise ValueError(f"Unknown sigma_max value: {config['sigma_max']}")

    # Model
    model = GBMFinancialDiffusion(config)

    if resume_path:
        model.load(resume_path)
        print(f"Resumed from {resume_path}")

    if warmstart_path:
        model.load_weights_only(warmstart_path)
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

    # Generate
    n_gen = config.get("n_generate", 120)
    sampler = config.get("sampler", "pc")
    if sampler == "karras":
        generated = model.generate_karras(n_samples=n_gen, seq_len=config["seq_len"])
    elif sampler == "em":
        generated = model.generate_em(n_samples=n_gen, seq_len=config["seq_len"])
    elif sampler == "ode":
        generated = model.generate_ode(n_samples=n_gen, seq_len=config["seq_len"])
    else:
        generated = model.generate(n_samples=n_gen, seq_len=config["seq_len"])
    np.save(os.path.join(exp_dir, "generated_data.npy"), generated)

    # Get ALL real data for comparison (not just n_gen samples)
    # Hill estimator is highly sample-size sensitive:
    #   120 samples → α ≈ 6.5,  2551 samples → α ≈ 9.95
    # Using a small subset artificially deflates α_real to match α_gen.
    real_data = []
    for batch in train_loader:
        real_data.append(batch.numpy())
    real_data = np.concatenate(real_data, axis=0)

    # Evaluate
    gen_results, real_results = model.evaluate(generated, real_data, save_dir=exp_dir)

    # Full diagnostics plot
    mode = "log_price" if sde_type == "gbm" else "log_return"
    plot_diagnostics(generated, real_data, mode=mode,
                     save_path=os.path.join(exp_dir, "diagnostics.png"))

    # Pathwise diagnostics (Audit D expansion)
    plot_pathwise_diagnostics(generated, real_data, mode=mode,
                              save_path=os.path.join(exp_dir, "pathwise_diagnostics.png"))

    # Cross-sectional mean path diagnostic (z-score diagnosis)
    plot_mean_path_diagnostic(generated, real_data, mode=mode,
                              save_path=os.path.join(exp_dir, "mean_path_diagnostic.png"))

    # Save results
    results = {
        "sde_type": sde_type,
        "schedule": schedule,
        "alpha_gen": gen_results["heavy_tail"]["alpha"],
        "alpha_real": real_results["heavy_tail"]["alpha"] if real_results else None,
        "beta_gen": gen_results["volatility_clustering"]["beta"],
        "beta_real": real_results["volatility_clustering"]["beta"] if real_results else None,
        "n_train": data_info["n_train"],
        "n_stocks": data_info["n_stocks"],
        "train_time_hours": train_time / 3600,
        "config": {k: v for k, v in config.items()
                   if isinstance(v, (int, float, str, bool))},
        # Effective training parameters (includes derived/defaulted values)
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
    }

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
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
    parser.add_argument("--no-anchor", action="store_true",
                        help="Disable anchor masking (include pos 0 in loss)")
    parser.add_argument("--n-reverse", type=int, default=None,
                        help="Override number of reverse SDE steps for generation")
    parser.add_argument("--loss-weighting", type=str, default=None,
                        choices=["uniform", "min_snr_5", "min_snr_3", "min_snr_1", "likelihood"],
                        help="Loss weighting strategy (default: uniform)")
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
                        choices=["pc", "karras", "em", "ode"],
                        help="Sampling method for generation (default: pc)")
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

    # Run
    if args.grid:
        run_grid(config, args.save_dir)
    else:
        run_experiment(config, args.save_dir, resume_path=args.resume,
                       warmstart_path=args.warmstart)


if __name__ == "__main__":
    main()
