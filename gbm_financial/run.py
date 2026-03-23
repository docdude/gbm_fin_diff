#!/usr/bin/env python3
"""
Main entry point for the GBM Financial Diffusion Model.

Reproduces the experiments from:
  "A diffusion-based generative model for financial time series via
   geometric Brownian motion" (arXiv:2507.19003)

Usage:
    # Run with default config (GBM + exponential schedule):
    python -m gbm_financial.run

    # Run specific SDE + schedule combination:
    python -m gbm_financial.run --sde_type gbm --schedule cosine

    # Run full experiment grid (all 9 combinations):
    python -m gbm_financial.run --experiment_grid

    # Quick test with synthetic data:
    python -m gbm_financial.run --use_synthetic --epochs 10 --n_reverse_steps 100

    # Generate from trained model:
    python -m gbm_financial.run --mode generate --checkpoint save/gbm_financial/final_model.pth
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import json

from gbm_financial.train import GBMFinancialDiffusion
from gbm_financial.data import get_dataloaders
from gbm_financial.metrics import evaluate_stylized_facts, plot_stylized_facts


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_single_experiment(config, save_dir):
    """Run a single training + generation + evaluation experiment.

    This implements the full pipeline from the paper:
    1. Load/prepare financial data (S&P 500 log-prices or log-returns)
    2. Train score network with denoising score matching
    3. Generate 120 synthetic time series via reverse SDE
    4. Evaluate stylized facts (heavy tails, vol clustering, leverage effect)
    """
    sde_type = config["sde_type"]
    schedule = config["schedule"]
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: SDE={sde_type.upper()}, Schedule={schedule}")
    print(f"{'='*70}")

    # 1. Data loading
    train_loader, val_loader, data_info = get_dataloaders(
        sde_type=sde_type,
        window_len=config["seq_len"],
        stride=config.get("stride", 400),
        batch_size=config["batch_size"],
        use_synthetic=config.get("use_synthetic", False),
        min_years=config.get("min_years", 40),
        cache_dir=os.path.join(save_dir, "data"),
        num_workers=0,
    )
    print(f"Data: {data_info['n_train']} train, {data_info['n_val']} val sequences")
    print(f"  mode={data_info['mode']}, seq_len={data_info['seq_len']}")

    # 2. Initialize model
    model = GBMFinancialDiffusion(config)

    # 3. Train
    exp_dir = os.path.join(save_dir, f"{sde_type}_{schedule}")
    model.train(train_loader, val_loader, save_dir=exp_dir)

    # 4. Generate synthetic data
    n_gen = config.get("n_generate", 120)
    generated = model.generate(n_samples=n_gen, seq_len=config["seq_len"])

    # Save generated data
    np.save(os.path.join(exp_dir, "generated_data.npy"), generated)

    # 5. Evaluate
    # Get some real data for comparison
    real_data = None
    for batch in train_loader:
        real_data = batch.numpy()
        if len(real_data) >= n_gen:
            real_data = real_data[:n_gen]
            break

    gen_results, real_results = model.evaluate(generated, real_data, save_dir=exp_dir)

    # Save results
    results_summary = {
        "sde_type": sde_type,
        "schedule": schedule,
        "heavy_tail_alpha": gen_results["heavy_tail"]["alpha"],
        "vol_clustering_beta": gen_results["volatility_clustering"]["beta"],
        "leverage_lag0": gen_results["leverage_effect"]["leverage_correlation"][0],
        "n_generated": n_gen,
        "config": {k: v for k, v in config.items()
                   if isinstance(v, (int, float, str, bool))},
    }
    if real_results:
        results_summary["real_heavy_tail_alpha"] = real_results["heavy_tail"]["alpha"]
        results_summary["real_vol_clustering_beta"] = real_results["volatility_clustering"]["beta"]

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    return results_summary


def run_experiment_grid(base_config, save_dir):
    """Run all 9 SDE × schedule combinations from the paper (Table in Section 4).

    Grid:
      - SDEs: VE, VP, GBM
      - Schedules: linear, exponential, cosine
    """
    sde_types = ["ve", "vp", "gbm"]
    schedules = ["linear", "exponential", "cosine"]

    all_results = []

    for sde_type in sde_types:
        for schedule in schedules:
            config = base_config.copy()
            config["sde_type"] = sde_type
            config["schedule"] = schedule

            try:
                result = run_single_experiment(config, save_dir)
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

    # Print comparison table
    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'SDE':<6} {'Schedule':<12} {'α (tail)':<12} {'β (vol.cl.)':<12} {'Lev(0)':<10}")
    print("-" * 56)
    for r in all_results:
        if "error" in r:
            print(f"{r['sde_type']:<6} {r['schedule']:<12} ERROR: {r['error'][:40]}")
        else:
            alpha = r.get("heavy_tail_alpha", float("nan"))
            beta = r.get("vol_clustering_beta", float("nan"))
            lev = r.get("leverage_lag0", float("nan"))
            print(f"{r['sde_type']:<6} {r['schedule']:<12} {alpha:<12.2f} {beta:<12.3f} {lev:<10.4f}")
    print(f"\nReference: S&P 500 empirical α ≈ 4.35")
    print(f"{'='*80}")

    # Save summary
    with open(os.path.join(save_dir, "grid_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def generate_from_checkpoint(config, checkpoint_path, save_dir):
    """Load a trained model and generate samples."""
    model = GBMFinancialDiffusion(config)
    model.load(checkpoint_path)

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

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "generated_data.npy"), generated)

    gen_results, _ = model.evaluate(generated, save_dir=save_dir)
    return generated, gen_results


def main():
    parser = argparse.ArgumentParser(
        description="GBM Financial Diffusion Model (arXiv:2507.19003)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "generate", "evaluate"],
                        help="Run mode")
    parser.add_argument("--experiment_grid", action="store_true",
                        help="Run full 3×3 experiment grid")
    parser.add_argument("--sde_type", type=str, default=None,
                        choices=["ve", "vp", "gbm"])
    parser.add_argument("--schedule", type=str, default=None,
                        choices=["linear", "exponential", "cosine"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--n_reverse_steps", type=int, default=None)
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic GBM data (no download)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for generate/evaluate)")
    parser.add_argument("--save_dir", type=str, default="save/gbm_financial",
                        help="Output directory")
    parser.add_argument("--sampler", type=str, default=None,
                        choices=["pc", "karras", "em", "ode"],
                        help="Sampling method for generation (default: pc)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command-line arguments
    for key in ["sde_type", "schedule", "epochs", "batch_size", "seq_len", "n_reverse_steps"]:
        val = getattr(args, key)
        if val is not None:
            config[key] = val
    if args.use_synthetic:
        config["use_synthetic"] = True
    if args.sampler:
        config["sampler"] = args.sampler
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else args.device

    print("Configuration:")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")

    # Run
    if args.experiment_grid:
        run_experiment_grid(config, args.save_dir)
    elif args.mode == "train":
        run_single_experiment(config, args.save_dir)
    elif args.mode == "generate":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for generate mode")
            sys.exit(1)
        generate_from_checkpoint(config, args.checkpoint, args.save_dir)
    elif args.mode == "evaluate":
        data_path = os.path.join(args.save_dir, "generated_data.npy")
        if not os.path.exists(data_path):
            print(f"ERROR: No generated data found at {data_path}")
            sys.exit(1)
        data = np.load(data_path)
        mode = "log_price" if config["sde_type"] == "gbm" else "log_return"
        evaluate_stylized_facts(data, mode=mode)


if __name__ == "__main__":
    main()
