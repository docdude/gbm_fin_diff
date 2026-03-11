# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Config D: Paper's actual approach - GBM on cumulative log-prices, sigma_max=1.0, no z-score.

Paper (arXiv:2507.19003):
  - Section 3.1: X_0 = {log s_1, ..., log s_L} (cumulative log-prices)
  - Section 4: sigma_min=0.01, sigma_max=1.0, N=2000, 1000 epochs
  - GBM SDE in price space = VE SDE in log-price space
  - GBM outperforms VE/VP because cumulative log-price representation
    induces heteroskedasticity when mapped back to prices via exp()

This is the configuration we've never tried:
  - log-prices + sigma_max=1.0 + NO normalization
"""
import os
import sys
import time
import numpy as np
import torch

# Allow running from scripts/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gbm_financial.train import GBMFinancialDiffusion
from gbm_financial.data import get_dataloaders
from gbm_financial.metrics import (evaluate_stylized_facts, plot_diagnostics,
                                   plot_pathwise_diagnostics,
                                   plot_mean_path_diagnostic)

CONFIG = {
    # GBM SDE -> routes to data_mode="log_price" (cumulative log-prices)
    "sde_type": "gbm",
    "schedule": "exponential",
    "sigma_min": 0.01,
    "sigma_max": 1.0,          # Paper Section 4: exact value
    "n_reverse_steps": 500,    # Reduced for CPU speed (paper uses 2000)

    # Small architecture for CPU
    "channels": 64,
    "diff_emb_dim": 128,
    "feat_emb_dim": 32,
    "time_emb_dim": 128,
    "n_layers": 2,
    "n_heads": 4,

    # Short sequences for CPU speed
    "seq_len": 256,
    "window_len": 256,
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "ema_decay": 0.999,
    "likelihood_weighting": False,

    # Data - stride=200 for manageable batch count on CPU
    "stride": 200,
    "min_years": 40,
    "use_synthetic": False,
    "normalize_data": False,   # NO z-score (paper doesn't mention normalization)

    # Generation
    "n_generate": 60,
}


def main():
    save_dir = "save/gbm_logprice_sigma1"
    exp_dir = os.path.join(save_dir, "gbm_exponential")
    os.makedirs(exp_dir, exist_ok=True)

    print("=" * 60)
    print("CONFIG D: GBM on log-prices | sigma_max=1.0 | no z-score")
    print("  (Paper's actual approach)")
    print("=" * 60)

    # Data - sde_type="gbm" -> mode="log_price" automatically
    train_loader, val_loader, data_info = get_dataloaders(
        sde_type="gbm",
        window_len=CONFIG["seq_len"],
        stride=CONFIG["stride"],
        batch_size=CONFIG["batch_size"],
        min_years=CONFIG.get("min_years", 40),
        num_workers=0,
    )

    # Print data stats
    sample = next(iter(train_loader))
    print(f"\nData: {data_info['n_train']} train, {data_info['n_val']} val "
          f"({data_info['n_stocks']} stocks)")
    print(f"  Sample shape: {sample.shape}")
    print(f"  Sample stats: mean={sample.mean():.4f}, std={sample.std():.4f}, "
          f"min={sample.min():.4f}, max={sample.max():.4f}")
    print(f"  sigma_max/data_std ratio: {CONFIG['sigma_max'] / sample.std():.2f}x")
    print(f"  NOTE: ratio < 3 means prior N(0,1) does NOT fully cover data")
    print(f"  This matches the paper's sigma_max=1.0 on log-prices")

    n_batches = len(train_loader)
    total_steps = CONFIG["epochs"] * n_batches
    print(f"  {n_batches} batches/epoch -> {total_steps} total steps")

    # Model
    model = GBMFinancialDiffusion(CONFIG)

    # Train
    start = time.time()
    model.train(train_loader, val_loader, save_dir=exp_dir)
    elapsed = time.time() - start
    print(f"\nTraining: {elapsed / 60:.1f} min")

    # Generate
    n_gen = CONFIG["n_generate"]
    generated = model.generate(n_samples=n_gen, seq_len=CONFIG["seq_len"])
    np.save(os.path.join(exp_dir, "generated_data.npy"), generated)

    # Get real data for comparison
    real_data = []
    for batch in train_loader:
        real_data.append(batch.numpy())
        if sum(len(b) for b in real_data) >= n_gen:
            break
    real_data = np.concatenate(real_data, axis=0)[:n_gen]

    # Evaluate
    gen_results, real_results = model.evaluate(generated, real_data, save_dir=exp_dir)

    # Additional plots
    mode = "log_price"
    plot_diagnostics(generated, real_data, mode=mode,
                     save_path=os.path.join(exp_dir, "diagnostics.png"))
    plot_pathwise_diagnostics(generated, real_data, mode=mode,
                              save_path=os.path.join(exp_dir, "pathwise_diagnostics.png"))
    plot_mean_path_diagnostic(generated, real_data, mode=mode,
                              save_path=os.path.join(exp_dir, "mean_path_diagnostic.png"))

    print(f"\nResults saved to {exp_dir}/")
    print("\nCompare with VE-on-log-returns run in save/ve_logreturns_quick/")


if __name__ == "__main__":
    main()
