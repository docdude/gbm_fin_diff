#!/usr/bin/env python3
"""Load checkpoint and generate + plot (skip training)."""
import os
import sys
import numpy as np
import torch

# Allow running from scripts/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gbm_financial.train import GBMFinancialDiffusion
from gbm_financial.data import load_csv_data, create_subsequences
from gbm_financial.metrics import (
    evaluate_stylized_facts, plot_stylized_facts, plot_diagnostics
)

SAVE_DIR = "save/plot_from_checkpoint"

config = {
    "channels": 64, "diff_emb_dim": 128, "feat_emb_dim": 32,
    "time_emb_dim": 64, "n_layers": 2, "n_heads": 4,
    "sde_type": "gbm", "schedule": "exponential",
    "sigma_min": 0.01, "sigma_max": 1.0,
    "n_reverse_steps": 500, "epochs": 200,
    "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-6,
    "ema_decay": 0.999, "likelihood_weighting": False, "seq_len": 256,
}

# Load data
stock_data = load_csv_data("data/sp500.csv")
sequences = create_subsequences(stock_data, window_len=256, stride=25, mode="log_price")
real_data = sequences[:50]

# Load model
pipeline = GBMFinancialDiffusion(config)
pipeline.load(os.path.join(SAVE_DIR, "model.pth"))

# Generate
generated = pipeline.generate(n_samples=50, seq_len=256, batch_size=25)
np.save(os.path.join(SAVE_DIR, "generated.npy"), generated)

# Stats
gen_returns = np.diff(generated, axis=-1)
real_returns = np.diff(real_data, axis=-1)
print(f"\nLog-price — Real std: {real_data.std():.4f}, Gen std: {generated.std():.4f}, "
      f"ratio: {generated.std()/real_data.std():.1f}x")
print(f"Returns — Real std: {real_returns.std():.6f}, Gen std: {gen_returns.std():.6f}, "
      f"ratio: {gen_returns.std()/real_returns.std():.1f}x")

# Plot diagnostics
plot_diagnostics(generated, real_data, mode="log_price",
                 save_path=os.path.join(SAVE_DIR, "diagnostics.png"))

# Evaluate
print("\n--- Generated ---")
gen_results = evaluate_stylized_facts(generated, mode="log_price")
print("\n--- Real ---")
real_results = evaluate_stylized_facts(real_data, mode="log_price")

plot_stylized_facts(gen_results, real_results,
                    save_path=os.path.join(SAVE_DIR, "stylized_facts.png"))

print(f"\nPlots saved to {SAVE_DIR}/")
