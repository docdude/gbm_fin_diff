#!/usr/bin/env python3
"""Evaluate a checkpoint mid-training: generate samples & compute pathwise metrics."""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gbm_financial.train import GBMFinancialDiffusion
from gbm_financial.train_l4 import PAPER_CONFIG
from gbm_financial.data import get_dataloaders, compute_sigma_max
from gbm_financial.metrics import (compute_pathwise_diagnostics,
                                   print_pathwise_summary,
                                   compute_log_returns)

# ── Config ──────────────────────────────────────────────────────────────
CHECKPOINT = "save/gbm_financial/best_model(1).pth"
N_GENERATE = 20           # fewer for CPU speed
N_REVERSE = 500           # fewer steps for CPU (still reasonable quality)
SAVE_DIR = "save/gbm_financial/eval_paper_600"

# ── Load real data ──────────────────────────────────────────────────────
config = dict(PAPER_CONFIG)

print("Loading data...")
# Use a non-existent CSV path so it falls through to pkl cache
train_loader, val_loader, dataset_info = get_dataloaders(
    csv_path="data/sp500_SKIP.csv",          # skip CSV, use pkl cache
    cache_dir="/opt/CSDI/data/financial",     # where the 105-ticker pkl lives
    window_len=config["window_len"],
    stride=config["stride"],
    batch_size=config["batch_size"],
    sde_type="gbm",
    min_years=config["min_years"],
)

# Auto σ_max
sigma_max, _ = compute_sigma_max(train_loader)
config["sigma_max"] = sigma_max
print(f"σ_max (auto) = {sigma_max:.2f}")

# Gather real data for comparison
real_data = []
for batch in train_loader:
    real_data.append(batch.numpy())
for batch in val_loader:
    real_data.append(batch.numpy())
real_data = np.concatenate(real_data, axis=0)
print(f"Real data: {real_data.shape}")

# ── Load model ──────────────────────────────────────────────────────────
print(f"\nLoading checkpoint: {CHECKPOINT}")
config_gen = dict(config)
config_gen["n_reverse_steps"] = N_REVERSE
config_gen["n_generate"] = N_GENERATE

device = torch.device("cpu")  # local GPU incompatible, use CPU
model = GBMFinancialDiffusion(config_gen, device=device)
model.load(CHECKPOINT)
print("Checkpoint loaded.")

# ── Generate ────────────────────────────────────────────────────────────
generated = model.generate(n_samples=N_GENERATE, batch_size=5)
os.makedirs(SAVE_DIR, exist_ok=True)
np.save(os.path.join(SAVE_DIR, "generated_data.npy"), generated)

# ── Quick scalar metrics (return AC, path std, endpoints) ───────────────
def quick_scalar_comparison(gen, real, label=""):
    """Compute the key scalar metrics we tracked in the QUICK run."""
    ret_r = compute_log_returns(real, mode="log_price")
    ret_g = compute_log_returns(gen, mode="log_price")

    def ac1(r):
        """Return AC(1) across all paths."""
        acs = []
        for i in range(r.shape[0]):
            if r.shape[1] > 1:
                c = np.corrcoef(r[i, :-1], r[i, 1:])[0, 1]
                acs.append(c)
        return np.nanmean(acs)

    metrics = {
        "Return std":       (np.std(ret_r), np.std(ret_g)),
        "Return AC(1)":     (ac1(ret_r), ac1(ret_g)),
        "Path std":         (np.std(real), np.std(gen)),
        "Endpoint mean":    (np.mean(real[:, -1]), np.mean(gen[:, -1])),
        "Endpoint std":     (np.std(real[:, -1]), np.std(gen[:, -1])),
        "Total variation":  (np.mean(np.sum(np.abs(np.diff(real, axis=1)), axis=1)),
                             np.mean(np.sum(np.abs(np.diff(gen, axis=1)), axis=1))),
        "Quadratic var":    (np.mean(np.sum(np.diff(real, axis=1)**2, axis=1)),
                             np.mean(np.sum(np.diff(gen, axis=1)**2, axis=1))),
        "Max drawdown":     (np.mean([np.min(real[i] - np.maximum.accumulate(real[i]))
                                       for i in range(real.shape[0])]),
                             np.mean([np.min(gen[i] - np.maximum.accumulate(gen[i]))
                                       for i in range(gen.shape[0])])),
    }

    print(f"\n{'='*70}")
    print(f"SCALAR METRICS {label}")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Real':>12} {'Generated':>12} {'Ratio':>8} {'Status':>8}")
    print("-" * 70)
    for name, (r, g) in metrics.items():
        if abs(r) > 1e-12:
            ratio = g / r
        else:
            ratio = float('nan')
        if 0.8 <= abs(ratio) <= 1.2:
            status = "OK"
        elif 0.5 <= abs(ratio) <= 2.0:
            status = "~"
        else:
            status = "!!"
        print(f"{name:<25} {r:>12.4f} {g:>12.4f} {ratio:>8.2f} {status:>8}")
    return metrics

paper_metrics = quick_scalar_comparison(generated, real_data, "(PAPER @ ~epoch 600)")

# ── Full pathwise diagnostics ────────────────────────────────────────
gen_pw = compute_pathwise_diagnostics(generated, mode="log_price")
real_pw = compute_pathwise_diagnostics(real_data, mode="log_price")
print_pathwise_summary(gen_pw, real_pw)

# ── Compare against QUICK_CONFIG results ─────────────────────────────
quick_gen_path = "save/gbm_financial/generated_data.npy"
if os.path.exists(quick_gen_path):
    quick_gen = np.load(quick_gen_path)
    print(f"\n{'='*70}")
    print("QUICK vs PAPER — key scalar metrics")
    print(f"{'='*70}")
    quick_scalar_comparison(quick_gen, real_data, "(QUICK)")

# ── Evaluate (plots + stylized facts) ──────────────────────────────────
model.evaluate(generated, real_data, save_dir=SAVE_DIR)
print(f"\nPlots saved to {SAVE_DIR}/")
