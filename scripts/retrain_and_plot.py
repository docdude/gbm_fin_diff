#!/usr/bin/env python3
"""
Retrain from scratch with all fixes, generate, and plot diagnostics.

Fixes applied:
  1. dim_feedforward = 4*channels (not hardcoded 64)
  2. Data standardization (normalize to zero mean, unit std)
  3. Probability flow ODE sampler (no stochastic noise accumulation)
  4. N=2000 reverse steps (paper setting)
"""
import os
import sys
import time
import numpy as np
import torch

# Allow running from scripts/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gbm_financial.train import GBMFinancialDiffusion
from gbm_financial.data import load_csv_data, create_subsequences
from gbm_financial.metrics import (
    evaluate_stylized_facts, plot_stylized_facts, plot_diagnostics
)
from torch.utils.data import DataLoader, TensorDataset

SAVE_DIR = "save/retrain_v3"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Configuration ===
SEQ_LEN = 256
EPOCHS = 300
N_REVERSE = 2000

config = {
    "channels": 64,
    "diff_emb_dim": 128,
    "feat_emb_dim": 32,
    "time_emb_dim": 64,
    "n_layers": 2,
    "n_heads": 4,
    "sde_type": "gbm",
    "schedule": "exponential",
    "sigma_min": 0.01,
    "sigma_max": 1.0,
    "n_reverse_steps": N_REVERSE,
    "epochs": EPOCHS,
    "batch_size": 16,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "ema_decay": 0.999,
    "likelihood_weighting": False,
    "seq_len": SEQ_LEN,
    "normalize_data": True,
}

# === Load data ===
stock_data = load_csv_data("data/sp500.csv")
sequences = create_subsequences(stock_data, window_len=SEQ_LEN, stride=25, mode="log_price")
print(f"Total sequences: {sequences.shape}")
print(f"Data stats: mean={sequences.mean():.4f}, std={sequences.std():.4f}")
print(f"Return stats: mean={np.diff(sequences, axis=-1).mean():.6f}, "
      f"std={np.diff(sequences, axis=-1).std():.6f}")

# Train/val split
np.random.seed(42)
perm = np.random.permutation(len(sequences))
n_train = int(len(sequences) * 0.9)
train_seqs = sequences[perm[:n_train]]
val_seqs = sequences[perm[n_train:]]
real_data = sequences[:60]

train_ds = TensorDataset(torch.from_numpy(train_seqs).float())
val_ds = TensorDataset(torch.from_numpy(val_seqs).float())
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# Wrap DataLoader to yield (B, L) instead of ((B, L),)
class UnwrapLoader:
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for (batch,) in self.loader:
            yield batch
    def __len__(self):
        return len(self.loader)

train_loader_unwrap = UnwrapLoader(train_loader)
val_loader_unwrap = UnwrapLoader(val_loader)

# === Train ===
pipeline = GBMFinancialDiffusion(config)

print(f"\n{'='*60}")
print(f"Training: {EPOCHS} epochs, seq_len={SEQ_LEN}, N={N_REVERSE}")
print(f"normalize_data={config['normalize_data']}")
print(f"{'='*60}")

start = time.time()
pipeline.train(train_loader_unwrap, val_loader_unwrap, save_dir=SAVE_DIR)
elapsed = time.time() - start
print(f"Training time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

# === Generate: both SDE and ODE ===
print("\n=== Generating with ODE (deterministic, N=2000) ===")
gen_ode = pipeline.generate_ode(n_samples=60, seq_len=SEQ_LEN, batch_size=20)

print("\n=== Generating with SDE (stochastic, N=500) ===")
pipeline.config["n_reverse_steps"] = 500
pipeline.sde.N = 500
gen_sde = pipeline.generate(n_samples=60, seq_len=SEQ_LEN, batch_size=20)

# Save
np.save(os.path.join(SAVE_DIR, "gen_sde.npy"), gen_sde)
np.save(os.path.join(SAVE_DIR, "gen_ode.npy"), gen_ode)

# === Compare stats ===
print("\n" + "="*70)
print("COMPARISON: Real vs Generated (SDE) vs Generated (ODE)")
print("="*70)

real_rets = np.diff(real_data, axis=-1)
sde_rets = np.diff(gen_sde, axis=-1)
ode_rets = np.diff(gen_ode, axis=-1)

from scipy.stats import kurtosis

print(f"{'':12s}  {'LogP std':>10s}  {'Ret std':>10s}  {'Kurtosis':>10s}")
print("-"*50)
print(f"{'Real':12s}  {real_data.std():10.4f}  {real_rets.std():10.6f}  "
      f"{kurtosis(real_rets.flatten()):10.2f}")
print(f"{'Gen (SDE)':12s}  {gen_sde.std():10.4f}  {sde_rets.std():10.6f}  "
      f"{kurtosis(sde_rets.flatten()):10.2f}")
print(f"{'Gen (ODE)':12s}  {gen_ode.std():10.4f}  {ode_rets.std():10.6f}  "
      f"{kurtosis(ode_rets.flatten()):10.2f}")
print(f"\nRet std ratio: SDE/real={sde_rets.std()/real_rets.std():.2f}x, "
      f"ODE/real={ode_rets.std()/real_rets.std():.2f}x")

# === Plot diagnostics ===
print("\n--- Plotting ODE diagnostics ---")
plot_diagnostics(gen_ode, real_data, mode="log_price",
                 save_path=os.path.join(SAVE_DIR, "diagnostics_ode.png"))

print("--- Plotting SDE diagnostics ---")
plot_diagnostics(gen_sde, real_data, mode="log_price",
                 save_path=os.path.join(SAVE_DIR, "diagnostics_sde.png"))

# === Side-by-side return distribution comparison ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

clip = np.percentile(np.abs(real_rets.flatten()), 99.5) * 2
bins = np.linspace(-clip, clip, 60)

for ax, (rets, label, color) in zip(axes, [
    (real_rets, "Real", "red"),
    (sde_rets, "Generated (SDE, N=500)", "blue"),
    (ode_rets, "Generated (ODE, N=2000)", "green"),
]):
    flat = rets.flatten()
    ax.hist(flat, bins=bins, density=True, alpha=0.7, color=color, label=label)
    x = np.linspace(-clip, clip, 200)
    ax.plot(x, norm.pdf(x, flat.mean(), flat.std()), 'k--', linewidth=1.5,
            label=f'N(0, {flat.std():.4f})')
    ax.set_title(f"{label}\nstd={flat.std():.5f}, kurt={kurtosis(flat):.1f}")
    ax.set_xlabel("Log-return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_xlim(-clip, clip)

plt.suptitle("Return Distribution: Real vs SDE vs ODE", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "return_distributions.png"), dpi=150,
            bbox_inches="tight")
plt.close()

# === Overlay histogram ===
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(real_rets.flatten(), bins=bins, density=True, alpha=0.5, color="red", label="Real")
ax.hist(ode_rets.flatten(), bins=bins, density=True, alpha=0.5, color="blue", label="ODE")
ax.hist(sde_rets.flatten(), bins=bins, density=True, alpha=0.3, color="green", label="SDE")
ax.set_xlabel("Log-return")
ax.set_ylabel("Density")
ax.set_title("Return Distribution Overlay")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "return_overlay.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll plots saved to {SAVE_DIR}/")

# === Evaluate stylized facts ===
print("\n--- ODE Stylized Facts ---")
evaluate_stylized_facts(gen_ode, mode="log_price")
print("\n--- Real Stylized Facts ---")
evaluate_stylized_facts(real_data, mode="log_price")

print(f"\n{'='*60}")
print("DONE — check plots in:", SAVE_DIR)
print(f"{'='*60}")
