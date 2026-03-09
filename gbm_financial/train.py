"""
Training, sampling, and evaluation pipeline for GBM financial diffusion model.

Combines:
  - Continuous-time score matching loss (from score_sde framework)
  - CSDI-based score network (adapted per paper Section 3.1.1)
  - VE/VP/GBM forward SDEs (paper Section 3.1)

Training procedure (paper Section 4):
  - Batch size: 64
  - Epochs: 1000
  - Optimizer: Adam with lr scheduling
  - T = 1, σ_min = 0.01, σ_max = 1.0
  - Continuous-time score matching with noise prediction

Sampling:
  - Euler-Maruyama on reverse SDE
  - N = 2000 discretization steps
  - Generate 120 samples of length 2048
"""

import os
import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from .sde import get_sde, get_sigma
from .score_network import FinancialScoreNetwork
from .metrics import (evaluate_stylized_facts, plot_stylized_facts,
                      plot_diagnostics, plot_pathwise_diagnostics)


class EMA:
    """Exponential Moving Average of model parameters.

    Standard technique for diffusion models (score_sde uses decay=0.999).
    Maintains a shadow copy of parameters: θ_ema = decay * θ_ema + (1-decay) * θ.
    The EMA copy typically produces higher-quality samples than the raw weights.

    Usage:
        ema = EMA(model, decay=0.999)
        # after each optimizer step:
        ema.update()
        # for generation:
        ema.apply_shadow()   # swap in EMA weights
        samples = model.generate()
        ema.restore()        # swap back training weights
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # Initialize shadow params as copies of current params
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters after an optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self):
        """Swap model params with EMA shadow params (for generation/eval)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original training params after generation/eval."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}


class GBMFinancialDiffusion:
    """Complete training and generation pipeline.

    This class ties together the SDE framework (from score_sde),
    the score network (from CSDI), and the evaluation metrics.
    """

    def __init__(self, config, device=None):
        """
        Args:
            config: dict with all configuration parameters
            device: torch device (auto-detected if None)
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SDE
        self.sde = get_sde(
            sde_type=config["sde_type"],
            schedule=config["schedule"],
            sigma_min=config["sigma_min"],
            sigma_max=config["sigma_max"],
            N=config["n_reverse_steps"],
        )

        # Score network — subclasses CSDI_base, applies paper's modifications
        self.model = FinancialScoreNetwork(config, self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Score network: {n_params:,} parameters")

        # EMA — standard for diffusion models (score_sde uses decay=0.999)
        # Paper doesn't mention EMA, but it's a free quality improvement.
        ema_decay = config.get("ema_decay", 0.999)
        self.use_ema = ema_decay > 0
        if self.use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
            print(f"EMA enabled: decay={ema_decay}")
        else:
            self.ema = None

        # Likelihood weighting for VE/GBM SDEs
        # score_sde uses λ(t) = σ_t² to ensure equal contribution across noise scales
        self.use_likelihood_weighting = config.get("likelihood_weighting", False)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-6)
        )

        # LR scheduler
        epochs = config["epochs"]
        p1 = int(0.75 * epochs)
        p2 = int(0.9 * epochs)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[p1, p2], gamma=0.1
        )

        # Data mode — with validation
        self.data_mode = "log_price" if config["sde_type"] == "gbm" else "log_return"
        if config["sde_type"] == "gbm":
            print(f"  GBM SDE → data_mode=log_price (cumulative log-returns)")
        else:
            print(f"  {config['sde_type'].upper()} SDE → data_mode=log_return (raw log-returns)")

        # Data standardization: normalize training data to zero mean, unit std.
        # This ensures the noise schedule (sigma_min=0.01..sigma_max=1.0) is
        # well-matched to the data scale. Without this, data with std=0.17 has
        # SNR<1 for most of the diffusion schedule, making low-noise regimes
        # almost unreachable by the model.
        # After generation, samples are denormalized back to original scale.
        self.data_mean = 0.0
        self.data_std = 1.0
        self.normalize_data = config.get("normalize_data", True)

        # Anchor-zero masking: when data_mode='log_price', each subsequence is
        # anchored at 0 via window - window[0].  The first timestep is always
        # deterministic.  If mask_anchor=True, exclude it from the denoising loss
        # so the network does not waste capacity learning an always-zero anchor.
        self.mask_anchor = config.get("mask_anchor", True) and self.data_mode == "log_price"
        if self.mask_anchor:
            print(f"  Anchor masking: ON (first timestep excluded from loss)")

    def compute_loss(self, x_0, eps=1e-5):
        """Compute continuous-time denoising score matching loss.

        Following score_sde's get_sde_loss_fn:
            1. Sample t ~ U(eps, T)
            2. Sample ε ~ N(0, I)
            3. Compute x_t = mean + std * ε
            4. Predict noise ε_θ(x_t, t)
            5. Loss = ||ε - ε_θ||² (weighted by σ² for VE)

        For VE/GBM: x_t = x_0 + σ_t * ε → target = ε
        For VP:     x_t = √α_t * x_0 + √(1-α_t) * ε → target = ε

        When mask_anchor=True and data_mode='log_price', the loss at position 0
        is zeroed out because the first timestep is deterministic (always 0).
        This prevents the model from overfitting to the boundary condition at
        the expense of learning local dynamics in the interior of the path.

        Args:
            x_0: (B, L) clean data
            eps: minimum time value for numerical stability
        Returns:
            scalar loss
        """
        B = x_0.shape[0]

        # Sample diffusion time uniformly
        t = torch.rand(B, device=self.device) * (self.sde.T - eps) + eps  # U(eps, T)

        # Sample noise
        z = torch.randn_like(x_0)

        # Compute perturbed data: x_t = mean + std * z
        mean, std = self.sde.marginal_prob(x_0, t)
        # std has shape (B,) → expand for broadcasting
        if std.dim() == 1:
            x_t = mean + std.unsqueeze(-1) * z
        else:
            x_t = mean + std * z

        # Predict noise
        eps_pred = self.model(x_t, t)

        # Denoising score matching loss
        # For VE/GBM: loss = ||ε - ε_θ||² (equivalent to ||σ·score + ε||²)
        # For VP: same ε-prediction loss
        losses = (z - eps_pred) ** 2

        # Mask anchor position: the first timestep is deterministic (always 0)
        # when training on anchored log-price paths.  Zeroing its loss prevents
        # the network from overfitting to the boundary condition.
        if self.mask_anchor:
            losses = losses.clone()
            losses[:, 0] = 0.0

        # Optional likelihood weighting: λ(t) = g(t)²/σ_t²
        # From score_sde: loss = g² * ||score + z/σ||² = (g²/σ²) * ||z - ε_θ||²
        # For VE+exponential: g²/σ² = 2·ln(σ_max/σ_min) (constant — no effect)
        # For other schedules: reweights to emphasize low-noise regime
        if self.use_likelihood_weighting:
            _, diffusion = self.sde.sde(x_0, t)  # g(t)
            g2 = diffusion.pow(2)  # g(t)²
            weights = g2 / std.pow(2).clamp(min=1e-12)  # g²/σ²
            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)
            losses = losses * weights

        loss = losses.mean()

        return loss

    def train(self, train_loader, val_loader=None, save_dir="save/gbm_financial"):
        """Train the score network.

        Args:
            train_loader: DataLoader with (B, L) tensors
            val_loader: optional validation DataLoader
            save_dir: directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.config
        epochs = config["epochs"]
        best_val_loss = float("inf")

        # Compute data statistics for standardization
        if self.normalize_data:
            all_data = []
            for batch in train_loader:
                all_data.append(batch)
            all_data = torch.cat(all_data, dim=0)
            self.data_mean = all_data.mean().item()
            self.data_std = all_data.std().item()
            if self.data_std < 1e-8:
                self.data_std = 1.0
            print(f"  Data standardization: mean={self.data_mean:.4f}, std={self.data_std:.4f}")

        print(f"\nTraining: SDE={config['sde_type']}, schedule={config['schedule']}")
        print(f"  epochs={epochs}, batch_size={config['batch_size']}, lr={config['lr']}")
        print(f"  σ_min={config['sigma_min']}, σ_max={config['sigma_max']}")
        print(f"  device={self.device}")
        print(f"  data mode={self.data_mode}")
        if self.normalize_data:
            print(f"  data normalization: ON (mean={self.data_mean:.4f}, std={self.data_std:.4f})")
        else:
            print(f"  data normalization: OFF")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}",
                        mininterval=5.0, leave=False)
            for batch in pbar:
                x_0 = batch.to(self.device)  # (B, L)

                # Standardize if enabled
                if self.normalize_data:
                    x_0 = (x_0 - self.data_mean) / self.data_std

                self.optimizer.zero_grad()
                loss = self.compute_loss(x_0)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                # Update EMA after each optimizer step
                if self.use_ema:
                    self.ema.update()

                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{epoch_loss / n_batches:.6f}")

            self.scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
            val_loss_str = ""
            if val_loader is not None and (epoch + 1) % 20 == 0:
                val_loss = self._validate(val_loader)
                val_loss_str = f", val_loss={val_loss:.4e}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save(os.path.join(save_dir, "best_model.pth"))

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch + 1:4d}/{epochs}: loss={avg_loss:.4e}, lr={lr:.2e}{val_loss_str}")

            # Save checkpoint periodically
            if (epoch + 1) % 100 == 0:
                self.save(os.path.join(save_dir, f"checkpoint_epoch{epoch + 1}.pth"))

        # Final save
        self.save(os.path.join(save_dir, "final_model.pth"))
        print(f"\nTraining complete. Models saved to {save_dir}")

    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x_0 = batch.to(self.device)
                loss = self.compute_loss(x_0)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def generate(self, n_samples=120, seq_len=None, batch_size=None):
        """Generate synthetic financial time series via reverse SDE.

        Uses Euler-Maruyama integration of the reverse-time SDE:
            dX = -g²(t) * score(X, t) dt + g(t) dW̄

        With noise prediction: score = -ε_θ / σ_t

        Paper settings: N=2000 reverse steps, T=1.
        Uses EMA weights if available (better sample quality).

        Args:
            n_samples: number of sequences to generate (paper: 120)
            seq_len: sequence length (default from config: 2048)
            batch_size: generation batch size
        Returns:
            numpy array of shape (n_samples, seq_len)
        """
        # Swap in EMA weights for generation (higher quality)
        if self.use_ema:
            self.ema.apply_shadow()

        self.model.eval()
        seq_len = seq_len or self.config["seq_len"]
        batch_size = batch_size or min(n_samples, 32)
        N = self.config["n_reverse_steps"]
        T = self.sde.T
        eps = 1e-3  # Don't integrate all the way to t=0 for stability

        all_samples = []
        n_remaining = n_samples

        print(f"\nGenerating {n_samples} samples (N={N} reverse steps)...")

        while n_remaining > 0:
            B = min(batch_size, n_remaining)

            # Sample from prior: x_T ~ p_T
            x = self.sde.prior_sampling((B, seq_len)).to(self.device)

            # Time steps: T → eps
            timesteps = torch.linspace(T, eps, N, device=self.device)

            for i in tqdm(range(N - 1), desc="Reverse SDE", leave=False, mininterval=2.0):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr  # negative

                # Current time as batch
                t_batch = torch.full((B,), t_curr, device=self.device)

                # Get noise prediction
                eps_pred = self.model(x, t_batch)

                # Convert to score: s = -ε_θ / σ_t
                _, std = self.sde.marginal_prob(x, t_batch)
                if std.dim() == 1:
                    score = -eps_pred / std.unsqueeze(-1).clamp(min=1e-8)
                else:
                    score = -eps_pred / std.clamp(min=1e-8)

                # Get SDE coefficients
                drift, diffusion = self.sde.sde(x, t_batch)

                # Reverse SDE: dX = [f - g² · score] dt + g dW̄
                if diffusion.dim() == 1:
                    g_sq = diffusion.pow(2).unsqueeze(-1)
                    g = diffusion.unsqueeze(-1)
                else:
                    g_sq = diffusion.pow(2)
                    g = diffusion

                rev_drift = drift - g_sq * score
                x_mean = x + rev_drift * dt

                # Add noise (except at the last step)
                if i < N - 2:
                    noise = torch.randn_like(x)
                    x = x_mean + g * torch.sqrt(torch.abs(dt)) * noise
                else:
                    x = x_mean

            all_samples.append(x.cpu().numpy())
            n_remaining -= B

        samples = np.concatenate(all_samples, axis=0)[:n_samples]

        # Denormalize back to original data scale
        if self.normalize_data:
            samples = samples * self.data_std + self.data_mean

        print(f"Generated {samples.shape[0]} samples of length {samples.shape[1]}")

        # Restore training weights after generation
        if self.use_ema:
            self.ema.restore()

        return samples

    @torch.no_grad()
    def generate_ode(self, n_samples=120, seq_len=None, batch_size=None):
        """Generate via probability flow ODE (deterministic, no stochastic noise).

        Probability flow ODE (Song et al. 2021, Eq. 6):
            dX = [f(X,t) - ½ g²(t) · score(X,t)] dt

        Advantages over reverse SDE:
          - No noise accumulation → much sharper outputs
          - Better for undertrained models (no noise to counterbalance)
          - Deterministic given the initial noise sample

        Args:
            n_samples: number of sequences to generate
            seq_len: sequence length
            batch_size: generation batch size
        Returns:
            numpy array of shape (n_samples, seq_len)
        """
        if self.use_ema:
            self.ema.apply_shadow()

        self.model.eval()
        seq_len = seq_len or self.config["seq_len"]
        batch_size = batch_size or min(n_samples, 32)
        N = self.config["n_reverse_steps"]
        T = self.sde.T
        eps = 1e-3

        all_samples = []
        n_remaining = n_samples

        print(f"\nGenerating {n_samples} samples (ODE, N={N} steps)...")

        while n_remaining > 0:
            B = min(batch_size, n_remaining)
            x = self.sde.prior_sampling((B, seq_len)).to(self.device)

            timesteps = torch.linspace(T, eps, N, device=self.device)

            for i in tqdm(range(N - 1), desc="PF-ODE", leave=False, mininterval=2.0):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr  # negative

                t_batch = torch.full((B,), t_curr, device=self.device)

                # Noise prediction → score
                eps_pred = self.model(x, t_batch)
                _, std = self.sde.marginal_prob(x, t_batch)
                if std.dim() == 1:
                    score = -eps_pred / std.unsqueeze(-1).clamp(min=1e-8)
                else:
                    score = -eps_pred / std.clamp(min=1e-8)

                # SDE coefficients
                drift, diffusion = self.sde.sde(x, t_batch)
                if diffusion.dim() == 1:
                    g_sq = diffusion.pow(2).unsqueeze(-1)
                else:
                    g_sq = diffusion.pow(2)

                # PF-ODE: dX = [f - ½ g² · score] dt  (NO noise term)
                ode_drift = drift - 0.5 * g_sq * score
                x = x + ode_drift * dt

            all_samples.append(x.cpu().numpy())
            n_remaining -= B

        samples = np.concatenate(all_samples, axis=0)[:n_samples]

        # Denormalize
        if self.normalize_data:
            samples = samples * self.data_std + self.data_mean

        print(f"Generated {samples.shape[0]} samples (ODE)")

        if self.use_ema:
            self.ema.restore()

        return samples

    def evaluate(self, generated_data, real_data=None, save_dir=None):
        """Evaluate generated data against stylized facts.

        Args:
            generated_data: (N, L) numpy array from generate()
            real_data: optional (N, L) numpy array of real data for comparison
            save_dir: if provided, save plots here
        Returns:
            results dict
        """
        print("\n--- Evaluating generated data ---")
        gen_results = evaluate_stylized_facts(generated_data, mode=self.data_mode)

        real_results = None
        if real_data is not None:
            print("\n--- Evaluating real data ---")
            real_results = evaluate_stylized_facts(real_data, mode=self.data_mode)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_stylized_facts(
                gen_results, real_results,
                save_path=os.path.join(save_dir, "stylized_facts.png")
            )
            # 6-panel diagnostic plot (distribution, QQ, sample paths + stylized facts)
            if real_data is not None:
                plot_diagnostics(
                    generated_data, real_data, mode=self.data_mode,
                    save_path=os.path.join(save_dir, "diagnostics.png")
                )
                # 8-panel pathwise diagnostics (Audit D expansion)
                plot_pathwise_diagnostics(
                    generated_data, real_data, mode=self.data_mode,
                    save_path=os.path.join(save_dir, "pathwise_diagnostics.png")
                )

        return gen_results, real_results

    def save(self, path):
        """Save model checkpoint (includes EMA state if enabled)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "data_mean": self.data_mean,
            "data_std": self.data_std,
        }
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        """Load model checkpoint (restores EMA state if present)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass
        if self.use_ema and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            print(f"  EMA state restored")
        # Restore data normalization stats
        if "data_mean" in checkpoint:
            self.data_mean = checkpoint["data_mean"]
            self.data_std = checkpoint["data_std"]
            print(f"  Data norm: mean={self.data_mean:.4f}, std={self.data_std:.4f}")
        print(f"Loaded checkpoint from {path}")
