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
from torch.utils.tensorboard import SummaryWriter

from .sde import get_sde, get_sigma, _expand
from .score_network import FinancialScoreNetwork
from .metrics import (evaluate_stylized_facts, plot_stylized_facts,
                      plot_diagnostics, plot_pathwise_diagnostics,
                      plot_mean_path_diagnostic)


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

    def __init__(self, model, decay=0.999, use_num_updates=True):
        self.model = model
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.num_updates = 0
        self.shadow = {}
        self.backup = {}
        # Initialize shadow params as copies of current params
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters after an optimizer step.

        When use_num_updates=True, ramps decay from ~0.1 to target over
        the first ~9000 steps.  This prevents the EMA from being polluted
        by random initial weights (matches score_sde_pytorch).
        """
        if self.use_num_updates:
            self.num_updates += 1
            decay = min(self.decay,
                        (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(decay).add_(
                    param.data, alpha=1.0 - decay
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
        return {
            'shadow': {name: tensor.clone() for name, tensor in self.shadow.items()},
            'num_updates': self.num_updates,
        }

    def load_state_dict(self, state_dict):
        # Backward compat: old checkpoints store shadow params directly
        if isinstance(state_dict, dict) and 'shadow' in state_dict:
            self.shadow = {name: tensor.clone()
                           for name, tensor in state_dict['shadow'].items()}
            self.num_updates = state_dict.get('num_updates', 0)
        else:
            self.shadow = {name: tensor.clone()
                           for name, tensor in state_dict.items()}
            self.num_updates = 0


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

        # Loss weighting strategy:
        #   "uniform"    — standard unweighted ε-prediction loss (default)
        #   "min_snr_5"  — min(SNR, γ=5) weighting: upweights low-noise timesteps
        #                  where fine-grained structure (QV) is determined
        #   "min_snr_1"  — more aggressive variant with γ=1
        #   "likelihood" — g²/σ² weighting (constant for VE+exponential)
        # Backward compat: likelihood_weighting: true → "likelihood"
        if config.get("likelihood_weighting", False):
            self.loss_weighting = "likelihood"
        else:
            self.loss_weighting = config.get("loss_weighting", "uniform")
        if self.loss_weighting != "uniform":
            print(f"  Loss weighting: {self.loss_weighting}")

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-6)
        )

        # LR scheduler
        epochs = config["epochs"]
        lr_schedule = config.get("lr_schedule", "multistep")
        if lr_schedule == "cosine":
            lr_min = config.get("lr_min", 0.0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=lr_min
            )
            print(f"  LR schedule: cosine (lr_min={lr_min})")
        else:
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

        # Data standardization.
        #
        # For GBM/log_price mode: DISABLED by default (matching the paper).
        # Anchored log-price paths have a natural scale where σ_min=0.01
        # matches daily return magnitude (~0.012).  σ_max is auto-computed
        # from the data (typically 5–10) to ensure the VE SDE prior N(0, σ_max²)
        # covers the actual marginal at t=T.  With σ_max=1.0 on raw paths
        # (range [-1, 3]), KL(marginal ∥ prior) = 0.16 → severe mismatch.
        #
        # Global z-score normalization of cumulative paths induces
        # mean-reversion (the score learns a restoring force toward the
        # dataset-average path shape).  If scale adjustment is needed,
        # prefer tuning σ_min/σ_max rather than normalizing trajectories.
        #
        # For VE/VP on raw log-returns: z-score may still be useful since
        # return distributions are roughly stationary and have no drift.
        self.data_mean = 0.0
        self.data_std = 1.0
        # Per-path stats for per_path normalization mode
        self.path_means = None  # (N_train,) array
        self.path_stds = None   # (N_train,) array

        # Normalization mode: "none", "global", "per_path"
        # - none: raw data (default for GBM/log_price)
        # - global: single (mean, std) across all data (legacy normalize_data=True)
        # - per_path: each window normalized independently; empirical (μ,σ) stored
        #   for denormalization at generation time
        default_normalize = self.data_mode != "log_price"
        # Support both old normalize_data bool and new normalize_mode string
        if "normalize_mode" in config:
            self.normalize_mode = config["normalize_mode"]
        elif config.get("normalize_data", default_normalize):
            self.normalize_mode = "global"
        else:
            self.normalize_mode = "none"
        # Legacy compat
        self.normalize_data = self.normalize_mode == "global"

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

        # Apply loss weighting
        if self.loss_weighting == "likelihood":
            _, diffusion = self.sde.sde(x_0, t)  # g(t)
            g2 = diffusion.pow(2)
            weights = g2 / std.pow(2).clamp(min=1e-12)  # g²/σ²
            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)
            losses = losses * weights
        elif self.loss_weighting.startswith("min_snr"):
            # min-SNR-γ: weight = min(γ, 1/σ²) — upweights low-noise regime
            # For VE: SNR ∝ 1/σ², so this caps the max weight at γ
            gamma = float(self.loss_weighting.split("_")[-1])
            snr = 1.0 / std.pow(2).clamp(min=1e-12)
            weights = torch.clamp(snr, max=gamma)
            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)
            losses = losses * weights

        loss = losses.mean()

        return loss

    def compute_loss_detailed(self, x_0, eps=1e-5):
        """Compute loss with per-t and per-position diagnostics.

        Returns dict with:
          - loss: scalar (same as compute_loss)
          - loss_by_t: {low, mid, high} — loss decomposed by diffusion time
          - loss_by_pos: {early, mid, late} — loss decomposed by sequence position
          - score_stats: {cos_sim, eps_pred_norm, eps_true_norm}
        """
        B, L = x_0.shape

        t = torch.rand(B, device=self.device) * (self.sde.T - eps) + eps
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, t)
        if std.dim() == 1:
            x_t = mean + std.unsqueeze(-1) * z
        else:
            x_t = mean + std * z

        eps_pred = self.model(x_t, t)
        per_sample_pos = (z - eps_pred) ** 2  # (B, L)

        if self.mask_anchor:
            per_sample_pos = per_sample_pos.clone()
            per_sample_pos[:, 0] = 0.0

        # Overall loss
        loss = per_sample_pos.mean()

        # --- Loss by diffusion time t ---
        per_sample = per_sample_pos.mean(dim=1)  # (B,)
        t_np = t.detach().cpu().numpy()
        ps = per_sample.detach().cpu().numpy()
        low_mask = t_np < 0.33
        mid_mask = (t_np >= 0.33) & (t_np < 0.67)
        high_mask = t_np >= 0.67
        loss_by_t = {
            "low": float(ps[low_mask].mean()) if low_mask.any() else float('nan'),
            "mid": float(ps[mid_mask].mean()) if mid_mask.any() else float('nan'),
            "high": float(ps[high_mask].mean()) if high_mask.any() else float('nan'),
        }

        # --- Loss by sequence position ---
        pos_losses = per_sample_pos.mean(dim=0).detach().cpu().numpy()  # (L,)
        third = L // 3
        loss_by_pos = {
            "early": float(pos_losses[:third].mean()),
            "mid": float(pos_losses[third:2*third].mean()),
            "late": float(pos_losses[2*third:].mean()),
        }

        # --- Score / prediction statistics ---
        with torch.no_grad():
            flat_z = z.reshape(B, -1)
            flat_pred = eps_pred.reshape(B, -1)
            cos = torch.nn.functional.cosine_similarity(flat_z, flat_pred, dim=1)
            score_stats = {
                "cos_sim": float(cos.mean().cpu()),
                "eps_pred_norm": float(flat_pred.norm(dim=1).mean().cpu()),
                "eps_true_norm": float(flat_z.norm(dim=1).mean().cpu()),
            }

        return {
            "loss": loss,
            "loss_by_t": loss_by_t,
            "loss_by_pos": loss_by_pos,
            "score_stats": score_stats,
        }

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
        start_epoch = getattr(self, '_resume_epoch', None) or 0
        best_val_loss = getattr(self, '_resume_best_val_loss', None) or float("inf")
        self._best_epoch = start_epoch  # track which epoch produced best_model
        if start_epoch > 0:
            print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss}")

        # TensorBoard
        tb_dir = os.path.join(save_dir, "tb")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"  TensorBoard: {tb_dir}")
        global_step = 0
        diag_every = config.get("diag_every_epochs", 10)  # detailed diagnostics interval

        # Compute data statistics for standardization
        if self.normalize_mode == "global":
            all_data = []
            for batch in train_loader:
                all_data.append(batch)
            all_data = torch.cat(all_data, dim=0)
            self.data_mean = all_data.mean().item()
            self.data_std = all_data.std().item()
            if self.data_std < 1e-8:
                self.data_std = 1.0
            print(f"  Data standardization (global): mean={self.data_mean:.4f}, std={self.data_std:.4f}")
        elif self.normalize_mode == "per_path":
            all_data = []
            for batch in train_loader:
                all_data.append(batch)
            all_data = torch.cat(all_data, dim=0)  # (N, L)
            self.path_means = all_data.mean(dim=-1).numpy()  # (N,)
            self.path_stds = all_data.std(dim=-1).numpy()    # (N,)
            self.path_stds = np.clip(self.path_stds, 1e-8, None)
            print(f"  Data standardization (per-path): {len(self.path_means)} windows")
            print(f"    mu: [{self.path_means.min():.3f}, {self.path_means.max():.3f}], "
                  f"mean={self.path_means.mean():.3f}")
            print(f"    sigma: [{self.path_stds.min():.3f}, {self.path_stds.max():.3f}], "
                  f"mean={self.path_stds.mean():.3f}")

        print(f"\nTraining: SDE={config['sde_type']}, schedule={config['schedule']}")
        print(f"  epochs={epochs}, batch_size={config['batch_size']}, lr={config['lr']}")
        print(f"  σ_min={config['sigma_min']}, σ_max={config['sigma_max']}")
        print(f"  device={self.device}")
        print(f"  data mode={self.data_mode}")
        print(f"  normalization: {self.normalize_mode}")

        for epoch in range(start_epoch, epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}",
                        mininterval=5.0, leave=False)
            for batch in pbar:
                x_0 = batch.to(self.device)  # (B, L)

                # Standardize if enabled
                if self.normalize_mode == "global":
                    x_0 = (x_0 - self.data_mean) / self.data_std
                elif self.normalize_mode == "per_path":
                    # Per-path z-score: each sample normalized independently
                    mu = x_0.mean(dim=-1, keepdim=True)
                    sigma = x_0.std(dim=-1, keepdim=True).clamp(min=1e-8)
                    x_0 = (x_0 - mu) / sigma

                self.optimizer.zero_grad()
                loss = self.compute_loss(x_0)
                loss.backward()

                # Gradient norm (before clipping)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                # Update EMA after each optimizer step
                if self.use_ema:
                    self.ema.update()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1
                pbar.set_postfix(loss=f"{epoch_loss / n_batches:.6f}")

                # Per-step logging (lightweight)
                writer.add_scalar("loss/train_step", loss.item(), global_step)
                writer.add_scalar("grad/norm", grad_norm, global_step)

            self.scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            lr = self.optimizer.param_groups[0]["lr"]

            # Per-epoch logging
            writer.add_scalar("loss/train", avg_loss, epoch + 1)
            writer.add_scalar("lr", lr, epoch + 1)

            # Validation
            val_loss_str = ""
            if val_loader is not None and (epoch + 1) % 20 == 0:
                val_loss = self._validate(val_loader)
                val_loss_str = f", val_loss={val_loss:.4e}"
                writer.add_scalar("loss/val", val_loss, epoch + 1)

                # EMA validation — use this for best_model selection since
                # generate() uses EMA weights via ema.apply_shadow()
                if self.use_ema:
                    self.ema.apply_shadow()
                    ema_val = self._validate(val_loader)
                    self.ema.restore()
                    writer.add_scalar("loss/val_ema", ema_val, epoch + 1)
                    writer.add_scalar("ema/gap", val_loss - ema_val, epoch + 1)
                    val_loss_str += f", ema_val={ema_val:.4e}"
                    # Select best model by EMA val_loss (matches generation weights)
                    if ema_val < best_val_loss:
                        # Keep history: rename previous best before overwriting
                        best_path = os.path.join(save_dir, "best_model.pth")
                        if os.path.exists(best_path):
                            prev_name = f"best_model_ep{self._best_epoch}_val{best_val_loss:.4f}.pth"
                            os.rename(best_path, os.path.join(save_dir, prev_name))
                        best_val_loss = ema_val
                        self._best_epoch = epoch + 1
                        self.save(best_path,
                                  epoch=epoch + 1, best_val_loss=best_val_loss)
                        print(f"    ★ New best EMA val_loss: {ema_val:.4e}")
                else:
                    if val_loss < best_val_loss:
                        best_path = os.path.join(save_dir, "best_model.pth")
                        if os.path.exists(best_path):
                            prev_name = f"best_model_ep{self._best_epoch}_val{best_val_loss:.4f}.pth"
                            os.rename(best_path, os.path.join(save_dir, prev_name))
                        best_val_loss = val_loss
                        self._best_epoch = epoch + 1
                        self.save(best_path,
                                  epoch=epoch + 1, best_val_loss=best_val_loss)

            # Detailed diagnostics (loss by t, by position, score stats)
            if (epoch + 1) % diag_every == 0 or epoch == 0:
                self.model.eval()
                with torch.no_grad():
                    diag_batch = next(iter(train_loader)).to(self.device)
                    if self.normalize_mode == "global":
                        diag_batch = (diag_batch - self.data_mean) / self.data_std
                    elif self.normalize_mode == "per_path":
                        mu = diag_batch.mean(dim=-1, keepdim=True)
                        sigma = diag_batch.std(dim=-1, keepdim=True).clamp(min=1e-8)
                        diag_batch = (diag_batch - mu) / sigma
                    diag = self.compute_loss_detailed(diag_batch)
                self.model.train()

                for band, val in diag["loss_by_t"].items():
                    writer.add_scalar(f"loss_by_t/{band}", val, epoch + 1)
                for region, val in diag["loss_by_pos"].items():
                    writer.add_scalar(f"loss_by_pos/{region}", val, epoch + 1)
                for stat, val in diag["score_stats"].items():
                    writer.add_scalar(f"score/{stat}", val, epoch + 1)

                # EMA val loss logged above in validation block

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1:4d}/{epochs}: loss={avg_loss:.4e}, lr={lr:.2e}{val_loss_str}")

            # Save checkpoint periodically
            ckpt_every = config.get("checkpoint_every", 100)
            if (epoch + 1) % ckpt_every == 0:
                self.save(os.path.join(save_dir, f"checkpoint_epoch{epoch + 1}.pth"),
                          epoch=epoch + 1, best_val_loss=best_val_loss)

        # Final save
        self.save(os.path.join(save_dir, "final_model.pth"),
                  epoch=epochs, best_val_loss=best_val_loss)
        writer.close()
        print(f"\nTraining complete. Models saved to {save_dir}")
        print(f"TensorBoard logs: tensorboard --logdir {tb_dir}")

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
        """Generate via Predictor-Corrector sampling (score_sde VE default).

        Predictor: ReverseDiffusion — exact discrete σ² steps via sde.discretize()
        Corrector: Langevin MCMC — adaptive step size with target SNR

        This matches score_sde_pytorch's default VE sampler:
          predictor = reverse_diffusion
          corrector = langevin
          snr = 0.16, n_steps_each = 1, noise_removal = True

        Score_sde PC loop (Song et al. 2021 Algorithm 1):
          for i in 0..N-1:
            x, x_mean = corrector(x, t_i)    # Langevin refinement
            x, x_mean = predictor(x, t_i)    # Reverse diffusion step
          return x_mean                      # noise removal
        """
        if self.use_ema:
            self.ema.apply_shadow()

        self.model.eval()
        seq_len = seq_len or self.config["seq_len"]
        batch_size = batch_size or min(n_samples, 32)
        N = self.config["n_reverse_steps"]  # = sde.N
        T = self.sde.T
        eps = self.config.get("sampling_eps", 1e-5)

        # PC sampler parameters (score_sde VE defaults)
        snr = self.config.get("pc_snr", 0.16)
        n_corrector_steps = self.config.get("pc_corrector_steps", 1)

        all_samples = []
        n_remaining = n_samples
        nfe = N * (n_corrector_steps + 1)

        print(f"\nGenerating {n_samples} samples "
              f"(PC sampler, N={N}, snr={snr}, NFE={nfe})...")

        # Pre-move discrete sigmas to device once
        discrete_sigmas = self.sde.discrete_sigmas.to(self.device)

        while n_remaining > 0:
            B = min(batch_size, n_remaining)
            x = self.sde.prior_sampling((B, seq_len)).to(self.device)
            timesteps = torch.linspace(T, eps, N, device=self.device)

            for i in tqdm(range(N), desc="PC sampling", leave=False,
                          mininterval=2.0):
                t_curr = timesteps[i]
                t_batch = torch.full((B,), t_curr, device=self.device)

                # === Corrector: Langevin MCMC ===
                for _ in range(n_corrector_steps):
                    eps_pred = self.model(x, t_batch)
                    _, std = self.sde.marginal_prob(x, t_batch)
                    score = -eps_pred / _expand(std, eps_pred).clamp(min=1e-8)

                    noise = torch.randn_like(x)
                    grad_norm = torch.norm(
                        score.reshape(B, -1), dim=-1).mean()
                    noise_norm = torch.norm(
                        noise.reshape(B, -1), dim=-1).mean()
                    # VE: alpha = 1.0 (no scaling)
                    step_size = (snr * noise_norm / grad_norm) ** 2 * 2
                    x_mean = x + step_size * score
                    x = x_mean + torch.sqrt(step_size * 2) * noise

                # === Predictor: Reverse Diffusion ===
                eps_pred = self.model(x, t_batch)
                _, std = self.sde.marginal_prob(x, t_batch)
                score = -eps_pred / _expand(std, eps_pred).clamp(min=1e-8)

                # Discrete VE sigma steps (SMLD discretization)
                idx = int(t_curr.item() * (N - 1) / T)
                sigma = discrete_sigmas[idx]
                adjacent_sigma = (discrete_sigmas[idx - 1]
                                  if idx > 0
                                  else torch.zeros_like(sigma))
                # G = sqrt(σ²_i - σ²_{i-1})
                G_sq = sigma ** 2 - adjacent_sigma ** 2
                G = torch.sqrt(G_sq)

                # Reverse step: x_mean = x + G² · score
                z = torch.randn_like(x)
                x_mean = x + _expand(G_sq, x) * score
                x = x_mean + _expand(G, x) * z

            # Noise removal: use denoised x_mean at final step
            x = x_mean

            all_samples.append(x.cpu().numpy())
            n_remaining -= B

        samples = np.concatenate(all_samples, axis=0)[:n_samples]

        # Denormalize back to original data scale
        if self.normalize_mode == "global":
            samples = samples * self.data_std + self.data_mean
        elif self.normalize_mode == "per_path" and self.path_means is not None:
            indices = np.random.randint(0, len(self.path_means),
                                        size=samples.shape[0])
            mu = self.path_means[indices]
            sigma = self.path_stds[indices]
            samples = samples * sigma[:, None] + mu[:, None]
            print(f"  Per-path denorm: sampled {samples.shape[0]} "
                  f"(mu, sigma) pairs")

        print(f"Generated {samples.shape[0]} samples of length "
              f"{samples.shape[1]}")

        if self.use_ema:
            self.ema.restore()

        return samples

    @torch.no_grad()
    def generate_em(self, n_samples=120, seq_len=None, batch_size=None):
        """Generate via Euler-Maruyama on reverse SDE (legacy sampler).

        Kept for comparison. The PC sampler (generate()) is recommended.
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

        print(f"\nGenerating {n_samples} samples (EM, N={N} steps)...")

        while n_remaining > 0:
            B = min(batch_size, n_remaining)
            x = self.sde.prior_sampling((B, seq_len)).to(self.device)
            timesteps = torch.linspace(T, eps, N, device=self.device)

            for i in tqdm(range(N - 1), desc="Reverse SDE", leave=False,
                          mininterval=2.0):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr  # negative

                t_batch = torch.full((B,), t_curr, device=self.device)
                eps_pred = self.model(x, t_batch)
                _, std = self.sde.marginal_prob(x, t_batch)
                score = -eps_pred / _expand(std, eps_pred).clamp(min=1e-8)

                drift, diffusion = self.sde.sde(x, t_batch)
                g_sq = _expand(diffusion, x) ** 2
                g = _expand(diffusion, x)

                rev_drift = drift - g_sq * score
                x_mean = x + rev_drift * dt

                if i < N - 2:
                    noise = torch.randn_like(x)
                    x = x_mean + g * torch.sqrt(torch.abs(dt)) * noise
                else:
                    x = x_mean

            all_samples.append(x.cpu().numpy())
            n_remaining -= B

        samples = np.concatenate(all_samples, axis=0)[:n_samples]

        if self.normalize_mode == "global":
            samples = samples * self.data_std + self.data_mean
        elif self.normalize_mode == "per_path" and self.path_means is not None:
            indices = np.random.randint(0, len(self.path_means),
                                        size=samples.shape[0])
            mu = self.path_means[indices]
            sigma = self.path_stds[indices]
            samples = samples * sigma[:, None] + mu[:, None]

        print(f"Generated {samples.shape[0]} samples of length "
              f"{samples.shape[1]}")

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
        if self.normalize_mode == "global":
            samples = samples * self.data_std + self.data_mean
        elif self.normalize_mode == "per_path" and self.path_means is not None:
            indices = np.random.randint(0, len(self.path_means), size=samples.shape[0])
            mu = self.path_means[indices]
            sigma = self.path_stds[indices]
            samples = samples * sigma[:, None] + mu[:, None]

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
                # Cross-sectional mean path diagnostic (z-score diagnosis)
                plot_mean_path_diagnostic(
                    generated_data, real_data, mode=self.data_mode,
                    save_path=os.path.join(save_dir, "mean_path_diagnostic.png")
                )

        return gen_results, real_results

    def save(self, path, epoch=None, best_val_loss=None):
        """Save model checkpoint (includes EMA state if enabled)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "data_mean": self.data_mean,
            "data_std": self.data_std,
            "normalize_mode": self.normalize_mode,
        }
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if best_val_loss is not None:
            checkpoint["best_val_loss"] = best_val_loss
        if self.path_means is not None:
            checkpoint["path_means"] = self.path_means
            checkpoint["path_stds"] = self.path_stds
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        """Load model checkpoint (restores EMA state if present)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass
        if self.use_ema and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            print(f"  EMA state restored")
        # Restore epoch tracking
        self._resume_epoch = checkpoint.get("epoch", None)
        self._resume_best_val_loss = checkpoint.get("best_val_loss", None)
        if self._resume_epoch is not None:
            print(f"  Resume from epoch {self._resume_epoch}")
        # Restore data normalization stats
        if "data_mean" in checkpoint:
            self.data_mean = checkpoint["data_mean"]
            self.data_std = checkpoint["data_std"]
            print(f"  Data norm: mean={self.data_mean:.4f}, std={self.data_std:.4f}")
        if "path_means" in checkpoint:
            self.path_means = checkpoint["path_means"]
            self.path_stds = checkpoint["path_stds"]
            self.normalize_mode = checkpoint.get("normalize_mode", "per_path")
            print(f"  Per-path stats restored: {len(self.path_means)} windows")
        print(f"Loaded checkpoint from {path}")
