"""Score network for GBM financial diffusion — subclassing the original CSDI.

Paper Section 3.1.1: "We adopt the neural network architecture originally developed
in the Conditional Score-based Diffusion Model (CSDI) framework [6]"

This module subclasses CSDI_base (main_model.py) to inherit:
  - diff_CSDI score network (diff_models.py) with Transformer + gated residual blocks
  - embed_layer (feature embedding)
  - time_embedding() (sinusoidal position encoding for side info)
  - get_side_info() (constructs conditioning tensor for residual blocks)

Two modifications from the paper:
  1. ContinuousDiffusionEmbedding — replaces CSDI's discrete-step DiffusionEmbedding
     (lookup table indexed by integer step) with on-the-fly sinusoidal embedding from
     continuous t ∈ [0,1], needed for continuous-time SDE training (score_sde framework).
  2. PositionalEncoding before Transformer — "explicit positional encodings are applied
     prior to the Transformer layers" (paper Section 3.1.1).

Plus the paper's hyperparameter changes (Section 4):
  channels: 64→128, diff_emb: 128→256, feat_emb: 16→64

Everything else — ResidualBlock gating, skip connections, feature/time Transformer
attention, side_info construction, temporal embeddings — is inherited unchanged.

SDE framework note: score_sde (Song et al.) is JAX/Flax and cannot be imported
directly into PyTorch — the SDE classes are ported in sde.py using the same math.
The continuous-time DSM loss and Euler-Maruyama sampler are in train.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import vendored CSDI components
from .vendor.csdi import CSDI_base, ResidualBlock


# ---------------------------------------------------------------------------
# Only structurally new component: continuous-time diffusion embedding
# ---------------------------------------------------------------------------
# Replaces CSDI's DiffusionEmbedding (diff_models.py lines 33-56) which uses a
# precomputed lookup table: self.embedding[diffusion_step]
#
# The paper uses continuous-time SDEs (score_sde framework), so we compute the
# sinusoidal embedding on-the-fly from continuous t ∈ [0, 1].
#
# Preserves CSDI's structure:
#   Frequency basis: 10^(i/(D-1) * 4) for i=0..D-1  (from _build_embedding)
#   Projection: Linear → SiLU → Linear → SiLU        (from forward)
# ---------------------------------------------------------------------------

class ContinuousDiffusionEmbedding(nn.Module):
    """Drop-in replacement for CSDI's DiffusionEmbedding that accepts continuous t.

    CSDI original (diff_models.py):
        embedding = _build_embedding(num_steps, dim)   # precomputed table
        x = self.embedding[diffusion_step]             # integer index

    This version:
        x = sinusoidal(t * 1000, frequencies)          # continuous t ∈ [0, 1]

    Same projection MLP and frequency basis in both cases.
    """

    def __init__(self, embedding_dim=256, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        half_dim = embedding_dim // 2
        # Same frequency basis as CSDI's DiffusionEmbedding._build_embedding
        frequencies = 10.0 ** (torch.arange(half_dim).float() / (half_dim - 1) * 4.0)
        self.register_buffer("frequencies", frequencies)
        # Same MLP projection as CSDI's DiffusionEmbedding
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, t):
        """
        Args:
            t: (B,) continuous diffusion time in [0, 1]
        Returns:
            (B, projection_dim) embedding vector
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # Scale to [0, 1000] for comparable frequency variation to CSDI's integer steps
        t_scaled = t.float() * 1000.0
        table = t_scaled.unsqueeze(-1) * self.frequencies.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(table), torch.cos(table)], dim=-1)  # (B, embedding_dim)
        # Same activation path as CSDI DiffusionEmbedding.forward
        x = self.projection1(emb)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


# ---------------------------------------------------------------------------
# Paper modification #1: Positional encoding before Transformer layers
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.

    Paper: "explicit positional encodings are applied prior to the Transformer layers"
    """

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (seq_len, batch, d_model) — TransformerEncoder input format."""
        return x + self.pe[: x.size(0)].unsqueeze(1)


class ResidualBlockWithPosEnc(ResidualBlock):
    """CSDI ResidualBlock with positional encoding added before Transformer.

    Subclasses the original ResidualBlock (diff_models.py) and overrides only
    forward_time() to insert positional encoding. All other logic — gating,
    conditioning, feature attention, skip connections — is inherited unchanged.

    Fixes vs vanilla CSDI:
      - dim_feedforward: CSDI hardcodes 64 regardless of channel size.
        Standard Transformer uses 4*d_model. With K=1 (univariate), the
        time_layer Transformer is the ONLY attention pathway (feature_layer
        is no-op), so adequate FFN capacity is critical.
    """

    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__(side_dim, channels, diffusion_embedding_dim, nheads, is_linear)
        self.pos_encoding = PositionalEncoding(channels)

        # Fix: replace the time_layer Transformer with proper dim_feedforward.
        # CSDI's get_torch_trans() hardcodes dim_feedforward=64, which creates
        # a severe bottleneck when channels > 64 (ratio < 1:1 vs standard 4:1).
        # With K=1 (univariate), forward_feature is a no-op, so time_layer is
        # the model's ONLY temporal attention — it needs full capacity.
        if not is_linear:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=nheads,
                dim_feedforward=4 * channels,  # Standard 4x expansion
                activation="gelu",
                batch_first=False,
            )
            self.time_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = y.permute(2, 0, 1)       # (L, B*K, C)
            y = self.pos_encoding(y)      # ← Paper modification #1
            y = self.time_layer(y)        # TransformerEncoder (inherited from diff_models.py)
            y = y.permute(1, 2, 0)        # (B*K, C, L)

        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


# ---------------------------------------------------------------------------
# Score network: subclass of CSDI_base
# ---------------------------------------------------------------------------

class FinancialScoreNetwork(CSDI_base):
    """Score network s_θ(x_t, t) for financial time series.

    Subclasses CSDI_base (main_model.py) to inherit:
      - self.diffmodel    (diff_CSDI instance — Transformer + gated residual blocks)
      - self.embed_layer  (feature embedding for side info)
      - time_embedding()  (sinusoidal position encoding)
      - get_side_info()   (constructs conditioning tensor)

    Then applies the paper's two modifications:
      1. Replaces diffmodel.diffusion_embedding → ContinuousDiffusionEmbedding
      2. Replaces diffmodel.residual_layers → ResidualBlockWithPosEnc

    The network predicts NOISE ε (not the score directly).
    Score is recovered as: s_θ = -ε_θ / σ_t

    Paper hyperparameters (Section 3.1.1 / Section 4):
        channels=128, diff_emb_dim=256, feat_emb_dim=64,
        time_emb_dim=128, n_layers=4, n_heads=8
    """

    def __init__(self, config, device):
        """
        Args:
            config: flat dict with keys: channels, diff_emb_dim, feat_emb_dim,
                    time_emb_dim, n_layers, n_heads (from config.yaml)
            device: torch device
        """
        # Map flat config → CSDI_base's nested config format
        csdi_config = {
            "model": {
                "timeemb": config.get("time_emb_dim", 128),
                "featureemb": config.get("feat_emb_dim", 64),
                "is_unconditional": 1,        # unconditional generation
                "target_strategy": "random",   # unused but required by CSDI_base
            },
            "diffusion": {
                "layers": config.get("n_layers", 4),
                "channels": config.get("channels", 128),
                "nheads": config.get("n_heads", 8),
                "diffusion_embedding_dim": config.get("diff_emb_dim", 256),
                "is_linear": False,
                # DDPM schedule params — required by CSDI_base.__init__ but unused
                # (we use continuous-time SDE training from score_sde instead)
                "beta_start": 0.0001,
                "beta_end": 0.5,
                "num_steps": 50,
                "schedule": "quad",
            },
        }

        # CSDI_base.__init__ creates:
        #   self.embed_layer  = nn.Embedding(1, feat_emb_dim)
        #   self.diffmodel    = diff_CSDI(config_diff, inputdim=1)
        #   self.emb_time_dim, self.emb_feature_dim, self.emb_total_dim
        #   self.alpha_torch  (DDPM schedule — unused with SDE training)
        super().__init__(target_dim=1, config=csdi_config, device=device)

        # --- Paper modification #1: continuous diffusion embedding ---
        # Replace diff_CSDI's discrete DiffusionEmbedding (lookup table)
        # with ContinuousDiffusionEmbedding (on-the-fly from continuous t)
        diff_emb_dim = csdi_config["diffusion"]["diffusion_embedding_dim"]
        self.diffmodel.diffusion_embedding = ContinuousDiffusionEmbedding(
            embedding_dim=diff_emb_dim,
            projection_dim=diff_emb_dim,
        )

        # --- Paper modification #2: positional encoding in residual blocks ---
        # Replace diff_CSDI's ResidualBlock instances with ResidualBlockWithPosEnc
        # (subclass that adds pos encoding before Transformer, inherits everything else)
        cfg = csdi_config["diffusion"]
        self.diffmodel.residual_layers = nn.ModuleList([
            ResidualBlockWithPosEnc(
                side_dim=cfg["side_dim"],     # set by CSDI_base.__init__
                channels=cfg["channels"],
                diffusion_embedding_dim=cfg["diffusion_embedding_dim"],
                nheads=cfg["nheads"],
                is_linear=cfg["is_linear"],
            )
            for _ in range(cfg["layers"])
        ])

        # Move everything to the target device
        self.to(device)

    def score_forward(self, x_t, t):
        """Forward pass for continuous-time SDE score estimation.

        Reshapes (B, L) univariate input to CSDI's (B, inputdim, K, L) format,
        constructs side_info using inherited CSDI_base methods, and calls
        self.diffmodel (diff_CSDI.forward).

        Args:
            x_t: (B, L) noisy time series
            t:   (B,) continuous diffusion time in [0, 1]
        Returns:
            epsilon_pred: (B, L) predicted noise
        """
        B, L = x_t.shape

        # Reshape to CSDI 4D format: (B, inputdim=1, K=1, L)
        x = x_t.unsqueeze(1).unsqueeze(2)

        # Construct side_info using inherited CSDI_base methods:
        #   time_embedding() — main_model.py line 49
        #   get_side_info()  — main_model.py line 96
        observed_tp = torch.arange(L, device=x_t.device).float().unsqueeze(0).expand(B, -1)
        cond_mask = torch.ones(B, 1, L, device=x_t.device)
        side_info = self.get_side_info(observed_tp, cond_mask)  # (B, side_dim, K=1, L)

        # Call diff_CSDI.forward() — continuous t goes to ContinuousDiffusionEmbedding
        predicted = self.diffmodel(x, side_info, t)  # (B, K=1, L)

        return predicted.squeeze(1)  # (B, L)

    def forward(self, x_t, t):
        """Override CSDI_base.forward for continuous-time SDE interface.

        CSDI_base.forward() expects a batch dict and does DDPM noising internally.
        We override to accept (x_t, t) directly for use with SDE training in train.py.
        """
        return self.score_forward(x_t, t)
