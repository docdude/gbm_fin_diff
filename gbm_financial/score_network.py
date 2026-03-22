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


class WaveNetTemporalBlock(nn.Module):
    """Dilated causal convolution stack with gated activations (WaveNet-style).

    Processes temporal sequences through exponentially dilated causal Conv1d
    layers with tanh * sigmoid gating — identical to the gating in CSDI's
    ResidualBlock but operating locally instead of globally.

    Architecture mirrors the WaveNet GAN generator from notebooks 07/12:
      - Dilated causal Conv1d (kernel=3, dilation=2^i)
      - Gated activation: tanh(conv_filter) * sigmoid(conv_gate)
      - Per-layer 1×1 skip projection, summed across layers
      - Per-layer 1×1 residual projection for cascading

    Uses SYMMETRIC (bidirectional) padding, not causal. The score model
    sees the entire noisy sequence — unlike an autoregressive generator,
    each position needs context from BOTH past and future to predict noise.

    For seq_len=2048 with dilation_rates=(1,2,4,8,16,32,64,128,256):
      receptive field = sum(d)*(k-1)+1 = 511*2+1 = 1023 timesteps
      (511 in each direction with symmetric padding)
    """

    def __init__(self, channels, dilation_rates=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                 kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.layers = nn.ModuleList()
        for d in dilation_rates:
            self.layers.append(nn.ModuleDict({
                'conv_filter': nn.Conv1d(channels, channels, kernel_size,
                                         padding=0, dilation=d),
                'conv_gate':   nn.Conv1d(channels, channels, kernel_size,
                                         padding=0, dilation=d),
                'res_conv':    nn.Conv1d(channels, channels, 1),
                'skip_conv':   nn.Conv1d(channels, channels, 1),
            }))
        # skip_conv uses default Kaiming init (not zero) so that the
        # WaveNet output is immediately non-trivial and the gate receives
        # meaningful gradient signal from the start.

    def forward(self, x):
        """x: (B, C, L) → (B, C, L).  Bidirectional: each position sees both past and future."""
        skip_sum = torch.zeros_like(x)
        for layer in self.layers:
            # Symmetric (non-causal) padding: score model is bidirectional
            # Unlike autoregressive WaveNet GAN, the score network sees the
            # entire noisy sequence — every position needs context from BOTH
            # directions to estimate noise accurately.
            total_pad = (self.kernel_size - 1) * layer['conv_filter'].dilation[0]
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            x_pad = F.pad(x, (pad_left, pad_right))
            h = torch.tanh(layer['conv_filter'](x_pad)) * \
                torch.sigmoid(layer['conv_gate'](x_pad))
            skip_sum = skip_sum + layer['skip_conv'](h)
            x = x + layer['res_conv'](h)
        return skip_sum


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

    Optional WaveNet branch (wavenet_branch=True):
      Adds a parallel dilated causal convolution stack that processes the
      temporal signal alongside the Transformer. The two outputs are merged
      via a learnable gate (initialized to 0 so the model starts identical
      to the original Transformer-only architecture).
    """

    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads,
                 is_linear=False, wavenet_branch=False, wavenet_dilation_rates=None):
        super().__init__(side_dim, channels, diffusion_embedding_dim, nheads, is_linear)
        self.pos_encoding = PositionalEncoding(channels)
        self.use_wavenet = wavenet_branch

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

        # Optional: parallel WaveNet dilated causal conv branch
        if self.use_wavenet:
            if wavenet_dilation_rates is None:
                wavenet_dilation_rates = (1, 2, 4, 8, 16, 32, 64, 128, 256)
            self.wavenet_block = WaveNetTemporalBlock(
                channels, dilation_rates=wavenet_dilation_rates)
            # Learnable mix gate — initialized to -2.2 (sigmoid ≈ 0.1)
            # so WaveNet starts with small influence, preserving the
            # pretrained Transformer, while providing enough gradient
            # signal for the gate to open as WaveNet learns.
            self.wavenet_gate = nn.Parameter(torch.tensor(-2.2))

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            # Transformer branch (global attention)
            y_t = y.permute(2, 0, 1)       # (L, B*K, C)
            y_t = self.pos_encoding(y_t)    # ← Paper modification #1
            y_t = self.time_layer(y_t)      # TransformerEncoder
            y_t = y_t.permute(1, 2, 0)      # (B*K, C, L)

            if self.use_wavenet:
                # WaveNet branch (local dilated causal convolutions)
                y_w = self.wavenet_block(y)   # (B*K, C, L)
                y = y_t + torch.sigmoid(self.wavenet_gate) * y_w
            else:
                y = y_t

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
        #
        # Optional WaveNet branch: when config["wavenet_branch"]=True, each
        # residual block gets a parallel dilated causal conv stack. Defaults
        # to False for backward compatibility with existing checkpoints.
        use_wavenet = config.get("wavenet_branch", False)
        wavenet_dilations = config.get("wavenet_dilation_rates", None)
        if use_wavenet:
            dil_str = wavenet_dilations or (1,2,4,8,16,32,64,128,256)
            print(f"  WaveNet branch enabled: dilations={list(dil_str)}")

        cfg = csdi_config["diffusion"]
        self.diffmodel.residual_layers = nn.ModuleList([
            ResidualBlockWithPosEnc(
                side_dim=cfg["side_dim"],     # set by CSDI_base.__init__
                channels=cfg["channels"],
                diffusion_embedding_dim=cfg["diffusion_embedding_dim"],
                nheads=cfg["nheads"],
                is_linear=cfg["is_linear"],
                wavenet_branch=use_wavenet,
                wavenet_dilation_rates=wavenet_dilations,
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
