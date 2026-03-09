"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Based on score_sde_pytorch (Song et al., 2021) with the following additions:
  - GBMSDE: Geometric Brownian Motion SDE for financial time series
    (Kim et al., arXiv:2507.19003)
  - Shape-agnostic broadcasting in reverse() to support 2D (B,L),
    3D (B,K,L) and 4D (B,C,H,W) tensors
  - get_sde() factory function
  - get_sigma() noise schedule helper (linear/exponential/cosine)

Original: https://github.com/yang-song/score_sde_pytorch
License: Apache 2.0
"""
import abc
import math
import torch
import numpy as np


def _expand(t, x):
  """Expand t of shape (B,) to match x's number of dimensions for broadcasting.

  Works for any tensor shape: (B,L), (B,K,L), (B,C,H,W), etc.
  """
  while t.dim() < x.dim():
    t = t.unsqueeze(-1)
  return t


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probability flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - _expand(diffusion, x) ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - _expand(G, x) ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * _expand(beta_t, x) * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = _expand(torch.exp(log_mean_coeff), x) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z.reshape(z.shape[0], -1) ** 2, dim=1) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = _expand(torch.sqrt(alpha), x) * x - x
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * _expand(beta_t, x) * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = _expand(torch.exp(log_mean_coeff), x) * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z.reshape(z.shape[0], -1) ** 2, dim=1) / 2.


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - \
        torch.sum(z.reshape(z.shape[0], -1) ** 2, dim=1) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G


# ==========================================================================
# GBM SDE for financial time series (arXiv:2507.19003)
# ==========================================================================

class GBMSDE(VESDE):
  """GBM-based SDE: dS_t = ½σ²_t S_t dt + σ_t S_t dW_t.

  In log-price space X_t = log(S_t), this reduces to the VE SDE:
      dX_t = σ_t dW_t

  The key difference from plain VE is that this SDE operates on LOG-PRICES.
  When converted back to prices (exp), the multiplicative noise structure
  naturally induces heteroskedasticity characteristic of financial markets.

  This is the paper's main contribution (Section 3.1).

  Paper states σ_min=0.01, σ_max=1.0, but σ_max=1.0 only works for
  normalized data.  For raw anchored log-price paths (range [-1, 3]),
  σ_max should be auto-computed from data pairwise distances (typically
  5–10) so the prior N(0, σ_max²) fully obscures the signal at t=T.
  See train_l4.py's auto-compute logic.
  """

  def __init__(self, sigma_min=0.01, sigma_max=5.0, N=2000):
    super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, N=N)


# ==========================================================================
# Noise schedule helpers (for the paper's 3 schedules)
# ==========================================================================

def get_sigma(t: torch.Tensor, schedule: str,
              sigma_min: float = 0.01, sigma_max: float = 1.0) -> torch.Tensor:
  """Compute σ_t for a given noise schedule.

  Args:
      t: diffusion time in [0, 1], shape (B,)
      schedule: one of 'linear', 'exponential', 'cosine'
      sigma_min, sigma_max: bounds on the noise level
  Returns:
      sigma_t of shape (B,)
  """
  if schedule == "linear":
    sigma_sq = sigma_min ** 2 + t * (sigma_max ** 2 - sigma_min ** 2)
    return torch.sqrt(sigma_sq)
  elif schedule == "exponential":
    return sigma_min * (sigma_max / sigma_min) ** t
  elif schedule == "cosine":
    return sigma_min + (sigma_max - sigma_min) * (1.0 - torch.cos(math.pi * t)) / 2.0
  else:
    raise ValueError(f"Unknown schedule: {schedule}")


# ==========================================================================
# Factory
# ==========================================================================

def get_sde(sde_type: str, schedule: str = "exponential",
            sigma_min: float = 0.01, sigma_max: float = 1.0,
            N: int = 2000) -> SDE:
  """Create an SDE instance.

  Args:
      sde_type: 've', 'vp', 'subvp', or 'gbm'
      schedule: 'linear', 'exponential', or 'cosine' (only used for display;
                all VE/GBM SDEs use the geometric schedule σ_min·(σ_max/σ_min)^t)
      sigma_min, sigma_max: noise bounds
      N: number of discretization steps
  """
  sde_type = sde_type.lower()
  if sde_type == "ve":
    return VESDE(sigma_min, sigma_max, N)
  elif sde_type == "vp":
    return VPSDE(beta_min=0.1, beta_max=20, N=N)
  elif sde_type == "subvp":
    return subVPSDE(beta_min=0.1, beta_max=20, N=N)
  elif sde_type == "gbm":
    return GBMSDE(sigma_min, sigma_max, N)
  else:
    raise ValueError(f"Unknown SDE type: {sde_type}. Choose from: ve, vp, subvp, gbm")
