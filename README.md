# GBM Financial Diffusion

A diffusion-based generative model for financial time series via geometric Brownian motion.

Implements the method from:

> **A diffusion-based generative model for financial time series via geometric Brownian motion**  
> arXiv:2507.19003

## Overview

This model generates realistic synthetic financial time series by using a **Geometric Brownian Motion (GBM)** forward SDE in log-price space, combined with a CSDI-based score network. The GBM forward process naturally matches the statistical properties of financial returns, yielding synthetic series that reproduce:

- **Heavy tails** (power-law with tail exponent α ≈ 4–5)
- **Volatility clustering** (slowly decaying autocorrelation of absolute returns)
- **Absence of linear autocorrelation** in raw returns

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/gbm_fin_diff.git
cd gbm_fin_diff

# Install (CPU or GPU)
pip install -e .

# Optional: for downloading stock data
pip install yfinance
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- NumPy, SciPy, Matplotlib, pandas, tqdm, PyYAML

## Quick Start

### Validate the implementation (no GPU required)

```bash
python -m gbm_financial.validate
```

Runs 5 diagnostic tests: oracle score, tensor shapes, loss convergence, noise prediction quality, and generation scale.

### Train on S&P 500 data

```bash
# Default: GBM SDE + exponential schedule, 1000 epochs
python -m gbm_financial.run

# Quick test with synthetic data
python -m gbm_financial.run --use_synthetic --epochs 10 --n_reverse_steps 100
```

### Run the full experiment grid (Table 1 in paper)

```bash
python -m gbm_financial.run --experiment_grid
```

This trains all 9 SDE × schedule combinations: {VE, VP, GBM} × {linear, exponential, cosine}.

## Project Structure

```
gbm_fin_diff/
├── gbm_financial/
│   ├── __init__.py
│   ├── config.yaml          # Default hyperparameters (paper Section 4)
│   ├── data.py              # Data pipeline: S&P 500 CSV → sliding windows
│   ├── metrics.py           # Stylized facts: Hill estimator, ACF, plots
│   ├── run.py               # CLI entry point
│   ├── score_network.py     # Score network (CSDI_base subclass)
│   ├── sde.py               # SDE library: VE, VP, GBM + reverse SDE
│   ├── train.py             # Training loop, Euler-Maruyama sampler
│   ├── train_l4.py          # Multi-stock training script (L4 GPU)
│   ├── validate.py          # 5-test validation suite
│   └── vendor/
│       ├── __init__.py
│       └── csdi.py          # Vendored CSDI components (Tashiro et al., 2021)
├── data/
│   └── sp500.csv            # S&P 500 daily prices (~6000 trading days)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Configuration

All hyperparameters are in [`gbm_financial/config.yaml`](gbm_financial/config.yaml). Key settings:

| Parameter | Default | Paper Reference |
|-----------|---------|-----------------|
| `sde_type` | `gbm` | Section 3.1 |
| `schedule` | `exponential` | Section 3.1 |
| `sigma_min` | 0.01 | Section 4 |
| `sigma_max` | 1.0 | Section 4 |
| `n_reverse_steps` | 2000 | Section 4 |
| `channels` | 128 | Section 3.1.1 |
| `epochs` | 1000 | Section 4 |
| `batch_size` | 64 | Section 4 |
| `seq_len` / `window_len` | 2048 | Section 3.1.2 |

## Method

**Forward SDE** (GBM in log-price space, equivalent to VE SDE):

$$dx = \sigma(t) \, dw, \quad \sigma(t) = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t$$

**Score matching loss** (continuous-time, denoising):

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| s_\theta(x_t, t) + \frac{\epsilon}{\sigma(t)} \right\|^2 \right]$$

**Sampling**: Reverse-time SDE solved with Euler-Maruyama (N=2000 steps).

## Vendored Dependencies

This repo is self-contained. The `vendor/` directory includes adapted code from:

- **CSDI** (Tashiro et al., 2021): Score network architecture (`diff_CSDI`, `CSDI_base`)
- **score_sde** (Song et al., 2021): SDE definitions (`VESDE`, `VPSDE`, `subVPSDE`)

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{gbm_diffusion_2025,
  title={A diffusion-based generative model for financial time series via geometric Brownian motion},
  year={2025},
  journal={arXiv preprint arXiv:2507.19003}
}
```
