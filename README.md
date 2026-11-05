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
git clone https://github.com/docdude/gbm_fin_diff.git
cd gbm_fin_diff

# Install (CPU or GPU)
pip install -e .

# Optional: for downloading stock data
pip install yfinance
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- NumPy, SciPy, Matplotlib, pandas, tqdm, PyYAML, TensorBoard
- Optional: yfinance (download), scikit-learn + pyarrow + arch (comparison harness)

## Quick Start

### Validate the implementation (no GPU required)

```bash
python -m gbm_financial.validate
```

Runs 6 diagnostic tests: oracle score, tensor shapes, loss convergence, noise prediction quality, generation scale, and a sampler oracle that checks every sampler × SDE × schedule against an exact Gaussian score (`--test samplers`).

### Full run + head-to-head comparison on one GPU

```bash
pip install -e ".[download,compare]"
bash scripts/l40s_pipeline.sh                  # ours vs vnhtg01, shared harness
RUN_ABLATION=1 bash scripts/l40s_pipeline.sh   # + paper σ_min=0.01 ablation
```

Auto-selects an L40S (override with `GPU=<index>`), is resumable, and writes
`save/comparison/REPORT.md`. Copy `data/financial/sp500_prices_raw.pkl` to the
training machine to train on identical data (otherwise it is downloaded).

### Compare any generator with the shared harness

```bash
python -m gbm_financial.compare --real save/fixed/gbm_exponential/real_train.npy \
    --holdout save/fixed/gbm_exponential/real_val.npy --input log_price \
    --gen ours=save/fixed/gbm_exponential/generated_seed0.npy,save/fixed/gbm_exponential/generated_seed1.npy \
    --baselines --out save/comparison/custom
```

All metrics are computed within windows (generated windows are independent),
averaged over sampling seeds, and reported next to iid-shuffle, block-bootstrap
and GARCH(1,1)-t baselines plus a held-out-ticker "real vs real" row.

### Train on S&P 500 data

```bash
# Paper architecture, GBM + exponential, multi-seed evaluation
python -m gbm_financial.train_l4 --save_dir save/fixed

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
│   ├── compare.py           # Shared comparison harness + baselines
│   ├── data.py              # Data pipeline: raw prices → sliding windows
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
| `sigma_min` | 0.001 (`train_l4`) | Section 4 uses 0.01 — see notes below |
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

**Sampling**: predictor-corrector (default), Euler–Maruyama, probability-flow ODE, or Karras/Heun, N=2000 steps. VE/GBM samplers step on a grid geometric in σ mapped through each schedule's exact inverse, and every sampler ends with a Tweedie denoise, so results no longer depend on `sampling_eps`.

### Implementation notes (deviations and clarifications)

- **Data representation:** every SDE trains on anchored log-price windows (`data_mode="log_price"`), as confirmed by the paper's author. Log-returns are used only for evaluation. `data_mode="log_return"` remains available as a legacy option.
- **GBM ≡ VE in log-price space:** with μ_t = ½σ_t² the GBM forward SDE is exactly the VE SDE, so on identical data the two are the same model. `GBMSDE` is kept as a separate class for bookkeeping only.
- **VP schedules:** β(t) = σ(t)² (author's parameterisation), with σ_VP ∈ [√β_min, √β_max] = [√0.1, √20]. The linear schedule reproduces the standard Song et al. VP SDE. VP data is globally standardised because its prior is N(0, 1).
- **σ_min = 1e-3 instead of 0.01:** on log-price paths, 0.01 is about one daily return (std ≈ 0.017), so the network never learned sub-daily structure. Use `--sigma_min 0.01` to reproduce the paper.
- **Raw prices:** `data/financial/sp500_prices_raw.pkl` stores unprocessed closes. Only one-day spike-and-revert bad ticks are repaired; real crashes (1987, AIG 2008, …) are kept. The legacy `sp500_prices.pkl` cache zeroed every return beyond 5σ and is ignored.
- **Validation split:** whole tickers are held out (`split="ticker"`), so overlapping windows cannot leak between train and validation.
- **Tail exponent:** α is a Hill estimate on the top 5% of |r|. The older √n-k estimate is still reported as `alpha_hill_sqrtn`.

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
