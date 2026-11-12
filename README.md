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

### Recommended generator (EDM + ε-model splice)

The best-validated setup combines two trained models that share the same noise
schedule (σ ∈ [1e-3, 12.56], exponential):

```bash
# 1. ε-prediction model (fine-scale structure)
python -m gbm_financial.train_l4 --save_dir save/fixed
# 2. EDM-preconditioned model (coarse structure: scale, drift, terminal spread)
python -m gbm_financial.train_l4 --save_dir save/edm_scalar --precond edm \
    --sigma_max 12.561595916748047
# 3. Generate/score: EDM Heun σ 12.6→0.5, EDM SDE 0.5→0.03, ε-model SDE below 0.03,
#    rejecting failed samples (lag-1 return ACF below the real 0.1th percentile)
python -m gbm_financial.train_l4 --eval_only --save_dir save/edm_scalar \
    --sampler hybrid --hybrid_sigma_switch 0.5 --reject_failed \
    --hybrid_fine_checkpoint save/fixed/gbm_exponential/best_model.pth --hybrid_fine_sigma 0.03
```

From Python: build `GBMFinancialDiffusion` from the run's `results.json` config, `load()` the
checkpoint, set the same sampler keys in `model.config`, and call `model.sample(n)`.

Results (3 seeds × 120 paths of 2048 days; 104 S&P stocks, 10 tickers held out):

| | std× | kurt | α | ACF r₁ | ACF\|r\| | β | leverage | volDisp | discAUC | NN× |
|---|---|---|---|---|---|---|---|---|---|---|
| real (train) | 1 | 41.5 | 2.90 | −0.010 | 0.156 | 0.40 | −5.2 | 0.29 | – | 1 |
| real held-out tickers | 1.04 | 18.7 | 2.99 | −0.013 | 0.141 | 0.38 | −4.0 | 0.25 | 0.843 | 1.02 |
| **splice (above)** | 0.90 | 24.7 | 2.67 | −0.022 | 0.157 | 0.34 | −4.4 | 0.39 | **0.843** | 1.13 |
| ε-model only | 0.70 | 18.0 | 2.88 | −0.019 | 0.157 | 0.40 | −6.8 | 0.23 | 0.897 | 0.89 |
| block bootstrap | 1.00 | 26.7 | 2.83 | −0.006 | 0.161 | 1.96 | −4.4 | 0.11 | 0.986 | 1.11 |
| GARCH(1,1)-t | 1.14 | 69 | 2.42 | 0.001 | 0.198 | 0.34 | +2.3 | 0.40 | 0.914 | 1.36 |

discAUC is a classifier's ability to tell generated from real windows; the held-out row is the
floor (a classifier also separates *different real tickers* at this window length). About 5% of
generated paths are rejected by the filter. Known residual gaps: return scale ×0.9, terminal
spread ×1.3, mean 2048-day drift 0.70 vs 0.95, clustering decay β slightly fast.

### Multi-asset synthetic OHLCV panels (NAS-100 → FinRL-Meta MACE/margin env)

Two models generate whole cross-sectional panels:

- **Market model** (unconditional, K=2): equal-weight market log-price + log VIX.
- **Ticker model** (K=7, last four channels observed): per-ticker anchored log-price,
  log(high/low) range and log volume (all normalised by causal trailing 252-day
  statistics), **conditioned on the market path** plus the ticker's trailing beta and
  correlation to the market. Every ticker in a scenario shares one market path,
  which is what produces realistic cross-asset correlation.

All channels are scaled per ticker so day-to-day changes have std 0.017. Training data
is Tiingo 1995 → 2024-05-31, which ends on the env's last training day, so no trade-period leakage.

Sampling-time corrections in `panel_assembly` (all default on, target era 2010-03-30+):

- **Era-matched market paths** (`--market_era`): the market model learns the full
  1995+ regime mix. Dot-com-style markets drive high beta dispersion and low correlation in
  the ticker model, so generated market paths are rejection-sampled with
  p(era | vol, VIX level, VIX variability) from a classifier fitted on real windows.
- **State-matched calendars** (`--random_calendar` to disable): ticker channels are in units
  of each ticker's trailing vol at the calendar start. Each scenario's start date is drawn
  from the 5 real dates whose market state (start VIX, early and full-window vol) is nearest
  to the generated path. A random date made tickers move 0.4–2× the market.
- **Filtered tail calibration** (`--tail_era`): the diffusion models produce about half
  the real rate of >6σ days. De-volatilised (EWMA) market shocks and ticker residual
  shocks are quantile-mapped onto the real target-era distribution, which keeps timing,
  vol clustering, betas and correlations intact. Range and volume move with the shock size.
- **Sector correlation** (`--no_sector_corr` to disable): generated ticker residuals are
  independent across tickers. Their normal scores are mixed with the real residual
  correlation matrix (Gaussian copula, shrunk 10% to I) before the tail mapping.
- **Market identity** (`--no_market_identity` to disable): the market channel is the
  equal-weight mean of these tickers, so a common daily shift enforces that exactly.

Raw model outputs are saved per scenario (`scenario_XXX_raw.npz`). `--from_raw DIR`
re-applies the ticker-side calibration on CPU in minutes, with no sampling.

**Leakage:** if the DRL side selects on a validation tail inside the generator's training
range (FinRL-Meta `optuna_vec_mace_v2` carves 2022-04-14 → 2024-05-31), build the bank with
generators that stop before it: `CUTOFF=2022-04-13 GPUS="0 1" scripts/build_cutoff_bank.sh`
(trains the 4 models with `--end_date`, then scenarios, eval and env export).

20-scenario bank vs 40 real 2010+ panels (`panel_eval`): pair corr .349 / .355,
PC1 share .371 / .378, beta dispersion .352 / .325, pooled kurtosis 9.6 / 8.9,
market vol .0127 / .0124. Turbulence is still high (median 108 / 78), partly from a few
2008-level stress scenarios.

```bash
# 1. data (FinRL-Meta venv, uses its Tiingo processor + repo .env)
~/FinRL-Meta/.venv/bin/python scripts/fetch_nas100_history.py
# 2. models (EDM + ε for each, then spliced at sampling time)
python -m gbm_financial.train_panel --model ticker --precond edm --save_dir save/panel/ticker_edm
python -m gbm_financial.train_panel --model ticker --save_dir save/panel/ticker_eps
python -m gbm_financial.train_panel --model market --precond edm --save_dir save/panel/market_edm
python -m gbm_financial.train_panel --model market --save_dir save/panel/market_eps
# 3. scenarios: 512-day panels for all 99 tickers on real trading calendars
python -m gbm_financial.panel_assembly --n_scenarios 50 --out save/panel/scenarios \
    --ticker_dir save/panel/ticker_edm --ticker_fine save/panel/ticker_eps/best_model.pth \
    --market_dir save/panel/market_edm --market_fine save/panel/market_eps/best_model.pth
    # defaults: --ticker_fine_sigma 0.1 --market_fine_sigma 0.03 --market_era/--tail_era 2010-03-30
python -m gbm_financial.panel_eval --scenarios save/panel/scenarios      # vs real panels
# 4. env configs via the env's own indicator/volatility/ADV/T-bill pipeline (FinRL-Meta venv)
~/FinRL-Meta/.venv/bin/python scripts/export_finrl_meta_env.py --scenarios save/panel/scenarios
```

Each `scenario_XXX.env.pkl` has the same structure as `MarketDataPreparator.create_env_config()`
(`price_array`, `tech_array`, `volatility_array`, `volume_array`, `adv20_array`, `tbill_rates`, …),
so it can stand in for the real train config. `--check_equivalence` rebuilds the real
train config through this exporter and matches the env's own output exactly
(indicators within 0.4%, from EMA warm-up). Use `--market real` to condition on real
market+VIX windows instead of the market model.

Data note: in the env's cached Tiingo data, `FB` is a constant 39.39 for 2010–2025
(the ticker was renamed META). The fetch script pulls `META` history under `FB`.

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
│   ├── panel_data.py        # Multichannel OHLCV + market/VIX panel windows
│   ├── panel_model.py       # Conditional multichannel diffusion (PanelDiffusion)
│   ├── train_panel.py       # Train/eval ticker + market panel models
│   ├── panel_assembly.py    # Scenario panels → OHLCV (FinRL-Meta raw schema)
│   ├── panel_eval.py        # Cross-sectional panel evaluation
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

**Sampling**: predictor-corrector (default), Euler–Maruyama, probability-flow ODE, Karras/Heun, or a hybrid (deterministic Heun at high noise, reverse SDE at low noise, optional model splice), N=2000 steps. VE/GBM samplers step on a grid geometric in σ mapped through each schedule's exact inverse, and every sampler ends with a Tweedie denoise, so results no longer depend on `sampling_eps`.

### Implementation notes (deviations and clarifications)

- **Data representation:** every SDE trains on anchored log-price windows (`data_mode="log_price"`), as confirmed by the paper's author. Log-returns are used only for evaluation. `data_mode="log_return"` remains available as a legacy option.
- **GBM ≡ VE in log-price space:** with μ_t = ½σ_t² the GBM forward SDE is exactly the VE SDE, so on identical data the two are the same model. `GBMSDE` is kept as a separate class for bookkeeping only.
- **VP schedules:** β(t) = σ(t)² (author's parameterisation), with σ_VP ∈ [√β_min, √β_max] = [√0.1, √20]. The linear schedule reproduces the standard Song et al. VP SDE. VP data is globally standardised because its prior is N(0, 1).
- **σ_min = 1e-3 instead of 0.01:** on log-price paths, 0.01 is about one daily return (std ≈ 0.017), so the network never learned sub-daily structure. Use `--sigma_min 0.01` to reproduce the paper.
- **Raw prices:** `data/financial/sp500_prices_raw.pkl` stores unprocessed closes. Only one-day spike-and-revert bad ticks are repaired; real crashes (1987, AIG 2008, …) are kept. The legacy `sp500_prices.pkl` cache zeroed every return beyond 5σ and is ignored.
- **Validation split:** whole tickers are held out (`split="ticker"`), so overlapping windows cannot leak between train and validation.
- **Tail exponent:** α is a Hill estimate on the top 5% of |r|. The older √n-k estimate is still reported as `alpha_hill_sqrtn`.
- **EDM preconditioning (`--precond edm`):** Karras et al. (2022) input/skip/output scaling and loss weighting around the same CSDI network. It fixes the ε-model's under-dispersion (return scale ×0.70) but produces a few failed samples under purely stochastic sampling, hence the hybrid sampler and `--reject_failed`.
- **Validation loss** uses stratified diffusion times and fixed noise, so `best_model.pth` selection is deterministic.

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
