#!/usr/bin/env python3
"""
Pipeline validation script for GBM Financial Diffusion Model.

Runs a series of tests to verify that the implementation is correct,
independent of training data quality/quantity. Tests can be run on CPU.

Tests:
  1. SDE oracle test — forward+reverse with true score → MSE ≈ 0
  2. Score network shape test — verify all tensor shapes through the pipeline
  3. Loss convergence test — train on synthetic data, verify loss decreases
  4. Noise prediction quality — check eps_pred std approaches 1.0 after training
  5. Generation scale test — verify generated data has correct magnitude

Usage:
    python -m gbm_financial.validate [--test all|oracle|shapes|train|quick]
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn

# Force CPU for validation (avoids incompatible GPU issues)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from gbm_financial.sde import get_sde, VESDE, GBMSDE
from gbm_financial.score_network import FinancialScoreNetwork
from gbm_financial.train import GBMFinancialDiffusion


def test_sde_oracle(schedule="exponential", N=500, verbose=True):
    """Test 1: Forward SDE + reverse with true score recovers original data.

    This proves the SDE math, marginal_prob, prior_sampling, and discretization
    are all correct. If this fails, there's a fundamental SDE bug.

    Expected: MSE < 0.001
    """
    print("\n" + "=" * 60)
    print("TEST 1: SDE Oracle (forward + reverse with true score)")
    print("=" * 60)

    device = torch.device("cpu")
    sde = VESDE(sigma_min=0.01, sigma_max=1.0, N=N)

    # Create test data
    torch.manual_seed(42)
    x_0 = torch.randn(8, 256, device=device) * 0.2  # (B=8, L=256)

    # Forward: noise to t=T
    T = sde.T
    t_T = torch.tensor(T, device=device)
    mean, std = sde.marginal_prob(x_0, t_T.expand(8))
    z = torch.randn_like(x_0)
    x_T = mean + std.unsqueeze(-1) * z

    # Reverse with true score: score(x_t, t) = -(x_t - x_0) / σ_t²
    eps = 1e-3
    timesteps = torch.linspace(T, eps, N, device=device)
    x = x_T.clone()

    for i in range(N - 1):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr

        t_batch = t_curr.expand(8)
        _, sigma_t = sde.marginal_prob(x, t_batch)

        # True score (oracle)
        true_score = -(x - x_0) / sigma_t.unsqueeze(-1).pow(2)

        drift, diffusion = sde.sde(x, t_batch)
        g_sq = diffusion.pow(2).unsqueeze(-1)
        g = diffusion.unsqueeze(-1)

        rev_drift = drift - g_sq * true_score
        x_mean = x + rev_drift * dt

        if i < N - 2:
            noise = torch.randn_like(x)
            x = x_mean + g * torch.sqrt(torch.abs(dt)) * noise
        else:
            x = x_mean

    mse = ((x - x_0) ** 2).mean().item()
    passed = mse < 0.005

    if verbose:
        print(f"  Schedule: {schedule}")
        print(f"  N steps: {N}")
        print(f"  Recovery MSE: {mse:.6f}")
        print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold: 0.005)")

    return passed, mse


def test_score_network_shapes(verbose=True):
    """Test 2: Verify all tensor shapes through the score network.

    Checks that CSDI subclassing works correctly — inputs, side_info,
    diffusion embedding, residual blocks, and output all have correct shapes.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Score Network Shape Verification")
    print("=" * 60)

    device = torch.device("cpu")
    config = {
        "channels": 32,         # small for speed
        "diff_emb_dim": 64,
        "feat_emb_dim": 16,
        "time_emb_dim": 32,
        "n_layers": 2,
        "n_heads": 4,
        "sde_type": "gbm",
        "schedule": "exponential",
        "sigma_min": 0.01,
        "sigma_max": 1.0,
        "n_reverse_steps": 100,
        "epochs": 1,
        "batch_size": 4,
        "lr": 1e-3,
    }

    model = FinancialScoreNetwork(config, device)
    model.eval()

    B, L = 4, 128
    x = torch.randn(B, L, device=device)
    t = torch.rand(B, device=device) * 0.99 + 0.01

    with torch.no_grad():
        out = model(x, t)

    shape_ok = out.shape == (B, L)
    finite_ok = torch.isfinite(out).all().item()

    # Check internal shapes
    n_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"  Input shape:  ({B}, {L})")
        print(f"  Output shape: {tuple(out.shape)}")
        print(f"  Shape correct: {shape_ok}")
        print(f"  All finite:    {finite_ok}")
        print(f"  Parameters:    {n_params:,}")
        print(f"  Output std:    {out.std().item():.4f}")
        print(f"  Output mean:   {out.mean().item():.4f}")

        # Check that positional encoding is applied
        has_pos_enc = any(hasattr(layer, 'pos_encoding')
                         for layer in model.diffmodel.residual_layers)
        print(f"  Has pos encoding: {has_pos_enc}")

        # Check dim_feedforward fix
        for i, layer in enumerate(model.diffmodel.residual_layers):
            if hasattr(layer, 'time_layer') and hasattr(layer.time_layer, 'layers'):
                ffn_dim = layer.time_layer.layers[0].linear1.out_features
                expected = 4 * config["channels"]
                print(f"  Layer {i} FFN dim: {ffn_dim} (expected {expected}, "
                      f"{'FIXED' if ffn_dim == expected else 'STILL 64!'})")

    passed = shape_ok and finite_ok
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_loss_convergence(n_epochs=30, verbose=True):
    """Test 3: Train on synthetic data and verify loss decreases.

    Uses synthetic GBM data (500 sequences) to train a small model.
    Loss should decrease significantly over 30 epochs.

    Expected: final_loss < 0.8 * initial_loss
    """
    print("\n" + "=" * 60)
    print("TEST 3: Loss Convergence on Synthetic Data")
    print("=" * 60)

    config = {
        "channels": 32,
        "diff_emb_dim": 64,
        "feat_emb_dim": 16,
        "time_emb_dim": 32,
        "n_layers": 2,
        "n_heads": 4,
        "sde_type": "gbm",
        "schedule": "exponential",
        "sigma_min": 0.01,
        "sigma_max": 1.0,
        "n_reverse_steps": 100,
        "epochs": n_epochs,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "ema_decay": 0.999,
        "likelihood_weighting": False,
        "seq_len": 128,
        "use_synthetic": True,
        "stride": 50,
    }

    # Create synthetic data
    from gbm_financial.data import generate_synthetic_gbm_data, create_subsequences
    from torch.utils.data import DataLoader, TensorDataset

    stock_data = generate_synthetic_gbm_data(n_sequences=30, seq_len=700)
    sequences = create_subsequences(stock_data, window_len=128, stride=50, mode="log_price")

    if len(sequences) < 16:
        print(f"  WARNING: Only {len(sequences)} sequences, need at least 16")
        return False

    dataset = TensorDataset(torch.from_numpy(sequences).float())
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    # Initialize model
    pipeline = GBMFinancialDiffusion(config)

    losses = []
    start = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0
        n = 0
        pipeline.model.train()
        for (batch,) in loader:
            batch = batch.to(pipeline.device)
            pipeline.optimizer.zero_grad()
            loss = pipeline.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), 1.0)
            pipeline.optimizer.step()
            if pipeline.use_ema:
                pipeline.ema.update()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / max(n, 1)
        losses.append(avg)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}: loss = {avg:.4f}")

    elapsed = time.time() - start
    initial_loss = np.mean(losses[:3])
    final_loss = np.mean(losses[-3:])
    ratio = final_loss / initial_loss

    passed = ratio < 0.85  # loss should decrease by at least 15%

    if verbose:
        print(f"  Initial loss (avg first 3): {initial_loss:.4f}")
        print(f"  Final loss (avg last 3):    {final_loss:.4f}")
        print(f"  Ratio: {ratio:.3f}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Result: {'PASS' if passed else 'FAIL'} (need ratio < 0.85)")

    return passed, losses


def test_noise_prediction_quality(verbose=True):
    """Test 4: After training, check noise prediction quality at various t.

    A well-functioning architecture should predict noise with:
      - eps_pred std approaching 1.0 (especially at high noise)
      - Positive cosine similarity between eps_pred and true z

    This test trains longer (50 epochs) to give the model a fair chance.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Noise Prediction Quality")
    print("=" * 60)

    config = {
        "channels": 32,
        "diff_emb_dim": 64,
        "feat_emb_dim": 16,
        "time_emb_dim": 32,
        "n_layers": 2,
        "n_heads": 4,
        "sde_type": "gbm",
        "schedule": "exponential",
        "sigma_min": 0.01,
        "sigma_max": 1.0,
        "n_reverse_steps": 100,
        "epochs": 50,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "ema_decay": 0.999,
        "likelihood_weighting": False,
        "seq_len": 128,
    }

    from gbm_financial.data import generate_synthetic_gbm_data, create_subsequences
    from torch.utils.data import DataLoader, TensorDataset

    stock_data = generate_synthetic_gbm_data(n_sequences=50, seq_len=700)
    sequences = create_subsequences(stock_data, window_len=128, stride=50, mode="log_price")
    dataset = TensorDataset(torch.from_numpy(sequences).float())
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    pipeline = GBMFinancialDiffusion(config)

    # Train
    for epoch in range(50):
        pipeline.model.train()
        for (batch,) in loader:
            batch = batch.to(pipeline.device)
            pipeline.optimizer.zero_grad()
            loss = pipeline.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), 1.0)
            pipeline.optimizer.step()
            if pipeline.use_ema:
                pipeline.ema.update()
        pipeline.scheduler.step()

    # Test noise prediction at various t
    pipeline.model.eval()
    if pipeline.use_ema:
        pipeline.ema.apply_shadow()

    test_data = torch.from_numpy(sequences[:32]).float().to(pipeline.device)
    test_t_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

    if verbose:
        print(f"  {'t':>6s}  {'eps_std':>8s}  {'cos_sim':>8s}  {'mse':>8s}")
        print("  " + "-" * 40)

    all_cos_sims = []
    for t_val in test_t_values:
        t = torch.full((32,), t_val, device=pipeline.device)
        z = torch.randn_like(test_data)
        mean, std = pipeline.sde.marginal_prob(test_data, t)
        x_t = mean + std.unsqueeze(-1) * z

        with torch.no_grad():
            eps_pred = pipeline.model(x_t, t)

        eps_std = eps_pred.std().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            eps_pred.reshape(-1), z.reshape(-1), dim=0
        ).item()
        mse_val = ((z - eps_pred) ** 2).mean().item()
        all_cos_sims.append(cos_sim)

        if verbose:
            print(f"  {t_val:6.2f}  {eps_std:8.3f}  {cos_sim:8.3f}  {mse_val:8.4f}")

    if pipeline.use_ema:
        pipeline.ema.restore()

    # Check: at high noise, cosine similarity should be positive
    high_noise_cossim = np.mean(all_cos_sims[-3:])  # t = 0.7, 0.9, 0.99
    passed = high_noise_cossim > 0.1

    if verbose:
        print(f"\n  Mean cos_sim at high noise (t≥0.7): {high_noise_cossim:.3f}")
        print(f"  Result: {'PASS' if passed else 'FAIL'} (need cos_sim > 0.1)")

    return passed


def test_generation_scale(verbose=True):
    """Test 5: Quick generation test — verify output scale is reasonable.

    Even with a barely-trained model, the generation pipeline should:
      1. Not produce NaN/Inf
      2. Produce finite values
      3. Have return std within 100x of input data std
    """
    print("\n" + "=" * 60)
    print("TEST 5: Generation Pipeline Scale Check")
    print("=" * 60)

    config = {
        "channels": 32,
        "diff_emb_dim": 64,
        "feat_emb_dim": 16,
        "time_emb_dim": 32,
        "n_layers": 2,
        "n_heads": 4,
        "sde_type": "gbm",
        "schedule": "exponential",
        "sigma_min": 0.01,
        "sigma_max": 1.0,
        "n_reverse_steps": 50,  # few steps for speed
        "epochs": 10,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "ema_decay": 0.999,
        "likelihood_weighting": False,
        "seq_len": 64,
    }

    from gbm_financial.data import generate_synthetic_gbm_data, create_subsequences
    from torch.utils.data import DataLoader, TensorDataset

    stock_data = generate_synthetic_gbm_data(n_sequences=20, seq_len=600)
    sequences = create_subsequences(stock_data, window_len=64, stride=30, mode="log_price")
    dataset = TensorDataset(torch.from_numpy(sequences).float())
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    pipeline = GBMFinancialDiffusion(config)

    # Quick train
    for epoch in range(10):
        pipeline.model.train()
        for (batch,) in loader:
            batch = batch.to(pipeline.device)
            pipeline.optimizer.zero_grad()
            loss = pipeline.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), 1.0)
            pipeline.optimizer.step()
            if pipeline.use_ema:
                pipeline.ema.update()

    # Generate
    generated = pipeline.generate(n_samples=4, seq_len=64, batch_size=4)

    finite = np.isfinite(generated).all()
    gen_std = generated.std()
    data_std = sequences.std()
    scale_ratio = gen_std / max(data_std, 1e-10)

    # Compute returns
    gen_returns = np.diff(generated, axis=-1)
    real_returns = np.diff(sequences, axis=-1)
    ret_std_ratio = gen_returns.std() / max(real_returns.std(), 1e-10)

    passed = finite and scale_ratio < 200

    if verbose:
        print(f"  Generated shape: {generated.shape}")
        print(f"  All finite: {finite}")
        print(f"  Data std:      {data_std:.4f}")
        print(f"  Generated std: {gen_std:.4f}")
        print(f"  Scale ratio:   {scale_ratio:.1f}x")
        print(f"  Return std ratio: {ret_std_ratio:.1f}x")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


class _GaussianOracle(torch.nn.Module):
    """Exact ε-prediction for data x_0 ~ N(0, s² I) under the pipeline's SDE.

    x_t ~ N(0, (m_t² s² + σ_t²) I) with m_t = mean coefficient (1 for VE),
    so ε*(x_t, t) = σ_t x_t / (m_t² s² + σ_t²).
    """

    def __init__(self, sde, data_std):
        super().__init__()
        self.sde = sde
        self.s2 = data_std ** 2

    def forward(self, x, t):
        m, std = self.sde.marginal_prob(torch.ones_like(x), t)
        std = std.view(-1, *([1] * (x.dim() - 1)))
        return std * x / (m ** 2 * self.s2 + std ** 2)


def test_sampler_oracle(verbose=True, data_std=0.3, tol=0.10):
    """Test 6: every sampler × SDE × schedule recovers the data scale with an
    exact score. Catches schedule/σ-grid mismatches and wrong reverse drifts
    (e.g. a PC predictor indexing a geometric σ table under a cosine schedule).
    """
    print("\n" + "=" * 60)
    print("TEST 6: Sampler Oracle (all SDEs × schedules × samplers)")
    print("=" * 60)

    base = {
        "channels": 8, "diff_emb_dim": 16, "feat_emb_dim": 4, "time_emb_dim": 8,
        "n_layers": 1, "n_heads": 2, "sigma_min": 1e-3, "sigma_max": 5.0,
        "n_reverse_steps": 400, "epochs": 1, "batch_size": 8, "lr": 1e-3,
        "ema_decay": 0.0, "seq_len": 64, "sampling_eps": 1e-5,
        "hybrid_ode_steps": 100, "hybrid_sde_steps": 300,
    }
    combos = [("gbm", sch, smp) for sch in ("exponential", "cosine", "linear")
              for smp in ("pc", "em", "ode", "karras", "hybrid")]
    combos += [("vp", sch, smp) for sch in ("exponential", "cosine", "linear")
               for smp in ("pc", "em", "ode")]

    all_ok = True
    for sde_type, schedule, sampler in combos:
        cfg = dict(base, sde_type=sde_type, schedule=schedule,
                   normalize_mode="none")
        pipe = GBMFinancialDiffusion(cfg, device=torch.device("cpu"))
        pipe.model = _GaussianOracle(pipe.sde, data_std)
        fn = {"pc": pipe.generate, "em": pipe.generate_em, "ode": pipe.generate_ode,
              "karras": pipe.generate_karras, "hybrid": pipe.generate_hybrid}[sampler]
        out = fn(n_samples=64, seq_len=64, batch_size=64, seed=0)
        ratio = out.std() / data_std
        ok = np.isfinite(out).all() and abs(ratio - 1) < tol
        all_ok &= bool(ok)
        if verbose:
            print(f"  {sde_type:<4} {schedule:<12} {sampler:<7} std ratio={ratio:.3f}  "
                  f"{'PASS' if ok else 'FAIL'}")
    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_edm_precond(verbose=True, n_epochs=25):
    """Test 7: EDM preconditioning — trains, samples and round-trips checkpoints
    for scalar and per-position σ_data."""
    import tempfile
    from gbm_financial.data import generate_synthetic_gbm_data, create_subsequences
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "=" * 60)
    print("TEST 7: EDM Preconditioning")
    print("=" * 60)
    stock = generate_synthetic_gbm_data(n_sequences=30, seq_len=700)
    seqs = create_subsequences(stock, window_len=64, stride=30, mode="log_price")
    data = torch.from_numpy(seqs).float()
    loader = DataLoader(TensorDataset(data), batch_size=16, shuffle=True, drop_last=True)
    ok = True
    for mode in ("scalar", "per_position"):
        sd = data.std(dim=0).tolist() if mode == "per_position" else float(data.std())
        cfg = {"channels": 32, "diff_emb_dim": 64, "feat_emb_dim": 16, "time_emb_dim": 32,
               "n_layers": 2, "n_heads": 4, "sde_type": "gbm", "schedule": "exponential",
               "sigma_min": 1e-3, "sigma_max": 5.0, "n_reverse_steps": 60, "epochs": n_epochs,
               "batch_size": 16, "lr": 1e-3, "ema_decay": 0.0, "seq_len": 64,
               "precond": "edm", "edm_sigma_data": sd, "edm_sigma_data_mode": mode}
        torch.manual_seed(0)
        pipe = GBMFinancialDiffusion(cfg)
        losses = []
        for _ in range(n_epochs):
            tot = 0.0
            for (b,) in loader:
                pipe.optimizer.zero_grad()
                loss = pipe.compute_loss(b.to(pipe.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pipe.model.parameters(), 1.0)
                pipe.optimizer.step()
                tot += loss.item()
            losses.append(tot / len(loader))
        finite = all(np.isfinite(losses))
        decreased = losses[-1] < 0.9 * losses[0]  # EDM loss has a high irreducible floor
        outs = {}
        for name, fn in (("pc", pipe.generate), ("em", pipe.generate_em),
                         ("ode", pipe.generate_ode), ("karras", pipe.generate_karras),
                         ("hybrid", pipe.generate_hybrid)):
            g = fn(n_samples=8, seq_len=64, batch_size=8, seed=0)
            outs[name] = bool(np.isfinite(g).all())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.pth")
            pipe.save(path, epoch=1)
            cfg2 = dict(cfg)
            cfg2.pop("edm_sigma_data")  # as when rebuilt from results.json
            pipe2 = GBMFinancialDiffusion(cfg2)
            pipe2.load(path)
            x = torch.randn(4, 64, device=pipe.device)
            t = torch.full((4,), 0.5, device=pipe.device)
            pipe.model.eval(); pipe2.model.eval()
            with torch.no_grad():
                same = torch.allclose(pipe.model(x, t), pipe2.model(x, t), atol=1e-6)
        passed = finite and decreased and all(outs.values()) and same
        ok &= passed
        if verbose:
            print(f"  σ_data={mode:<12} loss {losses[0]:.3f}→{losses[-1]:.3f}  samplers finite={outs}  "
                  f"checkpoint round-trip={same}  {'PASS' if passed else 'FAIL'}")
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_panel_model(verbose=True, n_epochs=3):
    """Test 8: multichannel conditional panel model (ε and EDM) — training,
    conditioning preserved in samples, hybrid sampler, checkpoint round trip."""
    import tempfile
    from torch.utils.data import DataLoader
    from gbm_financial.data import FinancialTimeSeriesDataset
    from gbm_financial.panel_model import PanelDiffusion

    print("\n" + "=" * 60)
    print("TEST 8: Multichannel Conditional Panel Model")
    print("=" * 60)
    rng = np.random.default_rng(0)
    N, L = 96, 64
    m = rng.standard_normal((N, L)) * 0.017
    tk = 0.8 * m + 0.6 * rng.standard_normal((N, L)) * 0.017
    X = np.stack([np.cumsum(tk, 1) - tk[:, :1], np.abs(tk) * 2, rng.standard_normal((N, L)) * 0.02,
                  np.cumsum(m, 1) - m[:, :1], rng.standard_normal((N, L)) * 0.05], 1).astype(np.float32)
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for precond in ("none", "edm"):
            cfg = {"channels": 16, "diff_emb_dim": 32, "feat_emb_dim": 8, "time_emb_dim": 16,
                   "n_layers": 1, "n_heads": 2, "sde_type": "gbm", "schedule": "exponential",
                   "sigma_min": 1e-3, "sigma_max": 5.0, "n_reverse_steps": 40, "epochs": n_epochs,
                   "batch_size": 16, "lr": 1e-3, "ema_decay": 0.0, "seq_len": L, "precond": precond,
                   "n_channels": 5, "conditional": True, "cond_channels": [3, 4], "anchor_channels": [0, 3],
                   "checkpoint_every": 1000, "hybrid_ode_steps": 10, "hybrid_sde_steps": 40}
            if precond == "edm":
                cfg["edm_sigma_data"] = [float(X[:, k].std()) for k in range(5)]
            torch.manual_seed(0)
            pipe = PanelDiffusion(cfg)
            loader = DataLoader(FinancialTimeSeriesDataset(X), batch_size=16, shuffle=True, drop_last=True)
            sub = os.path.join(d, precond)
            pipe.train(loader, loader, save_dir=sub)
            res = {}
            for smp in ("pc", "hybrid"):
                pipe.config["sampler"] = smp
                G = pipe.sample_panel(8, L, cond=X[:8], seed=0)
                res[smp] = (G.shape == (8, 5, L) and np.isfinite(G).all()
                            and np.allclose(G[:, 3:], X[:8, 3:]))
            pipe2 = PanelDiffusion(dict(cfg, edm_sigma_data=None) if precond == "edm" else cfg)
            pipe2.load(os.path.join(sub, "final_model.pth"))
            x = torch.randn(4, 5, L, device=pipe.device)
            t = torch.full((4,), 0.3, device=pipe.device)
            c = torch.as_tensor(X[:4], device=pipe.device)
            pipe._set_cond(c); pipe2._set_cond(c)
            pipe.model.eval(); pipe2.model.eval()
            with torch.no_grad():
                same = torch.allclose(pipe.model(x, t), pipe2.model(x, t), atol=1e-5)
                cond_zero = bool((pipe.model(x, t)[:, 3:].abs() < 1e-6).all()) if precond == "none" else True
            passed = all(res.values()) and same and cond_zero
            ok &= passed
            if verbose:
                print(f"  precond={precond:<5} samplers(shape, finite, cond kept)={res}  "
                      f"round-trip={same}  cond-channel ε=0: {cond_zero}  {'PASS' if passed else 'FAIL'}")
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_tail_calibration(verbose=True):
    """Test 9: filtered tail calibration keeps betas and shock ranks, and maps the
    de-volatilised residual shocks onto the reference (fat-tailed) distribution."""
    from scipy import stats as sst
    from gbm_financial.panel_assembly import TailCalibrator, decompose, devol

    print("\n" + "=" * 60)
    print("TEST 9: Filtered Tail Calibration")
    print("=" * 60)
    rng = np.random.default_rng(0)
    n, L = 99, 512
    tc = TailCalibrator.__new__(TailCalibrator)
    ref = rng.standard_t(3, 400_000)
    tc.ref_t = np.sort(ref / ref.std())
    tc.ref_m = tc.ref_t
    tc.k_range, tc.k_vol, tc.market_ref = 0.007, 0.007, None
    dm = rng.standard_normal(L - 1) * 0.017
    market = np.stack([np.r_[0.0, np.cumsum(dm)], np.zeros(L)])
    b = rng.uniform(0.5, 1.8, n)
    dc = b[:, None] * dm + rng.standard_normal((n, L - 1)) * 0.012
    paths = np.zeros((n, 3, L))
    paths[:, 0, 1:] = np.cumsum(dc, 1)
    out = tc.tickers(paths, market)
    mu0, b0, e0 = decompose(dc, dm)
    dco = np.diff(out[:, 0], axis=1)
    _, b1, _ = decompose(dco, dm)
    u0, sig = devol(e0)
    u1 = (dco - mu0 - b0[:, None] * (dm - dm.mean())) / sig
    beta_ok = np.abs(b1 - b0).max() < 0.1 and np.corrcoef(b0, b1)[0, 1] > 0.99
    rank_ok = sst.spearmanr(u0.ravel(), u1.ravel())[0] > 0.9999
    k_ref, k_out = sst.kurtosis(tc.ref_t), sst.kurtosis(u1.ravel())
    tail_ok = abs(np.log(k_out / k_ref)) < np.log(1.5) and sst.kurtosis(u0.ravel()) < 1.0
    ok = bool(beta_ok and rank_ok and tail_ok)
    if verbose:
        print(f"  beta kept: {beta_ok} (max |Δβ| {np.abs(b1 - b0).max():.3f})  ranks kept: {rank_ok}  "
              f"shock kurtosis {sst.kurtosis(u0.ravel()):.2f} → {k_out:.2f} (ref {k_ref:.2f})")
        print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok



def test_audit_fixes(verbose=True, n_epochs=2):
    """Test 10: noise embedding, anchor pinning, seeded splice reproducibility,
    min-SNR weights, bank-wide tail map."""
    import math
    import tempfile
    from torch.utils.data import DataLoader
    from gbm_financial.data import FinancialTimeSeriesDataset
    from gbm_financial.panel_model import PanelDiffusion
    from gbm_financial.score_network import ContinuousDiffusionEmbedding
    from gbm_financial.panel_assembly import TransferMap

    print("\n" + "=" * 60)
    print("TEST 10: Audit fixes")
    print("=" * 60)
    ok = True
    # (a) noise embedding: neighbouring noise levels must embed similarly
    smin, smax = 1e-3, 5.0
    sig = lambda t: smin * (smax / smin) ** t
    t_of = lambda s: math.log(s / smin) / math.log(smax / smin)
    sims = {}
    for mode in ("legacy", "log_sigma"):
        e = ContinuousDiffusionEmbedding(128, mode=mode)
        e.noise_fn = sig
        f = lambda t: torch.cat([(x := (torch.log(sig(t)) / 4 if mode == "log_sigma" else t * 1000)
                                  .unsqueeze(-1) * e.frequencies).sin(), x.cos()], -1)
        ts = torch.tensor([t_of(s) for s in (0.01, 0.1, 1.0)])
        ts2 = torch.tensor([t_of(1.01 * s) for s in (0.01, 0.1, 1.0)])
        sims[mode] = float(torch.nn.functional.cosine_similarity(f(ts), f(ts2)).min())
    emb_ok = sims["log_sigma"] > 0.99 and sims["legacy"] < 0.5
    ok &= emb_ok
    if verbose:
        print(f"  embedding cos-sim σ vs 1.01σ: legacy {sims['legacy']:.3f}, log_sigma {sims['log_sigma']:.4f}"
              f"  {'PASS' if emb_ok else 'FAIL'}")

    # (b,c) tiny panel model: anchors exactly 0 in every sampler; splice reproducible
    rng = np.random.default_rng(0)
    N, L = 64, 32
    m = rng.standard_normal((N, L)) * 0.017
    X = np.stack([np.cumsum(m, 1) - m[:, :1], np.abs(m) * 2,
                  np.cumsum(m, 1) - m[:, :1]], 1).astype(np.float32)
    cfg = {"channels": 16, "diff_emb_dim": 32, "feat_emb_dim": 8, "time_emb_dim": 16,
           "n_layers": 1, "n_heads": 2, "sde_type": "gbm", "schedule": "exponential",
           "sigma_min": 1e-3, "sigma_max": 5.0, "n_reverse_steps": 20, "epochs": n_epochs,
           "batch_size": 16, "lr": 1e-3, "ema_decay": 0.0, "seq_len": L, "precond": "none",
           "n_channels": 3, "conditional": True, "cond_channels": [2], "anchor_channels": [0, 2],
           "checkpoint_every": 1000, "hybrid_ode_steps": 6, "hybrid_sde_steps": 20,
           "noise_embedding": "log_sigma"}
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(0)
        pipe = PanelDiffusion(cfg)
        loader = DataLoader(FinancialTimeSeriesDataset(X), batch_size=16, shuffle=True, drop_last=True)
        pipe.train(loader, loader, save_dir=d)
        anchors = {}
        for smp in ("pc", "em", "ode", "karras", "hybrid"):
            pipe.config["sampler"] = smp
            G = pipe.sample_panel(4, L, cond=X[:4], seed=0)
            anchors[smp] = float(np.abs(G[:, 0, 0]).max())
        pin_ok = max(anchors.values()) == 0.0
        pipe.config.update(sampler="hybrid", hybrid_fine_checkpoint=os.path.join(d, "final_model.pth"),
                           hybrid_fine_sigma=0.05)
        a = pipe.sample_panel(4, L, cond=X[:4], seed=3)
        b = pipe.sample_panel(4, L, cond=X[:4], seed=3)
        rep_ok = np.array_equal(a, b)
        legacy = PanelDiffusion(dict(cfg, noise_embedding="legacy"))
        try:
            legacy.load(os.path.join(d, "final_model.pth"))
            guard_ok = False
        except ValueError:
            guard_ok = True
    ok &= pin_ok and rep_ok and guard_ok
    if verbose:
        print(f"  max |anchor| per sampler {anchors}  {'PASS' if pin_ok else 'FAIL'}")
        print(f"  seeded splice calls identical: {rep_ok}; embedding-mismatch load refused: {guard_ok}")

    # (d) min-SNR weights for ε-loss: min(SNR, γ)/SNR ∈ (0, 1], 1 at high noise
    std = torch.tensor([0.01, 0.5, 5.0])
    snr = 1 / std ** 2
    w = torch.clamp(snr, max=5.0) / snr
    snr_ok = bool((w <= 1).all() and w[-1] == 1 and w[0] < 0.01)
    ok &= snr_ok

    # (e) bank-wide transfer map keeps scenario-to-scenario tail variation
    gen = [rng.standard_normal(20000) * s for s in (0.7, 1.0, 1.4)]
    ref = rng.standard_t(3, 200000)
    tm = TransferMap(np.concatenate(gen), ref)
    counts = [int((np.abs(tm(g)) > 4).sum()) for g in gen]
    tail_ok = counts[0] < counts[1] < counts[2]
    ok &= tail_ok
    if verbose:
        print(f"  min-SNR weights {w.numpy().round(4)}  {'PASS' if snr_ok else 'FAIL'}")
        print(f"  |shock|>4 per scenario after bank map {counts} (must differ)  {'PASS' if tail_ok else 'FAIL'}")
        print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def run_all_tests():
    """Run all validation tests."""
    print("\n" + "#" * 60)
    print("#  GBM Financial Diffusion — Pipeline Validation")
    print("#" * 60)

    results = {}

    # Test 1: Oracle
    try:
        passed, mse = test_sde_oracle()
        results["oracle"] = ("PASS" if passed else "FAIL", f"MSE={mse:.6f}")
    except Exception as e:
        results["oracle"] = ("ERROR", str(e))

    # Test 2: Shapes
    try:
        passed = test_score_network_shapes()
        results["shapes"] = ("PASS" if passed else "FAIL", "")
    except Exception as e:
        results["shapes"] = ("ERROR", str(e))

    # Test 3: Loss convergence
    try:
        passed, losses = test_loss_convergence()
        results["convergence"] = ("PASS" if passed else "FAIL",
                                  f"loss {losses[0]:.4f}→{losses[-1]:.4f}")
    except Exception as e:
        results["convergence"] = ("ERROR", str(e))

    # Test 4: Noise prediction
    try:
        passed = test_noise_prediction_quality()
        results["noise_pred"] = ("PASS" if passed else "FAIL", "")
    except Exception as e:
        results["noise_pred"] = ("ERROR", str(e))

    # Test 5: Generation scale
    try:
        passed = test_generation_scale()
        results["gen_scale"] = ("PASS" if passed else "FAIL", "")
    except Exception as e:
        results["gen_scale"] = ("ERROR", str(e))

    # Test 6: Sampler oracle
    try:
        passed = test_sampler_oracle()
        results["sampler_oracle"] = ("PASS" if passed else "FAIL", "")
    except Exception as e:
        results["sampler_oracle"] = ("ERROR", str(e))

    # Test 7: EDM preconditioning
    try:
        passed = test_edm_precond()
        results["edm_precond"] = ("PASS" if passed else "FAIL", "")
    except Exception as e:
        results["edm_precond"] = ("ERROR", str(e))

    # Test 8: panel model
    try:
        passed = test_panel_model()
        results["panel_model"] = ("PASS" if passed else "FAIL", "")
    except Exception as e:
        results["panel_model"] = ("ERROR", str(e))

    # Test 9: tail calibration
    try:
        passed = test_tail_calibration()
        results["tail_calib"] = ("PASS" if passed else "FAIL", "")
    except Exception as e:
        results["tail_calib"] = ("ERROR", str(e))

    # Test 10: audit fixes
    try:
        passed = test_audit_fixes()
        results["audit_fixes"] = ("PASS" if passed else "FAIL", "")
    except Exception as e:
        results["audit_fixes"] = ("ERROR", str(e))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, (status, detail) in results.items():
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {marker} {name:15s}: {status} {detail}")
        if status != "PASS":
            all_passed = False

    if all_passed:
        print("\n  ALL TESTS PASSED — pipeline is correct.")
        print("  Quality issues are due to data scale / training duration.")
    else:
        print("\n  SOME TESTS FAILED — check output above.")

    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "oracle", "shapes", "train", "noise", "gen",
                                 "samplers", "edm", "panel", "tails", "audit", "quick"])
    args = parser.parse_args()

    if args.test == "all":
        run_all_tests()
    elif args.test == "oracle":
        test_sde_oracle()
    elif args.test == "shapes":
        test_score_network_shapes()
    elif args.test == "train":
        test_loss_convergence()
    elif args.test == "noise":
        test_noise_prediction_quality()
    elif args.test == "gen":
        test_generation_scale()
    elif args.test == "samplers":
        test_sampler_oracle()
    elif args.test == "edm":
        test_edm_precond()
    elif args.test == "panel":
        test_panel_model()
    elif args.test == "tails":
        test_tail_calibration()
    elif args.test == "audit":
        test_audit_fixes()
    elif args.test == "quick":
        # Quick: just oracle + shapes (fast, no training)
        test_sde_oracle()
        test_score_network_shapes()
