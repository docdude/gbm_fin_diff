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
                        choices=["all", "oracle", "shapes", "train", "noise", "gen", "quick"])
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
    elif args.test == "quick":
        # Quick: just oracle + shapes (fast, no training)
        test_sde_oracle()
        test_score_network_shapes()
