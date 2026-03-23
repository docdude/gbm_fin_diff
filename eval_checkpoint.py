#!/usr/bin/env python3
"""Evaluate a checkpoint: generate samples with selectable sampler & compute metrics.

Supports PC (Langevin), EM, and ODE samplers. Designed to run against a
best_model.pth while training is still in progress.

Usage:
    # PC sampler (default, matches score_sde VE):
    python eval_checkpoint.py --checkpoint save/gbm_financial_langevin/gbm_exponential/best_model.pth

    # Predictor-only (no Langevin corrector):
    python eval_checkpoint.py --checkpoint ... --corrector-steps 0

    # EM sampler (legacy):
    python eval_checkpoint.py --checkpoint ... --sampler em

    # Sweep SNR values:
    python eval_checkpoint.py --checkpoint ... --snr 0.05

    # Custom output dir:
    python eval_checkpoint.py --checkpoint ... --save-dir save/eval_run1
"""

import argparse
import sys
import os
import json
import numpy as np
import torch
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from gbm_financial.train import GBMFinancialDiffusion
from gbm_financial.data import get_dataloaders, compute_sigma_max
from gbm_financial.metrics import (compute_pathwise_diagnostics,
                                   print_pathwise_summary,
                                   compute_distribution_distances,
                                   print_distribution_distances,
                                   compute_log_returns,
                                   evaluate_stylized_facts,
                                   plot_stylized_facts,
                                   plot_diagnostics,
                                   plot_pathwise_diagnostics,
                                   plot_mean_path_diagnostic)


def quick_scalar_comparison(gen, real, label=""):
    """Key scalar metrics for quick comparison."""
    ret_r = compute_log_returns(real, mode="log_price")
    ret_g = compute_log_returns(gen, mode="log_price")

    def ac1(r):
        acs = []
        for i in range(r.shape[0]):
            if r.shape[1] > 1:
                c = np.corrcoef(r[i, :-1], r[i, 1:])[0, 1]
                acs.append(c)
        return np.nanmean(acs)

    metrics = {
        "Return std":       (np.std(ret_r), np.std(ret_g)),
        "Return AC(1)":     (ac1(ret_r), ac1(ret_g)),
        "Path std":         (np.std(real), np.std(gen)),
        "Endpoint mean":    (np.mean(real[:, -1]), np.mean(gen[:, -1])),
        "Endpoint std":     (np.std(real[:, -1]), np.std(gen[:, -1])),
        "Total variation":  (np.mean(np.sum(np.abs(np.diff(real, axis=1)), axis=1)),
                             np.mean(np.sum(np.abs(np.diff(gen, axis=1)), axis=1))),
        "Quadratic var":    (np.mean(np.sum(np.diff(real, axis=1)**2, axis=1)),
                             np.mean(np.sum(np.diff(gen, axis=1)**2, axis=1))),
        "Max drawdown":     (np.mean([np.min(real[i] - np.maximum.accumulate(real[i]))
                                       for i in range(real.shape[0])]),
                             np.mean([np.min(gen[i] - np.maximum.accumulate(gen[i]))
                                       for i in range(gen.shape[0])])),
    }

    print(f"\n{'='*70}")
    print(f"SCALAR METRICS {label}")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Real':>12} {'Generated':>12} {'Ratio':>8} {'Status':>8}")
    print("-" * 70)
    for name, (r, g) in metrics.items():
        if abs(r) > 1e-12:
            ratio = g / r
        else:
            ratio = float('nan')
        if 0.8 <= abs(ratio) <= 1.2:
            status = "OK"
        elif 0.5 <= abs(ratio) <= 2.0:
            status = "~"
        else:
            status = "!!"
        print(f"{name:<25} {r:>12.4f} {g:>12.4f} {ratio:>8.2f} {status:>8}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with selectable sampler")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pth or checkpoint_epochN.pth")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory (default: alongside checkpoint)")
    parser.add_argument("--sampler", type=str, default="pc",
                        choices=["pc", "em", "ode", "karras"],
                        help="Sampling method: pc (Langevin), em (Euler-Maruyama), ode, karras (Heun 2nd-order)")
    parser.add_argument("--karras-rho", type=float, default=None,
                        help="Karras sigma schedule exponent (default: 7)")
    parser.add_argument("--n-generate", type=int, default=60)
    parser.add_argument("--n-reverse", type=int, default=2000)
    parser.add_argument("--snr", type=float, default=0.16,
                        help="Langevin corrector SNR (PC sampler only)")
    parser.add_argument("--corrector-steps", type=int, default=0,
                        help="Langevin corrector steps per predictor step (0 = predictor-only)")
    parser.add_argument("--eps", type=float, default=None,
                        help="Sampling epsilon (min noise level). PC default=1e-5, EM default=1e-3")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=None,
                        help="Override data stride (default: from checkpoint config)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plot files")
    args = parser.parse_args()

    # Resolve save dir
    if args.save_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        suffix = args.sampler
        if args.sampler == "pc" and args.corrector_steps == 0:
            suffix = "pc_pred_only"
        elif args.sampler == "pc":
            suffix = f"pc_snr{args.snr}"
        elif args.sampler == "karras":
            suffix = f"karras_rho{config.get('karras_rho', 7) if args.karras_rho is None else args.karras_rho}"
        if args.eps is not None:
            suffix += f"_eps{args.eps}"
        args.save_dir = os.path.join(ckpt_dir, f"eval_{suffix}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Try to load config from results.json next to checkpoint, else from checkpoint
    ckpt_dir = os.path.dirname(args.checkpoint)
    results_json = os.path.join(ckpt_dir, "results.json")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if os.path.exists(results_json):
        with open(results_json) as f:
            config = json.load(f).get("config", {})
        print(f"Config from {results_json}")
    elif "config" in ckpt:
        config = ckpt["config"]
        print(f"Config from checkpoint")
    else:
        print("ERROR: No config found in results.json or checkpoint")
        sys.exit(1)

    # Merge architecture keys from checkpoint config that may be missing
    # from results.json (e.g. wavenet_branch added after results.json was written)
    ckpt_config = ckpt.get("config", {})
    for key in ("wavenet_branch", "wavenet_dilation_rates", "film_conditioning"):
        if key not in config and key in ckpt_config:
            config[key] = ckpt_config[key]
            print(f"  Merged '{key}={ckpt_config[key]}' from checkpoint config")

    # Override sampling params
    config["n_reverse_steps"] = args.n_reverse
    config["pc_snr"] = args.snr
    config["pc_corrector_steps"] = args.corrector_steps
    if args.eps is not None:
        config["sampling_eps"] = args.eps
    if args.karras_rho is not None:
        config["karras_rho"] = args.karras_rho

    # Load data
    print("Loading data...")
    stride = args.stride or config.get("stride", 400)
    window_len = config.get("seq_len", config.get("window_len", 2048))

    train_loader, val_loader, dataset_info = get_dataloaders(
        sde_type=config.get("sde_type", "gbm"),
        window_len=window_len,
        stride=stride,
        batch_size=config.get("batch_size", 64),
        min_years=config.get("min_years", 40),
        num_workers=2,
    )

    # σ_max: prefer saved numeric value from checkpoint/config (deterministic).
    # Only recompute from data if missing or set to "auto".
    saved_sigma = config.get("sigma_max")
    if isinstance(saved_sigma, (int, float)) and saved_sigma > 0:
        print(f"σ_max (from config) = {saved_sigma:.2f}")
    else:
        sigma_max, _ = compute_sigma_max(train_loader)
        config["sigma_max"] = sigma_max
        print(f"σ_max (auto-computed) = {sigma_max:.2f}")

    # Gather real data
    real_data = np.concatenate([b.numpy() for b in train_loader], axis=0)
    print(f"Real data: {real_data.shape}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GBMFinancialDiffusion(config, device=device)
    model.load(args.checkpoint)
    print("Checkpoint loaded.")

    # Generate
    sampler_label = args.sampler.upper()
    if args.sampler == "pc":
        if args.corrector_steps == 0:
            sampler_label = "PC (predictor-only)"
        else:
            sampler_label = f"PC (snr={args.snr}, corr_steps={args.corrector_steps})"
        generated = model.generate(n_samples=args.n_generate,
                                   seq_len=window_len,
                                   batch_size=args.batch_size)
    elif args.sampler == "em":
        generated = model.generate_em(n_samples=args.n_generate,
                                      seq_len=window_len,
                                      batch_size=args.batch_size)
    elif args.sampler == "ode":
        generated = model.generate_ode(n_samples=args.n_generate,
                                       seq_len=window_len,
                                       batch_size=args.batch_size)
    elif args.sampler == "karras":
        sampler_label = f"Karras/Heun (ρ={config.get('karras_rho', 7)})"
        generated = model.generate_karras(n_samples=args.n_generate,
                                          seq_len=window_len,
                                          batch_size=args.batch_size)

    np.save(os.path.join(args.save_dir, "generated_data.npy"), generated)

    # Scalar metrics
    scalar_metrics = quick_scalar_comparison(generated, real_data, f"({sampler_label})")

    # Pathwise diagnostics
    gen_pw = compute_pathwise_diagnostics(generated, mode="log_price")
    real_pw = compute_pathwise_diagnostics(real_data, mode="log_price")
    print_pathwise_summary(gen_pw, real_pw)

    # Distribution distances (KS + Wasserstein)
    dist_distances = compute_distribution_distances(
        gen_pw, real_pw, gen_data=generated, real_data=real_data)
    print_distribution_distances(dist_distances)

    # Stylized facts + plots
    gen_results, real_results = model.evaluate(generated, real_data, save_dir=args.save_dir)

    if not args.no_plots:
        mode = "log_price" if config.get("sde_type") == "gbm" else "log_return"
        plot_diagnostics(generated, real_data, mode=mode,
                         save_path=os.path.join(args.save_dir, "diagnostics.png"))
        plot_pathwise_diagnostics(generated, real_data, mode=mode,
                                  save_path=os.path.join(args.save_dir, "pathwise_diagnostics.png"))
        plot_mean_path_diagnostic(generated, real_data, mode=mode,
                                   save_path=os.path.join(args.save_dir, "mean_path_diagnostic.png"))

    # Save eval metadata
    eval_meta = {
        "checkpoint": args.checkpoint,
        "sampler": args.sampler,
        "n_reverse": args.n_reverse,
        "snr": args.snr,
        "corrector_steps": args.corrector_steps,
        "eps": args.eps,
        "n_generate": args.n_generate,
        "sigma_max": config["sigma_max"],
        "n_real": real_data.shape[0],
    }
    with open(os.path.join(args.save_dir, "eval_meta.json"), "w") as f:
        json.dump(eval_meta, f, indent=2)

    # ── Persistent eval log (append one record per run) ──
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "sampler": sampler_label,
        "eps": args.eps,
        "snr": args.snr,
        "corrector_steps": args.corrector_steps,
        "n_reverse": args.n_reverse,
        "n_generate": args.n_generate,
        "sigma_max": config["sigma_max"],
        "save_dir": args.save_dir,
    }
    # Scalar metrics (real, gen, ratio)
    for name, (r, g) in scalar_metrics.items():
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        ratio = g / r if abs(r) > 1e-12 else float('nan')
        log_record[f"{key}_real"] = float(r)
        log_record[f"{key}_gen"] = float(g)
        log_record[f"{key}_ratio"] = float(ratio)
    # Pathwise diagnostics (ratios)
    pw_pairs = [
        ("rolling_vol_mean", "rolling_vol", "mean_vol"),
        ("vol_of_vol", "rolling_vol", "vol_of_vol"),
        ("burst_dur_mean", "burst_duration", "mean"),
        ("max_dd_mean", "drawdown", "mean_depth"),
        ("dd_dur_mean", "drawdown", "mean_duration"),
        ("turning_pt_density", "turning_points", "mean_density"),
        ("qv_pw_mean", "roughness", "mean_qv"),
        ("tv_returns_mean", "roughness", "mean_tv_returns"),
    ]
    for log_key, pw_group, pw_field in pw_pairs:
        rv = real_pw[pw_group][pw_field]
        gv = gen_pw[pw_group][pw_field]
        log_record[f"{log_key}_real"] = float(rv)
        log_record[f"{log_key}_gen"] = float(gv)
        log_record[f"{log_key}_ratio"] = float(gv / rv) if abs(rv) > 1e-12 else float('nan')
    # Stylized facts
    log_record["alpha_gen"] = float(gen_results["heavy_tail"]["alpha"])
    log_record["alpha_real"] = float(real_results["heavy_tail"]["alpha"]) if real_results else None
    log_record["beta_gen"] = float(gen_results["volatility_clustering"]["beta"])
    log_record["beta_real"] = float(real_results["volatility_clustering"]["beta"]) if real_results else None
    lev = gen_results["leverage_effect"]["leverage_correlation"]
    log_record["leverage_L1_gen"] = float(lev[1]) if len(lev) > 1 else None
    # Distribution distances
    for dist_name, dist_vals in dist_distances.items():
        log_record[f"ks_{dist_name}"] = dist_vals["ks_stat"]
        log_record[f"ks_pval_{dist_name}"] = dist_vals["ks_pval"]
        log_record[f"wass_{dist_name}"] = dist_vals["wasserstein"]

    # Append to project-root log file
    log_path = os.path.join(os.path.dirname(__file__), "eval_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_record) + "\n")
    print(f"  → Appended to {log_path}")

    print(f"\nResults saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
