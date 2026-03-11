"""Re-evaluate generated_data.npy from L4 run with full pathwise diagnostics.

Usage:
  1. Download generated_data.npy from L4 to this machine
  2. python eval_l4_data.py <path_to_generated_data.npy>

Computes:
  - Stylized facts (alpha, beta, leverage)
  - Pathwise diagnostics (terminal sigma, mean path, vol-of-vol, bursts, drawdowns)
  - All plots saved next to the .npy file
"""

import sys
import os
import json
import numpy as np
import pickle

# Allow running from scripts/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbm_financial.metrics import (
    evaluate_stylized_facts,
    plot_stylized_facts,
    plot_diagnostics,
    plot_pathwise_diagnostics,
    plot_mean_path_diagnostic,
    compute_log_returns,
    compute_rolling_volatility,
    compute_burst_duration,
    compute_drawdown_stats,
    compute_path_roughness,
)
from gbm_financial.data import create_subsequences


def load_real_data(window_len=2048, stride=400):
    """Load real data for comparison."""
    cache_file = "data/financial/sp500_prices.pkl"
    if not os.path.exists(cache_file):
        print("No cached real data found at", cache_file)
        return None
    with open(cache_file, "rb") as f:
        stock_data = pickle.load(f)
    seqs = create_subsequences(stock_data, window_len=window_len, stride=stride, mode="log_price")
    return seqs


def compute_pathwise_summary(generated, real, mode="log_price"):
    """Compute all pathwise metrics and return summary dict."""
    gen_returns = compute_log_returns(generated, mode=mode)
    real_returns = compute_log_returns(real, mode=mode)

    # Terminal sigma ratio
    gen_terminal_std = np.std(generated[:, -1])
    real_terminal_std = np.std(real[:, -1])
    terminal_sigma_ratio = gen_terminal_std / max(real_terminal_std, 1e-12)

    # Mean path stats
    gen_mean_path = np.mean(generated, axis=0)
    real_mean_path = np.mean(real, axis=0)
    gen_std_path = np.std(generated, axis=0)
    real_std_path = np.std(real, axis=0)

    # Rolling vol
    gen_rv = compute_rolling_volatility(gen_returns)
    real_rv = compute_rolling_volatility(real_returns)
    vol_of_vol_ratio = gen_rv["vol_of_vol"] / max(real_rv["vol_of_vol"], 1e-12)

    # Burst duration
    gen_burst = compute_burst_duration(gen_returns)
    real_burst = compute_burst_duration(real_returns)
    burst_ratio = gen_burst["mean"] / max(real_burst["mean"], 1e-12)

    # Drawdowns (on log-price paths)
    if mode == "log_price":
        gen_dd = compute_drawdown_stats(generated)
        real_dd = compute_drawdown_stats(real)
        dd_ratio = gen_dd["mean_depth"] / max(real_dd["mean_depth"], 1e-12)
    else:
        gen_dd = {"mean_depth": np.nan}
        real_dd = {"mean_depth": np.nan}
        dd_ratio = np.nan

    # Path roughness (QV)
    gen_rough = compute_path_roughness(gen_returns)
    real_rough = compute_path_roughness(real_returns)
    qv_ratio = gen_rough["mean_qv"] / max(real_rough["mean_qv"], 1e-12)

    summary = {
        "terminal_sigma_gen": float(gen_terminal_std),
        "terminal_sigma_real": float(real_terminal_std),
        "terminal_sigma_ratio": float(terminal_sigma_ratio),
        "mean_path_final_gen": float(gen_mean_path[-1]),
        "mean_path_final_real": float(real_mean_path[-1]),
        "spread_final_gen": float(gen_std_path[-1]),
        "spread_final_real": float(real_std_path[-1]),
        "vol_of_vol_gen": float(gen_rv["vol_of_vol"]),
        "vol_of_vol_real": float(real_rv["vol_of_vol"]),
        "vol_of_vol_ratio": float(vol_of_vol_ratio),
        "burst_mean_gen": float(gen_burst["mean"]),
        "burst_mean_real": float(real_burst["mean"]),
        "burst_duration_ratio": float(burst_ratio),
        "drawdown_depth_gen": float(gen_dd["mean_depth"]),
        "drawdown_depth_real": float(real_dd["mean_depth"]),
        "drawdown_ratio": float(dd_ratio),
        "qv_gen": float(gen_rough["mean_qv"]),
        "qv_real": float(real_rough["mean_qv"]),
        "qv_ratio": float(qv_ratio),
    }
    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_l4_data.py <generated_data.npy>")
        sys.exit(1)

    npy_path = sys.argv[1]
    generated = np.load(npy_path)
    print(f"Loaded generated data: {generated.shape} from {npy_path}")

    save_dir = os.path.dirname(os.path.abspath(npy_path))
    mode = "log_price"  # L4 run used GBM = log_price mode

    # Stylized facts
    print("\n=== Stylized Facts (generated) ===")
    gen_results = evaluate_stylized_facts(generated, mode=mode)

    # Real data
    real = load_real_data(window_len=generated.shape[1])
    real_results = None
    if real is not None:
        print(f"\nReal data: {real.shape}")
        print("\n=== Stylized Facts (real) ===")
        real_results = evaluate_stylized_facts(real, mode=mode)

        # Pathwise diagnostics
        print("\n=== Pathwise Diagnostics ===")
        pathwise = compute_pathwise_summary(generated, real, mode=mode)

        for k, v in sorted(pathwise.items()):
            if "ratio" in k:
                flag = ""
                if isinstance(v, float) and v < 0.6:
                    flag = " << WARNING: squashed"
                elif isinstance(v, float) and v > 1.5:
                    flag = " >> WARNING: inflated"
                print(f"  {k:30s} = {v:.4f}{flag}")
            else:
                print(f"  {k:30s} = {v:.4f}")

        # Plots
        print(f"\nSaving plots to {save_dir}/")
        plot_stylized_facts(gen_results, real_results,
                            save_path=os.path.join(save_dir, "stylized_facts.png"))
        plot_diagnostics(generated, real, mode=mode,
                         save_path=os.path.join(save_dir, "diagnostics.png"))
        plot_pathwise_diagnostics(generated, real, mode=mode,
                                  save_path=os.path.join(save_dir, "pathwise_diagnostics.png"))
        plot_mean_path_diagnostic(generated, real, mode=mode,
                                   save_path=os.path.join(save_dir, "mean_path_diagnostic.png"))

        # Save full results
        full_results = {
            "stylized_facts_gen": {
                "alpha": gen_results["heavy_tail"]["alpha"],
                "beta": gen_results["volatility_clustering"]["beta"],
                "vol_acf_1": gen_results["volatility_clustering"]["autocorrelation"][0],
                "leverage_0": gen_results["leverage_effect"]["leverage_correlation"][0],
                "leverage_1": gen_results["leverage_effect"]["leverage_correlation"][1],
            },
            "stylized_facts_real": {
                "alpha": real_results["heavy_tail"]["alpha"],
                "beta": real_results["volatility_clustering"]["beta"],
                "vol_acf_1": real_results["volatility_clustering"]["autocorrelation"][0],
                "leverage_0": real_results["leverage_effect"]["leverage_correlation"][0],
                "leverage_1": real_results["leverage_effect"]["leverage_correlation"][1],
            },
            "pathwise": pathwise,
        }
        out_path = os.path.join(save_dir, "full_eval_results.json")
        with open(out_path, "w") as f:
            json.dump(full_results, f, indent=2, default=lambda x: float(x))
        print(f"Saved full results to {out_path}")
    else:
        print("No real data available — skipping comparison plots")

    print("\nDone.")


if __name__ == "__main__":
    main()
