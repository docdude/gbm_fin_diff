#!/usr/bin/env python3
"""Train / evaluate panel models for market-conditioned synthetic OHLCV.

Two models:
  --model market   unconditional: [market log-price, log VIX]           (K=2)
  --model ticker   [log-price, log-range, rel. log-volume | market, VIX]  (K=5,
                   last two channels observed)

Examples (single GPU):
    python -m gbm_financial.train_panel --model ticker --precond edm --save_dir save/panel/ticker_edm
    python -m gbm_financial.train_panel --model ticker --save_dir save/panel/ticker_eps
    python -m gbm_financial.train_panel --model market --precond edm --save_dir save/panel/market_edm
    python -m gbm_financial.train_panel --model ticker --eval_only --save_dir save/panel/ticker_edm \\
        --sampler hybrid --hybrid_sigma_switch 0.5 \\
        --hybrid_fine_checkpoint save/panel/ticker_eps/best_model.pth --hybrid_fine_sigma 0.03
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare import compare_returns
from .data import FinancialTimeSeriesDataset, ticker_split
from .panel_data import PanelData, TICKER_CHANNELS, MARKET_CHANNELS
from .panel_model import PanelDiffusion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_CONFIG = {
    "sde_type": "gbm", "schedule": "exponential",
    "sigma_min": 1e-3, "sigma_max": "auto", "n_reverse_steps": 2000,
    "channels": 128, "diff_emb_dim": 256, "feat_emb_dim": 64, "time_emb_dim": 128,
    "n_layers": 4, "n_heads": 8,
    "seq_len": 512, "batch_size": 64, "epochs": 400, "lr": 1e-3, "weight_decay": 1e-6,
    "ema_decay": 0.999, "checkpoint_every": 50, "precond": "none",
    "sampling_eps": 1e-5, "pc_corrector_steps": 0, "sampler": "pc",
    "hybrid_ode_steps": 100, "hybrid_sde_steps": 1000,
    "data_mode": "log_price", "normalize_mode": "none",
    "seed": 0, "train_ratio": 0.9,
}


def build_data(args, config):
    P = PanelData(args.ohlcv, args.vix, end_date=args.end_date)
    train_t, val_t = ticker_split(P.tickers, config["train_ratio"], config["seed"])
    L = config["seq_len"]
    if args.model == "ticker":
        Xtr, _ = P.ticker_windows(train_t, L=L, stride=args.stride)
        Xva, _ = P.ticker_windows(val_t, L=L, stride=args.stride)
        names = list(TICKER_CHANNELS) + list(MARKET_CHANNELS)
        config.update(n_channels=5, conditional=True, cond_channels=[3, 4], anchor_channels=[0, 3])
    else:
        X, meta = P.market_windows(L=L, stride=args.stride, n_baskets=args.n_baskets, seed=config["seed"])
        # time-based hold-out for the single market series: last 10% of window starts
        starts = np.array([m[1] for m in meta])
        cut = np.quantile(np.unique(starts).astype("datetime64[D]").astype(float), 0.9)
        is_val = starts.astype("datetime64[D]").astype(float) >= cut
        Xtr, Xva = X[~is_val], X[is_val & np.array([m[0] == "EW" for m in meta])]
        names = list(MARKET_CHANNELS)
        config.update(n_channels=2, conditional=False, cond_channels=[], anchor_channels=[0])
    config["channel_names"] = names
    config["val_tickers"] = val_t if args.model == "ticker" else []
    print(f"{args.model} windows: train {Xtr.shape}, val {Xva.shape}; {P.summary()}")
    return P, Xtr, Xva


def auto_sigma(config, Xtr):
    tgt = [k for k in range(Xtr.shape[1]) if k not in config["cond_channels"]]
    if config.get("sigma_max") == "auto":
        s = 0.0
        for k in tgt:
            x = Xtr[:1000, k]
            rng = np.percentile(x.max(0) - x.min(0), 99)
            s = max(s, 3.0 * max(x.std(), rng / 2))
        config["sigma_max"] = float(max(s, 5.0))
    if config.get("precond") == "edm" and config.get("edm_sigma_data") is None:
        sd = [float(Xtr[:, k].std()) if k in tgt else 1.0 for k in range(Xtr.shape[1])]
        config["edm_sigma_data"] = sd
    print(f"σ_max={config['sigma_max']:.3f}" + (f"  σ_data={np.round(config['edm_sigma_data'], 4).tolist()}"
                                                if config.get("edm_sigma_data") else ""))


def _ac1(x):
    xc = x - x.mean(1, keepdims=True)
    return (xc[:, 1:] * xc[:, :-1]).mean(1) / np.maximum(xc.var(1), 1e-20)


def _wcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).mean(1) / np.maximum(a.std(1) * b.std(1), 1e-20)


def panel_metrics(G, R, model):
    """Channel-level and cross-channel metrics, generated G vs real R (N, K, L)."""
    out = {}
    if model == "ticker":
        rg, rr = np.diff(G[:, 0], axis=1), np.diff(R[:, 0], axis=1)
        mg, mr = np.diff(G[:, 3], axis=1), np.diff(R[:, 3], axis=1)
        for name, f in [("corr_ret_market", lambda X, r, m: _wcorr(r, m)),
                        ("corr_absret_range", lambda X, r, m: _wcorr(np.abs(r), X[:, 1, 1:])),
                        ("corr_absret_vol", lambda X, r, m: _wcorr(np.abs(r), X[:, 2, 1:])),
                        ("corr_range_vol", lambda X, r, m: _wcorr(X[:, 1], X[:, 2])),
                        ("ac1_range", lambda X, r, m: _ac1(X[:, 1])),
                        ("ac1_vol", lambda X, r, m: _ac1(X[:, 2])),
                        ("range_level_std", lambda X, r, m: X[:, 1].std(1)),
                        ("vol_level_std", lambda X, r, m: X[:, 2].std(1)),
                        ("corr_rvol_vix", lambda X, r, m: _wcorr(_roll_vol(r), X[:, 4, 20:]))]:
            out[name] = float(np.mean(f(G, rg, mg)))
            out[name + "_real"] = float(np.mean(f(R, rr, mr)))
        out["beta_market"] = float(np.mean((rg * mg).mean(1) / np.maximum(mg.var(1), 1e-20)))
        out["beta_market_real"] = float(np.mean((rr * mr).mean(1) / np.maximum(mr.var(1), 1e-20)))
    else:
        rg, rr = np.diff(G[:, 0], axis=1), np.diff(R[:, 0], axis=1)
        out["corr_rvol_vix"] = float(np.mean(_wcorr(_roll_vol(rg), G[:, 1, 20:])))
        out["corr_rvol_vix_real"] = float(np.mean(_wcorr(_roll_vol(rr), R[:, 1, 20:])))
        out["corr_ret_dvix"] = float(np.mean(_wcorr(rg, np.diff(G[:, 1], axis=1))))
        out["corr_ret_dvix_real"] = float(np.mean(_wcorr(rr, np.diff(R[:, 1], axis=1))))
        out["vix_level_std"] = float(G[:, 1].std(1).mean())
        out["vix_level_std_real"] = float(R[:, 1].std(1).mean())
        out["ac1_dvix"] = float(np.mean(_ac1(np.diff(G[:, 1], axis=1))))
        out["ac1_dvix_real"] = float(np.mean(_ac1(np.diff(R[:, 1], axis=1))))
    return out


def _roll_vol(r, w=20):
    """Trailing w-day realised vol of returns r (N, L-1) → (N, L-w); entry i covers
    returns of days i+1 … i+w, i.e. it is aligned with day i+w of the window."""
    c = np.cumsum(np.concatenate([np.zeros((len(r), 1)), r ** 2], axis=1), axis=1)
    return np.sqrt((c[:, w:] - c[:, :-w]) / w)


def evaluate(model, config, Xtr, Xva, out_dir, n_seeds=3, n_per_seed=None):
    os.makedirs(out_dir, exist_ok=True)
    L = config["seq_len"]
    rng = np.random.default_rng(0)
    if config["model"] == "ticker":
        n = min(n_per_seed or len(Xva), len(Xva))
        ref_rows = rng.choice(len(Xva), n, replace=False)
        ref = Xva[ref_rows]
    else:
        n = n_per_seed or 120
        ref = Xva
    real_tr_ret = np.diff(Xtr[:, 0], axis=1)
    real_va_ret = np.diff(Xva[:, 0], axis=1)
    results, gens = [], []
    for sd in range(n_seeds):
        t0 = time.time()
        cond = ref if config["model"] == "ticker" else None
        G = model.sample_panel(n, L, cond=cond, seed=sd)
        np.save(os.path.join(out_dir, f"generated_seed{sd}.npy"), G)
        gens.append(G)
        met = compare_returns(np.diff(G[:, 0], axis=1), real_tr_ret, real_va_ret, seed=sd)
        met.update(panel_metrics(G, ref if config["model"] == "ticker" else Xva, config["model"]))
        met["rejection_rate"] = getattr(model, "last_rejection_rate", 0.0)
        results.append(met)
        print(f"  seed {sd}: {len(G)} samples in {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(out_dir, "real_ref.npy"), ref)
    agg = {k: {"mean": float(np.mean([r[k] for r in results])),
               "std": float(np.std([r[k] for r in results]))}
           for k in results[0] if isinstance(results[0][k], (int, float))}
    with open(os.path.join(out_dir, "panel_eval.json"), "w") as f:
        json.dump(agg, f, indent=2)
    keys = [k for k in agg if not k.endswith("_real")]
    print(f"\n{'metric':<26}{'generated':>14}{'real':>10}")
    for k in keys:
        rk = k + "_real"
        real = f"{agg[rk]['mean']:>10.4g}" if rk in agg else " " * 10
        print(f"{k:<26}{agg[k]['mean']:>10.4g}±{agg[k]['std']:<4.2g}{real}")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["ticker", "market"], required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--ohlcv", default=os.path.join(ROOT, "data/nas100/tiingo_ohlcv.csv.gz"))
    ap.add_argument("--vix", default=os.path.join(ROOT, "data/nas100/vix.csv"))
    ap.add_argument("--end_date", default="2024-05-31")
    ap.add_argument("--stride", type=int, default=None, help="default 64 (ticker) / 16 (market)")
    ap.add_argument("--n_baskets", type=int, default=8, help="market model basket augmentation")
    ap.add_argument("--precond", choices=["none", "edm"], default="none")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--sigma_min", type=float, default=None)
    ap.add_argument("--sigma_max", type=float, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--n_eval", type=int, default=None, help="samples per seed")
    ap.add_argument("--sampler", default=None, choices=["pc", "em", "ode", "karras", "hybrid"])
    ap.add_argument("--hybrid_sigma_switch", type=float, default=None)
    ap.add_argument("--hybrid_fine_checkpoint", default=None)
    ap.add_argument("--hybrid_fine_sigma", type=float, default=None)
    ap.add_argument("--reject_failed", action="store_true")
    ap.add_argument("--eval_tag", default=None)
    args = ap.parse_args()
    args.stride = args.stride or (64 if args.model == "ticker" else 16)

    config = dict(BASE_CONFIG, model=args.model)
    for k in ("precond", "epochs", "seq_len", "sigma_min", "sigma_max"):
        if getattr(args, k) is not None:
            config[k] = getattr(args, k)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    P, Xtr, Xva = build_data(args, config)
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, "real_train_sample.npy"), Xtr[np.random.default_rng(0).choice(
        len(Xtr), min(2000, len(Xtr)), replace=False)])

    if args.eval_only:
        with open(os.path.join(args.save_dir, "results.json")) as f:
            saved = json.load(f)["config"]
        config.update(saved)
        for k in ("sampler", "hybrid_sigma_switch", "hybrid_fine_checkpoint", "hybrid_fine_sigma"):
            if getattr(args, k) is not None:
                config[k] = getattr(args, k)
        model = PanelDiffusion(config)
        ck = os.path.join(args.save_dir, "best_model.pth")
        model.load(ck if os.path.exists(ck) else os.path.join(args.save_dir, "final_model.pth"))
    else:
        auto_sigma(config, Xtr)
        model = PanelDiffusion(config)
        if args.resume:
            model.load(args.resume)
        tl = DataLoader(FinancialTimeSeriesDataset(Xtr), batch_size=config["batch_size"],
                        shuffle=True, drop_last=True)
        vl = DataLoader(FinancialTimeSeriesDataset(Xva), batch_size=config["batch_size"])
        print(f"{len(tl)} batches/epoch × {config['epochs']} epochs")
        t0 = time.time()
        model.train(tl, vl, save_dir=args.save_dir)
        config["train_time_hours"] = (time.time() - t0) / 3600
        with open(os.path.join(args.save_dir, "results.json"), "w") as f:
            json.dump({"config": config}, f, indent=2, default=str)
        best = os.path.join(args.save_dir, "best_model.pth")
        if os.path.exists(best):
            model.load(best)

    if args.reject_failed:
        ac = model.path_lag1_autocorr(Xtr)
        model.config["reject_ac1_below"] = float(np.percentile(ac, 0.1))
    tag = args.eval_tag or (config.get("sampler", "pc") + (
        f"_{config.get('hybrid_sigma_switch')}" if config.get("sampler") == "hybrid" else "") + (
        f"_splice{config.get('hybrid_fine_sigma')}" if config.get("hybrid_fine_checkpoint") else "") + (
        "_filter" if args.reject_failed else ""))
    evaluate(model, config, Xtr, Xva, os.path.join(args.save_dir, f"eval_{tag}"),
             n_seeds=args.n_seeds, n_per_seed=args.n_eval)


if __name__ == "__main__":
    main()
