"""Assemble synthetic multi-asset OHLCV panels from the market and ticker models.

Pipeline per scenario (all tickers share one market path):
  1. market path  [market log-price, log VIX]: sampled from the market model, or
     taken from a real train-period window (``market="real"``)
  2. every ticker's [log-price, log-range, rel. log-volume] sampled conditioned
     on that market path
  3. inversion to OHLCV with per-ticker statistics; high/low placement and the
     overnight gap are drawn from empirical (train-period) distributions

Output CSVs use the FinRL-Meta raw schema (date, open, high, low, close,
adjusted_close, volume, tic, day) on a real trading calendar segment, so the
env's own indicator / volatility / ADV pipeline can be applied unchanged
(see scripts/export_finrl_meta_env.py).

CLI:
    python -m gbm_financial.panel_assembly --n_scenarios 20 --out save/panel/scenarios \\
        --ticker_dir save/panel/ticker_edm --ticker_fine save/panel/ticker_eps/best_model.pth \\
        --market_dir save/panel/market_edm --market_fine save/panel/market_eps/best_model.pth
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from .panel_data import PanelData, SCALE, RANGE_FLOOR
from .panel_model import PanelDiffusion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZBINS = np.array([-np.inf, -1.5, -0.5, -0.15, 0.15, 0.5, 1.5, np.inf])


def load_model(exp_dir, fine_ckpt=None, sampler="hybrid", switch=0.5, fine_sigma=0.03,
               reject_ac1_below=None):
    with open(os.path.join(exp_dir, "results.json")) as f:
        cfg = json.load(f)["config"]
    cfg.update(sampler=sampler, hybrid_sigma_switch=switch)
    if fine_ckpt:
        cfg.update(hybrid_fine_checkpoint=fine_ckpt, hybrid_fine_sigma=fine_sigma)
    if reject_ac1_below is not None:
        cfg["reject_ac1_below"] = reject_ac1_below
    m = PanelDiffusion(cfg)
    ck = os.path.join(exp_dir, "best_model.pth")
    m.load(ck if os.path.exists(ck) else os.path.join(exp_dir, "final_model.pth"))
    return m


class EmpiricalOHLC:
    """Train-period distributions used to place high/low around close and the open."""

    def __init__(self, P: PanelData):
        z = P.ret / P.stats["ret_std"]
        with np.errstate(divide="ignore", invalid="ignore"):
            hl = np.log(P.high / P.low)
            a = np.log(P.high / P.close) / hl
        ok = np.isfinite(z) & np.isfinite(a) & (hl > RANGE_FLOOR)
        zb = np.digitize(z[ok], ZBINS) - 1
        self.a_pools = [np.clip(a[ok][zb == b], 0, 1) for b in range(len(ZBINS) - 1)]
        gap = np.full_like(P.close, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap[1:] = np.log(P.open[1:] / P.close[:-1]) / P.stats["ret_std"]
        self.gap_pool = gap[np.isfinite(gap)]
        self.gap_pool = self.gap_pool[np.abs(self.gap_pool) < 10]

    def sample(self, z, rng):
        a = np.empty_like(z)
        zb = np.digitize(z, ZBINS) - 1
        for b, pool in enumerate(self.a_pools):
            m = zb == b
            if m.any():
                a[m] = rng.choice(pool, m.sum())
        g = rng.choice(self.gap_pool, z.shape)
        return a, g


class PanelGenerator:

    def __init__(self, P, ticker_model, market_model=None, seed=0):
        self.P = P
        self.tm = ticker_model
        self.mm = market_model
        self.L = int(ticker_model.config["seq_len"])
        self.emp = EmpiricalOHLC(P)
        self.rng = np.random.default_rng(seed)

    # -- market -------------------------------------------------------------
    def market_path(self, mode="model", start_idx=None, seed=None):
        if mode == "real":
            return self.P.market_window(start_idx, self.L)
        if self.mm is None:
            raise ValueError("market='model' needs a market model")
        return self.mm.sample_panel(1, self.L, cond=None, seed=seed)[0]

    # -- tickers ------------------------------------------------------------
    def ticker_paths(self, market, tickers, seed=None):
        n = len(tickers)
        cond = np.zeros((n, 5, self.L), dtype=np.float32)
        cond[:, 3:] = market[None]
        return self.tm.sample_panel(n, self.L, cond=cond, seed=seed, chunk=n)[:, :3]

    # -- inversion ----------------------------------------------------------
    def _anchor(self, j, ref_idx):
        P = self.P
        c = P.close[:ref_idx + 1, j]
        ok = np.where(np.isfinite(c))[0]
        k = ok[-1] if len(ok) else np.where(np.isfinite(P.close[:, j]))[0][0]
        base = P.lvol_base[:ref_idx + 1, j]
        okb = np.where(np.isfinite(base))[0]
        b = base[okb[-1]] if len(okb) else np.nanmedian(P.lvol_base[:, j])
        return P.close[k, j], b

    def to_ohlcv(self, paths, market, tickers, dates, ref_idx):
        P, S = self.P, self.P.stats
        rows = []
        for i, tic in enumerate(tickers):
            j = P.tickers.index(tic)
            c0, lr, lv = paths[i]
            ret = np.diff(c0) / SCALE * S["ret_std"][j]
            c_start, vbase = self._anchor(j, ref_idx)
            close = c_start * np.exp(np.r_[0.0, np.cumsum(ret)])
            hl = np.exp(lr / SCALE * S["lrange_dstd"][j] + S["lrange_mean"][j])
            z = np.r_[0.0, ret / S["ret_std"][j]]
            a, g = self.emp.sample(z, self.rng)
            high = close * np.exp(a * hl)
            low = high * np.exp(-hl)
            prev = np.r_[close[0], close[:-1]]
            open_ = np.clip(prev * np.exp(g * S["ret_std"][j]), low, high)
            volume = np.exp(lv / SCALE * S["lvol_dstd"][j] + S["lvol_mean"][j] + vbase)
            rows.append(pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                                      "close": close, "adjusted_close": close,
                                      "volume": np.round(volume), "tic": tic}))
        df = pd.concat(rows, ignore_index=True)
        df["day"] = pd.to_datetime(df["date"]).dt.dayofweek
        mret = np.diff(market[0]) / SCALE * P.mstats["mret_std"]
        mk = pd.DataFrame({"date": dates,
                           "market_index": 100 * np.exp(np.r_[0.0, np.cumsum(mret)]),
                           "vix": np.exp(market[1] / SCALE * P.mstats["lvix_dstd"] + P.mstats["lvix_mean"])})
        return df.sort_values(["date", "tic"]).reset_index(drop=True), mk

    def scenario(self, tickers, start_idx, market="model", seed=None):
        dates = self.P.dates[start_idx:start_idx + self.L]
        mpath = self.market_path(market, start_idx, seed)
        tpaths = self.ticker_paths(mpath, tickers, seed)
        return self.to_ohlcv(tpaths, mpath, tickers, dates, max(start_idx - 1, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_scenarios", type=int, default=10)
    ap.add_argument("--ticker_dir", required=True)
    ap.add_argument("--ticker_fine", default=None)
    ap.add_argument("--market_dir", default=None)
    ap.add_argument("--market_fine", default=None)
    ap.add_argument("--ticker_fine_sigma", type=float, default=0.3,
                    help="σ below which the ε ticker model takes over (best in eval: 0.3)")
    ap.add_argument("--market_fine_sigma", type=float, default=0.3)
    ap.add_argument("--switch", type=float, default=0.5, help="Heun→SDE switch σ (hybrid sampler)")
    ap.add_argument("--market", choices=["model", "real"], default="model")
    ap.add_argument("--calendar_start", default="2010-03-30",
                    help="scenario calendars are drawn from real train dates ≥ this (FRED T-bill coverage)")
    ap.add_argument("--tickers", default=None, help="comma list; default: all 99 env tickers")
    ap.add_argument("--ohlcv", default=os.path.join(ROOT, "data/nas100/tiingo_ohlcv.csv.gz"))
    ap.add_argument("--vix", default=os.path.join(ROOT, "data/nas100/vix.csv"))
    ap.add_argument("--end_date", default="2024-05-31")
    ap.add_argument("--no_filter", action="store_true", help="disable failed-sample rejection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = PanelData(args.ohlcv, args.vix, end_date=args.end_date)
    thr = None
    if not args.no_filter:
        c = P.c_ret[1:]
        thr_list = []
        for j in range(c.shape[1]):
            x = c[:, j][np.isfinite(c[:, j])]
            for s in range(0, len(x) - 512, 128):
                w = x[s:s + 511]
                w = w - w.mean()
                thr_list.append((w[1:] * w[:-1]).mean() / max(w.var(), 1e-20))
        thr = float(np.percentile(thr_list, 0.1))
    tm = load_model(args.ticker_dir, args.ticker_fine, switch=args.switch,
                    fine_sigma=args.ticker_fine_sigma, reject_ac1_below=thr)
    mm = (load_model(args.market_dir, args.market_fine, switch=args.switch,
                     fine_sigma=args.market_fine_sigma) if args.market == "model" else None)
    gen = PanelGenerator(P, tm, mm, seed=args.seed)
    tickers = args.tickers.split(",") if args.tickers else list(P.tickers)
    cal_ok = np.where(P.dates >= args.calendar_start)[0]
    starts = np.random.default_rng(args.seed).choice(
        cal_ok[cal_ok <= len(P.dates) - gen.L], args.n_scenarios, replace=False)
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for k, s in enumerate(sorted(starts)):
        df, mk = gen.scenario(tickers, int(s), market=args.market, seed=args.seed + 7919 * k)
        f = os.path.join(args.out, f"scenario_{k:03d}.csv.gz")
        df.to_csv(f, index=False)
        mk.to_csv(os.path.join(args.out, f"scenario_{k:03d}_market.csv"), index=False)
        manifest.append({"file": os.path.basename(f), "calendar_start": str(P.dates[s]),
                         "calendar_end": str(P.dates[s + gen.L - 1]), "market": args.market,
                         "rejection_rate": getattr(tm, "last_rejection_rate", 0.0)})
        print(f"[{k + 1}/{args.n_scenarios}] {f}: {df.tic.nunique()} tickers × {df.date.nunique()} days "
              f"({manifest[-1]['calendar_start']} → {manifest[-1]['calendar_end']})", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"args": vars(args), "scenarios": manifest}, f, indent=2)


if __name__ == "__main__":
    main()
