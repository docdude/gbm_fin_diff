#!/usr/bin/env python3
"""Convert raw OHLCV panels (synthetic scenarios) into FinRL-Meta MACE/margin env configs.

Runs under the FinRL-Meta environment and applies the env's own pipeline
(MarketDataPreparator._fetch_data post-processing + create_env_config):
TA-Lib indicators, 20-day volatility, ADV20, first-valid-date filter, ffill,
and FRED DTB3 T-bill rates aligned to the dates.

    ~/FinRL-Meta/.venv/bin/python scripts/export_finrl_meta_env.py \\
        --scenarios save/panel/scenarios --finrl_meta ~/FinRL-Meta

Writes <scenario>.env.pkl dicts with keys date_list, price_array, tech_array,
volatility_array, volume_array, adv20_array, tbill_rates, tic_list — the same
structure as MarketDataPreparator.create_env_config(), so they can be passed to
MACEVecEnv(config=...) / MACEStockTradingEnv in place of the real train config.

--check_equivalence rebuilds the env's real TRAIN config through this pipeline
from the env's cached raw OHLCV and compares it with create_env_config().
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_env_config(raw, indicators, cache_dir, fred_api_key=None, rf_ticker="fred:DTB3"):
    from meta.data_processors._base import _Base, DataSource, IndicatorLib
    from meta.data_processors.fred import fetch_fred_series_df

    p = _Base(DataSource.tiingo, raw.date.min(), raw.date.max(), "1d")
    p.dataframe = raw[["date", "open", "high", "low", "close", "adjusted_close", "volume", "tic", "day"]].copy()
    p.add_technical_indicator(indicators, select_stockstats_talib=IndicatorLib.TALIB,
                              drop_na_timesteps=False)
    df = p.dataframe.sort_values(by=["time", "tic"]).reset_index(drop=True).rename(columns={"time": "date"})
    # --- identical to MarketDataPreparator._fetch_data post-processing ---
    df["daily_return"] = df.groupby("tic")["close"].pct_change()
    df["volatility"] = df.groupby("tic")["daily_return"].transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=2).std())
    df["adv20"] = df.groupby("tic")["volume"].transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=1).mean())
    df = df.drop(["index"], axis=1, errors="ignore")
    cols = list(indicators) + ["volatility", "adv20", "daily_return"]
    valid_start = df.dropna(subset=cols)["date"].min()
    df = df[df["date"] >= valid_start].copy()
    df = df.groupby("tic", group_keys=False).apply(lambda x: x.ffill())
    df = df.fillna(0).reset_index(drop=True)
    # --- identical to create_env_config ---
    df = df.sort_values(["date", "tic"], ignore_index=True)
    date_list = df["date"].unique()
    tic_list = df["tic"].unique()
    price, tech, vola, vol, adv = [], [], [], [], []
    for d, g in df.groupby("date", sort=True):
        price.append(g["close"].values)
        tech.append(g[list(indicators)].values.flatten())
        vola.append(g["volatility"].values)
        vol.append(g["volume"].values)
        adv.append(g["adv20"].values)
    rf = fetch_fred_series_df(series_ticker=rf_ticker, market_dates=list(date_list),
                              cache_dir=cache_dir, api_key=fred_api_key)
    rf = rf[rf["date"].isin(date_list)]
    return {"date_list": date_list, "price_array": np.array(price), "tech_array": np.array(tech),
            "volatility_array": np.array(vola), "volume_array": np.array(vol),
            "adv20_array": np.array(adv), "tbill_rates": rf["close"].values, "tic_list": tic_list}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=None, help="directory with scenario_*.csv.gz")
    ap.add_argument("--finrl_meta", default=os.path.expanduser("~/FinRL-Meta"))
    ap.add_argument("--dotenv", default=os.path.join(ROOT, ".env"))
    ap.add_argument("--check_equivalence", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.finrl_meta)
    from meta.config import INDICATORS
    cache_dir = os.path.join(args.finrl_meta, "data")
    fred_key = None
    if os.path.exists(args.dotenv):
        for line in open(args.dotenv):
            if line.startswith("FRED_API_KEY="):
                fred_key = line.split("=", 1)[1].strip().strip('"').strip("'")

    if args.check_equivalence:
        from finrl.config_tickers import NAS_100_TICKER
        from meta.env_market_impact.envs.market_data import Split
        from meta.env_market_impact.vec.data_prep import build_vec_tiingo_market_data_preparator
        prep = build_vec_tiingo_market_data_preparator(
            tickers=NAS_100_TICKER, start_date="2010-01-01", end_date="2026-01-01",
            tech_indicators=INDICATORS, train_ratio=0.9, benchmark_ticker="QQEW")
        ref = prep.create_env_config(Split.TRAIN)
        raw = prep._market_df  # the processed panel the env itself loaded (raw OHLCV columns kept)
        raw = raw[raw.tic.isin(prep.tickers) & (raw.date <= ref["date_list"][-1])].copy()
        raw["day"] = pd.to_datetime(raw["date"]).dt.dayofweek
        mine = build_env_config(raw, INDICATORS, cache_dir, fred_key)
        # skip indicator warm-up (env computed indicators from an earlier start)
        common = np.intersect1d(mine["date_list"], ref["date_list"])[150:]
        i_m = np.searchsorted(mine["date_list"], common)
        i_r = np.searchsorted(ref["date_list"], common)
        for k in ("price_array", "tech_array", "volatility_array", "volume_array", "adv20_array"):
            a, b = mine[k][i_m], ref[k][i_r]
            print(f"{k:<18} max rel err {np.nanmax(np.abs(a - b) / (np.abs(b) + 1e-8)):.2e}  "
                  f"({len(common)} common dates of {len(ref['date_list'])})")
        print("tbill max abs err", np.abs(mine["tbill_rates"][i_m] - ref["tbill_rates"][i_r]).max())
        return

    files = sorted(glob.glob(os.path.join(args.scenarios, "scenario_*[0-9].csv.gz")))
    for f in files:
        raw = pd.read_csv(f)
        cfg = build_env_config(raw, INDICATORS, cache_dir, fred_key)
        out = f.replace(".csv.gz", ".env.pkl")
        with open(out, "wb") as fh:
            pickle.dump(cfg, fh)
        print(f"{os.path.basename(out)}: {cfg['price_array'].shape[0]} steps × "
              f"{cfg['price_array'].shape[1]} tickers, tech {cfg['tech_array'].shape}")


if __name__ == "__main__":
    main()
