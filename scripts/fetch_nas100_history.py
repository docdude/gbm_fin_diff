#!/usr/bin/env python3
"""Fetch long daily OHLCV history for the FinRL-Meta NAS-100 universe via Tiingo.

Reuses FinRL-Meta's Tiingo processor (credentials from the repo .env, adjusted
OHLCV). Run with the FinRL-Meta environment, e.g.:

    ~/FinRL-Meta/.venv/bin/python scripts/fetch_nas100_history.py \\
        --finrl_meta ~/FinRL-Meta --end 2024-05-31

Writes one CSV per ticker to data/nas100/tiingo/ (resumable: existing files are
skipped) and a combined data/nas100/tiingo_ohlcv.csv.gz. The default end date is
the last day of the MACE env train split, so no trade-period data is fetched.
"""
import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finrl_meta", default=os.path.expanduser("~/FinRL-Meta"))
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--end", default="2024-05-31", help="inclusive; env train split ends 2024-05-31")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "nas100"))
    ap.add_argument("--dotenv", default=os.path.join(ROOT, ".env"))
    ap.add_argument("--rate_limit_wait", type=int, default=65,
                    help="seconds to wait after HTTP 429 before retrying")
    args = ap.parse_args()

    sys.path.insert(0, args.finrl_meta)
    from finrl.config_tickers import NAS_100_TICKER
    from meta.data_processors.tiingo import Tiingo

    tickers = [t for t in NAS_100_TICKER if t != "WLTW"]  # no provider data (env drops it too)
    tdir = os.path.join(args.out_dir, "tiingo")
    os.makedirs(tdir, exist_ok=True)
    proc = Tiingo("tiingo", args.start, args.end, "1d", DOTENV_PATH=args.dotenv, max_retry=1)

    failed = []
    for i, tic in enumerate(tickers):
        path = os.path.join(tdir, f"{tic}.csv")
        if os.path.exists(path):
            continue
        for attempt in range(6):
            try:
                raw = proc._fetch_ticker(tic)
                break
            except Exception as err:
                status = getattr(getattr(err, "response", None), "status_code", None)
                if status == 429 and attempt < 5:
                    print(f"  rate-limited on {tic}; waiting {args.rate_limit_wait}s", flush=True)
                    time.sleep(args.rate_limit_wait)
                    continue
                print(f"  {tic}: failed ({err})", flush=True)
                raw = None
                break
        if raw is None or raw.empty:
            failed.append(tic)
            continue
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
            "open": raw["adjOpen"], "high": raw["adjHigh"], "low": raw["adjLow"],
            "close": raw["adjClose"], "volume": raw["adjVolume"], "tic": tic,
        }).dropna()
        df.to_csv(path, index=False)
        print(f"[{i + 1}/{len(tickers)}] {tic}: {len(df)} rows {df.date.iloc[0]} → {df.date.iloc[-1]}",
              flush=True)

    frames = [pd.read_csv(os.path.join(tdir, f)) for f in sorted(os.listdir(tdir)) if f.endswith(".csv")]
    allp = pd.concat(frames, ignore_index=True).sort_values(["date", "tic"])
    out = os.path.join(args.out_dir, "tiingo_ohlcv.csv.gz")
    allp.to_csv(out, index=False)
    print(f"Saved {out}: {allp.tic.nunique()} tickers, {len(allp)} rows, "
          f"{allp.date.min()} → {allp.date.max()}; failed: {failed}")


if __name__ == "__main__":
    main()
