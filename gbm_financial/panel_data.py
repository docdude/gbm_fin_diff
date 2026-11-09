"""Multichannel panel data for market-conditioned synthetic OHLCV generation.

Per ticker, aligned on the trading calendar:
    c0  anchored log-price      cumsum of vol-normalised log returns (0 at window start)
    c1  log range               log(log(H/L)), centred/scaled per ticker
    c2  relative log volume     log V − trailing 252-day median of log V (causal), scaled
Market (conditioning / market-model targets):
    m0  anchored market log-price  equal-weight mean of available tickers' returns
    m1  log VIX                    centred/scaled

Every channel is scaled so its day-to-day change has std SCALE (=0.017, the
S&P daily-return scale the noise schedule was calibrated on); per-ticker
statistics are stored so generated channels can be mapped back to OHLCV.

Only data up to ``end_date`` is used (default: last day of the MACE env train
split), so the generator never sees the env's trade period.
"""

import warnings

import numpy as np
import pandas as pd

SCALE = 0.017
TICKER_CHANNELS = ("logprice", "logrange", "logvol")
MARKET_CHANNELS = ("market_logprice", "log_vix")
RANGE_FLOOR = 2e-3        # floor for log(H/L) (degenerate H==L days in old data)
VOL_BASELINE_DAYS = 252
VOL_BASELINE_MIN = 60


def _repair_spikes(close, spike=0.4, tol=0.1):
    """Repair one-day spike-and-revert bad ticks; returns (close, repaired_mask)."""
    c = close.copy()
    r = np.diff(np.log(c))
    fixed = np.zeros(len(c), dtype=bool)
    for i in np.where(np.abs(r[:-1]) > spike)[0]:
        if np.isfinite(r[i + 1]) and abs(r[i] + r[i + 1]) < tol * abs(r[i]):
            fixed[i + 1] = True
            r[i + 1] += r[i]
            r[i] = 0.0
    if fixed.any():
        c[1:] = c[0] * np.exp(np.cumsum(r))
    return c, fixed


class PanelData:
    """Wide (date × ticker) OHLCV arrays + derived channels and normalisation stats."""

    def __init__(self, ohlcv_path, vix_path, end_date="2024-05-31", start_date=None):
        df = pd.read_csv(ohlcv_path)
        df = df[df.date <= end_date]
        if start_date:
            df = df[df.date >= start_date]
        vix = pd.read_csv(vix_path).set_index("date")["vix"]
        vix = vix[vix.index <= end_date]

        self.dates = np.array(sorted(set(df.date) & set(vix.index)))
        self.tickers = sorted(df.tic.unique())
        wide = {k: df.pivot(index="date", columns="tic", values=k).reindex(self.dates)[self.tickers]
                for k in ("open", "high", "low", "close", "volume")}
        T, N = len(self.dates), len(self.tickers)

        close = wide["close"].to_numpy(float)
        high = wide["high"].to_numpy(float)
        low = wide["low"].to_numpy(float)
        open_ = wide["open"].to_numpy(float)
        vol = wide["volume"].to_numpy(float)
        self.n_repaired = 0
        for j in range(N):
            ok = np.isfinite(close[:, j]) & (close[:, j] > 0)
            idx = np.where(ok)[0]
            if len(idx) < 3:
                continue
            c_new, fixed = _repair_spikes(close[idx, j])
            if fixed.any():
                ratio = c_new / close[idx, j]
                for a in (high, low, open_):
                    a[idx, j] *= ratio
                close[idx, j] = c_new
                self.n_repaired += int(fixed.sum())
        self.close, self.high, self.low, self.open, self.volume = close, high, low, open_, vol

        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.full((T, N), np.nan)
            ret[1:] = np.log(close[1:] / close[:-1])
            rng = np.log(np.maximum(np.log(high / low), RANGE_FLOOR))
            logv = np.log(np.where(vol > 0, vol, np.nan))
        base = (pd.DataFrame(logv).shift(1)
                .rolling(VOL_BASELINE_DAYS, min_periods=VOL_BASELINE_MIN).median().to_numpy())
        lvrel = logv - base
        self.ret, self.lrange, self.lvol_base, self.lvrel = ret, rng, base, lvrel

        # Per-ticker normalisation statistics (training data only)
        def dstd(x):
            return np.nanstd(np.diff(x, axis=0), axis=0)
        self.stats = {
            "ret_std": np.nanstd(ret, axis=0),
            "lrange_mean": np.nanmean(rng, axis=0),
            "lrange_dstd": dstd(rng),
            "lvol_mean": np.nanmean(lvrel, axis=0),
            "lvol_dstd": dstd(lvrel),
        }

        # Equal-weight market of available tickers + VIX
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mret = np.nanmean(ret, axis=1)
        self.mret[0] = 0.0
        lv = np.log(vix.reindex(self.dates).to_numpy(float))
        self.vix = np.exp(lv)
        self.mstats = {
            "mret_std": float(np.nanstd(self.mret[1:])),
            "lvix_mean": float(np.nanmean(lv)),
            "lvix_dstd": float(np.nanstd(np.diff(lv))),
        }
        self.log_vix = lv

        # Scaled channels (T, N) / (T,)
        s = self.stats
        self.c_ret = ret / s["ret_std"] * SCALE
        self.c_range = (rng - s["lrange_mean"]) / s["lrange_dstd"] * SCALE
        self.c_vol = (lvrel - s["lvol_mean"]) / s["lvol_dstd"] * SCALE
        self.m_ret = self.mret / self.mstats["mret_std"] * SCALE
        self.m_vix = (lv - self.mstats["lvix_mean"]) / self.mstats["lvix_dstd"] * SCALE

    # ------------------------------------------------------------------
    def ticker_valid(self, j):
        """Boolean (T,) mask of days where every ticker channel is defined."""
        return (np.isfinite(self.c_ret[:, j]) & np.isfinite(self.c_range[:, j])
                & np.isfinite(self.c_vol[:, j]))

    def market_window(self, start, L, mret_scaled=None):
        m = self.m_ret if mret_scaled is None else mret_scaled
        r = m[start:start + L].copy()
        r[0] = 0.0
        return np.stack([np.cumsum(r), self.m_vix[start:start + L]])

    def ticker_windows(self, tickers, L=512, stride=64):
        """(N, 5, L) windows [c0, c1, c2, m0, m1] + metadata for the given tickers."""
        out, meta = [], []
        for tic in tickers:
            j = self.tickers.index(tic)
            ok = self.ticker_valid(j)
            T = len(self.dates)
            starts = range(0, T - L + 1, stride)
            for s in starts:
                if not ok[s + 1:s + L].all() or not np.isfinite(self.c_range[s, j]) \
                        or not np.isfinite(self.c_vol[s, j]):
                    continue
                r = self.c_ret[s:s + L, j].copy()
                r[0] = 0.0
                tw = np.stack([np.cumsum(r), self.c_range[s:s + L, j], self.c_vol[s:s + L, j]])
                out.append(np.concatenate([tw, self.market_window(s, L)]))
                meta.append((tic, self.dates[s]))
        X = np.asarray(out, dtype=np.float32)
        return X, meta

    def market_windows(self, L=512, stride=16, n_baskets=1, basket_min=20, seed=0):
        """(N, 2, L) market windows; n_baskets>1 adds random equal-weight baskets
        (each rescaled to the market's daily scale) as data augmentation."""
        rng = np.random.default_rng(seed)
        T, N = self.ret.shape
        out, meta = [], []
        for s in range(0, T - L + 1, stride):
            if not np.isfinite(self.m_vix[s:s + L]).all():
                continue
            out.append(self.market_window(s, L))
            meta.append(("EW", self.dates[s]))
            avail = np.where(np.isfinite(self.ret[s + 1:s + L]).all(axis=0))[0]
            for b in range(n_baskets - 1):
                if len(avail) < basket_min:
                    break
                k = rng.integers(basket_min, len(avail) + 1)
                pick = rng.choice(avail, k, replace=False)
                br = self.ret[s:s + L, pick].mean(axis=1)
                br = br / np.nanstd(br[1:]) * SCALE
                out.append(self.market_window(s, L, mret_scaled=np.r_[np.zeros(s), br,
                                                                       np.zeros(T - s - L)]))
                meta.append((f"basket{b}", self.dates[s]))
        return np.asarray(out, dtype=np.float32), meta

    def summary(self):
        return {"dates": (self.dates[0], self.dates[-1], len(self.dates)),
                "tickers": len(self.tickers), "repaired_ticks": self.n_repaired,
                "median_ret_std": float(np.nanmedian(self.stats["ret_std"])),
                "market_ret_std": self.mstats["mret_std"]}
