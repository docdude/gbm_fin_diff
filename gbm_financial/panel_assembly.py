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
import glob
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


def market_path_features(X, mstats):
    """Regime features of market windows X (n, 2, L) [market log-price, log VIX]."""
    r = np.diff(X[:, 0], axis=1) / SCALE * mstats["mret_std"]
    lvix = X[:, 1] / SCALE * mstats["lvix_dstd"] + mstats["lvix_mean"]
    rc = r - r.mean(1, keepdims=True)
    sd = rc.std(1)
    a = np.abs(r) - np.abs(r).mean(1, keepdims=True)
    acf_abs = np.mean([(a[:, :-k] * a[:, k:]).mean(1) / (a ** 2).mean(1) for k in range(1, 11)], axis=0)
    cum = np.cumsum(r, axis=1)
    mdd = (cum - np.maximum.accumulate(cum, axis=1)).min(1)
    kurt = (rc ** 4).mean(1) / sd ** 4 - 3
    skew = (rc ** 3).mean(1) / sd ** 3
    return np.column_stack([np.log(sd), np.arcsinh(kurt), skew, acf_abs, lvix.mean(1), lvix.std(1),
                            mdd, np.log(np.exp(lvix).mean(1) / (sd * np.sqrt(252) * 100))])


class EraSelector:
    """Rejection sampler restricting generated market paths to a target era.

    The market model learns the full-history regime mixture (e.g. dot-com era:
    high vol, high beta dispersion, low correlation). A logistic classifier on
    regime features of REAL windows, labelled by start date ≥ era_start, gives
    p(era | x); since the model's paths approximate the full-history mixture,
    accepting a path with probability p(era | x) yields the target era's
    regime distribution (density ratio q_era/q_all = p(era | x) / π_era).

    Only regime-level features (log vol, VIX level and variability) are used:
    tail-shape features (kurtosis, |r| clustering) differ by era too, but the model
    under-produces them in every era, so selecting on them over-selects calm paths."""

    FEATURES = [0, 4, 5]

    def __init__(self, P, L, era_start, stride=8):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        starts = [s for s in range(0, len(P.dates) - L + 1, stride)
                  if np.isfinite(P.m_vix[s:s + L]).all()]
        X = np.stack([P.market_window(s, L) for s in starts])
        y = P.dates[starts] >= era_start
        self.mstats = P.mstats
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        self.clf.fit(market_path_features(X, P.mstats)[:, self.FEATURES], y)
        self.prior = float(y.mean())

    def accept_prob(self, X):
        return self.clf.predict_proba(market_path_features(X, self.mstats)[:, self.FEATURES])[:, 1]


EWMA_LAMBDA = 0.94

# Regime labels on the traded episode (the env drops ~59 indicator warm-up days)
EPISODE_DAYS = 453
CRASH_20D, BEAR_ANN, BULL_ANN, STATE_63D = -0.20, -0.10, 0.15, 0.05
REGIMES = ("bull", "sideways", "bear", "crash")


def regime_stats(index, vix=None):
    """Regime label + summary of a price index path, in order: bear (annualised
    ≤ −10%), crash (worst 20-day return ≤ −20%, i.e. a sharp drop that does not
    turn into a bear), bull (annualised ≥ +15%), else sideways. 1995→2022 real
    NAS-100 windows: bull 45%, sideways 21%, bear 10%, crash 24%."""
    r = np.diff(np.log(np.asarray(index, float)))
    n = len(r)
    c0 = np.r_[0.0, np.cumsum(r)]
    mdd = float(np.exp((c0 - np.maximum.accumulate(c0)).min()) - 1)
    w20 = float(np.exp((c0[20:] - c0[:-20]).min()) - 1) if n > 21 else np.nan
    ann = float(np.exp(c0[-1] * 252 / n) - 1)
    lab = ("bear" if ann <= BEAR_ANN else "crash" if w20 <= CRASH_20D
           else "bull" if ann >= BULL_ANN else "sideways")
    r63 = np.exp(c0[63:] - c0[:-63]) - 1 if n > 63 else np.array([0.0])
    return {"label": lab, "ann_return": ann, "ann_vol": float(r.std() * np.sqrt(252)),
            "max_drawdown": mdd, "worst_20d": w20,
            "vix_mean": float(np.nanmean(vix)) if vix is not None else np.nan,
            "days_bull": float((r63 > STATE_63D).mean()), "days_bear": float((r63 < -STATE_63D).mean()),
            "days_sideways": float((np.abs(r63) <= STATE_63D).mean())}


def market_regime(path, mstats, episode_days=EPISODE_DAYS):
    """Regime label of a (2, L) market channel path over its last episode_days."""
    r = np.diff(path[0, -episode_days:]) / SCALE * mstats["mret_std"]
    return regime_stats(np.exp(np.r_[0.0, np.cumsum(r)]))["label"]


def parse_regime_mix(spec, n):
    """'balanced' or 'bull=0.4,sideways=0.3,...' → integer quotas summing to n."""
    if spec == "balanced":
        w = {k: 1.0 for k in REGIMES}
    else:
        w = {k: float(v) for k, v in (kv.split("=") for kv in spec.split(","))}
        bad = set(w) - set(REGIMES)
        if bad:
            raise ValueError(f"unknown regimes {bad}; use {REGIMES}")
    tot = sum(w.values())
    raw = {k: n * v / tot for k, v in w.items()}
    q = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - q[k], reverse=True)[:n - sum(q.values())]:
        q[k] += 1
    return q


class CalendarMatcher:
    """Choose a real calendar start whose market state matches a generated market path.

    Ticker channels are generated in units of each ticker's trailing vol at the
    calendar start, and the ticker model learned them alongside real market paths
    that start from that state. Pairing a calm path with a turbulent date (or vice
    versa) makes tickers move 0.4-2x the market, so the start is drawn from the k
    real dates nearest in (start log VIX, log vol of first 63 days, log vol of window)."""

    def __init__(self, P, L, cal_starts, k=5):
        self.mstats, self.k = P.mstats, k
        self.starts = np.asarray(cal_starts)
        F = self._features(np.stack([P.market_window(s, L) for s in self.starts]))
        self.mu, self.sd = F.mean(0), F.std(0)
        self.F = (F - self.mu) / self.sd

    def _features(self, X):
        r = np.diff(X[:, 0], axis=1) / SCALE * self.mstats["mret_std"]
        lvix0 = X[:, 1, :5].mean(1) / SCALE * self.mstats["lvix_dstd"]
        return np.column_stack([lvix0, np.log(r[:, :63].std(1)), np.log(r.std(1))])

    def match(self, path, rng):
        f = (self._features(path[None])[0] - self.mu) / self.sd
        d = ((self.F - f) ** 2).sum(1)
        return int(self.starts[rng.choice(np.argsort(d)[:self.k])])


def devol(x, lam=EWMA_LAMBDA):
    """Causal EWMA de-volatilisation of zero-mean shocks x (n, T) → (u, sig)."""
    v = np.empty_like(x)
    prev = (x ** 2).mean(-1)
    for t in range(x.shape[-1]):
        v[:, t] = prev
        prev = lam * prev + (1 - lam) * x[:, t] ** 2
    sig = np.sqrt(np.maximum(v, 1e-20))
    return x / sig, sig


def decompose(dc, dm):
    """dc (n, T) ticker increments, dm (T,) market increments → mean, beta, residual."""
    dmc = dm - dm.mean()
    mu = dc.mean(1, keepdims=True)
    b = (dc - mu) @ dmc / (dmc ** 2).sum()
    return mu, b, dc - mu - b[:, None] * dmc


def quantile_map(u, ref_sorted):
    """Map u to ref's distribution through u's own pooled empirical CDF."""
    flat = u.ravel()
    q = (np.argsort(np.argsort(flat)) + 0.5) / flat.size
    grid = (np.arange(ref_sorted.size) + 0.5) / ref_sorted.size
    return np.interp(q, grid, ref_sorted).reshape(u.shape)


class TailCalibrator:
    """Filtered quantile mapping of generated shocks onto real (target-era) shocks.

    Diffusion samples under-produce rare large moves (≈½ the real rate of >6σ days).
    Returns are split into mean + beta·market + residual, the residual (or the
    market return itself) is de-volatilised with a causal EWMA, and the shocks u are
    quantile-mapped to the pooled real distribution of the same statistic computed
    with the identical pipeline. Timing, volatility clustering, betas and
    correlations are kept; only the conditional tail shape changes. Log-range and
    log-volume channels move with |u| using slopes fitted on real data."""

    def __init__(self, P, L, era_start, stride=128, corr_shrink=0.1):
        T = len(P.dates)
        e0 = int(np.searchsorted(P.dates, era_start))
        starts = [s for s in range(e0, T - L + 1, stride)
                  if np.isfinite(P.m_vix[s:s + L]).all()]
        um, ut, du_lr, du_lv = [], [], [], []
        N = len(P.tickers)
        S, n_pair = np.zeros((N, N)), np.zeros((N, N))
        for s in starts:
            dm = np.diff(P.market_window(s, L)[0])
            um.append(devol((dm - dm.mean())[None])[0][0])
            U = np.zeros((N, L - 1))
            ok = np.zeros(N, dtype=bool)
            for j in range(N):
                if not P.ticker_valid(j)[s + 1:s + L].all() or not (P.volume[s:s + L, j] > 0).all():
                    continue
                st = P.window_stats(j, s, fallback=False)
                if st is None:
                    continue
                c = P.ticker_channels(j, s, L, st)
                if not np.isfinite(c).all():
                    continue
                u = devol(decompose(np.diff(c[0])[None], dm)[2])[0][0]
                ut.append(u)
                du_lr.append(np.c_[np.abs(u), c[1, 1:]])
                du_lv.append(np.c_[np.abs(u), c[2, 1:]])
                U[j], ok[j] = u, True
            if (s - e0) % (L - 1) < stride:      # ~non-overlapping windows for the correlation
                S += U @ U.T
                n_pair += np.outer(ok, ok) * (L - 1)
        self.ref_m = np.sort(np.concatenate(um))
        self.ref_t = np.sort(np.concatenate(ut))
        self.k_range = self._slope(np.concatenate(du_lr))
        self.k_vol = self._slope(np.concatenate(du_lv))
        self.market_ref = None
        self.res_corr = self._corr(S, n_pair, corr_shrink)

    @staticmethod
    def _corr(S, n, shrink):
        """Residual-shock correlation of real tickers (sector structure), shrunk to I."""
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = np.where(n > 0, S / n, 0.0)
            d = np.sqrt(np.diag(cov))
            C = np.where(np.outer(d, d) > 0, cov / np.outer(d, d), 0.0)
        C = (1 - shrink) * C + shrink * np.eye(len(C))
        w, V = np.linalg.eigh(C)
        C = (V * np.maximum(w, 1e-3)) @ V.T
        d = np.sqrt(np.diag(C))
        return C / np.outer(d, d)

    @staticmethod
    def _slope(a):
        x = a[:, 0] - a[:, 0].mean()
        return float((x * (a[:, 1] - a[:, 1].mean())).sum() / (x ** 2).sum())

    def fit_market_bank(self, bank):
        """Pooled generated market shocks (CDF source for market mapping)."""
        dm = np.diff(bank[:, 0], axis=1)
        self.market_ref = devol(dm - dm.mean(1, keepdims=True))[0]

    def market(self, path):
        dm = np.diff(path[0])
        mu = dm.mean()
        u, sig = devol((dm - mu)[None])
        pool = np.concatenate([u, self.market_ref]) if self.market_ref is not None else u
        u2 = quantile_map(pool, self.ref_m)[:1]
        out = path.copy()
        out[0] = path[0, 0] + np.r_[0.0, np.cumsum(mu + u2[0] * sig[0])]
        return out

    def tickers(self, paths, market, idx=None):
        """paths (n, 3, L) generated [c0, c1, c2]; market (2, L) conditioning path;
        idx: indices of the n tickers in P.tickers (enables residual correlation)."""
        dc = np.diff(paths[:, 0], axis=1)
        dm = np.diff(market[0])
        mu, b, e = decompose(dc, dm)
        u, sig = devol(e)
        if idx is not None and getattr(self, "res_corr", None) is not None:
            # Gaussian copula: mix normal scores so sector peers co-move without
            # copying each other's idiosyncratic jumps; tails restored by the map below
            from scipy.stats import norm
            flat = u.ravel()
            z = norm.ppf((np.argsort(np.argsort(flat)) + 0.5) / flat.size).reshape(u.shape)
            w, V = np.linalg.eigh(self.res_corr[np.ix_(idx, idx)])
            u_mixed = (V * np.sqrt(np.maximum(w, 0))) @ V.T @ z
        else:
            u_mixed = u
        u2 = quantile_map(u_mixed, self.ref_t)
        out = paths.copy()
        out[:, 0, 1:] = paths[:, 0, :1] + np.cumsum(mu + b[:, None] * (dm - dm.mean()) + u2 * sig, axis=1)
        d = np.abs(u2) - np.abs(u)
        out[:, 1, 1:] += self.k_range * d
        out[:, 2, 1:] += self.k_vol * d
        return out


class PanelGenerator:

    def __init__(self, P, ticker_model, market_model=None, seed=0, era=None, tails=None, seq_len=None):
        self.P = P
        self.tm = ticker_model
        self.mm = market_model
        self.L = int(ticker_model.config["seq_len"] if ticker_model is not None else seq_len)
        self.emp = EmpiricalOHLC(P)
        self.rng = np.random.default_rng(seed)
        self.era = era
        self.tails = tails
        self.market_identity = True
        self.last_market_accept = None

    # -- market -------------------------------------------------------------
    def market_path(self, mode="model", start_idx=None, seed=None, batch=8, max_rounds=50):
        if mode == "real":
            return self.P.market_window(start_idx, self.L)
        if self.mm is None:
            raise ValueError("market='model' needs a market model")
        if self.era is None:
            return self.mm.sample_panel(1, self.L, cond=None, seed=seed)[0]
        rng = np.random.default_rng(seed)
        for k in range(max_rounds):
            G = self.mm.sample_panel(batch, self.L, cond=None,
                                     seed=None if seed is None else seed + 104729 * k)
            p = self.era.accept_prob(G)
            ok = np.where(rng.random(batch) < p)[0]
            if len(ok):
                self.last_market_accept = float(p[ok[0]])
                return G[ok[0]]
        raise RuntimeError("era selection accepted no market path")

    def market_bank_by_regime(self, quotas, seed=0, stress=("bear", "crash"),
                              batch=8, max_candidates=4000):
        """Sample market paths until each regime quota is filled.

        Paths are labelled on the traded episode (regime_stats rules). Era selection
        still applies, except for regimes in ``stress`` (bear/crash paths may come from
        the model's full 1995+ history, since 2010+ windows contain almost none).
        Returns (paths, labels, era_probs)."""
        need = {k: v for k, v in quotas.items() if v > 0}
        got = {k: 0 for k in need}
        rng = np.random.default_rng(seed + 17)
        paths, labels, probs, seen = [], [], [], {k: 0 for k in REGIMES}
        k = 0
        while any(got[r] < need[r] for r in need):
            if k * batch >= max_candidates:
                raise RuntimeError(f"regime quotas not filled after {k * batch} candidates: "
                                   f"got {got}, need {need}, seen {seen}")
            G = self.mm.sample_panel(batch, self.L, cond=None, seed=seed + 104729 * k)
            p = self.era.accept_prob(G) if self.era is not None else np.ones(batch)
            for g, pi in zip(G, p):
                lab = market_regime(g, self.P.mstats)
                seen[lab] += 1
                if got.get(lab, 0) >= need.get(lab, 0):
                    continue
                if lab not in stress and rng.random() >= pi:
                    continue
                got[lab] += 1
                paths.append(g)
                labels.append(lab)
                probs.append(float(pi))
            k += 1
        print(f"regime bank: {k * batch} candidates, labels seen {seen}, kept {got}", flush=True)
        return np.stack(paths), labels, probs

    # -- tickers ------------------------------------------------------------
    def ticker_paths(self, market, tickers, stats, seed=None):
        n = len(tickers)
        K = int(self.tm.config["n_channels"])
        cond = np.zeros((n, K, self.L), dtype=np.float32)
        cond[:, 3:5] = market[None]
        if K == 7:
            for i, st in enumerate(stats):
                cond[i, 5:] = self.P.state_channels(st, self.L)
        return self.tm.sample_panel(n, self.L, cond=cond, seed=seed, chunk=n)[:, :3]

    # -- inversion ----------------------------------------------------------
    def _anchor_close(self, j, ref_idx):
        c = self.P.close[:ref_idx + 1, j]
        ok = np.where(np.isfinite(c))[0]
        k = ok[-1] if len(ok) else np.where(np.isfinite(self.P.close[:, j]))[0][0]
        return self.P.close[k, j]

    def stats_for(self, tickers, s):
        return [self.P.window_stats(self.P.tickers.index(t), s) for t in tickers]

    def enforce_market_identity(self, paths, market, stats):
        """The market channel IS the equal-weight mean of these tickers' returns, so
        shift every ticker's daily return by (market − EW of generated tickers).
        A common shift: dispersion and residual structure are unchanged."""
        vol = np.array([st["vol"] for st in stats])[:, None]
        r = np.diff(paths[:, 0], axis=1) / SCALE * vol
        m = np.diff(market[0]) / SCALE * self.P.mstats["mret_std"]
        r = r + (m - r.mean(0))[None]
        out = paths.copy()
        out[:, 0, 1:] = paths[:, 0, :1] + np.cumsum(r / vol * SCALE, axis=1)
        return out

    def to_ohlcv(self, paths, market, tickers, dates, ref_idx, stats=None):
        """Invert channels to OHLCV. stats: per-ticker window_stats at the window
        start (default: computed at ref_idx + 1)."""
        P, S = self.P, self.P.stats
        if stats is None:
            stats = self.stats_for(tickers, ref_idx + 1)
        rows = []
        for i, tic in enumerate(tickers):
            j = P.tickers.index(tic)
            st = stats[i]
            c0, lr, lv = paths[i]
            ret = np.diff(c0) / SCALE * st["vol"]
            c_start = self._anchor_close(j, ref_idx)
            close = c_start * np.exp(np.r_[0.0, np.cumsum(ret)])
            hl = np.exp(lr / SCALE * S["lrange_dstd"][j] + st["rmean"])
            z = np.r_[0.0, ret / st["vol"]]
            a, g = self.emp.sample(z, self.rng)
            high = close * np.exp(a * hl)
            low = high * np.exp(-hl)
            prev = np.r_[close[0], close[:-1]]
            open_ = np.clip(prev * np.exp(g * st["vol"]), low, high)
            volume = np.exp(lv / SCALE * S["lvol_dstd"][j] + st["vbase"])
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

    def scenario(self, tickers, start_idx, market="model", seed=None, mpath=None):
        """One scenario; mpath: pre-generated (2, L) market path (overrides ``market``)."""
        dates = self.P.dates[start_idx:start_idx + self.L]
        if mpath is None:
            mpath = self.market_path(market, start_idx, seed)
        mraw = mpath
        if self.tails is not None and market == "model":
            mpath = self.tails.market(mpath)
        stats = self.stats_for(tickers, start_idx)
        tpaths = self.ticker_paths(mpath, tickers, stats, seed)
        self.last_raw = {"market_raw": mraw, "market": mpath, "tickers_raw": tpaths}
        if self.tails is not None:
            tpaths = self.tails.tickers(tpaths, mpath, [self.P.tickers.index(t) for t in tickers])
        if self.market_identity:
            tpaths = self.enforce_market_identity(tpaths, mpath, stats)
        return self.to_ohlcv(tpaths, mpath, tickers, dates, max(start_idx - 1, 0), stats)


def reassemble(args, P):
    """Rebuild scenarios from saved raw model outputs (``scenario_*_raw.npz``) with the
    current ticker-side calibration; no sampling. Tickers stay conditioned on the
    saved (already calibrated) market path."""
    files = sorted(glob.glob(os.path.join(args.from_raw, "scenario_*_raw.npz")))
    if not files:
        raise FileNotFoundError(f"no scenario_*_raw.npz in {args.from_raw}")
    L = int(np.load(files[0])["market"].shape[-1])
    gen = PanelGenerator(P, None, seed=args.seed, seq_len=L)
    if args.tail_era.lower() != "none":
        gen.tails = TailCalibrator(P, L, args.tail_era)
        if args.no_sector_corr:
            gen.tails.res_corr = None
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for k, f in enumerate(files):
        z = np.load(f)
        s, tickers, mpath, tp = int(z["start_idx"]), list(z["tickers"]), z["market"], z["tickers_raw"]
        if gen.tails is not None:
            tp = gen.tails.tickers(tp, mpath, [P.tickers.index(t) for t in tickers])
        stats = gen.stats_for(tickers, s)
        if not args.no_market_identity:
            tp = gen.enforce_market_identity(tp, mpath, stats)
        df, mk = gen.to_ohlcv(tp, mpath, tickers, P.dates[s:s + L], max(s - 1, 0), stats)
        out = os.path.join(args.out, f"scenario_{k:03d}.csv.gz")
        df.to_csv(out, index=False)
        mk.to_csv(os.path.join(args.out, f"scenario_{k:03d}_market.csv"), index=False)
        np.savez_compressed(os.path.join(args.out, f"scenario_{k:03d}_raw.npz"),
                            **{key: z[key] for key in z.files})
        manifest.append({"file": os.path.basename(out), "source": f, "calendar_start": str(P.dates[s]),
                         "calendar_end": str(P.dates[s + L - 1])})
        print(f"[{k + 1}/{len(files)}] {out}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"args": vars(args), "scenarios": manifest}, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_scenarios", type=int, default=10)
    ap.add_argument("--ticker_dir", default=None, help="required unless --from_raw")
    ap.add_argument("--ticker_fine", default=None)
    ap.add_argument("--market_dir", default=None)
    ap.add_argument("--market_fine", default=None)
    ap.add_argument("--ticker_fine_sigma", type=float, default=0.1,
                    help="σ below which the ε ticker model takes over (0.1: vol ratio 1.00 on real markets)")
    ap.add_argument("--market_fine_sigma", type=float, default=0.03,
                    help="0.03: best |r| clustering / kurtosis in the market sampler sweep")
    ap.add_argument("--switch", type=float, default=0.5, help="Heun→SDE switch σ (hybrid sampler)")
    ap.add_argument("--market", choices=["model", "real"], default="model")
    ap.add_argument("--market_era", default="2010-03-30",
                    help="keep generated market paths whose regime matches real windows starting on/after "
                         "this date (rejection sampling); 'none' = full 1995+ regime mixture")
    ap.add_argument("--tail_era", default="2010-03-30",
                    help="filtered tail calibration of market/ticker shocks to real windows starting "
                         "on/after this date; 'none' disables")
    ap.add_argument("--calendar_start", default="2010-03-30",
                    help="scenario calendars are drawn from real train dates ≥ this (FRED T-bill coverage)")
    ap.add_argument("--random_calendar", action="store_true",
                    help="draw calendar starts at random instead of matching each market path's state")
    ap.add_argument("--no_sector_corr", action="store_true",
                    help="skip injecting the real residual (sector) correlation into ticker shocks")
    ap.add_argument("--no_market_identity", action="store_true",
                    help="skip forcing the tickers' equal-weight return to equal the market path")
    ap.add_argument("--from_raw", default=None,
                    help="re-assemble scenarios from DIR/scenario_*_raw.npz (no sampling, no models)")
    ap.add_argument("--regime_mix", default=None,
                    help="'balanced' or e.g. 'bull=0.35,sideways=0.35,bear=0.15,crash=0.15': fill "
                         "per-regime quotas of market paths (labels on the traded episode)")
    ap.add_argument("--stress_full_history", action="store_true",
                    help="with --regime_mix: bear/crash paths bypass the era selection")
    ap.add_argument("--max_candidates", type=int, default=4000,
                    help="with --regime_mix: market paths to try before giving up")
    ap.add_argument("--tickers", default=None, help="comma list; default: all 99 env tickers")
    ap.add_argument("--ohlcv", default=os.path.join(ROOT, "data/nas100/tiingo_ohlcv.csv.gz"))
    ap.add_argument("--vix", default=os.path.join(ROOT, "data/nas100/vix.csv"))
    ap.add_argument("--end_date", default="2024-05-31")
    ap.add_argument("--no_filter", action="store_true", help="disable failed-sample rejection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = PanelData(args.ohlcv, args.vix, end_date=args.end_date)
    if args.from_raw:
        return reassemble(args, P)
    if not args.ticker_dir:
        ap.error("--ticker_dir is required unless --from_raw is given")
    thr = None
    if not args.no_filter:
        c = P.ret[1:]
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
    era = None
    if args.market == "model" and args.market_era.lower() != "none":
        era = EraSelector(P, tm.config["seq_len"], args.market_era)
        print(f"market era selection: windows ≥ {args.market_era} (prior {era.prior:.2f})", flush=True)
    gen = PanelGenerator(P, tm, mm, seed=args.seed, era=era)
    gen.market_identity = not args.no_market_identity
    if args.tail_era.lower() != "none":
        gen.tails = TailCalibrator(P, gen.L, args.tail_era)
        if args.no_sector_corr:
            gen.tails.res_corr = None
        print(f"tail calibration: real shocks ≥ {args.tail_era} "
              f"(k_range {gen.tails.k_range:.3f}, k_vol {gen.tails.k_vol:.3f})", flush=True)
    tickers = args.tickers.split(",") if args.tickers else list(P.tickers)
    cal_ok = np.where(P.dates >= args.calendar_start)[0]
    cal_ok = cal_ok[cal_ok <= len(P.dates) - gen.L]
    starts = sorted(np.random.default_rng(args.seed).choice(cal_ok, args.n_scenarios, replace=False))
    bank, bank_p, bank_lab = None, [], []
    if args.market == "model":
        if args.regime_mix:
            quotas = parse_regime_mix(args.regime_mix, args.n_scenarios)
            stress = ("bear", "crash") if args.stress_full_history else ()
            bank, bank_lab, bank_p = gen.market_bank_by_regime(
                quotas, seed=args.seed, stress=stress, max_candidates=args.max_candidates)
        else:
            paths = []
            for k in range(args.n_scenarios):
                paths.append(gen.market_path("model", seed=args.seed + 7919 * k))
                bank_p.append(gen.last_market_accept)
            bank = np.stack(paths)
            bank_lab = [market_regime(p, P.mstats) for p in bank]
        if gen.tails is not None:
            gen.tails.fit_market_bank(bank)
        if not args.random_calendar:
            cm = CalendarMatcher(P, gen.L, cal_ok)
            crng = np.random.default_rng(args.seed + 1)
            starts = [cm.match(p, crng) for p in bank]
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for k, s in enumerate(starts):
        df, mk = gen.scenario(tickers, int(s), market=args.market, seed=args.seed + 7919 * k,
                              mpath=None if bank is None else bank[k])
        f = os.path.join(args.out, f"scenario_{k:03d}.csv.gz")
        df.to_csv(f, index=False)
        mk.to_csv(os.path.join(args.out, f"scenario_{k:03d}_market.csv"), index=False)
        np.savez_compressed(os.path.join(args.out, f"scenario_{k:03d}_raw.npz"),
                            start_idx=int(s), tickers=np.array(tickers), **gen.last_raw)
        manifest.append({"file": os.path.basename(f), "calendar_start": str(P.dates[s]),
                         "calendar_end": str(P.dates[s + gen.L - 1]), "market": args.market,
                         "rejection_rate": getattr(tm, "last_rejection_rate", 0.0),
                         "market_era_prob": bank_p[k] if bank_p else None,
                         "market_regime": bank_lab[k] if bank_lab else None})
        print(f"[{k + 1}/{args.n_scenarios}] {f}: {df.tic.nunique()} tickers × {df.date.nunique()} days "
              f"({manifest[-1]['calendar_start']} → {manifest[-1]['calendar_end']})", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"args": vars(args), "scenarios": manifest}, f, indent=2)


if __name__ == "__main__":
    main()
