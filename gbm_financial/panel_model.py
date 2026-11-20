"""Multichannel (panel) diffusion model: per-ticker OHLCV channels conditioned on
market channels, or an unconditional market model.

PanelDiffusion reuses GBMFinancialDiffusion's SDEs, samplers (PC/EM/ODE/Karras/
hybrid + splice), EMA, checkpointing and training loop; it overrides the loss
for (B, K, L) tensors with conditioning and anchor masks.

Config keys (in addition to the usual ones):
    n_channels       K
    conditional      True if some channels are observed
    cond_channels    list of channel indices that are observed (conditioning)
    anchor_channels  channel indices anchored at 0 at t=0 (log-price channels)
    channel_names    for logging
"""

import numpy as np
import torch

from .sde import _expand
from .train import GBMFinancialDiffusion


class PanelDiffusion(GBMFinancialDiffusion):

    def __init__(self, config, device=None):
        config = dict(config)
        config.setdefault("data_mode", "log_price")
        config.setdefault("normalize_mode", "none")
        config["mask_anchor"] = False  # anchors handled per channel below
        super().__init__(config, device=device)
        K = int(config["n_channels"])
        self.K = K
        cm = torch.zeros(K)
        for c in config.get("cond_channels", []):
            cm[c] = 1.0
        self.cond_mask = cm.to(self.device)
        self.anchor_channels = list(config.get("anchor_channels", [0]))
        self.channel_names = config.get("channel_names", [f"ch{k}" for k in range(K)])
        self._cond_batch = None

    # ------------------------------------------------------------------
    def _prior_shape(self, B, seq_len):
        return (B, self.K, seq_len)

    pin_anchor = True

    def _anchor_index(self, x):
        """Position 0 of generated anchor channels (observed channels are masked
        out of the network input, so they need no pinning)."""
        cm = self.cond_mask.cpu().numpy()
        ch = [c for c in self.anchor_channels if cm[c] == 0]
        return (slice(None), ch, 0) if ch else None

    def _set_cond(self, x0):
        """Condition the network(s) on the observed channels of x0 (B, K, L)."""
        self._cond_batch = x0
        self.model.set_condition(x0, self.cond_mask)
        fine = getattr(self, "_fine", None)
        if fine is not None:
            fine.set_condition(x0, self.cond_mask)

    def _loss_mask(self, B, L):
        m = (1 - self.cond_mask).view(1, -1, 1).expand(B, -1, L).clone()
        for c in self.anchor_channels:
            m[:, c, 0] = 0.0
        return m

    def compute_loss(self, x_0, eps=1e-5, t=None, z=None):
        B, K, L = x_0.shape
        if t is None:
            t = torch.rand(B, device=self.device) * (self.sde.T - eps) + eps
        if z is None:
            z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, t)
        x_t = mean + _expand(std, x_0) * z
        self._set_cond(x_0)
        eps_pred = self.model(x_t, t)
        losses = (z - eps_pred) ** 2
        if self.loss_weighting == "edm":
            sd = self.model.sigma_data
            sd2 = (sd.view(1, -1, 1) if sd.dim() else sd) ** 2
            s2 = std.view(-1, 1, 1) ** 2
            losses = losses * (s2 + sd2) / sd2.clamp(min=1e-12)
        elif self.loss_weighting != "uniform":
            raise ValueError(f"loss_weighting={self.loss_weighting} not supported for panels")
        mask = self._loss_mask(B, L)
        return (losses * mask).sum() / mask.sum()

    def compute_loss_detailed(self, x_0, eps=1e-5):
        B, K, L = x_0.shape
        t = torch.rand(B, device=self.device) * (self.sde.T - eps) + eps
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, t)
        x_t = mean + _expand(std, x_0) * z
        self._set_cond(x_0)
        eps_pred = self.model(x_t, t)
        se = (z - eps_pred) ** 2 * self._loss_mask(B, L)
        per = se.sum(dim=(1, 2)) / self._loss_mask(B, L).sum(dim=(1, 2))
        tt = t.detach().cpu().numpy()
        ps = per.detach().cpu().numpy()
        bands = {"low": tt < 0.33, "mid": (tt >= 0.33) & (tt < 0.67), "high": tt >= 0.67}
        loss_by_t = {k: float(ps[m].mean()) if m.any() else float("nan") for k, m in bands.items()}
        loss_by_pos = {}
        for k in range(K):
            if self.cond_mask[k] == 0:
                loss_by_pos[self.channel_names[k]] = float(se[:, k].mean().detach().cpu())
        tgt = (1 - self.cond_mask).bool()
        with torch.no_grad():
            zf, pf = z[:, tgt].reshape(B, -1), eps_pred[:, tgt].reshape(B, -1)
            cos = torch.nn.functional.cosine_similarity(zf, pf, dim=1)
            score_stats = {"cos_sim": float(cos.mean()), "eps_pred_norm": float(pf.norm(dim=1).mean()),
                           "eps_true_norm": float(zf.norm(dim=1).mean())}
        return {"loss": per.mean(), "loss_by_t": loss_by_t, "loss_by_pos": loss_by_pos,
                "score_stats": score_stats}

    # ------------------------------------------------------------------
    def _fine_net(self):
        fine = super()._fine_net()
        if fine is not None and self._cond_batch is not None:
            fine.set_condition(self._cond_batch, self.cond_mask)
        return fine

    def sample_panel(self, n_samples, seq_len, cond=None, seed=None, chunk=64):
        """Generate (n, K, L) windows.

        cond: (n, K, L) array whose cond channels are used (others ignored), or
        None for an unconditional model. Output cond channels are set to cond.
        """
        if cond is not None:
            cond = np.asarray(cond, dtype=np.float32)
            assert cond.shape[0] == n_samples
        out = []
        for i in range(0, n_samples, chunk):
            n = min(chunk, n_samples - i)
            if cond is not None:
                c = torch.as_tensor(cond[i:i + n], device=self.device)
                self._set_cond(c)
            s = None if seed is None else seed + i
            g = self.sample(n_samples=n, seq_len=seq_len, batch_size=n, seed=s)
            if cond is not None:
                cm = self.cond_mask.cpu().numpy().astype(bool)
                g[:, cm] = cond[i:i + n][:, cm]
            out.append(g)
        return np.concatenate(out, axis=0)

    def sample(self, n_samples=120, seq_len=None, batch_size=None, seed=None,
               sampler=None, max_rounds=10):
        """Like GBMFinancialDiffusion.sample, but rejection keeps conditioning aligned:
        failed rows are resampled with the same conditioning rows."""
        thr = self.config.get("reject_ac1_below")
        if thr is None or self._cond_batch is None or not bool(self.cond_mask.any()):
            return super().sample(n_samples, seq_len, batch_size, seed, sampler, max_rounds)
        cond_all = self._cond_batch
        result = np.empty((n_samples, self.K, seq_len), dtype=np.float32)
        todo = np.arange(n_samples)
        n_drawn = n_rej = 0
        cfg_thr = self.config.pop("reject_ac1_below")
        try:
            for r in range(max_rounds):
                self._set_cond(cond_all[todo])
                s = None if seed is None else seed + 1000 * r
                g = super().sample(len(todo), seq_len, len(todo), s, sampler)
                ok = self.path_lag1_autocorr(g) >= thr
                n_drawn += len(g)
                n_rej += int((~ok).sum())
                result[todo[ok]] = g[ok]
                todo = todo[~ok]
                if len(todo) == 0:
                    break
            if len(todo):
                result[todo] = g[~ok]  # keep last attempt rather than fail
        finally:
            self.config["reject_ac1_below"] = cfg_thr
            self._set_cond(cond_all)
        self.last_rejection_rate = n_rej / max(n_drawn, 1)
        if n_rej:
            print(f"  Rejected {n_rej}/{n_drawn} failed samples ({100 * self.last_rejection_rate:.1f}%)")
        return result
