"""Recursion-depth probe for the HRM / TRM (H_cycles x L_cycles) ablation.

Records, for every recursion state z^k produced inside the H/L loop:
  - cos_prev   cosine between z^k and z^(k-1)            -- fixed-point convergence
  - rel_resid  ||z^k - z^(k-1)|| / ||z^(k-1)||           -- does step k still do work
  - rms        scale of z^k                              -- drift / blowup
  - eff_rank   participation ratio of the token cov.     -- convergence vs collapse
  - zgrad      ||dLoss/dz^k||                            -- credit assignment through depth

`zgrad` only exists for steps inside the truncated-BPTT window (HRM runs the recursion
prefix under no_grad); the absence of the other steps is itself the measurement.

The probe is activated through a module-global slot rather than a forward() kwarg so that
no model signature changes. The slot is only ever set on the eager path in pretrain.py, so
the compiled `train_batch` always traces `probe is None` and never recompiles.
"""

from typing import Optional
from contextlib import contextmanager
import math
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
import pydantic


_EPS = 1e-12

# Metric key -> (numerator, divisor) with a reduction kind:
#   "mean"  reduced as sum(num)/sum(div)   -- activation statistics
#   "sumsq" reduced as sqrt(sum(num))      -- squared norms (FSDP2 grads are SUM-reduced)
_KIND_MEAN = "mean"
_KIND_SUMSQ = "sumsq"

_PARAM_BUCKETS = ("H_level", "L_level", "embed_tokens", "lm_head")


class InstrumentationConfig(pydantic.BaseModel):
    enabled: bool = True
    interval: int = 100  # Must be a multiple of log_interval; probe steps run eager (slow).

    token_sample: Optional[int] = 8192  # Strided token subsample for activation stats.
    log_eff_rank: bool = True
    log_zgrad: bool = True
    log_param_grads: bool = True


class RecursionProbe:
    def __init__(self, config: InstrumentationConfig) -> None:
        self.config = config
        self.records: dict[str, tuple[Tensor, Tensor]] = {}
        self.kinds: dict[str, str] = {}
        self._in_graph: dict[str, int] = {"H": 0, "L": 0}

    # ---- record helpers ----
    def _add(self, key: str, num: Tensor, div: Tensor, kind: str) -> None:
        prev = self.records.get(key)
        if prev is not None:
            num = prev[0] + num
            div = prev[1] + div
        self.records[key] = (num, div)
        self.kinds[key] = kind

    def _mean(self, key: str, value: Tensor) -> None:
        self._add(key, value.detach().float().reshape(()), torch.ones((), device=value.device), _KIND_MEAN)

    def _sumsq(self, key: str, value: Tensor) -> None:
        self._add(key, value.detach().float().reshape(()), torch.zeros((), device=value.device), _KIND_SUMSQ)

    def record_scalar(self, key: str, value: float, device: torch.device) -> None:
        self._mean(key, torch.tensor(float(value), device=device))

    def _subsample(self, z: Tensor, stride: int, n: int) -> Tensor:
        return z[::stride][:n].float() if stride > 1 else z[:n].float()

    # ---- recursion-step statistics ----
    @torch.no_grad()
    def _record_stats(self, tag: str, z_in: Tensor, z_out: Tensor) -> None:
        n_tokens = z_out.shape[0]
        limit = self.config.token_sample or n_tokens
        stride = max(1, n_tokens // limit) if n_tokens > limit else 1
        out = self._subsample(z_out.detach(), stride, limit)

        self._mean(f"zstat/{tag}/rms", out.pow(2).mean().sqrt())

        if self.config.log_eff_rank and out.shape[0] > 1:
            centered = out - out.mean(dim=0, keepdim=True)
            cov = centered.T @ centered
            trace = torch.diagonal(cov).sum()
            # Participation ratio (sum lambda)^2 / sum lambda^2; cov is symmetric so
            # sum lambda^2 == ||cov||_F^2. Computed per-rank (cov is too large to reduce).
            self._mean(f"zstat/{tag}/eff_rank", trace.pow(2) / cov.pow(2).sum().clamp_min(_EPS))

        # z_L starts as the 1-D `zL_init` buffer, so the first L step has no predecessor state.
        if z_in.dim() < 2:
            return

        prev = self._subsample(z_in.detach(), stride, limit)
        self._mean(f"zstat/{tag}/cos_prev", F.cosine_similarity(out, prev, dim=-1).mean())
        self._mean(f"zstat/{tag}/rel_resid",
                   ((out - prev).norm(dim=-1) / prev.norm(dim=-1).clamp_min(_EPS)).mean())

    def record(self, role: str, idx: int, z_in: Tensor, z_out: Tensor) -> None:
        tag = f"{role}/{'cycle' if role == 'H' else 'step'}{idx:02d}"
        self._record_stats(tag, z_in, z_out)

        if self.config.log_zgrad and z_out.requires_grad:
            self._in_graph[role] = self._in_graph.get(role, 0) + 1
            key = f"zgrad/{tag}"
            z_out.register_hook(lambda g, key=key: self._sumsq(key, g.float().pow(2).sum()))

    # ---- parameter gradient norms ----
    @torch.no_grad()
    def finalize_grads(self, model: nn.Module) -> None:
        device = next(model.parameters()).device
        for role, count in self._in_graph.items():
            self.record_scalar(f"zgrad/{role}/in_graph_steps", count, device)

        if not self.config.log_param_grads:
            return

        totals: dict[str, Optional[Tensor]] = {k: None for k in ("total",) + _PARAM_BUCKETS}
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            # FSDP2 shards are disjoint and gradients are SUM-reduced
            # (set_gradient_divide_factor(1.0)), so shard sum-of-squares + all-reduce is exact.
            local = grad.to_local() if hasattr(grad, "to_local") else grad
            sq = local.float().pow(2).sum()

            keys = ["total"] + [b for b in _PARAM_BUCKETS if f"{b}." in name]
            for key in keys:
                totals[key] = sq if totals[key] is None else totals[key] + sq

        for key, value in totals.items():
            if value is not None:
                self._sumsq(f"gradnorm/{key}", value)


# ---- active probe slot ----

_ACTIVE_PROBE: Optional[RecursionProbe] = None


def get_active_probe() -> Optional[RecursionProbe]:
    return _ACTIVE_PROBE


@contextmanager
def active_probe(probe: RecursionProbe):
    global _ACTIVE_PROBE
    _ACTIVE_PROBE = probe
    try:
        yield probe
    finally:
        _ACTIVE_PROBE = None


# ---- reduction & derived summaries ----

@torch.inference_mode()
def reduce_probe_metrics(probe: RecursionProbe, prefix: str = "probe/") -> dict[str, float]:
    """Collective: must be called on every rank. Only rank 0 gets a populated dict."""
    keys = list(sorted(probe.records.keys()))  # Sort keys to guarantee all processes use the same order.
    if not keys:
        return {}

    values = torch.stack([probe.records[k][0] for k in keys] + [probe.records[k][1] for k in keys])
    if dist.is_initialized():
        dist.reduce(values, dst=0)
        if dist.get_rank() != 0:
            return {}

    num, div = (x.cpu().numpy().tolist() for x in values.chunk(2, dim=-1))
    return {prefix + k: (math.sqrt(max(num[i], 0.0)) if probe.kinds[k] == _KIND_SUMSQ else num[i] / div[i])
            for i, k in enumerate(keys)}


_ZSTAT_RE = re.compile(r"^probe/zstat/(?P<role>[HL])/(?:cycle|step)(?P<idx>\d+)/(?P<stat>\w+)$")
_ZGRAD_RE = re.compile(r"^probe/zgrad/(?P<role>[HL])/(?:cycle|step)(?P<idx>\d+)$")


def derive_summaries(metrics: dict[str, float], prefix: str = "probe/summary/") -> dict[str, float]:
    """Config-independent scalars so the 9x2 sweep is comparable despite differing step counts."""
    profiles: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for key, value in metrics.items():
        match = _ZSTAT_RE.match(key) or _ZGRAD_RE.match(key)
        if match is not None:
            stat = match.groupdict().get("stat", "zgrad")
            profiles.setdefault((match["role"], stat), []).append((int(match["idx"]), value))

    out: dict[str, float] = {}
    for (role, stat), points in profiles.items():
        series = [v for _, v in sorted(points)]
        first, last = series[0], series[-1]
        out[f"{prefix}{role}/{stat}_first"] = first
        out[f"{prefix}{role}/{stat}_last"] = last
        out[f"{prefix}{role}/{stat}_mean"] = sum(series) / len(series)
        # >1 means the quantity grows with depth, <1 means it decays (residual contraction,
        # or gradient attenuation across the truncated-BPTT window).
        out[f"{prefix}{role}/{stat}_ratio"] = last / first if abs(first) > _EPS else float("nan")

    h_norm = metrics.get("probe/gradnorm/H_level")
    l_norm = metrics.get("probe/gradnorm/L_level")
    if h_norm is not None and l_norm is not None and l_norm > _EPS:
        out[f"{prefix}gradnorm_H_over_L"] = h_norm / l_norm

    return out
