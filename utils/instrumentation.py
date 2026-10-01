"""Recursion-depth probe for the HRM / TRM (H_cycles x L_cycles) ablation.

Records, for every recursion state z^k produced inside the H/L loop:
  - cos_prev   cosine between z^k and z^(k-1)            -- fixed-point convergence
  - rel_resid  ||z^k - z^(k-1)|| / ||z^(k-1)||           -- does step k still do work
  - rms        scale of z^k                              -- drift / blowup
  - eff_rank   participation ratio of the token cov.     -- convergence vs collapse
  - zgrad      ||dLoss/dz^k||                            -- credit assignment through depth
  - cos_x      cosine between z^k and the token embedding x -- survival of the raw input
  - gain       ||core(h+d+inj) - core(h+inj)|| / ||d||    -- does the incoming state still matter
  - prenorm_rms  residual RMS before the final norm      -- the washout signature of pre-norm

`zgrad` only exists for steps inside the truncated-BPTT window (HRM runs the recursion
prefix under no_grad); the absence of the other steps is itself the measurement.

Also records, per recursion step, how the two incoming signals are mixed. The H/L blocks fuse
additively (`core(hidden_states + input_injection)`), so the analogue of "attention over the
two inputs" is the projection share of each onto the fused state; share_h + share_inj == 1.

Finally, on a coarser interval and for a subset of (recursion step, layer), it recomputes the
attention matrix eagerly. FlexAttention is a fused kernel that never materialises softmax(QK^T),
so there is no hook to read it from -- recomputation from q/k/v is the only option. The mask is
an exact replica of the `score_mod` predicate in models/flash_attention_prefixlm_v2.py.

The probe is activated through a module-global slot rather than a forward() kwarg so that
no model signature changes. The slot is only ever set on the eager path in pretrain.py, so
the compiled `train_batch` always traces `probe is None` and never recompiles.
"""

from typing import Literal, Optional
from contextlib import contextmanager
import math
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
import pydantic


_EPS = 1e-12
_NEG_INF = float("-inf")

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
    log_fusion: bool = True       # Projection shares of hidden_states / input_injection.
    log_x_survival: bool = True   # How much of the token embedding survives at depth.

    # Sensitivity of each recursion step to its incoming state: one extra (no-grad) block forward per
    # step with the state perturbed by eps * rms(state). Activations are bf16, so eps must stay well
    # above bf16 rounding (see scripts/probe_gain_ckpt.py for the fp32 calibration).
    log_gain: bool = True
    gain_eps: float = 1.0  # bf16 matches fp32 within ~15% at 1.0; at 0.05 bf16 rounding reads 0.04 for a true 0.005.
    log_prenorm: bool = True      # Residual RMS before the Transformer's final norm.

    # Attention recompute. Far more expensive than the stats above, hence its own interval and a
    # restriction to the first/last recursion step of each role.
    log_attn: bool = True
    attn_interval: int = 500
    attn_query_sample: int = 256
    attn_layers: Literal["ends", "all"] = "ends"  # "ends" = layers {0, n//2, n-1}
    log_attn_drift: bool = True

    @pydantic.model_validator(mode="after")
    def _validate_attn_interval(self):
        # Attention is recorded by the same probe object that the z-stats use, so its steps have to
        # be a subset of the probe steps.
        if self.enabled and self.log_attn and self.attn_interval % self.interval != 0:
            raise ValueError(f"instrumentation.attn_interval ({self.attn_interval}) must be a "
                             f"multiple of instrumentation.interval ({self.interval}).")
        return self


class RecursionProbe:
    def __init__(self, config: InstrumentationConfig, with_attention: bool = False) -> None:
        self.config = config
        self.records: dict[str, tuple[Tensor, Tensor]] = {}
        self.kinds: dict[str, str] = {}
        self._in_graph: dict[str, int] = {"H": 0, "L": 0}

        self.attention = with_attention and config.log_attn
        self._tag: Optional[str] = None     # Current recursion step, set by begin_step().
        self._role: Optional[str] = None
        self._pos: Optional[str] = None     # "first" / "last" within the role, for drift.
        self._attn_step = False
        self._q_idx: Optional[Tensor] = None
        self._drift_cache: dict[str, Tensor] = {}

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

    def _stride(self, n_tokens: int) -> tuple[int, int]:
        limit = self.config.token_sample or n_tokens
        return (max(1, n_tokens // limit) if n_tokens > limit else 1), limit

    # ---- recursion-step context ----
    def begin_step(self, role: str, idx: int, n_steps: int) -> None:
        self._role = role
        self._tag = f"{role}/{'cycle' if role == 'H' else 'step'}{idx:02d}"
        # idx 0 wins when n_steps == 1 (e.g. H in the (1,12) config), so drift is simply absent
        # rather than a meaningless zero.
        self._pos = "first" if idx == 0 else ("last" if idx == n_steps - 1 else None)
        self._attn_step = self.attention and self._pos is not None

    def end_step(self) -> None:
        self._role = self._tag = self._pos = None
        self._attn_step = False

    # ---- recursion-step statistics ----
    @torch.no_grad()
    def _record_stats(self, tag: str, z_in: Tensor, z_out: Tensor) -> None:
        stride, limit = self._stride(z_out.shape[0])
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

    def record(self, role: str, idx: int, z_in: Tensor, z_out: Tensor, x: Optional[Tensor] = None) -> None:
        tag = f"{role}/{'cycle' if role == 'H' else 'step'}{idx:02d}"
        self._record_stats(tag, z_in, z_out)

        if x is not None and self.config.log_x_survival and z_out.dim() > 1:
            self._record_x_survival(tag, x, z_out)

        if self.config.log_zgrad and z_out.requires_grad:
            self._in_graph[role] = self._in_graph.get(role, 0) + 1
            key = f"zgrad/{tag}"
            z_out.register_hook(lambda g, key=key: self._sumsq(key, g.float().pow(2).sum()))

    @torch.no_grad()
    def _record_x_survival(self, tag: str, x: Tensor, z: Tensor) -> None:
        stride, limit = self._stride(z.shape[0])
        z_s = self._subsample(z.detach(), stride, limit)
        x_s = self._subsample(x.detach(), stride, limit)
        self._mean(f"xsurv/{tag}/cos_x", F.cosine_similarity(z_s, x_s, dim=-1).mean())
        self._mean(f"xsurv/{tag}/share_x",
                   ((x_s * z_s).sum(-1) / z_s.pow(2).sum(-1).clamp_min(_EPS)).mean())

    # ---- additive fusion of the two incoming signals ----
    @torch.no_grad()
    def record_fusion(self, hidden_states: Tensor, input_injection: Tensor, fused: Tensor) -> None:
        if not self.config.log_fusion or self._tag is None:
            return
        stride, limit = self._stride(fused.shape[0])
        s = self._subsample(fused.detach(), stride, limit)
        # zL_init is 1-D at the first L step; it broadcasts into every token row.
        h = self._subsample(hidden_states.detach(), stride, limit) if hidden_states.dim() > 1 \
            else hidden_states.detach().float().expand_as(s)
        inj = self._subsample(input_injection.detach(), stride, limit) if input_injection.dim() > 1 \
            else input_injection.detach().float().expand_as(s)

        s_sq = s.pow(2).sum(-1).clamp_min(_EPS)
        tag = self._tag
        # Projection shares onto the fused state; these sum to exactly 1 by construction.
        self._mean(f"fusion/{tag}/share_h", ((h * s).sum(-1) / s_sq).mean())
        self._mean(f"fusion/{tag}/share_inj", ((inj * s).sum(-1) / s_sq).mean())
        self._mean(f"fusion/{tag}/cos_h_inj", F.cosine_similarity(h, inj, dim=-1).mean())
        self._mean(f"fusion/{tag}/norm_ratio",
                   (inj.norm(dim=-1) / h.norm(dim=-1).clamp_min(_EPS)).mean())

    # ---- state sensitivity ----
    def record_gain(self, gain: Tensor) -> None:
        if self._tag is not None:
            self._mean(f"zstat/{self._tag}/gain", gain)

    @torch.no_grad()
    def record_prenorm(self, x: Tensor) -> None:
        if not self.config.log_prenorm or self._tag is None:
            return
        stride, limit = self._stride(x.shape[0])
        self._mean(f"zstat/{self._tag}/prenorm_rms", self._subsample(x.detach(), stride, limit).pow(2).mean().sqrt())

    # ---- attention ----
    def _query_idx(self, doc_ids: Tensor, n_tokens: int) -> Tensor:
        """Evenly spread query rows over the non-padding tokens, computed once and reused so that
        the drift comparison comes from identical rows. Padding carries doc_id == -1 and would
        otherwise attend only to other padding."""
        if self._q_idx is None:
            valid = (doc_ids >= 0).nonzero(as_tuple=True)[0]
            if valid.numel() == 0:
                valid = torch.arange(n_tokens, device=doc_ids.device)
            n = self.config.attn_query_sample
            if valid.numel() > n:
                valid = valid[torch.linspace(0, valid.numel() - 1, n, device=valid.device).long()]
            self._q_idx = valid
        return self._q_idx

    @torch._dynamo.disable()  # type: ignore[misc]
    @torch.no_grad()
    def record_attention(self, layer_idx: int, n_layers: int, q: Tensor, k: Tensor, v: Tensor,
                         gate: Tensor, is_causal: bool, seq_info: dict[str, Tensor]) -> None:
        if not self._attn_step or self._tag is None:
            return
        if self.config.attn_layers == "ends" and layer_idx not in (0, n_layers // 2, n_layers - 1):
            return

        n_tokens, _n_heads, head_dim = q.shape
        doc_ids = seq_info["doc_ids"].to(torch.int32).reshape(-1)[:n_tokens]
        prefix_ends = seq_info["prefix_ends"].to(torch.int32).reshape(-1)[:n_tokens]
        rows = self._query_idx(doc_ids, n_tokens)

        # Exact replica of the score_mod predicate in flash_attention_prefixlm_v2.py. Note that
        # prefix_ends is indexed by the KEY, not the query.
        kv = torch.arange(n_tokens, device=q.device, dtype=torch.int32)
        is_prefix_key = kv < prefix_ends
        causal = kv[None, :] <= rows[:, None].to(kv.dtype)
        allow = (doc_ids[rows][:, None] == doc_ids[None, :])
        allow &= causal if is_causal else (causal | is_prefix_key[None, :])

        scores = torch.einsum("qhd,khd->hqk", q[rows].float(), k.float()) / math.sqrt(head_dim)
        p = torch.softmax(scores.masked_fill(~allow[None], _NEG_INF), dim=-1)

        tag = f"attn/{self._tag}/layer{layer_idx:02d}"
        entropy = -(p * p.clamp_min(_EPS).log()).sum(-1)
        self._mean(f"{tag}/entropy", entropy.mean())
        # Row support varies with position under a causal mask, so also report entropy relative to
        # the uniform-over-allowed-keys maximum.
        self._mean(f"{tag}/entropy_norm",
                   (entropy / allow.sum(-1).clamp_min(2).float().log()[None]).mean())
        self._mean(f"{tag}/max_weight", p.amax(-1).mean())

        # Unbounded logit growth saturates the softmax and kills gradients through attention.
        allowed_scores = torch.where(allow[None], scores, scores.new_zeros(()))
        self._mean(f"{tag}/logit_absmax", allowed_scores.abs().amax())
        self._mean(f"{tag}/logit_rms",
                   (allowed_scores.pow(2).sum() / allow.sum().clamp_min(1) / p.shape[0]).sqrt())

        # A token can hold high attention weight yet contribute little if its value vector is small.
        eff = p * v.float().norm(dim=-1).transpose(0, 1)[:, None, :]
        eff = eff / eff.sum(-1, keepdim=True).clamp_min(_EPS)
        self._mean(f"{tag}/eff_entropy", -(eff * eff.clamp_min(_EPS).log()).sum(-1).mean())
        self._mean(f"{tag}/eff_max", eff.amax(-1).mean())

        # Response mass is 1 - prefix mass, so only one of the two is logged.
        self._mean(f"{tag}/mass_prefix", (p * is_prefix_key[None, None, :]).sum(-1).mean())
        self._mean(f"{tag}/gate_mean", torch.sigmoid(gate[rows].float()).mean())

        if self.config.log_attn_drift:
            self._record_drift(f"attn_drift/{self._role}/layer{layer_idx:02d}", p)

    @torch.no_grad()
    def _record_drift(self, key: str, p: Tensor) -> None:
        if self._pos == "first":
            self._drift_cache[key] = p.to(torch.float16)
            return
        prev = self._drift_cache.pop(key, None)
        if prev is None:
            return
        prev = prev.float()
        mean = 0.5 * (prev + p)
        kl = lambda a, b: (a * (a.clamp_min(_EPS).log() - b.clamp_min(_EPS).log())).sum(-1)
        self._mean(f"{key}/js_first_last", (0.5 * (kl(prev, mean) + kl(p, mean))).mean())
        self._mean(f"{key}/cos_first_last", F.cosine_similarity(prev, p, dim=-1).mean())

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


@contextmanager
def suspend_probe():
    """Hide the active probe for an auxiliary forward (e.g. the gain perturbation), so the records
    of the blocks it runs through are not counted twice."""
    global _ACTIVE_PROBE
    probe, _ACTIVE_PROBE = _ACTIVE_PROBE, None
    try:
        yield
    finally:
        _ACTIVE_PROBE = probe


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


_ZSTAT_RE = re.compile(r"^probe/(?:zstat|fusion|xsurv)/(?P<role>[HL])/(?:cycle|step)(?P<idx>\d+)/(?P<stat>\w+)$")
_ZGRAD_RE = re.compile(r"^probe/zgrad/(?P<role>[HL])/(?:cycle|step)(?P<idx>\d+)$")
_ATTN_RE = re.compile(r"^probe/attn/(?P<role>[HL])/(?:cycle|step)(?P<idx>\d+)/layer(?P<layer>\d+)/(?P<stat>\w+)$")


def _summarize(series: list[float], prefix: str, out: dict[str, float]) -> None:
    first, last = series[0], series[-1]
    out[f"{prefix}_first"] = first
    out[f"{prefix}_last"] = last
    out[f"{prefix}_mean"] = sum(series) / len(series)
    # >1 means the quantity grows with depth, <1 means it decays (residual contraction,
    # or gradient attenuation across the truncated-BPTT window).
    out[f"{prefix}_ratio"] = last / first if abs(first) > _EPS else float("nan")


def derive_summaries(metrics: dict[str, float], prefix: str = "probe/summary/") -> dict[str, float]:
    """Config-independent scalars so the 9x2 sweep is comparable despite differing step counts."""
    profiles: dict[tuple[str, str], list[tuple[int, float]]] = {}
    attn: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    for key, value in metrics.items():
        if (match := _ATTN_RE.match(key)) is not None:
            attn.setdefault((match["role"], match["layer"], match["stat"]), []).append((int(match["idx"]), value))
        elif (match := _ZSTAT_RE.match(key) or _ZGRAD_RE.match(key)) is not None:
            stat = match.groupdict().get("stat", "zgrad")
            profiles.setdefault((match["role"], stat), []).append((int(match["idx"]), value))

    out: dict[str, float] = {}
    for (role, stat), points in profiles.items():
        _summarize([v for _, v in sorted(points)], f"{prefix}{role}/{stat}", out)
    for (role, layer, stat), points in attn.items():
        _summarize([v for _, v in sorted(points)], f"{prefix}{role}/layer{layer}/{stat}", out)

    h_norm = metrics.get("probe/gradnorm/H_level")
    l_norm = metrics.get("probe/gradnorm/L_level")
    if h_norm is not None and l_norm is not None and l_norm > _EPS:
        out[f"{prefix}gradnorm_H_over_L"] = h_norm / l_norm

    return out
