from dataclasses import dataclass

from torch import Tensor
import torch.nn.functional as F


IGNORE_LABEL_ID = -100


def trunc_normal_init_(tensor: Tensor, std: float = 1.0):
    """Fast approximate truncated normal initialization. Fairly accurate."""

    return tensor.normal_().fmod_(3.0).mul_(1.014762601732121 * std)


def packing_sequence_sum(x: Tensor, cu_seqlens: Tensor):
    c = F.pad(x.cumsum(0), (1, 0))
    return c[cu_seqlens[1:]] - c[cu_seqlens[:-1]]


def resolve_bp_steps(H_cycles: int, L_cycles: int, H_bp_steps: int, L_bp_steps: int, full_backprop: bool = False) -> tuple[int, int]:
    """Resolve the backprop-through-time budget of a two-level recurrence.

    The two axes are counted by separate loop indices, so they have separate maxima:
    `H_bp_steps` counts the last N of the `H` H-block applications, while `L_bp_steps` counts
    the last N L-block applications across ALL H cycles (the L loop index is global over
    `0..H*L-1`). Hence "backprop through everything" is `(H_cycles, H_cycles * L_cycles)`.
    Note this is NOT the total block-application count `H*(L+1)` used for depth accounting.

    A negative value means "all cycles on that axis". Shared by HRM, TRM and the HF exporter
    so the three cannot drift apart.
    """
    H_max, L_max = H_cycles, H_cycles * L_cycles
    if full_backprop:
        return H_max, L_max

    H_bp = H_max if H_bp_steps < 0 else min(H_bp_steps, H_max)
    L_bp = L_max if L_bp_steps < 0 else min(L_bp_steps, L_max)
    return H_bp, L_bp


@dataclass
class WrappedTensor:
    value: Tensor


def wrap_tensor(value: Tensor) -> WrappedTensor:
    """Wrap a Tensor, so that FSDP2 won't see this Tensor, and do preprocessing such as moving to device and casting."""
    return WrappedTensor(value)


def unwrap_tensor(wrapped: Tensor | WrappedTensor) -> Tensor:
    return wrapped.value if isinstance(wrapped, WrappedTensor) else wrapped
