"""Validate the backprop-through-time budget of HRM / TRM.

Checks that (a) leaving the new fields unset reproduces the legacy warmup formula exactly,
(b) `full_backprop=True` really builds a graph through every block application, and
(c) the truncated regime builds a graph through exactly `H_bp + L_bp` of them.

Run:  python scripts/validate_bp_steps.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.common import resolve_bp_steps  # noqa: E402
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel  # noqa: E402
from models.baselines.trm_nocarry import TinyRecursiveModel  # noqa: E402


BASE = dict(max_seq_len=128, n_layers=2, hidden_size=64, num_heads=2, expansion=4.0,
            attn_type="causal", init_type="lecun_normal", norm_type="pre", norm_eps=1e-6,
            pos_emb_type="none")


def check_legacy_formula():
    """Unset H_bp_steps/L_bp_steps must reproduce the pre-change derivation bit-for-bit."""
    for H in range(1, 9):
        for L in range(1, 13):
            for bp_steps in range(2, 6):
                model = HierarchicalReasoningModel(BASE | dict(H_cycles=H, L_cycles=L))
                got = model._resolve_bp_steps(bp_steps)

                legacy_H = min(H, bp_steps - 1)
                legacy_L = bp_steps - legacy_H
                # The clamp is behaviour-preserving: `k >= H*L - L_bp` already saturates past H*L.
                want = (min(legacy_H, H), min(legacy_L, H * L))
                assert got == want, f"H={H} L={L} bp_steps={bp_steps}: got {got}, want {want}"
    print("[ok] legacy warmup formula preserved over H in 1..8, L in 1..12, bp_steps in 2..5")


def check_sentinel():
    for H, L in ((1, 1), (2, 3), (4, 6), (6, 12)):
        assert resolve_bp_steps(H, L, -1, -1, False) == (H, H * L)
        assert resolve_bp_steps(H, L, 0, 0, True) == (H, H * L)
        assert resolve_bp_steps(H, L, 99, 99, False) == (H, H * L), "budget must clamp to the maximum"
    print("[ok] -1 sentinel, full_backprop and clamping all agree")


def count_grad_blocks(model, **fwd_kwargs) -> int:
    """Count block applications that built a graph, by tagging each level's output."""
    n = 0

    def wrap(module):
        inner = module.forward

        def fwd(*args, **kwargs):
            nonlocal n
            out = inner(*args, **kwargs)
            n += int(out.requires_grad)
            return out

        module.forward = fwd

    levels = [getattr(model, name) for name in ("H_level", "L_level") if hasattr(model, name)]
    for level in {id(m): m for m in levels}.values():  # TRM shares one module across both roles.
        wrap(level)

    x = torch.randn(8, BASE["hidden_size"], requires_grad=True)
    model(None, x, **fwd_kwargs)
    return n


def check_hrm_counts():
    H, L = 3, 4
    total = H * (L + 1)

    full = HierarchicalReasoningModel(BASE | dict(H_cycles=H, L_cycles=L, full_backprop=True))
    n = count_grad_blocks(full, bp_steps=2)
    assert n == total, f"full_backprop HRM: {n} grad blocks, want {total}"

    explicit = HierarchicalReasoningModel(BASE | dict(H_cycles=H, L_cycles=L, H_bp_steps=-1, L_bp_steps=-1))
    assert count_grad_blocks(explicit, bp_steps=2) == total, "H_bp_steps=-1,L_bp_steps=-1 must equal full_backprop"

    trunc = HierarchicalReasoningModel(BASE | dict(H_cycles=H, L_cycles=L))
    H_bp, L_bp = trunc._resolve_bp_steps(5)
    n = count_grad_blocks(trunc, bp_steps=5)
    assert n == H_bp + L_bp, f"truncated HRM: {n} grad blocks, want {H_bp + L_bp}"
    assert n < total, "truncated regime must build fewer graphs than full backprop"
    print(f"[ok] HRM (3,4): full={total} blocks, truncated(bp_steps=5)={n} blocks")


def check_trm_counts():
    H, L = 3, 4
    total = H * (L + 1)

    full = TinyRecursiveModel(BASE | dict(H_cycles=H, L_cycles=L, full_backprop=True))
    assert full.H_bp_steps == H and full.L_bp_steps == H * L
    n = count_grad_blocks(full)
    assert n == total, f"full_backprop TRM: {n} grad blocks, want {total}"

    trunc = TinyRecursiveModel(BASE | dict(H_cycles=H, L_cycles=L, H_bp_steps=1, L_bp_steps=1))
    n = count_grad_blocks(trunc)
    assert n == 2, f"truncated TRM: {n} grad blocks, want 2"
    print(f"[ok] TRM (3,4): full={total} blocks, truncated(1,1)=2 blocks")


def check_mixed_axes_rejected():
    """Pinning one axis while leaving the other on the warmup schedule is ill-defined, not silent."""
    for kwargs in (dict(H_bp_steps=-1), dict(L_bp_steps=-1), dict(H_bp_steps=2), dict(L_bp_steps=3)):
        try:
            HierarchicalReasoningModel(BASE | dict(H_cycles=3, L_cycles=4) | kwargs)
        except ValueError:
            continue
        raise AssertionError(f"HRM accepted a half-specified bp budget: {kwargs}")

    try:
        HierarchicalReasoningModel(BASE | dict(H_cycles=3, L_cycles=4, full_backprop=True, H_bp_steps=1, L_bp_steps=1))
    except ValueError:
        pass
    else:
        raise AssertionError("HRM accepted full_backprop together with an explicit budget")
    print("[ok] half-specified and conflicting HRM bp budgets are rejected")


def check_grad_checkpointing():
    """Checkpointed and non-checkpointed backward must agree."""
    cfg = BASE | dict(H_cycles=2, L_cycles=3, full_backprop=True)
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        model = HierarchicalReasoningModel(cfg | dict(grad_checkpointing=ckpt))
        torch.manual_seed(1)
        x = torch.randn(8, BASE["hidden_size"])
        model(None, x, bp_steps=2)[1].square().sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))

    assert torch.allclose(grads[0], grads[1], atol=1e-5), "grad_checkpointing changed the gradients"
    print("[ok] grad_checkpointing is gradient-neutral")


if __name__ == "__main__":
    check_legacy_formula()
    check_sentinel()
    check_hrm_counts()
    check_trm_counts()
    check_mixed_axes_rejected()
    check_grad_checkpointing()
    print("\nAll backprop-budget checks passed.")
