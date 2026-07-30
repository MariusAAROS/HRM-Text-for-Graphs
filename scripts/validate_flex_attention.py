"""Validate the FlexAttention prefixLM port against a reference implementation.

Run on an A100 node (needs CUDA + Triton for FlexAttention):
    module purge && module load arch/a100 && module load pytorch-gpu/py3/2.8.0
    source $WORK/HRM-Text-for-Graphs/.venv/bin/activate
    python scripts/validate_flex_attention.py

Checks:
  1. Forward parity   : flash_attn_varlen_prefixlm vs a naive masked-softmax reference.
  2. Gradient parity  : dq/dk/dv match the reference under autograd.
Both use the packed layout + padding produced by compute_aux_seq_tensors_scalars,
including the is_causal=False (prefixLM) and is_causal=True (causal) paths.

The third "overfit one batch" sanity check is the short training smoke run:
    sbatch --array=0 slurm/train_graphqa.slurm    # watch train/loss fall in W&B/logs
"""
import numpy as np
import torch

from models.flash_attention_prefixlm_v2 import (
    compute_aux_seq_tensors_scalars,
    flash_attn_varlen_prefixlm,
)


def reference_prefixlm(q, k, v, is_causal, doc_ids, prefix_ends):
    """Dense masked-softmax reference. q/k/v: [S, H, D]."""
    S, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale  # [H, S, S]

    idx = torch.arange(S, device=q.device)
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    causal = idx[None, :] <= idx[:, None]
    if is_causal:
        allow = same_doc & causal
    else:
        allow = same_doc & (causal | (idx[None, :] < prefix_ends[None, :]))

    scores = scores.masked_fill(~allow[None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn, v)  # [S, H, D]
    return out


def build_batch(prefix_lens, causal_lens, batch_max_tokens, H, D, device, dtype):
    tensors, _ = compute_aux_seq_tensors_scalars(
        np.asarray(prefix_lens, dtype=np.int32),
        np.asarray(causal_lens, dtype=np.int32),
        batch_max_tokens,
    )
    doc_ids = torch.from_numpy(tensors["doc_ids"]).to(device)
    prefix_ends = torch.from_numpy(tensors["prefix_ends"]).to(device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    qkv = [torch.randn(batch_max_tokens, H, D, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
           for _ in range(3)]
    return qkv, doc_ids, prefix_ends


def run_case(is_causal, device, dtype, atol, rtol):
    prefix_lens = [5, 3, 8]
    causal_lens = [4, 6, 2]
    S = 128  # multiple of 128, > sum(total)=28, remainder is padding
    H, D = 4, 32

    (q, k, v), doc_ids, prefix_ends = build_batch(prefix_lens, causal_lens, S, H, D, device, dtype)
    for t in (q, k, v):
        t.requires_grad_(True)

    out = flash_attn_varlen_prefixlm(q, k, v, is_causal, doc_ids=doc_ids, prefix_ends=prefix_ends)
    ref = reference_prefixlm(q, k, v, is_causal, doc_ids, prefix_ends)

    # Compare only real tokens (padding rows are unused / undefined by design).
    total = int((doc_ids >= 0).sum())
    fwd_ok = torch.allclose(out[:total].float(), ref[:total].float(), atol=atol, rtol=rtol)
    fwd_err = (out[:total].float() - ref[:total].float()).abs().max().item()

    # Gradient parity: same upstream grad, compare dq/dk/dv on real tokens.
    up = torch.randn_like(out)
    gq, gk, gv = torch.autograd.grad(out, (q, k, v), up, retain_graph=True)
    rq, rk, rv = torch.autograd.grad(ref, (q, k, v), up)
    grad_ok = all(
        torch.allclose(a[:total].float(), b[:total].float(), atol=atol, rtol=rtol)
        for a, b in ((gq, rq), (gk, rk), (gv, rv))
    )
    grad_err = max((a[:total].float() - b[:total].float()).abs().max().item()
                   for a, b in ((gq, rq), (gk, rk), (gv, rv)))

    tag = "causal" if is_causal else "prefixlm"
    print(f"[{tag:8s} {str(dtype).split('.')[-1]:>8s}] "
          f"fwd {'OK ' if fwd_ok else 'FAIL'} (max |d|={fwd_err:.2e})  "
          f"grad {'OK ' if grad_ok else 'FAIL'} (max |d|={grad_err:.2e})")
    return fwd_ok and grad_ok


def main():
    assert torch.cuda.is_available(), "FlexAttention validation needs a CUDA (A100) device."
    device = "cuda"
    ok = True
    # fp32 for tight tolerances; bf16 for the training dtype (looser tolerances).
    ok &= run_case(False, device, torch.float32, atol=2e-4, rtol=2e-4)
    ok &= run_case(True, device, torch.float32, atol=2e-4, rtol=2e-4)
    ok &= run_case(False, device, torch.bfloat16, atol=2e-2, rtol=2e-2)
    ok &= run_case(True, device, torch.bfloat16, atol=2e-2, rtol=2e-2)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
