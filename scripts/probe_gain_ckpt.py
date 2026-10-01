"""Offline recursion probe of a trained checkpoint: is the recurrent state still used?

Runs one teacher-forced forward over a packed batch of eval samples with the recursion probe
active, and reports per recursion step:
  gain         ||core(h + d + inj) - core(h + inj)|| / ||d||, d = eps * rms(h) * N(0, I)
               (~0: the block output ignores its incoming state, so extra steps cannot carry
               information; see utils/instrumentation.py)
  prenorm_rms  RMS of the residual before the final norm (the pre-norm washout signature)
  rms          RMS of the recursion state

This is the source of truth for the gain table: it runs on any checkpoint (including the ones
trained before the in-training probe existed) and can run in fp32. Activations are bf16 in
training, whose rounding floor can hide a gain of ~0.005, so `--dtype fp32` and an `--eps` sweep
calibrate the in-training `instrumentation.gain_eps`.

Usage:
    python scripts/probe_gain_ckpt.py --ckpt_path <ckpt_dir> [<ckpt_dir> ...] \
        --data data/graphqa/hrm-text/standard/val.jsonl --dtype fp32 bf16 --eps 0.05 0.25 1.0 \
        [--fresh] [--csv data/results/recursion-hyp-probe.csv]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path

from models.layers import find_multiple
from models.flash_attention_prefixlm_v2 import compute_aux_seq_tensors_scalars
from pretrain import PretrainConfig, V1DatasetMeta, load_model_class
from simple_inference_engine import inference_load_checkpoint
from utils.instrumentation import (InstrumentationConfig, RecursionProbe, active_probe, derive_summaries,
                                   reduce_probe_metrics)

STEP_RE = re.compile(r"^probe/zstat/(?P<role>[HL])/(?:cycle|step)(?P<idx>\d+)/(?P<stat>gain|prenorm_rms|rms)$")


def fresh_model(ckpt_path: str, dtype: torch.dtype):
    """Same architecture as the checkpoint, randomly initialised (the reference for a healthy gain)."""
    with open(os.path.join(ckpt_path, "all_config.yaml")) as f:
        cfg = PretrainConfig(**yaml.safe_load(f))
    with open(os.path.join(ckpt_path, "train_metadata.yaml")) as f:
        meta = V1DatasetMeta(**yaml.safe_load(f))
    combined = cfg.arch.model_dump() | meta.model_dump() | cfg.data.model_dump()
    with torch.device("cuda"):
        model = load_model_class(cfg.arch.head)(load_model_class(cfg.arch.name)(combined), combined)
    return model.to(dtype).eval()


def packed_batch(ckpt, rows: list[dict], pad_multiple: int = 128) -> dict[str, torch.Tensor]:
    eoa = int(ckpt.tokenizer.convert_tokens_to_ids(ckpt.tokenizer_info["eoa"]))
    prompts = [ckpt.tokenize_prompt(r.get("condition", "direct"), r["instruction"]) for r in rows]
    answers = [np.asarray(ckpt.tokenizer(r["response"], add_special_tokens=False)["input_ids"] + [eoa]) for r in rows]
    prefix_lens = np.array([len(p) for p in prompts], dtype=np.int32)
    causal_lens = np.array([len(a) for a in answers], dtype=np.int32)
    S = find_multiple(int((prefix_lens + causal_lens).sum()), pad_multiple)

    inputs = np.zeros(S, dtype=np.int64)
    position_ids = np.zeros(S, dtype=np.int64)
    off = 0
    for p, a in zip(prompts, answers):
        seq = np.concatenate([p, a])
        inputs[off:off + len(seq)] = seq
        position_ids[off:off + len(seq)] = np.arange(len(seq))
        off += len(seq)
    aux, _ = compute_aux_seq_tensors_scalars(prefix_lens, causal_lens, S)
    return {k: torch.as_tensor(v, device="cuda") for k, v in
            {"inputs": inputs, "position_ids": position_ids,
             "doc_ids": aux["doc_ids"], "prefix_ends": aux["prefix_ends"]}.items()}


@torch.no_grad()
def probe_once(model, carry, batch, eps: float, seed: int) -> dict[str, float]:
    cfg = InstrumentationConfig(token_sample=None, log_eff_rank=False, log_zgrad=False, log_param_grads=False,
                                log_fusion=False, log_x_survival=False, log_attn=False, gain_eps=eps)
    probe = RecursionProbe(cfg)
    torch.manual_seed(seed)
    with active_probe(probe):
        model(carry, batch)
    return reduce_probe_metrics(probe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", nargs="+", required=True)
    ap.add_argument("--ckpt_epoch", type=int, default=None)
    ap.add_argument("--data", default="data/graphqa/hrm-text/standard/val.jsonl")
    ap.add_argument("--n_samples", type=int, default=12, help="Eval samples packed into the probed batch.")
    ap.add_argument("--dtype", nargs="+", default=["fp32"], choices=["fp32", "bf16"])
    ap.add_argument("--eps", nargs="+", type=float, default=[1.0])
    ap.add_argument("--L_cycles", type=int, default=None, help="Probe with an overridden L_cycles.")
    ap.add_argument("--fresh", action="store_true", help="Also probe a randomly initialised twin of each checkpoint.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="Append per-step rows here (replacing rows of the same key).")
    args = ap.parse_args()

    with open(args.data) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [rows[i] for i in np.random.default_rng(args.seed).choice(len(rows), args.n_samples, replace=False)]

    out = []
    for ckpt_path in args.ckpt_path:
        run = os.path.basename(os.path.normpath(ckpt_path))
        overrides = {"L_cycles": args.L_cycles} if args.L_cycles else {}
        ckpt = inference_load_checkpoint(ckpt_path, args.ckpt_epoch, True, arch_overrides=overrides)
        batch = packed_batch(ckpt, rows)
        variants = [("trained", ckpt.model)]
        if args.fresh:
            variants.append(("fresh", fresh_model(ckpt_path, torch.bfloat16)))
        for init, bf16_model in variants:
            for dtype in args.dtype:
                model = bf16_model.float() if dtype == "fp32" else bf16_model.to(torch.bfloat16)
                for eps in args.eps:
                    metrics = probe_once(model, ckpt.carry, batch, eps, args.seed)
                    summary = derive_summaries(metrics)
                    for key, value in metrics.items():
                        if (m := STEP_RE.match(key)) is not None:
                            out.append(dict(run=run, init=init, dtype=dtype, eps=eps, L_override=args.L_cycles or 0,
                                            role=m["role"], idx=int(m["idx"]), stat=m["stat"], value=value))
                    print(f"{run:40s} {init:7s} {dtype} eps={eps:<5} "
                          f"gain L={summary.get('probe/summary/L/gain_mean', float('nan')):.4f} "
                          f"H={summary.get('probe/summary/H/gain_mean', float('nan')):.4f}  "
                          f"prenorm L={summary.get('probe/summary/L/prenorm_rms_mean', float('nan')):.1f} "
                          f"H={summary.get('probe/summary/H/prenorm_rms_mean', float('nan')):.1f}", flush=True)
                bf16_model.to(torch.bfloat16)
        del ckpt, variants
        torch.cuda.empty_cache()

    if args.csv and out:
        df = pd.DataFrame(out)
        key = ["run", "init", "dtype", "eps", "L_override"]
        if os.path.exists(args.csv):
            old = pd.read_csv(args.csv)
            old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(args.csv, index=False)
        print(f"wrote {args.csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
