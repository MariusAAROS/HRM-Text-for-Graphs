"""Exact-match evaluation of a trained HRM-Text checkpoint on GraphQA.

GraphQA answers are short and deterministic (e.g. "Yes.", "No.", a node id,
a small list), so greedy decoding + normalized exact match is the natural
metric. There is no in-training eval loop in pretrain.py, so this is run
separately (see slurm/eval_graphqa.slurm).

The recursion depth (H_cycles / L_cycles) is read from the checkpoint's
all_config.yaml, so it automatically matches how the model was trained --
nothing to pass on the CLI.

Usage:
    python scripts/eval_graphqa.py \
        --ckpt_path $SCRATCH/graphqa/ckpts/graphqa_H2_L3 \
        --data data/graphqa/hrm-text/standard/test.jsonl \
        --use_ema \
        --out $SCRATCH/graphqa/ckpts/graphqa_H2_L3/eval_test.json

    # With wandb logging for cross-run comparison:
    python scripts/eval_graphqa.py \
        --ckpt_path $SCRATCH/graphqa/ckpts/graphqa_H2_L3 \
        --data data/graphqa/hrm-text/standard/test.jsonl \
        --use_ema \
        --wandb_project graphqa-eval \
        --out $SCRATCH/graphqa/ckpts/graphqa_H2_L3/eval_test.json
"""
import argparse
import json
import os
import re
import sys
from typing import Optional

import numpy as np
import torch
import wandb
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path

from models.layers import find_multiple
from simple_inference_engine import inference_load_checkpoint


def normalize(text: str) -> str:
    """Whitespace/case/trailing-punctuation-insensitive normalization."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .")


def read_config_cycles(ckpt_path: str) -> tuple[Optional[int], Optional[int]]:
    cfg_file = os.path.join(ckpt_path, "all_config.yaml")
    if not os.path.exists(cfg_file):
        return None, None
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    arch = cfg.get("arch", {})
    return arch.get("H_cycles"), arch.get("L_cycles")


@torch.inference_mode()
def generate(ckpt, condition: str, prompt: str, max_generation: int, pad_multiple: int = 128) -> str:
    """Greedy, cache-free generation.

    Rebuilds a single-document packed batch each step (prompt = prefix, generated
    tokens = causal) and reads the next-token logits. Avoids the KV-cache path so
    everything runs through the FlexAttention training forward. GraphQA answers are
    short, so the O(n^2) re-encode cost is negligible.
    """
    prompt_ids = ckpt.tokenize_prompt(condition, prompt)  # boq + condition + instruction + eoq
    prompt_len = int(prompt_ids.shape[0])
    stop_id = int(ckpt.tokenizer.convert_tokens_to_ids(ckpt.tokenizer_info["eoa"]))

    gen: list[int] = []
    for _ in range(max_generation):
        seq = np.concatenate([prompt_ids, np.asarray(gen, dtype=prompt_ids.dtype)]) if gen else prompt_ids
        total = int(seq.shape[0])
        S = find_multiple(total, pad_multiple)

        inputs = np.zeros(S, dtype=np.int64);       inputs[:total] = seq
        position_ids = np.zeros(S, dtype=np.int64); position_ids[:total] = np.arange(total)
        doc_ids = np.full(S, -1, dtype=np.int32);   doc_ids[:total] = 0
        prefix_ends = np.zeros(S, dtype=np.int32);  prefix_ends[:total] = prompt_len

        batch = {k: torch.as_tensor(v, device="cuda")
                 for k, v in {"inputs": inputs, "position_ids": position_ids,
                              "doc_ids": doc_ids, "prefix_ends": prefix_ends}.items()}
        _carry, logits = ckpt.model(ckpt.carry, batch)
        next_id = int(logits[total - 1].argmax(-1).item())
        if next_id == stop_id:
            break
        gen.append(next_id)

    return ckpt.decode_generation(np.asarray(gen, dtype=np.int64), stop_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="Directory with fsdp2_epoch_* + all_config.yaml.")
    ap.add_argument("--data", required=True, help="GraphQA JSONL (instruction/response/condition).")
    ap.add_argument("--ckpt_epoch", type=int, default=None, help="Epoch to load (default: latest).")
    ap.add_argument("--use_ema", action="store_true", help="Use EMA weights (recommended).")
    ap.add_argument("--max_generation", type=int, default=32, help="Max new tokens for the answer.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of eval samples.")
    ap.add_argument("--out", default=None, help="Where to write the JSON report.")
    # wandb (opt-in: only active when --wandb_project is set)
    ap.add_argument("--wandb_project", default=None, help="W&B project name. Enables wandb logging when set.")
    ap.add_argument("--wandb_entity", default=None, help="W&B entity (team or user). Uses default if omitted.")
    ap.add_argument("--wandb_name", default=None, help="W&B run name. Auto-generated from ckpt if omitted.")
    ap.add_argument("--wandb_tags", nargs="*", default=None, help="Optional W&B tags (e.g. --wandb_tags ablation ema).")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} eval samples from {args.data}")

    ckpt = inference_load_checkpoint(args.ckpt_path, args.ckpt_epoch, args.use_ema)
    H_cycles, L_cycles = read_config_cycles(args.ckpt_path)

    # ---- wandb init (opt-in) -----------------------------------------------
    use_wandb = args.wandb_project is not None
    if use_wandb:
        run_name = args.wandb_name or os.path.basename(os.path.normpath(args.ckpt_path))
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.wandb_tags,
            config={
                "ckpt_path": args.ckpt_path,
                "ckpt_epoch": args.ckpt_epoch,
                "data": args.data,
                "use_ema": args.use_ema,
                "max_generation": args.max_generation,
                "limit": args.limit,
                "H_cycles": H_cycles,
                "L_cycles": L_cycles,
                "ratio_L_over_H": (L_cycles / H_cycles) if (H_cycles and L_cycles) else None,
            },
        )

    results = []
    correct = 0
    for idx, r in enumerate(rows):
        gold = r["response"]
        pred = generate(ckpt, r.get("condition", "direct"), r["instruction"], args.max_generation)
        is_correct = normalize(pred) == normalize(gold)
        correct += int(is_correct)
        results.append({"id": idx, "gold": gold, "pred": pred, "correct": is_correct})

    accuracy = correct / len(rows) if rows else 0.0
    report = {
        "ckpt_path": args.ckpt_path,
        "data": args.data,
        "H_cycles": H_cycles,
        "L_cycles": L_cycles,
        "ratio_L_over_H": (L_cycles / H_cycles) if (H_cycles and L_cycles) else None,
        "n": len(rows),
        "correct": correct,
        "accuracy": accuracy,
        "use_ema": args.use_ema,
        "samples": results,
    }

    print(f"H_cycles={H_cycles} L_cycles={L_cycles}  "
          f"accuracy = {accuracy:.4f} ({correct}/{len(rows)})")

    # ---- wandb log ----------------------------------------------------------
    if use_wandb:
        wandb.log({"eval/accuracy": accuracy, "eval/correct": correct, "eval/n": len(rows)})

        # Per-sample table for drill-down in the wandb UI
        table = wandb.Table(columns=["id", "gold", "pred", "correct"])
        for r in results:
            table.add_data(r["id"], r["gold"], r["pred"], r["correct"])
        wandb.log({"eval/samples": table})
        wandb.finish()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
