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
"""
import argparse
import json
import os
import re
from typing import Optional

import yaml

from simple_inference_engine import inference_load_checkpoint, inference_generate


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="Directory with fsdp2_epoch_* + all_config.yaml.")
    ap.add_argument("--data", required=True, help="GraphQA JSONL (instruction/response/condition).")
    ap.add_argument("--ckpt_epoch", type=int, default=None, help="Epoch to load (default: latest).")
    ap.add_argument("--use_ema", action="store_true", help="Use EMA weights (recommended).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=2048, help="KV-cache length (prompt + generation).")
    ap.add_argument("--max_generation", type=int, default=32, help="Max new tokens for the answer.")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (deterministic).")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of eval samples.")
    ap.add_argument("--out", default=None, help="Where to write the JSON report.")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} eval samples from {args.data}")

    ckpt = inference_load_checkpoint(args.ckpt_path, args.ckpt_epoch, args.use_ema)

    def prompt_iter():
        for idx, r in enumerate(rows):
            condition = r.get("condition", "direct")
            yield idx, (condition, r["instruction"])

    predictions: dict[int, str] = {}
    for pid, text in inference_generate(
        ckpt,
        prompt_iter(),
        max_tokens=args.max_tokens,
        max_generation=args.max_generation,
        batch_size=args.batch_size,
        temp=args.temp,
    ):
        predictions[pid] = text

    H_cycles, L_cycles = read_config_cycles(args.ckpt_path)

    results = []
    correct = 0
    for idx, r in enumerate(rows):
        gold = r["response"]
        pred = predictions.get(idx, "")
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

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
