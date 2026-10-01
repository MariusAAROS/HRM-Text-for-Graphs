"""Export training curves of W&B runs to a tidy CSV (read-only).

Used for the H >= 4 collapse of the RQ2.1 grid: do those runs diverge (loss / grad spikes) or
train normally to a high plateau? The Jean Zay grid predates the recursion probe, so only the
train/* scalars exist there; local runs also carry probe/* keys (pass them with --keys).

Usage:
    python scripts/pull_wandb_curves.py --project HRM-GraphQA-Ablation \
        --runs graphqa_H1_L6 graphqa_H4_L3 graphqa_H4_L6_fullbp ... \
        --csv data/results/recursion-hyp-h4-curves.csv
"""
import argparse

import pandas as pd
import wandb

DEFAULT_KEYS = ["train/loss", "train/accuracy", "train/exact_accuracy", "train/lr", "bp_steps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--entity", default=None)
    ap.add_argument("--runs", nargs="+", required=True, help="W&B run names.")
    ap.add_argument("--keys", nargs="*", default=DEFAULT_KEYS)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    api = wandb.Api()
    wanted = set(args.runs)
    runs = [r for r in api.runs(f"{args.entity or api.default_entity}/{args.project}") if r.name in wanted]
    missing = wanted - {r.name for r in runs}
    if missing:
        print(f"not found in {args.project}: {sorted(missing)}")

    frames = []
    for run in runs:
        rows = [{k: r.get(k) for k in ["_step"] + args.keys} for r in run.scan_history()]
        df = pd.DataFrame(rows).dropna(subset=[k for k in args.keys if k.startswith("train/")][:1])
        df.insert(0, "run", run.name)
        df.insert(1, "project", args.project)
        frames.append(df)
        print(f"{run.name:28s} {len(df):5d} rows, final loss {df['train/loss'].iloc[-1] if 'train/loss' in df else float('nan'):.4f}")

    out = pd.concat(frames, ignore_index=True).rename(columns={"_step": "step"})
    out.to_csv(args.csv, index=False)
    print(f"wrote {args.csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
