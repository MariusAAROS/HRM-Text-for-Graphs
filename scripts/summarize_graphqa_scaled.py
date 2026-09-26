"""Summarize the scaled GraphQA runs (scripts/run_graphqa_scaled.sh).

Reads every ckpts/<run>/eval_<set>_<split>_ep<e>.json and joins the per-sample
results with the raw GraphQA JSON (same row order as the eval JSONL) to report
accuracy overall, excluding MaximumFlow, and per task.

"old" is the split behind report/figures/ablation_hl_grid.png, whose MaximumFlow
questions omit edge capacities (unanswerable); "fixed" is the same seed-42 split
regenerated with capacities, so its MaximumFlow column is meaningful.

Usage:
    python scripts/summarize_graphqa_scaled.py [--root /work/dfm/marius-ortega/graphqa_scaled] [--per-task]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

RAW = {
    "old": "/work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM/data/graphqa/baseline/standard",
    "fixed": "{root}/raw/baseline-fixed/standard",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/dfm/marius-ortega/graphqa_scaled")
    ap.add_argument("--per-task", action="store_true", help="Also print per-task accuracy.")
    args = ap.parse_args()

    tasks_cache = {}

    def tasks_for(set_name, split):
        key = (set_name, split)
        if key not in tasks_cache:
            with open(os.path.join(RAW[set_name].format(root=args.root), f"{split}.json")) as f:
                tasks_cache[key] = [r["task"] for r in json.load(f)]
        return tasks_cache[key]

    pattern = re.compile(r"eval_(\w+?)_(val|test)_ep(\d+)\.json$")
    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, "ckpts", "*", "eval_*_ep*.json"))):
        m = pattern.search(os.path.basename(path))
        if not m:
            continue
        set_name, split, epoch = m.group(1), m.group(2), int(m.group(3))
        with open(path) as f:
            report = json.load(f)
        tasks = tasks_for(set_name, split)
        per_task = defaultdict(lambda: [0, 0])
        for s in report["samples"]:
            t = tasks[s["id"]]
            per_task[t][0] += int(s["correct"])
            per_task[t][1] += 1
        no_mf = [v for t, v in per_task.items() if t != "MaximumFlow"]
        acc_no_mf = sum(c for c, _ in no_mf) / max(1, sum(n for _, n in no_mf))
        rows.append((os.path.basename(os.path.dirname(path)), set_name, split, epoch,
                     report["accuracy"], acc_no_mf, dict(per_task)))

    if not rows:
        print(f"No eval reports under {args.root}/ckpts")
        return

    print(f"{'run':28s} {'set':6s} {'split':5s} {'ep':>2s} {'acc':>6s} {'acc-MF':>6s}")
    for run, set_name, split, epoch, acc, acc_no_mf, _ in rows:
        print(f"{run:28s} {set_name:6s} {split:5s} {epoch:>2d} {acc:6.3f} {acc_no_mf:6.3f}")

    if args.per_task:
        task_names = sorted({t for r in rows for t in r[6]})
        print()
        print(f"{'run':28s} {'set':6s} {'split':5s} {'ep':>2s} " + " ".join(f"{t[:8]:>8s}" for t in task_names))
        for run, set_name, split, epoch, _, _, per_task in rows:
            cells = [f"{per_task[t][0] / per_task[t][1]:8.2f}" if t in per_task else f"{'-':>8s}" for t in task_names]
            print(f"{run:28s} {set_name:6s} {split:5s} {epoch:>2d} " + " ".join(cells))


if __name__ == "__main__":
    main()
