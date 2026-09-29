"""Log finished scaled-GraphQA eval reports (ckpts/<run>/eval_<set>_<split>_ep<e>.json) to wandb.

One wandb run per report, with the same keys eval_graphqa.py --wandb_project writes
(config H_cycles / L_cycles / ..., metrics eval/accuracy, eval/f1, ..., eval/samples table),
so the report notebooks' CSV-export workflow applies unchanged. Adds eval/acc_no_maxflow,
per-task eval/task/<Task> accuracies, and config fields model / eval_set / split / epoch.

Reports are re-used from disk (no GPU); a `<report>.wandb` marker next to each JSON
prevents logging it twice. Called by scripts/run_graphqa_scaled.sh after each eval.

Usage:
    python scripts/log_graphqa_evals_to_wandb.py [--root ...] [--project HRM-GraphQA-Scaled-eval] [--run <run name>]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import wandb

RAW = {
    "old": "/work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM/data/graphqa/baseline/standard",
    "fixed": "{root}/raw/baseline-fixed/standard",
}
PATTERN = re.compile(r"eval_(\w+?)_(val|test)_ep(\d+)\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/dfm/marius-ortega/graphqa_scaled")
    ap.add_argument("--project", default="HRM-GraphQA-Scaled-eval")
    ap.add_argument("--run", default=None, help="Only log reports of this run (default: all runs).")
    ap.add_argument("--tag", default="scaled", help="Experiment tag added to every logged run.")
    args = ap.parse_args()

    run_glob = args.run or "*"
    for path in sorted(glob.glob(os.path.join(args.root, "ckpts", run_glob, "eval_*_ep*.json"))):
        m = PATTERN.search(os.path.basename(path))
        if not m or os.path.exists(path + ".wandb"):
            continue
        set_name, split, epoch = m.group(1), m.group(2), int(m.group(3))
        run = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            report = json.load(f)
        # "<set>_Lov<k>" reports (L_cycles overridden at inference) share <set>'s questions. Sets without
        # GraphQA task labels (e.g. pointer chasing) are logged without the per-task breakdown.
        raw = RAW.get(set_name.split("_Lov")[0])
        if raw is not None:
            with open(os.path.join(raw.format(root=args.root), f"{split}.json")) as f:
                tasks = [r["task"] for r in json.load(f)]
        else:
            tasks = ["all"] * len(report["samples"])

        per_task = defaultdict(lambda: [0, 0])
        for s in report["samples"]:
            per_task[tasks[s["id"]]][0] += int(s["correct"])
            per_task[tasks[s["id"]]][1] += 1
        no_mf = [v for t, v in per_task.items() if t != "MaximumFlow"]

        H, L = report["H_cycles"], report["L_cycles"]
        model = "trm" if "_trm_" in run else "hrm"
        backprop = "full" if "_fullbp" in run else "truncated"
        wandb.init(
            project=args.project,
            name=f"{run}_{set_name}_{split}_ep{epoch}",
            tags=[f"H{H}", f"L{L}", model, set_name, split, f"ep{epoch}", backprop, args.tag],
            config={
                "ckpt_path": report["ckpt_path"],
                "ckpt_epoch": epoch,
                "data": report["data"],
                "use_ema": report["use_ema"],
                "H_cycles": H,
                "L_cycles": L,
                "ratio_L_over_H": report["ratio_L_over_H"],
                "model": model,
                "backprop": backprop,
                "train_run": run,
                "eval_set": set_name,
                "split": split,
                "epoch": epoch,
            },
        )
        wandb.log({
            "eval/accuracy": report["accuracy"], "eval/correct": report["correct"], "eval/n": report["n"],
            "eval/precision": report["precision"], "eval/recall": report["recall"], "eval/f1": report["f1"],
            "eval/hits_at_1": report["hits_at_1"],
            "eval/acc_no_maxflow": sum(c for c, _ in no_mf) / max(1, sum(n for _, n in no_mf)),
            **{f"eval/task/{t}": c / n for t, (c, n) in sorted(per_task.items())},
        })
        table = wandb.Table(columns=["id", "task", "gold", "pred", "correct", "precision", "recall", "f1", "hits_at_1"])
        for s in report["samples"]:
            table.add_data(s["id"], tasks[s["id"]], s["gold"], s["pred"], s["correct"],
                           s["precision"], s["recall"], s["f1"], s["hits_at_1"])
        wandb.log({"eval/samples": table})
        wandb.finish()

        open(path + ".wandb", "w").close()
        print(f"logged {run} {set_name}/{split} ep{epoch}: acc={report['accuracy']:.3f}")


if __name__ == "__main__":
    main()
