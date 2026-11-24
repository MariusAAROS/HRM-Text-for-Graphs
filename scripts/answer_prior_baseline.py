"""Per-task answer-prior baselines on GraphQA, and how far each trained model is from them.

For every task: the most frequent training answer (the "prior"), and the accuracy of always
giving it on val / test. For every evaluated model (from-scratch eval JSONs of this repo, and
optionally prediction CSVs of the fine-tuning repo): per-task accuracy, number of distinct
predictions, share of the most common prediction, share of predictions equal to the prior.
A model whose accuracy matches the prior and whose predictions are mostly the prior answer has
learned the label distribution of that task, not the graph.

Task labels come from Graph-Representation-Learning-for-LLM's data/graphqa/baseline/standard,
whose rows align one-to-one with data/graphqa/hrm-text/standard. An eval file is used only if
its gold answers match that split row for row (the capacity-annotated "fixed" split does not).

    python scripts/answer_prior_baseline.py [--grl_predictions '<glob>'] [--no_wandb]
"""
import argparse
import glob
import json
import os
from collections import Counter

import pandas as pd

GRL = "/work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM"
EVAL_GLOBS = [
    "/work/dfm/marius-ortega/graphqa_fullbp/ckpts/*/eval_*.json",
    "/work/dfm/marius-ortega/recursion_hyp/ckpts/*/eval_*.json",
    "/work/dfm/marius-ortega/graphqa_scaled/ckpts/*/eval_*.json",
]
OUT_DIR = "data/results"


def norm(answer):
    return str(answer).strip().lower().rstrip(".").strip()


def load_split(split):
    with open(os.path.join(GRL, "data/graphqa/baseline/standard", f"{split}.json")) as f:
        return pd.DataFrame([{"task": r["task"], "gold": norm(r["answer"])} for r in json.load(f)])


def prediction_stats(df, prior):
    """df: task, pred (normalized), correct (bool). One row per task."""
    rows = []
    for task, sub in df.groupby("task"):
        counts = Counter(sub["pred"])
        rows.append(dict(
            task=task, n=len(sub), acc=sub["correct"].mean(),
            n_unique_pred=len(counts), top_pred_share=counts.most_common(1)[0][1] / len(sub),
            share_eq_prior=(sub["pred"] == prior[task]).mean(),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grl_predictions", default=None,
                    help="glob of GRL predictions_epoch_*.csv files (task,pred,label columns)")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--project", default="HRM-Pretrained-Recursion")
    args = ap.parse_args()

    splits = {s: load_split(s) for s in ("train", "val", "test")}
    prior = splits["train"].groupby("task")["gold"].agg(lambda s: s.value_counts().index[0]).to_dict()

    baseline_rows = []
    for split in ("val", "test"):
        df = splits[split]
        for task, sub in df.groupby("task"):
            baseline_rows.append(dict(split=split, task=task, n=len(sub), prior_answer=prior[task],
                                      prior_acc=(sub["gold"] == prior[task]).mean(),
                                      majority_acc=sub["gold"].value_counts().iloc[0] / len(sub),
                                      n_unique_gold=sub["gold"].nunique()))
    baseline = pd.DataFrame(baseline_rows)

    run_rows, skipped = [], []
    for path in sorted(p for g in EVAL_GLOBS for p in glob.glob(g)):
        with open(path) as f:
            report = json.load(f)
        split = os.path.basename(report["data"]).removesuffix(".jsonl")
        if split not in ("val", "test"):
            continue
        samples = sorted(report["samples"], key=lambda s: s["id"])
        gold = splits[split]
        if len(samples) != len(gold) or any(norm(s["gold"]) != g for s, g in zip(samples, gold["gold"])):
            skipped.append(path)
            continue
        df = gold.assign(pred=[norm(s["pred"]) for s in samples], correct=[bool(s["correct"]) for s in samples])
        stats = prediction_stats(df, prior).assign(
            source="from-scratch", run=os.path.basename(os.path.dirname(path)),
            file=os.path.basename(path), split=split, H=report.get("H_cycles"), L=report.get("L_cycles"))
        run_rows.append(stats)

    if args.grl_predictions:
        for path in sorted(glob.glob(args.grl_predictions)):
            df = pd.read_csv(path, keep_default_na=False)
            df = df.assign(pred=df["pred"].map(norm), correct=df["pred"].map(norm) == df["label"].map(norm))
            stats = prediction_stats(df, prior).assign(
                source="grl", run=os.path.basename(os.path.dirname(path)), file=os.path.basename(path),
                split="val", H=None, L=None)
            run_rows.append(stats)

    runs = pd.concat(run_rows, ignore_index=True)
    runs = runs.merge(baseline[["split", "task", "prior_acc"]], on=["split", "task"], how="left")
    runs["acc_minus_prior"] = runs["acc"] - runs["prior_acc"]

    # A run is "diverged" when its overall accuracy is below 0.1 (the H >= 4 constant-output runs).
    overall = runs.assign(hits=runs["acc"] * runs["n"]).groupby(["source", "run", "file"])[["hits", "n"]].sum()
    diverged = overall.index[(overall["hits"] / overall["n"]) < 0.1]
    runs["diverged"] = runs.set_index(["source", "run", "file"]).index.isin(diverged)

    os.makedirs(OUT_DIR, exist_ok=True)
    baseline.to_csv(os.path.join(OUT_DIR, "answer-prior-baseline.csv"), index=False)
    runs.to_csv(os.path.join(OUT_DIR, "answer-prior-runs.csv"), index=False)

    # Console summary: test split, non-diverged from-scratch runs pooled, at their trained
    # depth (the eval_*Lov1* files re-evaluate a run with L overridden to 1).
    test = runs[(runs["split"] == "test") & (runs["source"] == "from-scratch") & ~runs["diverged"]
                & ~runs["file"].str.contains("Lov")]
    summary = test.groupby("task").agg(acc=("acc", "mean"), share_eq_prior=("share_eq_prior", "mean"),
                                       top_pred_share=("top_pred_share", "mean"),
                                       n_unique_pred=("n_unique_pred", "mean"))
    summary = baseline[baseline["split"] == "test"].set_index("task")[["prior_answer", "prior_acc", "n_unique_gold"]] \
        .join(summary).sort_values("acc", ascending=False)
    pd.set_option("display.width", 200)
    print(f"{test['run'].nunique()} non-diverged from-scratch runs on test "
          f"({len(diverged)} diverged files excluded, {len(skipped)} files skipped: split mismatch)")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.project, name="a-answer-prior", group="a", reinit=True,
                         config={"exp": "a", "n_eval_files": len(runs[["run", "file"]].drop_duplicates()),
                                 "n_skipped": len(skipped)})
        run.log({"prior_baseline": wandb.Table(dataframe=baseline),
                 "runs_per_task": wandb.Table(dataframe=runs.astype({"H": str, "L": str})),
                 "summary_test_from_scratch": wandb.Table(dataframe=summary.reset_index())})
        run.finish()


if __name__ == "__main__":
    main()
