"""Export the W&B project of the GraphQA-v2 multi-seed reruns to one flat CSV for report/paper_results.ipynb.

    python scripts/pull_v2_runs.py [--project HRM-Graph-v2] [--out data/results/v2_seeds/runs.csv]

One row per finished run (training and eval): name, group, job_type, state, created, runtime_s, the scalar
config keys (exp, backbone, regime, encoding, held_out_task, seed, kgqa_dataset, clip, H, L, init, backprop,
split, train_run, ...) and every numeric summary metric (val/exact_match, val/exact_match_<Task>,
eval/accuracy, eval/task/<Task>, epoch, ...). The notebook groups by the experiment keys and aggregates seeds.
Producers: GRL scripts/v2/jobs.py (runner.py / depth_runner.py) and scripts/run_recursion_hyp_v2.sh.
"""
import argparse
import os

import pandas as pd
import wandb

ENTITY = "m-ortega-p-le-l-onard-de-vinci"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="HRM-Graph-v2")
    ap.add_argument("--out", default="data/results/v2_seeds/runs.csv")
    ap.add_argument("--all-states", action="store_true", help="Keep running/crashed runs too.")
    args = ap.parse_args()

    api = wandb.Api(timeout=180)
    rows = []
    for run in api.runs(f"{ENTITY}/{args.project}", per_page=500):
        if run.state != "finished" and not args.all_states:
            continue
        row = {"name": run.name, "id": run.id, "group": run.group, "job_type": run.job_type, "state": run.state,
               "created": run.created_at, "runtime_s": run.summary.get("_runtime")}
        for k, v in run.config.items():
            if isinstance(v, (int, float, str, bool)) and not k.startswith("_"):
                row[k] = v
        values = run.summary._json_dict
        if not any(k in values for k in ("val/exact_match", "eval/accuracy")):
            # depth_runner.py evals often end with an unflushed summary: take the logged history
            values = {}
            for h in run.scan_history():
                values.update({k: v for k, v in h.items() if v is not None})
        for k, v in values.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.startswith("_"):
                row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("created")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} runs -> {args.out}")
    if "exp" in df:
        print(df.groupby(["exp", "job_type"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
