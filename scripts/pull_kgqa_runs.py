"""Add W&B runs to the KGQA result exports (data/results/metaqa-variants.csv, kqa-pro.csv).

Those CSVs are W&B UI exports (project Meta-ICL): one row per run, nested config groups as dict
reprs, summary metrics as columns. This script appends runs in the same format, keeping only the
columns the export already has. A run already in the file (same Name and Created) is replaced.

    python scripts/pull_kgqa_runs.py --csv data/results/metaqa-variants.csv --runs h6tkcksa rk56bxwb
"""
import argparse
from datetime import datetime, timezone

import pandas as pd
import wandb

PROJECT = "m-ortega-p-le-l-onard-de-vinci/Meta-ICL"


def run_row(run, columns):
    created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    row = {"Name": run.name, "State": run.state, "Created": created.strftime("%Y-%m-%dT%H:%M:%S.000Z")}
    row.update({k: str(v) if isinstance(v, (dict, list)) else v for k, v in run.config.items()})
    row.update({k: v for k, v in run.summary._json_dict.items() if isinstance(v, (int, float, str))})
    return {c: row.get(c) for c in columns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="W&B run ids")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    api = wandb.Api(timeout=120)
    rows = []
    for run_id in args.runs:
        run = api.run(f"{PROJECT}/{run_id}")
        assert run.state == "finished", f"{run_id} is {run.state}"
        row = run_row(run, df.columns)
        rows.append(row)
        print(f"{run_id} {run.name}: val/exact_match={row.get('val/exact_match')}")
    new = pd.DataFrame(rows, columns=df.columns)
    df = df[~df.set_index(["Name", "Created"]).index.isin(new.set_index(["Name", "Created"]).index)]
    pd.concat([df, new], ignore_index=True).to_csv(args.csv, index=False)
    print(f"wrote {args.csv}: {len(df) + len(new)} rows")


if __name__ == "__main__":
    main()
