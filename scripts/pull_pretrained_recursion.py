"""Export the pretrained-recursion runs (W&B project HRM-Pretrained-Recursion) to data/results.

Experiments (run by Graph-Representation-Learning-for-LLM, branch cross-repo, see
HANDOFF_pretrained_recursion.md):
  b       fine-tuned HRM-Text-1B checkpoints (trained at H2L3) evaluated at other (H, L)
  c       gain probe of the recurrent modules
  d       GraphQA fine-tune of the pretrained 1B at another (H, L)
  e       GraphQA H2L3 fine-tune continued at another (H, L)
  d-test / e-test   test exact match of the d / e checkpoints

Writes:
  pretrained-recursion-eval.csv    one row per evaluation run (b, d-test, e-test), all val/* metrics
                                   (the depth runner logs test-split metrics under val/*)
  pretrained-recursion-probe.csv   one row per (probe run, module, recursion step), copied from
                                   the probes' shared CSV
  pretrained-recursion-train.csv   validation curves of the d / e fine-tunes

    python scripts/pull_pretrained_recursion.py
"""
import ast
import os

import pandas as pd
import wandb

PROJECT = "m-ortega-p-le-l-onard-de-vinci/HRM-Pretrained-Recursion"
OUT_DIR = "data/results"
PROBE_CSV = "/work/dfm/marius-ortega/pretrained_recursion/results/probe_hf.csv"


def dataset_name(config):
    """Lightning logs the whole dataset config group over the flat `dataset` key."""
    dataset = config.get("dataset")
    if isinstance(dataset, str) and dataset.startswith("{"):
        dataset = ast.literal_eval(dataset)
    return dataset["name"] if isinstance(dataset, dict) else dataset


def source_of(config):
    """The fine-tune a run evaluates: the dataset for b, the d / e job name otherwise."""
    ckpt = config.get("source_ckpt", "")
    if "/pretrained_recursion/ckpts/" in ckpt:
        return os.path.basename(os.path.dirname(ckpt))
    return dataset_name(config)


def main():
    api = wandb.Api(timeout=120)
    runs = [r for r in api.runs(PROJECT) if r.state == "finished"]

    eval_rows, curve_rows = [], []
    for run in runs:
        config, summary = run.config, run.summary._json_dict
        if run.group in ("b", "d-test", "e-test"):
            values = summary
            if "val/exact_match" not in values:  # summary not flushed: take the logged history
                values = {}
                for row in run.scan_history():
                    values.update({k: v for k, v in row.items() if v is not None})
            metrics = {k.removeprefix("val/"): v for k, v in values.items()
                       if k.startswith("val/") and isinstance(v, (int, float))}
            eval_rows.append(dict(exp=run.group, run=run.name, source=source_of(config),
                                  dataset=dataset_name(config), H=config["H"], L=config["L"],
                                  trained_H=config["trained_H"], trained_L=config["trained_L"],
                                  **metrics))
        elif run.group in ("d", "e"):
            for row in run.scan_history():
                if row.get("val/exact_match") is None:
                    continue
                curve_rows.append(dict(exp=run.group, run=run.name, H=config["H"], L=config["L"],
                                       step=row.get("trainer/global_step"), epoch=row.get("epoch"),
                                       exact_match=row["val/exact_match"], loss=row.get("val/loss")))

    os.makedirs(OUT_DIR, exist_ok=True)
    # Probe rows are written by every probe run to the shared CSV; W&B holds the same tables.
    probe = pd.read_csv(PROBE_CSV)
    probe.to_csv(os.path.join(OUT_DIR, "pretrained-recursion-probe.csv"), index=False)
    print(f"probe: {len(probe)} rows")
    for name, rows in (("eval", eval_rows), ("train", curve_rows)):
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_DIR, f"pretrained-recursion-{name}.csv"), index=False)
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
