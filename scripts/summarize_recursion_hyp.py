"""Summarize the GraphQA recursion-hypothesis runs (scripts/run_recursion_hyp_local.sh) into one CSV.

Reads ckpts/<run>/eval_old[_Lov1]_<split>_ep<E>.json from two roots:
  recursion_hyp   graphqa_<arm>_H<H>_L<L>_s<seed>        arms peri, post, peri-2l, trm-matched
  graphqa_fullbp  graphqa_H<H>_L<L>_fullbp_local_s<seed>  the pre-norm reference arm ("pre")
and writes one row per (run, split, L at inference). "Lov1" evals rerun a trained model with
L_cycles = 1: if accuracy does not drop, the model does not use its L steps.

Usage:
    python scripts/summarize_recursion_hyp.py [--csv data/results/recursion-hyp-local.csv]
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

ARM_RE = re.compile(r"graphqa_(?P<arm>[a-z0-9-]+)_H(?P<H>\d+)_L(?P<L>\d+)_s(?P<seed>\d+)$")
PRE_RE = re.compile(r"graphqa_H(?P<H>\d+)_L(?P<L>\d+)_fullbp_local_s(?P<seed>\d+)$")
EVAL_RE = re.compile(r"eval_old(?P<lov>_Lov1)?_(?P<split>val|test)_ep\d+\.json$")
PRE_CELLS = {(1, 1), (1, 6), (1, 24), (3, 6)}  # the pre-norm references quoted against the new arms
N_LAYERS = {"trm-matched": 2, "peri-2l": 2}     # layers per module; 6 otherwise (size B, half_layers)


def load(root: str, name_re: re.Pattern, arm: str | None) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "ckpts", "*", "eval_old*_ep*.json"))):
        run = os.path.basename(os.path.dirname(path))
        m, e = name_re.match(run), EVAL_RE.search(os.path.basename(path))
        if not m or not e:
            continue
        H, L = int(m["H"]), int(m["L"])
        if arm == "pre" and (H, L) not in PRE_CELLS:
            continue
        run_arm = arm or m["arm"]
        with open(path) as f:
            report = json.load(f)
        rows.append(dict(run=run, arm=run_arm, n_layers=N_LAYERS.get(run_arm, 6), H=H, L=L, seed=int(m["seed"]),
                         split=e["split"], L_eval=1 if e["lov"] else L,
                         accuracy=report["accuracy"], correct=report["correct"], n=report["n"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp_root", default="/work/dfm/marius-ortega/recursion_hyp")
    ap.add_argument("--pre_root", default="/work/dfm/marius-ortega/graphqa_fullbp")
    ap.add_argument("--csv", default="data/results/recursion-hyp-local.csv")
    args = ap.parse_args()

    df = pd.DataFrame(load(args.pre_root, PRE_RE, "pre") + load(args.hyp_root, ARM_RE, None))
    if df.empty:
        raise SystemExit("no eval reports found")
    df = df.sort_values(["arm", "H", "L", "seed", "L_eval", "split"])
    df.to_csv(args.csv, index=False)
    print(f"wrote {args.csv} ({len(df)} rows)")

    pooled = df.groupby(["arm", "H", "L", "L_eval"])[["correct", "n"]].sum()  # val + test, all seeds
    print((pooled["correct"] / pooled["n"]).unstack("L_eval").round(3).to_string())


if __name__ == "__main__":
    main()
