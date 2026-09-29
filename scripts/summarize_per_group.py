"""Per-group accuracy (GraphQA task, pointer-chasing k, ...) of the local recursion runs.

Joins every ckpts/<run>/eval_<set>_<split>_ep<E>.json with per-sample labels (eval `id` = row
index into the eval file, as in scripts/summarize_graphqa_scaled.py) and writes a tidy CSV with
one row per (run, eval set, split, group). Run-name fields (H, L, regime, variant, seed, ...) come
from the named groups of --name-re. The splits are kept separate in the CSV so the reader can
pool val + test (770 GraphQA samples) and keep the binomial SE honest.

Labels:
  --labels <dir>  with {val,test}.json (a list of dicts) or {val,test}.jsonl, and --field the key
                  to group by; --field none groups everything into "all".
  --per-query     score each comma-separated answer of a multi-query response separately
                  (pointer chasing v2: "3, 11, 7, 0."), by position; `correct`/`n` then count queries.

Usage:
    # GraphQA per task, full-BP grid
    python scripts/summarize_per_group.py --root /work/dfm/marius-ortega/graphqa_fullbp \
        --labels /work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM/data/graphqa/baseline/standard \
        --field task --csv data/results/graphqa-fullbp-local-per-group.csv
    # Pointer chasing per k
    python scripts/summarize_per_group.py --root /work/dfm/marius-ortega/recursion_hyp --set ptr ptr_Lov1 \
        --labels data/pointer/hrm-text/standard --field k --name-re "$HYP_RE" --csv data/results/recursion-hyp-per-k.csv
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import pandas as pd

FULLBP_RE = r"graphqa_H(?P<H>\d+)_L(?P<L>\d+)_(?P<regime>fullbp|trunc)_local_s(?P<seed>\d+)$"
HYP_RE = r"(?P<task>graphqa|pointer2?)_(?P<variant>[\w-]+?)_H(?P<H>\d+)_L(?P<L>\d+)_s(?P<seed>\d+)$"
REPORT_RE = re.compile(r"eval_(?P<set>\w+?)_(?P<split>val|test)_ep(?P<epoch>\d+)\.json$")


def split_answers(text: str) -> list[str]:
    return [a.strip() for a in text.strip().rstrip(" .").split(",")]


def load_labels(labels_dir: str, split: str, field: str) -> list:
    json_path, jsonl_path = (os.path.join(labels_dir, f"{split}.{ext}") for ext in ("json", "jsonl"))
    if os.path.exists(json_path):
        with open(json_path) as f:
            rows = json.load(f)
    else:
        with open(jsonl_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return ["all" if field == "none" else r[field] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--field", default="task", help="Label key to group by, or 'none'.")
    ap.add_argument("--set", nargs="+", default=["old"], help="Eval report set names to include.")
    ap.add_argument("--name-re", default=FULLBP_RE, help=f"Run-name regex; its named groups become columns. "
                                                         f"Recursion-hyp runs: {HYP_RE!r}")
    ap.add_argument("--per-query", action="store_true", help="Score each comma-separated answer separately.")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    name_re = re.compile(args.name_re)
    labels_cache: dict[str, list] = {}
    out = []
    for path in sorted(glob.glob(os.path.join(args.root, "ckpts", "*", "eval_*_ep*.json"))):
        run = os.path.basename(os.path.dirname(path))
        m_run, m_rep = name_re.match(run), REPORT_RE.search(os.path.basename(path))
        if m_run is None or m_rep is None or m_rep["set"] not in args.set:
            continue
        split = m_rep["split"]
        if split not in labels_cache:
            labels_cache[split] = load_labels(args.labels, split, args.field)
        labels = labels_cache[split]
        with open(path) as f:
            report = json.load(f)
        if len(report["samples"]) != len(labels):
            raise SystemExit(f"{path}: {len(report['samples'])} samples vs {len(labels)} labels")

        counts = defaultdict(lambda: [0, 0])
        for s in report["samples"]:
            if args.per_query:
                gold, pred = split_answers(s["gold"]), split_answers(s["pred"])
                counts[labels[s["id"]]][0] += sum(g == p for g, p in zip(gold, pred))
                counts[labels[s["id"]]][1] += len(gold)
            else:
                counts[labels[s["id"]]][0] += int(s["correct"])
                counts[labels[s["id"]]][1] += 1
        if not args.per_query:
            assert sum(c for c, _ in counts.values()) == report["correct"], path  # reproduces the report

        fields = {k: (int(v) if v.isdigit() else v) for k, v in m_run.groupdict().items()}
        for group, (correct, n) in counts.items():
            out.append(dict(run=run, **fields, set=m_rep["set"], split=split, epoch=int(m_rep["epoch"]),
                            H_eval=report.get("H_cycles"), L_eval=report.get("L_cycles"),
                            group=group, correct=correct, n=n, accuracy=correct / n))

    if not out:
        raise SystemExit(f"no matching eval reports under {args.root}/ckpts")
    df = pd.DataFrame(out).sort_values(["run", "set", "split", "group"])
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"wrote {args.csv} ({len(df)} rows, {df['run'].nunique()} runs)")

    pooled = df.groupby(["run", "set", "group"])[["correct", "n"]].sum()
    pooled["acc"] = pooled["correct"] / pooled["n"]
    print(pooled["acc"].unstack("group").round(3).to_string())


if __name__ == "__main__":
    main()
