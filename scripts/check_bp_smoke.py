"""In-situ check of the backprop budget on the REAL training path (bf16, FSDP2, compiled blocks).

scripts/validate_bp_steps.py proves the math on a toy model; this reads the recursion probe of
actual pretrain.py runs from W&B (the bench runs of scripts/run_graphqa_fullbp_local.sh) and checks,
for every probed step:
  - full backprop:  bp/H_bp_steps == H, bp/L_bp_steps == H*L, in-graph H / L steps == H / H*L,
                    and a finite, nonzero dLoss/dz at the FIRST L step (the one truncation cuts first);
  - truncated:      in-graph H / L steps == the warmup budget (H_bp = min(H, bp-1), L_bp = bp - H_bp),
                    and no gradient at the first L step whenever it lies outside that budget.

Run names encode the regime: graphqa_H<H>_L<L>_{fullbp,trunc}_local_s<seed>.

Usage:
    python scripts/check_bp_smoke.py [--project HRM-GraphQA-FullBP-Local-smoke] [--run <name> ...]
"""
import argparse
import math
import re
import sys

import wandb

NAME_RE = re.compile(r"graphqa_H(?P<H>\d+)_L(?P<L>\d+)_(?P<regime>fullbp|trunc)_local_s\d+$")


def check_run(run) -> list[str]:
    m = NAME_RE.search(run.name)
    H, L, full = int(m["H"]), int(m["L"]), m["regime"] == "fullbp"
    keys = ["probe/bp/H_bp_steps", "probe/bp/L_bp_steps", "probe/zgrad/H/in_graph_steps",
            "probe/zgrad/L/in_graph_steps", "probe/zgrad/L/step00"]
    rows = [r for r in run.scan_history(keys=keys[:4]) if r.get("probe/bp/L_bp_steps") is not None]
    first_step = {r["_step"]: r.get("probe/zgrad/L/step00") for r in run.scan_history(keys=["probe/zgrad/L/step00"])}
    if not rows:
        return [f"{run.name}: no probed step logged (run too short? probe interval is 100 steps)"]

    errors = []
    for r in rows:
        step = r["_step"]
        H_bp, L_bp = int(r["probe/bp/H_bp_steps"]), int(r["probe/bp/L_bp_steps"])
        in_H, in_L = int(r["probe/zgrad/H/in_graph_steps"]), int(r["probe/zgrad/L/in_graph_steps"])
        g0 = first_step.get(step)
        if full:
            want = (H, H * L)
            if (H_bp, L_bp) != want or (in_H, in_L) != want:
                errors.append(f"{run.name} step {step}: budget {(H_bp, L_bp)} in-graph {(in_H, in_L)}, want {want}")
            if g0 is None or not math.isfinite(g0) or g0 <= 0:
                errors.append(f"{run.name} step {step}: dLoss/dz at L step 0 = {g0}, want finite > 0")
        else:
            if (in_H, in_L) != (H_bp, L_bp) or H_bp > min(H, 4) or H_bp + L_bp > 5:
                errors.append(f"{run.name} step {step}: in-graph {(in_H, in_L)} vs budget {(H_bp, L_bp)}")
            if L_bp < H * L and g0 is not None:
                errors.append(f"{run.name} step {step}: L step 0 has a gradient ({g0}) outside the truncated window")
    last = rows[-1]
    print(f"[{'ok' if not errors else 'FAIL'}] {run.name}: {len(rows)} probed steps, last step {last['_step']}: "
          f"budget ({int(last['probe/bp/H_bp_steps'])}, {int(last['probe/bp/L_bp_steps'])}), "
          f"in-graph ({int(last['probe/zgrad/H/in_graph_steps'])}, {int(last['probe/zgrad/L/in_graph_steps'])}), "
          f"zgrad L step0 = {first_step.get(last['_step'])}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="HRM-GraphQA-FullBP-Local-smoke")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--run", nargs="*", default=None, help="Only these run names (default: all matching runs).")
    args = ap.parse_args()

    api = wandb.Api()
    runs = [r for r in api.runs(f"{args.entity or api.default_entity}/{args.project}")
            if NAME_RE.search(r.name) and (not args.run or r.name in args.run)]
    if not runs:
        sys.exit(f"no matching runs in {args.project}")

    errors = [e for run in runs for e in check_run(run)]
    for e in errors:
        print("  " + e)
    if errors:
        sys.exit(f"{len(errors)} backprop-budget violations")
    print(f"\nAll {len(runs)} runs respect their backprop budget on the real training path.")


if __name__ == "__main__":
    main()
