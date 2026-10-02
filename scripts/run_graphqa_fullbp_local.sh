#!/bin/bash
# =====================================================================
# Full-backprop vs truncated-BPTT on a sparse subset of the HRM H/L grid
# (report/figures/ablation_hl_grid.png), run locally on 2 GPUs.
#
# Question: the H=1 row of the grid is flat in L. With the truncated budget
# (H_bp = min(H,4), L_bp = 5 - H_bp after warmup) only the last 4 L-steps get
# gradient at H=1 (4/6 at L=6, 4/24 at L=24). Does full backprop make L matter?
#
# Every cell is trained twice from scratch with the original small-grid recipe
# (config/cfg_graphqa.yaml: 3,080 samples x 20 epochs = 4,358 steps, seed 0):
#   graphqa_H<H>_L<L>_fullbp_local_s<seed>   arch.full_backprop=True compile_scope=block
#   graphqa_H<H>_L<L>_trunc_local_s<seed>    the grid's truncated regime, retrained locally
# H1L1 (and, after the bp warmup, H1L3 / H2L1) have identical gradient graphs in both
# regimes: they are the sanity controls for the full-backprop path.
#
# Grad checkpointing stays off (96 GB fits every cell). It is only valid with compiled
# blocks anyway: eager FlexAttention returns wrong gradients under checkpoint recompute
# (see scripts/validate_bp_steps.py; models/transformer.py refuses that combination).
#
# W&B: everything goes to dedicated projects so it never mixes with the Jean Zay grid
# (HRM-GraphQA-Ablation / graphqa-eval) or the scaled runs:
#   train  HRM-GraphQA-FullBP-Local    eval  HRM-GraphQA-FullBP-Local-eval
#   bench  HRM-GraphQA-FullBP-Local-smoke
#
# Two workers (one per GPU) claim runs from RUN_TABLE in order (longest first, stretch
# runs last). A run whose final checkpoint exists is skipped. With DEADLINE set, a worker
# does not start a run whose estimate (minutes, incl. eval) would end after it.
# The queue machinery lives in scripts/lib_local_queue.sh (shared with run_recursion_hyp_local.sh).
#
# Usage (from repo root):
#   BENCH=1 bash scripts/run_graphqa_fullbp_local.sh      # ~7 min per bench run, steps/s + peak memory
#   DEADLINE="2026-09-29 08:00" nohup setsid bash scripts/run_graphqa_fullbp_local.sh \
#       > /work/dfm/marius-ortega/graphqa_fullbp/logs/queue.log 2>&1 &
#   RUNS="graphqa_H1_L1_fullbp_local_s0" bash scripts/run_graphqa_fullbp_local.sh
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-/work/dfm/marius-ortega/graphqa_fullbp}"
DATA_DIR="${DATA_DIR:-$ROOT/std_prepared}"
EVAL_DIR="${EVAL_DIR:-data/graphqa/hrm-text/standard}"  # byte-identical to the "old" split of the grid figure
CONFIG=cfg_graphqa
PROJECT="${PROJECT:-HRM-GraphQA-FullBP-Local}"

# ---- Runs: name | estimated minutes (train + eval, on one RTX PRO 6000) | hydra overrides ----
# Longest first to minimise the makespan; seed-1 stretch runs last (dropped first by DEADLINE).
full()  { echo "graphqa_H$1_L$2_fullbp_local_s$3|$4|arch.H_cycles=$1 arch.L_cycles=$2 arch.full_backprop=True compile_scope=block seed=$3"; }
trunc() { echo "graphqa_H$1_L$2_trunc_local_s$3|$4|arch.H_cycles=$1 arch.L_cycles=$2 seed=$3"; }
# Estimates from BENCH=1 on an RTX PRO 6000 (full BP: 1.04 steps/s at H3L12 .. 12.4 at H1L1;
# truncated H3L6: 4.8 steps/s) for 4,358 steps, plus ~2 x (3 + 0.25 * H*(L+1)) min of eval.
RUN_TABLE=(
  "$(full 3 12 0 95)"
  "$(full 2 12 0 72)"
  "$(full 1 24 0 64)"
  "$(full 3 6 0 55)"
  "$(trunc 3 12 0 50)"
  "$(full 2 6 0 40)"
  "$(full 2 3 0 28)"
  "$(full 1 12 0 38)"
  "$(full 3 3 0 36)"
  "$(trunc 2 12 0 37)"
  "$(trunc 1 24 0 37)"
  "$(trunc 3 6 0 31)"
  "$(full 1 6 0 25)"
  "$(trunc 2 6 0 24)"
  "$(trunc 2 3 0 18)"
  "$(trunc 1 12 0 23)"
  "$(trunc 3 3 0 22)"
  "$(full 1 3 0 18)"
  "$(full 2 1 0 18)"
  "$(trunc 1 6 0 17)"
  "$(trunc 1 3 0 14)"
  "$(trunc 2 1 0 14)"
  "$(full 1 1 0 13)"
  "$(trunc 1 1 0 13)"
  # ---- H >= 4 rows (where the Jean Zay 1-step grid collapses): does full backprop rescue them? ----
  # Truncated reruns only at H4L6 / H6L6, to check the collapse reproduces locally; the other
  # 1-step cells come from the Jean Zay grid. 42 unrolled blocks (H6L6) still fit in 96 GB.
  "$(full 6 6 0 95)"
  "$(full 5 6 0 80)"
  "$(full 4 6 0 65)"
  "$(full 6 2 0 45)"
  "$(full 4 3 0 40)"
  "$(trunc 6 6 0 40)"
  "$(trunc 4 6 0 30)"
  "$(full 4 1 0 25)"
  # ---- stretch: second seed on the H=1 row ----
  "$(full 1 24 1 64)"
  "$(trunc 1 24 1 37)"
  "$(full 1 6 1 25)"
  "$(trunc 1 6 1 17)"
  "$(full 1 1 1 13)"
  "$(trunc 1 1 1 13)"
)
# Bench = heaviest cells (memory / throughput) + one truncated and one control run, which also
# serve as the in-situ backprop smoke test (scripts/check_bp_smoke.py reads their W&B history).
DEFAULT_BENCH_RUNS="graphqa_H3_L12_fullbp_local_s0 graphqa_H2_L12_fullbp_local_s0 graphqa_H1_L24_fullbp_local_s0 graphqa_H3_L6_fullbp_local_s0 graphqa_H3_L6_trunc_local_s0 graphqa_H1_L1_fullbp_local_s0"

# shellcheck source=scripts/lib_local_queue.sh
source scripts/lib_local_queue.sh

[[ "$BENCH" == "1" ]] || [[ -f "$DATA_DIR/metadata.json" ]] || { echo "No prepared data at $DATA_DIR" >&2; exit 1; }
queue_main
