#!/bin/bash
# =====================================================================
# Why doesn't recursion help? Minimal hypothesis tests, run locally on 2 GPUs.
#
#   H1 (collapse)  pre-norm lets the residual reach RMS ~300 before the final norm, so an L block
#                  barely depends on its incoming state (gain ~0.005, scripts/probe_gain_ckpt.py).
#                  B2: post / peri norm cap that growth -- does L start to matter?
#   TRM gap        our TRM lacks x injection, 1-cycle gradients, 2-layer modules and post norm.
#                  C2: TRM matched to the paper recipe (minus deep supervision).
#   H3 (task)      GraphQA may not need depth. B3/C3: pointer chasing, where depth = k hops.
#
# All GraphQA runs use the cfg_graphqa recipe (4,358 steps, seed 0) and full backprop unless
# noted; the pre-norm references are the runs of scripts/run_graphqa_fullbp_local.sh.
# Every L > 1 run is also evaluated with L_cycles = 1 at inference (eval_<set>_Lov1_*): no drop
# means the trained model does not use its extra L steps.
#
# Runs are named <task>_<variant>_H<H>_L<L>_s<seed>. PHASES selects the run tables:
#   b  B2 {post, peri} x {H1L1, H1L6} on GraphQA + B3 pre-norm H1L1 / H1L6 on pointer chasing
#   c  C1 2-layer modules with FIX norm, C2 TRM-matched on GraphQA. Needs FIX=post|peri, chosen
#      from B2 (higher L gain at H1L6).
#   p  pointer chasing v2 (N=16, 4 queries per graph; v1 stayed at chance, see cfg_pointer2.yaml):
#      pre H1L1 vs FIX H1L6, then FIX 2-layer H3L6 and TRM-matched H3L6.
#
# W&B: train HRM-RecursionHyp-Local, eval HRM-RecursionHyp-Local-eval, bench HRM-RecursionHyp-Local-smoke.
#
# Usage (from repo root):
#   BENCH=1 BENCH_SECONDS=300 PHASES="b c" FIX=peri RUNS="graphqa_peri_H1_L6_s0 graphqa_trm-matched_H3_L6_s0" \
#       bash scripts/run_recursion_hyp_local.sh
#   PHASES=b DEADLINE="2026-09-30 04:55" nohup setsid bash scripts/run_recursion_hyp_local.sh \
#       > /work/dfm/marius-ortega/recursion_hyp/logs/queue_b.log 2>&1 &
#   PHASES=c FIX=peri DEADLINE=... nohup setsid bash scripts/run_recursion_hyp_local.sh > .../queue_c.log 2>&1 &
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-/work/dfm/marius-ortega/recursion_hyp}"
DATA_DIR="${DATA_DIR:-/work/dfm/marius-ortega/graphqa_fullbp/std_prepared}"
EVAL_DIR="${EVAL_DIR:-data/graphqa/hrm-text/standard}"
POINTER_DATA_DIR="${POINTER_DATA_DIR:-$ROOT/pointer_prepared}"
POINTER_EVAL_DIR="${POINTER_EVAL_DIR:-data/pointer/hrm-text/standard}"
POINTER2_DATA_DIR="${POINTER2_DATA_DIR:-$ROOT/pointer2_prepared}"
POINTER2_EVAL_DIR="${POINTER2_EVAL_DIR:-data/pointer2/hrm-text/standard}"
CONFIG=cfg_graphqa
PROJECT="${PROJECT:-HRM-RecursionHyp-Local}"
EVAL_TAG="${EVAL_TAG:-recursion-hyp}"
PHASES="${PHASES:-b}"
FIX="${FIX:-}"

# GraphQA runs use cfg_graphqa + the std split ("old"); pointer_* runs cfg_pointer, pointer2_* cfg_pointer2.
run_env() {  # $1 = run name
  if [[ "$1" == pointer2_* ]]; then
    RUN_CONFIG=cfg_pointer2; RUN_DATA_DIR="$POINTER2_DATA_DIR"; RUN_EVAL_DIR="$POINTER2_EVAL_DIR"; RUN_EVAL_SET=ptr2
  elif [[ "$1" == pointer_* ]]; then
    RUN_CONFIG=cfg_pointer; RUN_DATA_DIR="$POINTER_DATA_DIR"; RUN_EVAL_DIR="$POINTER_EVAL_DIR"; RUN_EVAL_SET=ptr
  else
    RUN_CONFIG="$CONFIG"; RUN_DATA_DIR="$DATA_DIR"; RUN_EVAL_DIR="$EVAL_DIR"; RUN_EVAL_SET=old
  fi
  RUN_EPOCHS="$(epochs_of "$RUN_CONFIG")"
}

extra_evals() {  # $1 = run name: the L_cycles = 1 inference override for every L > 1 run
  local L="${1##*_L}"; L="${L%%_*}"
  (( L > 1 )) && eval_one "$1" "${RUN_EVAL_SET}_Lov1" --L_cycles 1
  return 0
}

# ---- Runs: name | estimated minutes (train + eval, one RTX PRO 6000) | hydra overrides ----
FULL="arch.full_backprop=True compile_scope=block"
hl() { echo "arch.H_cycles=$1 arch.L_cycles=$2"; }
# 12-layer HRM (6 per module) with a given norm.
hrm()   { echo "$1_$2_H$3_L$4_s0|$5|$(hl "$3" "$4") arch.norm_type=$2 $FULL"; }
# HRM with 2-layer modules: the recursion is the only source of depth.
hrm2l() { echo "$1_$2-2l_H$3_L$4_s0|$5|$(hl "$3" "$4") arch.n_layers=4 arch.norm_type=$2 $FULL"; }
# TRM matched to the paper: one 2-layer post-norm net, x re-injected into every L step, gradient
# through the last cycle only (its L steps + the H step). No deep supervision.
trm()   { echo "$1_trm-matched_H$2_L$3_s0|$4|arch/net@arch=trm $(hl "$2" "$3") arch.n_layers=2 arch.norm_type=post arch.inject_x=True arch.H_bp_steps=1 arch.L_bp_steps=$3"; }

RUN_TABLE=()
if [[ " $PHASES " == *" b "* ]]; then
  RUN_TABLE+=(
    "$(hrm graphqa post 1 6 32)"
    "$(hrm graphqa peri 1 6 32)"
    "$(hrm pointer pre 1 6 40)"
    "$(hrm pointer pre 1 1 22)"
    "$(hrm graphqa post 1 1 13)"
    "$(hrm graphqa peri 1 1 13)"
  )
fi
if [[ " $PHASES " == *" c "* ]]; then
  [[ "$FIX" == post || "$FIX" == peri ]] || { echo "PHASES=c needs FIX=post|peri" >&2; exit 1; }
  RUN_TABLE+=(
    "$(hrm2l graphqa "$FIX" 3 6 25)"
    "$(trm graphqa 3 6 25)"
    "$(hrm2l graphqa "$FIX" 1 6 15)"
    "$(trm graphqa 1 6 15)"
    "$(hrm2l graphqa "$FIX" 1 1 8)"
    "$(trm graphqa 1 1 8)"
  )
fi
if [[ " $PHASES " == *" p "* ]]; then
  [[ "$FIX" == post || "$FIX" == peri ]] || { echo "PHASES=p needs FIX=post|peri" >&2; exit 1; }
  RUN_TABLE+=(
    "$(hrm pointer2 pre 1 1 20)"
    "$(hrm pointer2 "$FIX" 1 6 35)"
    "$(hrm2l pointer2 "$FIX" 3 6 35)"
    "$(trm pointer2 3 6 30)"
  )
fi

# shellcheck source=scripts/lib_local_queue.sh
source scripts/lib_local_queue.sh

if [[ "$BENCH" != "1" ]]; then
  for d in "$DATA_DIR" "$POINTER_DATA_DIR" "$POINTER2_DATA_DIR"; do
    [[ -f "$d/metadata.json" ]] || echo "WARNING: no prepared data at $d yet (runs using it fail if still missing)" >&2
  done
fi
echo "Phases: $PHASES  fix: ${FIX:-none}  runs: ${#RUN_TABLE[@]}"
queue_main
