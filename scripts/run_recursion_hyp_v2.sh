#!/bin/bash
# =====================================================================
# recursion_hyp_per_task_1b on GraphQA v2, 3 seeds: the 12 (H, L) cells with H <= 3 of the
# full-backprop vs truncated-BPTT comparison (scripts/run_graphqa_fullbp_local.sh), retrained
# from scratch on the fixed data (MaxFlow capacities printed, deduped val/test; see
# Graph-Representation-Learning-for-LLM/src/datasets/graphqa/generate_v2.py).
#
#   rh_H<H>_L<L>_fullbp_s<seed>   arch.full_backprop=True compile_scope=block
#   rh_H<H>_L<L>_trunc_s<seed>    truncated BPTT (bp budget of cfg_graphqa)
#
# A seed changes the model init (pretrain.py seed=) and the data order: each seed trains on its
# own prepared copy of the train split (prepare_sft_data.py --seed), built on first use.
# Eval: final EMA checkpoint on v2 val + test, --max_generation 96 (32 truncated long
# ConnectedNodes / DisconnectedNodes answers).
#
# W&B: everything goes to the clean project HRM-Graph-v2 (training job_type default, evals
# job_type=eval). Group = run name without _s<seed>, so a group holds the 3 seeds of one cell.
# Flat config keys: exp=recursion_hyp encoding=adjacency data_version=v2 seed backprop.
#
# Usage (from repo root; GPUS="0 1" with two GPUs):
#   GPUS=0 nohup setsid bash scripts/run_recursion_hyp_v2.sh > $ROOT/logs/queue.log 2>&1 &
#   RUNS="rh_H1_L1_fullbp_s0" bash scripts/run_recursion_hyp_v2.sh
#   BENCH=1 RUNS="rh_H3_L12_fullbp_s0" bash scripts/run_recursion_hyp_v2.sh
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-/work/dfm/marius-ortega/hrm_graph_v2/recursion_hyp}"
TRAIN_JSONL="${TRAIN_JSONL:-data/graphqa/v2/adjacency/hrm-text/standard/train.jsonl}"
EVAL_DIR="${EVAL_DIR:-data/graphqa/v2/adjacency/hrm-text/standard}"
TOKENIZER="${TOKENIZER:-$(ls "$HOME"/.cache/huggingface/hub/models--sapientinc--HRM-Text-1B/snapshots/*/tokenizer.json | head -1)}"
DATA_DIR="$ROOT/prepared_s0"  # per-seed dirs, see run_env
CONFIG=cfg_graphqa
PROJECT="${PROJECT:-HRM-Graph-v2}"
EVAL_PROJECT="${EVAL_PROJECT:-$PROJECT}"
SMOKE_PROJECT="${SMOKE_PROJECT:-$PROJECT-smoke}"
EVAL_SET=v2
EVAL_TAG=recursion_hyp
EVAL_MAX_GEN="${EVAL_MAX_GEN:-96}"
EVAL_LOG_ARGS="--grouped --config exp=recursion_hyp encoding=adjacency data_version=v2"
GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-0 1 2}"

# ---- Runs: name | estimated minutes (train + eval, one RTX PRO 6000; measured on the v1 runs) | overrides ----
run() {  # $1 H, $2 L, $3 fullbp|trunc, $4 seed, $5 minutes
  local name="rh_H$1_L$2_$3_s$4" bp=""
  [[ "$3" == "fullbp" ]] && bp="arch.full_backprop=True compile_scope=block"
  echo "$name|$5|arch.H_cycles=$1 arch.L_cycles=$2 $bp seed=$4 ++wandb_group=rh_H$1_L$2_$3 ++wandb_job_type=train ++wandb_config.exp=recursion_hyp ++wandb_config.encoding=adjacency ++wandb_config.data_version=v2 ++wandb_config.seed=$4 ++wandb_config.backprop=$3"
}
# H L minutes(fullbp) minutes(trunc), longest first.
CELLS=(
  "3 12 131 82" "2 12 90 61" "1 24 87 57" "3 6 74 52" "2 6 51 37" "1 12 47 34"
  "3 3 44 34" "2 3 32 28" "1 6 27 24" "1 3 18 17" "2 1 18 18" "1 1 11 11"
)
RUN_TABLE=()
for seed in $SEEDS; do
  for cell in "${CELLS[@]}"; do
    read -r H L full_min trunc_min <<< "$cell"
    RUN_TABLE+=("$(run "$H" "$L" fullbp "$seed" "$full_min")" "$(run "$H" "$L" trunc "$seed" "$trunc_min")")
  done
done
DEFAULT_BENCH_RUNS="rh_H3_L12_fullbp_s0 rh_H1_L1_trunc_s0"

run_env() {  # $1 = run name: per-seed prepared train data
  local seed="${1##*_s}"
  RUN_CONFIG="$CONFIG"; RUN_DATA_DIR="$ROOT/prepared_s$seed"; RUN_EVAL_DIR="$EVAL_DIR"; RUN_EVAL_SET="$EVAL_SET"
  RUN_EPOCHS="$(epochs_of "$RUN_CONFIG")"
}

prepare_seed() {  # $1 = seed; idempotent
  local out="$ROOT/prepared_s$1"
  [[ -f "$out/metadata.json" ]] && return
  echo "[prepare] seed $1 -> $out"
  "$PY" scripts/prepare_sft_data.py --train "$TRAIN_JSONL" --tokenizer "$TOKENIZER" --output "$out" \
    --epochs "$(epochs_of "$CONFIG")" --seed "$1"
}

# shellcheck source=scripts/lib_local_queue.sh
source scripts/lib_local_queue.sh

[[ -f "$TRAIN_JSONL" ]] || { echo "Missing $TRAIN_JSONL (run scripts/format_graphqa_to_hrm.py on GraphQA v2)" >&2; exit 1; }
for seed in $SEEDS; do prepare_seed "$seed"; done
queue_main
