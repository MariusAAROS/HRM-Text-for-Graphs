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
PY="${PY:-$HOME/miniforge3/envs/hrm-text-graph-pt/bin/python}"
CONFIG=cfg_graphqa
EPOCHS="$(sed -n 's/^epochs: *\([0-9]*\).*/\1/p' config/${CONFIG}.yaml)"
PROJECT="${PROJECT:-HRM-GraphQA-FullBP-Local}"
EVAL_PROJECT="${EVAL_PROJECT:-${PROJECT}-eval}"
SMOKE_PROJECT="${SMOKE_PROJECT:-${PROJECT}-smoke}"
read -r -a GPUS <<< "${GPUS:-0 1}"
BENCH="${BENCH:-0}"
BENCH_SECONDS="${BENCH_SECONDS:-420}"
DEADLINE="${DEADLINE:-}"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_IDLE_MIB="${GPU_IDLE_MIB:-1024}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"  # never "disabled": all_config.yaml (needed by eval) is only written with a wandb run

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
  "$(full 1 12 0 38)"
  "$(full 3 3 0 36)"
  "$(trunc 2 12 0 37)"
  "$(trunc 1 24 0 37)"
  "$(trunc 3 6 0 31)"
  "$(full 1 6 0 25)"
  "$(trunc 2 6 0 24)"
  "$(trunc 1 12 0 23)"
  "$(trunc 3 3 0 22)"
  "$(full 1 3 0 18)"
  "$(full 2 1 0 18)"
  "$(trunc 1 6 0 17)"
  "$(trunc 1 3 0 14)"
  "$(trunc 2 1 0 14)"
  "$(full 1 1 0 13)"
  "$(trunc 1 1 0 13)"
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
RUNS="${RUNS:-}"  # space/comma-separated subset of run names; empty = all (bench: DEFAULT_BENCH_RUNS)
[[ "$BENCH" == "1" && -z "$RUNS" ]] && RUNS="$DEFAULT_BENCH_RUNS"

mkdir -p "$ROOT/logs" "$ROOT/ckpts"
CLAIMS="$ROOT/claims/$(date +%s)_$$"  # per invocation: a relaunch re-claims (trained runs are skipped anyway)
mkdir -p "$CLAIMS"

log() { echo "[$(date '+%F %T')] [gpu$GPU] $*"; }

gpu_mem_used() { nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }

wait_gpu_idle() {
  # Other containers' processes may not show up in --query-compute-apps, so also gate on memory.
  while :; do
    local apps mem
    apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
    mem="$(gpu_mem_used)"
    if [[ -z "$apps" ]] && (( mem < GPU_IDLE_MIB )); then
      return
    fi
    log "GPU busy (${mem} MiB used, pids: ${apps//$'\n'/,}); waiting ${POLL_SECONDS}s ..."
    sleep "$POLL_SECONDS"
  done
}

train_cmd() {  # $1 = run name, $2 = overrides, $3 = checkpoint path (or null), $4 = W&B project
  # shellcheck disable=SC2086
  echo "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
    --config-name "$CONFIG" \
    arch/size@arch=B \
    data.path="$DATA_DIR" \
    $2 \
    project_name="$4" \
    run_name="$1" \
    checkpoint_path="$3"
}

bench_run() {  # $1 = run name, $2 = overrides
  local log="$ROOT/logs/bench_$1.log" memlog="$ROOT/logs/bench_$1.mem"
  wait_gpu_idle
  log "[bench] $1 for ${BENCH_SECONDS}s"
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$memlog" &
  local mon=$!
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES="$GPU" timeout --signal=INT --kill-after=60 "$BENCH_SECONDS" \
    $(train_cmd "$1" "$2" null "$SMOKE_PROJECT") > "$log" 2>&1 || true
  kill "$mon" 2>/dev/null || true
  # tqdm writes "\r"-separated updates; the last one carries the smoothed rate and the total.
  local last rate total peak
  last="$(tr '\r' '\n' < "$log" | grep -E '[0-9]+/[0-9]+ \[' | tail -1 || true)"
  rate="$(grep -oE '[0-9.]+(it/s|s/it)' <<< "$last" || true)"
  total="$(grep -oE '/[0-9]+ \[' <<< "$last" | tr -dc '0-9' || true)"
  peak="$(sort -n "$memlog" | tail -1)"
  grep -q "OutOfMemoryError" "$log" && peak="OOM"
  "$PY" - "$1" "${rate:-?}" "${total:-0}" "${peak:-?}" <<'PY'
import sys
name, rate, total, peak = sys.argv[1], sys.argv[2], int(sys.argv[3] or 0), sys.argv[4]
if rate.endswith("it/s"):
    sps = float(rate[:-4])
elif rate.endswith("s/it"):
    sps = 1.0 / float(rate[:-4])
else:
    sps = None
eta = f"{total / sps / 60:.0f} min" if sps and total else "?"
print(f"[bench] {name:34s} {sps if sps else '?':>8} steps/s  total_steps={total}  train ETA={eta}  peak_mem={peak} MiB", flush=True)
PY
}

eval_run() {  # $1 = run name. Final checkpoint only, val + test, same settings as the grid figure.
  local ckpt="$ROOT/ckpts/$1"
  [[ -d "$ckpt/fsdp2_epoch_$EPOCHS" ]] || return 0
  for split in val test; do
    local out="$ckpt/eval_old_${split}_ep${EPOCHS}.json"
    [[ -f "$out" ]] && continue
    log "[eval] $1 epoch $EPOCHS on $split"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_graphqa.py \
      --ckpt_path "$ckpt" \
      --ckpt_epoch "$EPOCHS" \
      --data "$EVAL_DIR/$split.jsonl" \
      --use_ema \
      --max_generation 32 \
      --out "$out" > "$ROOT/logs/eval_$1_${split}.log" 2>&1 \
      || log "[eval] FAILED: $1 $split (see logs)"
  done
  "$PY" scripts/log_graphqa_evals_to_wandb.py --root "$ROOT" --project "$EVAL_PROJECT" --tag fullbp-local --run "$1" \
    > "$ROOT/logs/wandb_eval_$1.log" 2>&1 \
    || log "[eval] wandb logging FAILED: $1 (see $ROOT/logs/wandb_eval_$1.log)"
}

train_run() {  # $1 = run name, $2 = overrides
  local ckpt="$ROOT/ckpts/$1"
  if [[ -d "$ckpt/fsdp2_epoch_$EPOCHS" ]]; then
    log "[skip] $1 already trained"
  else
    wait_gpu_idle
    log "[train] $1  ($2)"
    # shellcheck disable=SC2046
    if ! CUDA_VISIBLE_DEVICES="$GPU" $(train_cmd "$1" "$2" "$ckpt" "$PROJECT") > "$ROOT/logs/train_$1.log" 2>&1; then
      log "[train] FAILED: $1 (see $ROOT/logs/train_$1.log)"
      return 1
    fi
    log "[train] done: $1"
  fi
  eval_run "$1"
}

worker() {  # $1 = GPU index
  GPU="$1"
  local entry name est overrides
  for entry in "${RUN_TABLE[@]}"; do
    IFS='|' read -r name est overrides <<< "$entry"
    if [[ -n "$RUNS" ]] && [[ ! " ${RUNS//,/ } " =~ " $name " ]]; then
      continue
    fi
    mkdir "$CLAIMS/$name" 2>/dev/null || continue  # atomic: the other worker already took it
    if [[ -n "$DEADLINE" ]] && [[ "$BENCH" != "1" ]] && [[ ! -d "$ROOT/ckpts/$name/fsdp2_epoch_$EPOCHS" ]] \
       && (( $(date +%s) + est * 60 > $(date -d "$DEADLINE" +%s) )); then
      log "[deadline] not starting $name (~${est} min would end after $DEADLINE)"
      continue
    fi
    if [[ "$BENCH" == "1" ]]; then
      bench_run "$name" "$overrides"
    else
      train_run "$name" "$overrides" || true  # one failed run should not stop the queue
    fi
  done
  log "worker finished"
}

[[ "$BENCH" == "1" ]] || [[ -f "$DATA_DIR/metadata.json" ]] || { echo "No prepared data at $DATA_DIR" >&2; exit 1; }
echo "Config: $CONFIG (epochs=$EPOCHS)  data: $DATA_DIR  root: $ROOT  gpus: ${GPUS[*]}  wandb: $WANDB_MODE"
echo "W&B projects: train=$PROJECT eval=$EVAL_PROJECT bench=$SMOKE_PROJECT  deadline: ${DEADLINE:-none}"

for g in "${GPUS[@]}"; do
  worker "$g" &
  sleep 20  # stagger startup (torchrun rendezvous, compile caches)
done
wait

echo "[$(date '+%F %T')] queue finished"
