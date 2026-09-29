#!/bin/bash
# =====================================================================
# Shared 2-GPU local run queue: sourced by scripts/run_graphqa_fullbp_local.sh and
# scripts/run_recursion_hyp_local.sh, which set the variables below and RUN_TABLE, then call
# `queue_main`.
#
# RUN_TABLE entries: "name|estimated minutes (train + eval)|hydra overrides".
# One worker per GPU claims entries in order (atomic mkdir). A run whose final checkpoint
# exists is skipped; with DEADLINE set, a worker does not start a run whose estimate would end
# after it. Every run is trained, then evaluated on val + test (final checkpoint, EMA).
#
# Required: ROOT DATA_DIR EVAL_DIR CONFIG PROJECT RUN_TABLE
# Optional: PY EVAL_PROJECT SMOKE_PROJECT EVAL_SET EVAL_TAG GPUS BENCH BENCH_SECONDS DEADLINE
#           POLL_SECONDS GPU_IDLE_MIB RUNS DEFAULT_BENCH_RUNS
# Hooks (define before queue_main to override):
#   run_env <name>     sets RUN_CONFIG RUN_DATA_DIR RUN_EVAL_DIR RUN_EVAL_SET RUN_EPOCHS for a run
#   extra_evals <name> extra evaluations after the standard ones (e.g. an L_cycles override)
# =====================================================================

PY="${PY:-$HOME/miniforge3/envs/hrm-text-graph-pt/bin/python}"
EVAL_PROJECT="${EVAL_PROJECT:-${PROJECT}-eval}"
SMOKE_PROJECT="${SMOKE_PROJECT:-${PROJECT}-smoke}"
EVAL_SET="${EVAL_SET:-old}"  # eval report name: eval_<set>_<split>_ep<E>.json
EVAL_TAG="${EVAL_TAG:-fullbp-local}"
read -r -a GPUS <<< "${GPUS:-0 1}"
BENCH="${BENCH:-0}"
BENCH_SECONDS="${BENCH_SECONDS:-420}"
DEADLINE="${DEADLINE:-}"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_IDLE_MIB="${GPU_IDLE_MIB:-1024}"
RUNS="${RUNS:-}"  # space/comma-separated subset of run names; empty = all (bench: DEFAULT_BENCH_RUNS)
[[ "$BENCH" == "1" && -z "$RUNS" ]] && RUNS="${DEFAULT_BENCH_RUNS:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"  # never "disabled": all_config.yaml (needed by eval) is only written with a wandb run

epochs_of() { sed -n 's/^epochs: *\([0-9]*\).*/\1/p' "config/$1.yaml"; }

if ! declare -F run_env > /dev/null; then
  run_env() {  # $1 = run name
    RUN_CONFIG="$CONFIG"; RUN_DATA_DIR="$DATA_DIR"; RUN_EVAL_DIR="$EVAL_DIR"; RUN_EVAL_SET="$EVAL_SET"
    RUN_EPOCHS="$(epochs_of "$RUN_CONFIG")"
  }
fi

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
    --config-name "$RUN_CONFIG" \
    arch/size@arch=B \
    data.path="$RUN_DATA_DIR" \
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

eval_one() {  # $1 = run name, $2 = eval set name, $3.. = extra eval_graphqa.py args. Final checkpoint, val + test.
  local name="$1" set="$2" ckpt="$ROOT/ckpts/$1"
  shift 2
  for split in val test; do
    local out="$ckpt/eval_${set}_${split}_ep${RUN_EPOCHS}.json"
    [[ -f "$out" ]] && continue
    log "[eval] $name epoch $RUN_EPOCHS on $split ($set)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_graphqa.py \
      --ckpt_path "$ckpt" \
      --ckpt_epoch "$RUN_EPOCHS" \
      --data "$RUN_EVAL_DIR/$split.jsonl" \
      --use_ema \
      --max_generation 32 \
      "$@" \
      --out "$out" > "$ROOT/logs/eval_${name}_${set}_${split}.log" 2>&1 \
      || log "[eval] FAILED: $name $set $split (see logs)"
  done
}

eval_run() {  # $1 = run name
  [[ -d "$ROOT/ckpts/$1/fsdp2_epoch_$RUN_EPOCHS" ]] || return 0
  eval_one "$1" "$RUN_EVAL_SET"
  if declare -F extra_evals > /dev/null; then
    extra_evals "$1"
  fi
  "$PY" scripts/log_graphqa_evals_to_wandb.py --root "$ROOT" --project "$EVAL_PROJECT" --tag "$EVAL_TAG" --run "$1" \
    > "$ROOT/logs/wandb_eval_$1.log" 2>&1 \
    || log "[eval] wandb logging FAILED: $1 (see $ROOT/logs/wandb_eval_$1.log)"
}

train_run() {  # $1 = run name, $2 = overrides
  local ckpt="$ROOT/ckpts/$1"
  if [[ -d "$ckpt/fsdp2_epoch_$RUN_EPOCHS" ]]; then
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
    run_env "$name"
    if [[ -n "$DEADLINE" ]] && [[ "$BENCH" != "1" ]] && [[ ! -d "$ROOT/ckpts/$name/fsdp2_epoch_$RUN_EPOCHS" ]] \
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

queue_main() {
  mkdir -p "$ROOT/logs" "$ROOT/ckpts"
  CLAIMS="$ROOT/claims/$(date +%s)_$$"  # per invocation: a relaunch re-claims (trained runs are skipped anyway)
  mkdir -p "$CLAIMS"

  echo "Config: $CONFIG  data: $DATA_DIR  root: $ROOT  gpus: ${GPUS[*]}  wandb: $WANDB_MODE"
  echo "W&B projects: train=$PROJECT eval=$EVAL_PROJECT bench=$SMOKE_PROJECT  deadline: ${DEADLINE:-none}"

  for g in "${GPUS[@]}"; do
    worker "$g" &
    sleep 20  # stagger startup (torchrun rendezvous, compile caches)
  done
  wait

  echo "[$(date '+%F %T')] queue finished"
}
