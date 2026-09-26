#!/bin/bash
# =====================================================================
# Scaled GraphQA runs (HRM + TRM at H1-L1, H1-L6, H3-L6), run locally on a
# single GPU, one after another. No SLURM: each run waits until the GPU is
# idle, so it never lands on top of someone else's job.
#
# Prerequisites:
#   - ~1M-sample train pool tokenized to $DATA_DIR with prepare_sft_data.py
#     --epochs equal to `epochs` in config/cfg_graphqa_scaled.yaml.
#   - Eval JSONL in $ROOT/jsonl/eval-{old,fixed}/standard/{val,test}.jsonl
#     (old = the split behind report/figures/ablation_hl_grid.png;
#      fixed = same seed-42 split regenerated with capacity-annotated MaximumFlow).
#
# Usage (from repo root):
#   nohup setsid bash scripts/run_graphqa_scaled.sh > $ROOT/logs/queue.log 2>&1 &
#   BENCH=1 bash scripts/run_graphqa_scaled.sh           # ~15 min per config, prints steps/s + peak memory
#   RUNS="graphqa_scaled_H1_L1" bash scripts/run_graphqa_scaled.sh
#
# A run whose final checkpoint exists is skipped. pretrain.py cannot resume mid-run
# faithfully (resume_from restarts the LR / bp-warmup schedule at step 0), so an
# interrupted run is retrained from scratch.
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-/work/dfm/marius-ortega/graphqa_scaled}"
DATA_DIR="${DATA_DIR:-$ROOT/prepared_1m}"
PY="${PY:-$HOME/miniforge3/envs/hrm-text-graph-pt/bin/python}"
CONFIG=cfg_graphqa_scaled
EPOCHS="$(sed -n 's/^epochs: *\([0-9]*\).*/\1/p' config/${CONFIG}.yaml)"
BENCH="${BENCH:-0}"
BENCH_SECONDS="${BENCH_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_IDLE_MIB="${GPU_IDLE_MIB:-1024}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"  # never "disabled": all_config.yaml (needed by eval) is only written with a wandb run

# ---- Runs: name | hydra overrides. Cheapest first so results arrive early. ----
# TRM bp steps mirror slurm/train_graphqa_trm.slurm: H_bp = min(H, 4), L_bp = min(5 - H_bp, L).
RUN_TABLE=(
  "graphqa_scaled_H1_L1|arch.H_cycles=1 arch.L_cycles=1"
  "graphqa_scaled_trm_H1_L1|arch/net@arch=trm +arch.half_layers=True arch.H_cycles=1 arch.L_cycles=1 arch.H_bp_steps=1 arch.L_bp_steps=1"
  "graphqa_scaled_H1_L6|arch.H_cycles=1 arch.L_cycles=6"
  "graphqa_scaled_trm_H1_L6|arch/net@arch=trm +arch.half_layers=True arch.H_cycles=1 arch.L_cycles=6 arch.H_bp_steps=1 arch.L_bp_steps=4"
  "graphqa_scaled_H3_L6|arch.H_cycles=3 arch.L_cycles=6"
  "graphqa_scaled_trm_H3_L6|arch/net@arch=trm +arch.half_layers=True arch.H_cycles=3 arch.L_cycles=6 arch.H_bp_steps=3 arch.L_bp_steps=2"
)
RUNS="${RUNS:-}"  # space/comma-separated subset of run names; empty = all

EVAL_SETS=(old fixed)

mkdir -p "$ROOT/logs" "$ROOT/ckpts"

gpu_mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

wait_gpu_idle() {
  # Other containers' processes may not show up in --query-compute-apps, so also gate on memory.
  while :; do
    local apps mem
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
    mem="$(gpu_mem_used)"
    if [[ -z "$apps" ]] && (( mem < GPU_IDLE_MIB )); then
      return
    fi
    echo "[$(date '+%F %T')] GPU busy (${mem} MiB used, pids: ${apps//$'\n'/,}); waiting ${POLL_SECONDS}s ..."
    sleep "$POLL_SECONDS"
  done
}

train_cmd() {  # $1 = run name, $2 = overrides, $3 = checkpoint path (or null)
  # shellcheck disable=SC2086
  echo "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
    --config-name "$CONFIG" \
    arch/size@arch=B \
    data.path="$DATA_DIR" \
    $2 \
    run_name="$1" \
    checkpoint_path="$3"
}

bench_run() {  # $1 = run name, $2 = overrides
  local log="$ROOT/logs/bench_$1.log" memlog="$ROOT/logs/bench_$1.mem"
  wait_gpu_idle
  echo "[$(date '+%F %T')] [bench] $1 for ${BENCH_SECONDS}s"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$memlog" &
  local mon=$!
  # shellcheck disable=SC2046
  WANDB_MODE=offline timeout --signal=INT --kill-after=60 "$BENCH_SECONDS" \
    $(train_cmd "$1" "$2" null) project_name=HRM-GraphQA-Scaled-bench > "$log" 2>&1 || true
  kill "$mon" 2>/dev/null || true
  # tqdm writes "\r"-separated updates; the last one carries the smoothed rate and the total.
  local last rate total
  last="$(tr '\r' '\n' < "$log" | grep -E '[0-9]+/[0-9]+ \[' | tail -1)"
  rate="$(grep -oE '[0-9.]+(it/s|s/it)' <<< "$last" || true)"
  total="$(grep -oE '/[0-9]+ \[' <<< "$last" | tr -dc '0-9' || true)"
  local peak
  peak="$(sort -n "$memlog" | tail -1)"
  "$PY" - "$1" "${rate:-?}" "${total:-0}" "${peak:-?}" <<'PY'
import sys
name, rate, total, peak = sys.argv[1], sys.argv[2], int(sys.argv[3] or 0), sys.argv[4]
if rate.endswith("it/s"):
    sps = float(rate[:-4])
elif rate.endswith("s/it"):
    sps = 1.0 / float(rate[:-4])
else:
    sps = None
eta = f"{total / sps / 3600:.1f} h" if sps and total else "?"
print(f"[bench] {name:28s} {sps if sps else '?':>8} steps/s  total_steps={total}  ETA={eta}  peak_mem={peak} MiB", flush=True)
PY
}

eval_run() {  # $1 = run name
  local ckpt="$ROOT/ckpts/$1"
  for e in $(seq 1 "$EPOCHS"); do
    [[ -d "$ckpt/fsdp2_epoch_$e" ]] || continue
    for set in "${EVAL_SETS[@]}"; do
      # Val at every epoch (learning curve); test only on the final checkpoint.
      local splits=(val)
      (( e == EPOCHS )) && splits=(val test)
      for split in "${splits[@]}"; do
        local out="$ckpt/eval_${set}_${split}_ep${e}.json"
        [[ -f "$out" ]] && continue
        echo "[$(date '+%F %T')] [eval] $1 epoch $e on $set/$split"
        "$PY" scripts/eval_graphqa.py \
          --ckpt_path "$ckpt" \
          --ckpt_epoch "$e" \
          --data "$ROOT/jsonl/eval-$set/standard/$split.jsonl" \
          --use_ema \
          --max_generation 32 \
          --out "$out" > "$ROOT/logs/eval_$1_${set}_${split}_ep${e}.log" 2>&1 \
          || echo "[eval] FAILED: $1 epoch $e $set/$split (see logs)"
      done
    done
  done
}

train_run() {  # $1 = run name, $2 = overrides
  local ckpt="$ROOT/ckpts/$1"
  if [[ -d "$ckpt/fsdp2_epoch_$EPOCHS" ]]; then
    echo "[$(date '+%F %T')] [skip] $1 already trained"
  else
    wait_gpu_idle
    echo "[$(date '+%F %T')] [train] $1  ($2)"
    # shellcheck disable=SC2046
    if ! $(train_cmd "$1" "$2" "$ckpt") > "$ROOT/logs/train_$1.log" 2>&1; then
      echo "[$(date '+%F %T')] [train] FAILED: $1 (see $ROOT/logs/train_$1.log)"
      return 1
    fi
    echo "[$(date '+%F %T')] [train] done: $1"
  fi
  eval_run "$1"
}

[[ "$BENCH" == "1" ]] || [[ -f "$DATA_DIR/metadata.json" ]] || { echo "No prepared data at $DATA_DIR" >&2; exit 1; }
echo "Config: $CONFIG (epochs=$EPOCHS)  data: $DATA_DIR  root: $ROOT  wandb: $WANDB_MODE"

for entry in "${RUN_TABLE[@]}"; do
  name="${entry%%|*}"
  overrides="${entry#*|}"
  if [[ -n "$RUNS" ]] && [[ ! " ${RUNS//,/ } " =~ " $name " ]]; then
    continue
  fi
  if [[ "$BENCH" == "1" ]]; then
    bench_run "$name" "$overrides"
  else
    train_run "$name" "$overrides" || true  # one failed run should not stop the queue
  fi
done

echo "[$(date '+%F %T')] queue finished"
