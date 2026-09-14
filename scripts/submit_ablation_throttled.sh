#!/bin/bash
# =====================================================================
# Throttled launcher for the (H,L) ablation sweeps in slurm/.
#
# Submits each (H,L) config as its own single-element job array, keeping at most
# MAX_INFLIGHT jobs in the queue at a time. Jean Zay caps parallel runs at 10;
# if that cap applies at SUBMIT time (as it does on the -dev QoS) then a whole
# `#SBATCH --array=0-15` is rejected outright, because every element counts as a
# queued job the moment it is submitted -- so the in-file `%10` throttle cannot
# help and submissions have to be drip-fed from outside.
#
# No edits to the .slurm scripts are needed: `sbatch --array=<i>` overrides the
# in-file array directive, and the script's own
# `H=${H_LIST[$SLURM_ARRAY_TASK_ID]}` resolves the config from that index.
#
# The (H,L) grid is parsed out of the target .slurm, never redeclared here, so
# the two cannot drift.
#
# Usage (from repo root on a Jean Zay login node):
#   bash scripts/submit_ablation_throttled.sh slurm/train_graphqa.slurm
#   DRY_RUN=1 bash scripts/submit_ablation_throttled.sh slurm/train_graphqa.slurm
#   MAX_INFLIGHT=4 bash scripts/submit_ablation_throttled.sh slurm/eval_graphqa.slurm
#   CONFIGS="H1_L1,H5_L6" bash scripts/submit_ablation_throttled.sh slurm/train_graphqa.slurm
#   CONFIGS="0,4,7" bash scripts/submit_ablation_throttled.sh slurm/train_graphqa.slurm
#   THROTTLE_SCOPE=name bash scripts/submit_ablation_throttled.sh slurm/train_graphqa.slurm
#
# A full GraphQA sweep is 16 jobs at up to 20h with 10 in flight, so this script
# must stay alive ~40h -- longer than a login-node session survives. Run it
# detached, or (preferred) from a small CPU job:
#   nohup bash scripts/submit_ablation_throttled.sh slurm/train_graphqa.slurm \
#         > logs/sweep_train_graphqa.log 2>&1 &
# =====================================================================
set -euo pipefail

# No brace in this message: a `}` inside ${1:?...} would close the expansion early.
SCRIPT="${1:?usage: submit_ablation_throttled.sh <path/to/train_or_eval_xxx.slurm>}"
[[ -f "$SCRIPT" ]] || { echo "No such script: $SCRIPT" >&2; exit 1; }

MAX_INFLIGHT="${MAX_INFLIGHT:-10}"        # Jean Zay parallel-run cap
POLL_SECONDS="${POLL_SECONDS:-300}"       # jobs run for hours; polling squeue every 30s is just noise
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"                       # resubmit indices already in the ledger
THROTTLE_SCOPE="${THROTTLE_SCOPE:-user}"  # user = all your jobs (the real cap) | name = this sweep only

# ---- Parse the grid out of the .slurm (single source of truth) -------
# `^H_LIST=` anchors to the live array; the commented-out `# [v1] H_LIST=` is skipped.
read -r -a H_LIST <<< "$(sed -n 's/^H_LIST=(\(.*\))/\1/p' "$SCRIPT" | head -1)"
read -r -a L_LIST <<< "$(sed -n 's/^L_LIST=(\(.*\))/\1/p' "$SCRIPT" | head -1)"
N=${#H_LIST[@]}

(( N > 0 )) || { echo "Could not parse H_LIST from $SCRIPT" >&2; exit 1; }
(( N == ${#L_LIST[@]} )) || {
  echo "Grid mismatch in $SCRIPT: ${N} H values vs ${#L_LIST[@]} L values" >&2; exit 1; }

# The in-file --array upper bound must cover the grid, otherwise a plain
# `sbatch $SCRIPT` (without this launcher) would silently skip configs.
ARRAY_MAX="$(sed -n 's/^#SBATCH --array=0-\([0-9]*\).*/\1/p' "$SCRIPT" | head -1)"
if [[ -n "$ARRAY_MAX" ]] && (( ARRAY_MAX != N - 1 )); then
  echo "WARNING: $SCRIPT declares --array=0-${ARRAY_MAX} but the grid has ${N} configs (0-$((N-1)))." >&2
fi

JOBNAME="$(sed -n 's/^#SBATCH --job-name=\([^ ]*\).*/\1/p' "$SCRIPT" | head -1)"
[[ -n "$JOBNAME" ]] || { echo "Could not read --job-name from $SCRIPT" >&2; exit 1; }

# ---- Which configs to submit ----------------------------------------
labels=()
for i in $(seq 0 $((N - 1))); do labels+=("H${H_LIST[$i]}_L${L_LIST[$i]}"); done

indices=()
if [[ -n "${CONFIGS:-}" ]]; then
  IFS=',' read -r -a want <<< "$CONFIGS"
  for w in "${want[@]}"; do
    w="${w// /}"
    if [[ "$w" =~ ^[0-9]+$ ]]; then
      (( w < N )) || { echo "Index out of range: $w (grid has $N configs)" >&2; exit 1; }
      indices+=("$w")
    else
      found=-1
      for j in "${!labels[@]}"; do [[ "${labels[$j]}" == "$w" ]] && { found=$j; break; }; done
      (( found >= 0 )) || { echo "Unknown config: $w (valid: ${labels[*]})" >&2; exit 1; }
      indices+=("$found")
    fi
  done
else
  indices=($(seq 0 $((N - 1))))
fi

# ---- Ledger: survive a launcher crash without resubmitting -----------
# Records indices already handed to sbatch. A job that FAILED stays in the
# ledger; delete its line (or set FORCE=1) to resubmit it.
mkdir -p logs
LEDGER="logs/.submitted_${JOBNAME}"
touch "$LEDGER"

already_submitted() {
  [[ "$FORCE" == "1" ]] && return 1
  grep -qE "^$1[[:space:]]" "$LEDGER"
}

# ---- Queue occupancy -------------------------------------------------
if ! command -v squeue >/dev/null 2>&1; then
  # Failing open here would dump the whole sweep into the queue at once.
  [[ "$DRY_RUN" == "1" ]] || { echo "squeue not found -- refusing to submit unthrottled." >&2; exit 1; }
fi

inflight() {
  if [[ "$THROTTLE_SCOPE" == "name" ]]; then
    squeue -u "$USER" -h -r -n "$JOBNAME" | wc -l
  else
    squeue -u "$USER" -h -r | wc -l
  fi
}

echo "Script      : $SCRIPT  (job-name: $JOBNAME)"
echo "Grid        : $N configs -- ${labels[*]}"
echo "Submitting  : ${#indices[@]} of them (indices: ${indices[*]})"
echo "Max inflight: $MAX_INFLIGHT  (scope: $THROTTLE_SCOPE, poll every ${POLL_SECONDS}s)"
echo "Ledger      : $LEDGER"
echo

for idx in "${indices[@]}"; do
  label="${labels[$idx]}"

  if already_submitted "$idx"; then
    echo "[skip] $label (index $idx) already in ledger"
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] sbatch --array=$idx $SCRIPT   # $label"
    continue
  fi

  while :; do
    n="$(inflight)"
    (( n < MAX_INFLIGHT )) || {
      echo "[throttle] ${n}/${MAX_INFLIGHT} jobs in queue; waiting ${POLL_SECONDS}s ..."
      sleep "$POLL_SECONDS"
      continue
    }
    break
  done

  jobid="$(sbatch --parsable --array="$idx" "$SCRIPT")"
  printf '%s\t%s\t%s\n' "$idx" "$label" "$jobid" >> "$LEDGER"
  echo "[submit] $label (index $idx) -> job $jobid"
done

echo "All submissions issued."
