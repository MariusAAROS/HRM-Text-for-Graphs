#!/bin/bash
# =====================================================================
# Evaluate the full-backprop GraphQA checkpoints (exact-match accuracy).
# Mirrors train_graphqa_fullbp.slurm / train_graphqa_trm_fullbp.slurm: one eval
# job per (family, H, L) cell.
#
# Run AFTER both full-backprop training arrays have produced checkpoints:
#   sbatch slurm/eval_graphqa_fullbp.slurm
# =====================================================================
#SBATCH --job-name=graphqa_fullbp_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=axa@a100          # <-- ADJUST: your A100 allocation
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-3                # indices 0-3 = TRM

set -euo pipefail

# Same 4 cells as training, once per family.
H_LIST=(2 3 4 6)
L_LIST=(6 6 6 6)
FAM_LIST=(trm trm trm trm)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
FAM=${FAM_LIST[$SLURM_ARRAY_TASK_ID]}

# Run names must match the training scripts exactly: HRM has no family infix.
if [[ "${FAM}" == "trm" ]]; then
  RUN=graphqa_trm_H${H}_L${L}_fullbp
else
  RUN=graphqa_H${H}_L${L}_fullbp
fi

module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0            # <-- must match training
source $WORK/HRM-Text-for-Graphs/.a100_venv/bin/activate

export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd $WORK/HRM-Text-for-Graphs
mkdir -p logs

CKPT_DIR=$SCRATCH/graphqa/ckpts/${RUN}
# Evaluate on the held-out test split (swap to val.jsonl for validation).
EVAL_DATA=data/graphqa/hrm-text/standard/test.jsonl

echo "[${RUN}] evaluating ${CKPT_DIR} on ${EVAL_DATA}"

srun python scripts/eval_graphqa.py \
  --ckpt_path ${CKPT_DIR} \
  --data ${EVAL_DATA} \
  --use_ema \
  --max_generation 32 \
  --wandb_project graphqa-eval \
  --wandb_tags H${H} L${L} ${FAM} fullbp ablation \
  --out ${CKPT_DIR}/eval_test.json
