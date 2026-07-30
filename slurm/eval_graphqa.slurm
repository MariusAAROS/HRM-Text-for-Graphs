#!/bin/bash
# =====================================================================
# Evaluate the trained GraphQA ablation checkpoints (exact-match accuracy).
# Mirrors the training job array: one eval job per (H,L) config.
#
# Run AFTER train_graphqa.slurm has produced checkpoints. Submit with:
#   sbatch slurm/eval_graphqa.slurm
# (or chain: sbatch --dependency=afterok:<train_jobid> slurm/eval_graphqa.slurm)
# =====================================================================
#SBATCH --job-name=hrm_graphqa_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=axa@h100          # <-- ADJUST: your H100 allocation
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --array=0-8

set -euo pipefail

# Same grid as training.
H_LIST=(2 4 6 8 1 2 3 4 6)
L_LIST=(3 6 9 12 12 6 4 3 2)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=graphqa_H${H}_L${L}

module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0            # <-- ADJUST to match training
source $WORK/HRM-Text/.venv/bin/activate

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
  --batch_size 32 \
  --max_generation 32 \
  --out ${CKPT_DIR}/eval_test.json
