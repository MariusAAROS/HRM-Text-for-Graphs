#!/bin/bash
# =====================================================================
# Evaluate the trained MetaQA ablation checkpoints (exact-match accuracy).
# Mirrors the training job array: one eval job per (H,L) config.
#
# Run AFTER train_metaqa.slurm has produced checkpoints. Submit with:
#   sbatch slurm/eval_metaqa.slurm
# (or chain: sbatch --dependency=afterok:<train_jobid> slurm/eval_metaqa.slurm)
# =====================================================================
#SBATCH --job-name=hrm_metaqa_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=axa@a100          # <-- ADJUST: your A100 allocation
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu_a100-dev       # 1194 long prompts at batch 5 does not fit the 2h dev QoS
#SBATCH --array=0-8

set -euo pipefail

# Same grid as training.
H_LIST=(2 4 6 8 1 2 3 4 6)
L_LIST=(3 6 9 12 12 6 4 3 2)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=metaqa_H${H}_L${L}

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

CKPT_DIR=$SCRATCH/metaqa/ckpts/${RUN}
# Evaluate on the held-out test split (swap to val.jsonl for validation).
EVAL_DATA=data/metaqa/hrm-text/standard/test.jsonl

echo "[${RUN}] evaluating ${CKPT_DIR} on ${EVAL_DATA}"

# batch_size 5: generate_batch packs the whole batch into ONE sequence, and
# MetaQA prompts average 966 tokens (max 1313). 5 keeps the packed length near
# GraphQA's ~4.7k tokens instead of blowing up the O(S^2) attention.
srun python scripts/eval_graphqa.py \
  --ckpt_path ${CKPT_DIR} \
  --data ${EVAL_DATA} \
  --use_ema \
  --max_generation 32 \
  --batch_size 5 \
  --wandb_project metaqa-eval \
  --wandb_tags H${H} L${L} ablation \
  --out ${CKPT_DIR}/eval_test.json
