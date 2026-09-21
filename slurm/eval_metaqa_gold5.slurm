#!/bin/bash
# =====================================================================
# Evaluate the trained metaqa-gold-5 checkpoints.
# Mirrors the training job array: one eval job per (H,L) config.
#
# scripts/eval_graphqa.py reports exact match plus set-based
# precision/recall/F1/hits@1 over the '|'-joined answer lists.
#
# Run AFTER train_metaqa_gold5.slurm has produced checkpoints. Submit with:
#   sbatch slurm/eval_metaqa_gold5.slurm
# (or chain: sbatch --dependency=afterok:<train_jobid> slurm/eval_metaqa_gold5.slurm)
# =====================================================================
#SBATCH --job-name=hrm_metaqa_gold5_eval
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
#SBATCH --qos=qos_gpu_a100-dev        # 128-token answers = 4x the forwards; the 2h dev QoS is too tight
#SBATCH --array=0-5

set -euo pipefail

VARIANT=metaqa-gold-5

# Same grid as training -- see train_metaqa_gold5.slurm for the index table.
H_LIST=(1 3 5 6 2 1)
L_LIST=(6 6 6 6 12 1)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=metaqa_gold5_H${H}_L${L}

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

CKPT_DIR=$SCRATCH/${VARIANT}/ckpts/${RUN}
# Evaluate on the held-out test split (swap to val.jsonl for validation).
EVAL_DATA=data/${VARIANT}/hrm-text/standard/test.jsonl

echo "[${RUN}] evaluating ${CKPT_DIR} on ${EVAL_DATA}"

# max_generation 128: gold-5 answers are up to 43 tokens (median 19), so the 32
# used for single-answer MetaQA would truncate the list and silently cap recall.
# batch_size 5 keeps the packed sequence near GraphQA's ~4.7k tokens (generate_batch
# packs the whole batch into ONE sequence, and attention is O(S^2)).
srun python scripts/eval_graphqa.py \
  --ckpt_path ${CKPT_DIR} \
  --data ${EVAL_DATA} \
  --use_ema \
  --max_generation 128 \
  --batch_size 5 \
  --wandb_project metaqa-gold5-eval \
  --wandb_tags gold5 H${H} L${L} ablation \
  --out ${CKPT_DIR}/eval_test.json
