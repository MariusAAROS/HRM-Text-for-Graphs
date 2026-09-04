#!/bin/bash
# =====================================================================
# Evaluate the trained MMLU ablation checkpoints.
# Mirrors the training job array: one eval job per (H,L) config.
#
# Goes through evaluation/main.py so scoring uses MMLU.compute_metrics
# (per-subject micro/macro averages, 1-token constrained generation).
# Metrics land in W&B as eval/MMLU/*.
#
# 14042 test prompts at batch_size 1 (long 5-shot contexts) -> the longest
# eval of the set; budget accordingly.
#
# Run AFTER train_mmlu.slurm. Submit with:
#   sbatch slurm/eval_mmlu.slurm
# =====================================================================
#SBATCH --job-name=hrm_mmlu_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=axa@a100          # <-- ADJUST: your A100 allocation
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --array=0-8

set -euo pipefail

# Same grid as training.
H_LIST=(2 4 6 8 1 2 3 4 6)
L_LIST=(3 6 9 12 12 6 4 3 2)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=mmlu_H${H}_L${L}

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

CKPT_DIR=$SCRATCH/mmlu/ckpts/${RUN}

echo "[${RUN}] evaluating ${CKPT_DIR} on the MMLU test split"

srun python -m evaluation.main \
  config=evaluation/config/scratch_benchmarking.yaml \
  ckpt_path=${CKPT_DIR} \
  run_only=[MMLU] \
  out=${CKPT_DIR}/eval_test.json \
  samples_out=${CKPT_DIR}/eval_test_samples.jsonl \
  wandb_project=mmlu-eval \
  wandb_name=${RUN} \
  wandb_tags=[H${H},L${L},ablation]
