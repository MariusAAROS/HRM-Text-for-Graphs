#!/bin/bash
# =====================================================================
# Evaluate the trained GSM8k ablation checkpoints.
# Mirrors the training job array: one eval job per (H,L) config.
#
# Unlike the KGQA evals this goes through evaluation/main.py so the real
# benchmark scorer (GSM8k.compute_metrics) is used rather than raw
# exact-match. Metrics land in W&B as eval/GSM8k/*.
#
# Run AFTER train_gsm8k.slurm. Submit with:
#   sbatch slurm/eval_gsm8k.slurm
# =====================================================================
#SBATCH --job-name=hrm_gsm8k_eval
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
#SBATCH --array=0-5

set -euo pipefail

# Same grid as training -- see train_gsm8k.slurm for the v2 index table.
# [v1] H_LIST=(2 4 6 8 1 2 3 4 6)
# [v1] L_LIST=(3 6 9 12 12 6 4 3 2)
H_LIST=(1 3 5 6 2 1)
L_LIST=(6 6 6 6 12 1)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=gsm8k_H${H}_L${L}

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

CKPT_DIR=$SCRATCH/gsm8k/ckpts/${RUN}

echo "[${RUN}] evaluating ${CKPT_DIR} on the GSM8k test split"

srun python -m evaluation.main \
  config=evaluation/config/scratch_benchmarking.yaml \
  ckpt_path=${CKPT_DIR} \
  run_only=[GSM8k] \
  out=${CKPT_DIR}/eval_test.json \
  samples_out=${CKPT_DIR}/eval_test_samples.jsonl \
  wandb_project=gsm8k-eval \
  wandb_name=${RUN} \
  wandb_tags=[H${H},L${L},ablation]
