#!/bin/bash
# =====================================================================
# Evaluate the trained TRM GraphQA ablation checkpoints (exact-match accuracy).
# Mirrors train_graphqa_trm.slurm: one eval job per (H,L) config.
#
# The TRM architecture (net, cycles, half_layers) is read back from each
# checkpoint's all_config.yaml by scripts/eval_graphqa.py, so no arch flags
# are needed here -- the same eval script that handles HRM handles TRM.
#
# Run AFTER train_graphqa_trm.slurm has produced checkpoints. Submit with:
#   sbatch slurm/eval_graphqa_trm.slurm
# (or chain: sbatch --dependency=afterok:<train_jobid> slurm/eval_graphqa_trm.slurm)
# =====================================================================
#SBATCH --job-name=trm_graphqa_eval
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
#SBATCH --array=0-8

set -euo pipefail

# Same grid as training.
H_LIST=(2 4 6 8 1 2 3 4 6)
L_LIST=(3 6 9 12 12 6 4 3 2)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=graphqa_trm_H${H}_L${L}

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
  --wandb_tags trm H${H} L${L} ablation \
  --out ${CKPT_DIR}/eval_test.json
