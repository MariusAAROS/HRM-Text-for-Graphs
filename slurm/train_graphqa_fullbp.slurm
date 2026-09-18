#!/bin/bash
# =====================================================================
# HRM-Text full-backprop arm on GraphQA (Jean Zay, A100 partition).
#
# De-confounds depth from gradient truncation. In the main ablation
# (slurm/train_graphqa.slurm) the bp budget is a fixed total of 5 steps split
# H-first, so L_bp = 5 - min(H,4) SHRINKS as H grows: L_bp = 2 at H=3 but 1 at H=4.
# The HRM accuracy cliff (0.400 -> 0.062) lands on exactly that transition, so
# "deeper hurts" and "less gradient signal hurts" are not separable there.
#
# This arm re-runs a subset straddling the cliff with backprop through EVERY cycle
# (arch.full_backprop=True), holding everything else identical to the baseline cell.
# If the cliff disappears, it was truncation; if it persists, it is depth.
#
# Gradient checkpointing is mandatory here: (6,6) full backprop keeps
# 6*(6+1) = 42 block applications x 6 layers = 252 layer boundaries live.
#
# Submit with:  sbatch slurm/train_graphqa_fullbp.slurm
# =====================================================================
#SBATCH --job-name=hrm_graphqa_fullbp
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
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-3                 # 4 (H,L) configs straddling the H=3 -> H=4 cliff

set -euo pipefail

# ---- (H,L) grid: L fixed at 6, H swept across the cliff --------------
# Holding L constant isolates H. Each cell has a truncated twin already trained by
# slurm/train_graphqa.slurm under the same seed / lr / batch size, so the only
# difference is the backprop budget.
#
#  idx   H    L   H*(L+1)   truncated twin's (H_bp, L_bp)
#    0   2    6      14     (2, 3)
#    1   3    6      21     (3, 2)
#    2   4    6      28     (4, 1)   <-- L_bp collapses to 1 here
#    3   6    6      42     (4, 1)
H_LIST=(2 3 4 6)
L_LIST=(6 6 6 6)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=graphqa_H${H}_L${L}_fullbp

# ---- Environment (Jean Zay A100) ------------------------------------
module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0            # <-- must match setup_venv_jeanzay.sh
source $WORK/HRM-Text-for-Graphs/.a100_venv/bin/activate     # <-- venv (FlexAttention, no FA3)

export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd $WORK/HRM-Text-for-Graphs
mkdir -p logs

# ---- Persistent paths (shared by train + eval) ----------------------
DATA_DIR=$SCRATCH/graphqa/std_prepared       # from scripts/prepare_graphqa.sh
CKPT_DIR=$SCRATCH/graphqa/ckpts/${RUN}       # _fullbp suffix: never collides with the truncated twin

echo "[${RUN}] H_cycles=${H} L_cycles=${L}  full_backprop=True  data=${DATA_DIR}  ckpt=${CKPT_DIR}"

# ---- Launch: single GPU, size B, from scratch -----------------------
# full_backprop=True also zeroes bp_warmup_ratio (enforced in the config validator), so
# the gradient budget is constant from step 1. Verify in W&B: the instrumentation probe
# logs bp/H_bp_steps and bp/L_bp_steps, which must read H and H*L respectively.
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
  --config-name cfg_graphqa \
  arch/size@arch=B \
  data.path=${DATA_DIR} \
  arch.H_cycles=${H} \
  arch.L_cycles=${L} \
  arch.full_backprop=True \
  arch.grad_checkpointing=True \
  run_name=${RUN} \
  checkpoint_path=${CKPT_DIR}
