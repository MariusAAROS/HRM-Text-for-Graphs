#!/bin/bash
# =====================================================================
# HRM-Text L/H recursion ablation on MMLU (Jean Zay, A100 partition).
#
# Same job layout as slurm/train_graphqa.slurm: one SLURM job per
# (H_cycles, L_cycles) config via a job array, size-B HRM from scratch,
# single A100.
#
# NOTE: MMLU auxiliary_train is 40.7M tokens, so cfg_mmlu.yaml trains for a
# SINGLE epoch (~9.9k steps) -- still 2.3x GraphQA's step count. If the deep
# configs (8,12)/(6,9) hit the 20h wall, re-run prepare_mmlu.sh with LIMIT set.
#
# Prerequisites (run once, on a LOGIN node):
#   - venv built with scripts/setup_venv_jeanzay.sh (torch FlexAttention, no FA3).
#   - scripts/prepare_mmlu.sh (tokenizer + HF cache + $SCRATCH/mmlu/std_prepared).
#
# Submit with:  sbatch slurm/train_mmlu.slurm
# =====================================================================
#SBATCH --job-name=hrm_mmlu
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
#SBATCH --array=0-5                 # 6 (H,L) configs (see grid below)

set -euo pipefail

# ---- (H,L) grid: array index -> config ------------------------------
# [v1] Axis 1  (fix ratio L/H = 1.5) :  (2,3) (4,6) (6,9) (8,12)
# [v1] Axis 2  (fix H*L ~= 12)       :  (1,12) (2,6) (3,4) (4,3) (6,2)
# [v1] H_LIST=(2 4 6 8 1 2 3 4 6)
# [v1] L_LIST=(3 6 9 12 12 6 4 3 2)
#
# v2 confirmation subset of the GraphQA 2-D surface (full 16-cell grid runs on
# GraphQA only; see train_graphqa.slurm). Together with the v1 cells (2,6) and (4,6)
# this completes the L=6 row for H = 1..6, i.e. the H main effect at fixed L, plus
# (2,12) off-row and the non-recurrent control (1,1).
#  idx   H    L   H*L    L/H   H*(L+1)
#    0   1    6     6   6.00       7
#    1   3    6    18   2.00      21
#    2   5    6    30   1.20      35
#    3   6    6    36   1.00      42
#    4   2   12    24   6.00      26
#    5   1    1     1   1.00       2
H_LIST=(1 3 5 6 2 1)
L_LIST=(6 6 6 6 12 1)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=mmlu_H${H}_L${L}

# ---- Environment (Jean Zay A100) ------------------------------------
module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0            # <-- must match setup_venv_jeanzay.sh
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

# ---- Persistent paths (shared by train + eval) ----------------------
DATA_DIR=$SCRATCH/mmlu/std_prepared          # from scripts/prepare_mmlu.sh
CKPT_DIR=$SCRATCH/mmlu/ckpts/${RUN}

echo "[${RUN}] H_cycles=${H} L_cycles=${L}  data=${DATA_DIR}  ckpt=${CKPT_DIR}"

# ---- Launch: single GPU, size B, from scratch -----------------------
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
  --config-name cfg_mmlu \
  arch/size@arch=B \
  data.path=${DATA_DIR} \
  arch.H_cycles=${H} \
  arch.L_cycles=${L} \
  run_name=${RUN} \
  checkpoint_path=${CKPT_DIR}
