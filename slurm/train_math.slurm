#!/bin/bash
# =====================================================================
# HRM-Text L/H recursion ablation on MATH (Jean Zay, A100 partition).
#
# Same job layout as slurm/train_graphqa.slurm: one SLURM job per
# (H_cycles, L_cycles) config via a job array, size-B HRM from scratch,
# single A100.
#
# Prerequisites (run once, on a LOGIN node):
#   - venv built with scripts/setup_venv_jeanzay.sh (torch FlexAttention, no FA3).
#   - scripts/prepare_math.sh (tokenizer + HF cache + $SCRATCH/math/std_prepared).
#
# Submit with:  sbatch slurm/train_math.slurm
# =====================================================================
#SBATCH --job-name=hrm_math
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=axa@a100          # <-- ADJUST: your A100 allocation
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --array=0-8                 # 9 (H,L) configs (see grid below)

set -euo pipefail

# ---- (H,L) grid: array index -> config ------------------------------
# Axis 1  (fix ratio L/H = 1.5) :  (2,3) (4,6) (6,9) (8,12)
# Axis 2  (fix H*L ~= 12)       :  (1,12) (2,6) (3,4) (4,3) (6,2)
H_LIST=(2 4 6 8 1 2 3 4 6)
L_LIST=(3 6 9 12 12 6 4 3 2)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=math_H${H}_L${L}

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
DATA_DIR=$SCRATCH/math/std_prepared          # from scripts/prepare_math.sh
CKPT_DIR=$SCRATCH/math/ckpts/${RUN}

echo "[${RUN}] H_cycles=${H} L_cycles=${L}  data=${DATA_DIR}  ckpt=${CKPT_DIR}"

# ---- Launch: single GPU, size B, from scratch -----------------------
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
  --config-name cfg_math \
  arch/size@arch=B \
  data.path=${DATA_DIR} \
  arch.H_cycles=${H} \
  arch.L_cycles=${L} \
  run_name=${RUN} \
  checkpoint_path=${CKPT_DIR}
