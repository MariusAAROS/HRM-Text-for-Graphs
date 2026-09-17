#!/bin/bash
# =====================================================================
# HRM-Text L/H recursion ablation on the metaqa-retrieved-1 variant
# (retrieved -- not gold -- subgraph, '|'-joined answer lists).
#
# Identical to slurm/train_metaqa.slurm except for the dataset paths:
# cfg_metaqa.yaml and the (H,L) grid are reused unchanged so the variant
# stays directly comparable to the single-answer MetaQA runs.
#
# Prerequisites (run once, on a LOGIN node):
#   - venv built with scripts/setup_venv_jeanzay.sh (torch FlexAttention, no FA3).
#   - bash scripts/prepare_metaqa.sh metaqa-retrieved-1
#
# Submit with:  sbatch slurm/train_metaqa_retrieved1.slurm
# =====================================================================
#SBATCH --job-name=hrm_metaqa_retrieved1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=axa@a100          # <-- ADJUST: your A100 allocation
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8           # A100 node: ~8 CPU cores per GPU
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-5                 # 6 (H,L) configs (see grid below)

set -euo pipefail

VARIANT=metaqa-retrieved-1

# ---- (H,L) grid: array index -> config ------------------------------
# Same 6-cell subset as slurm/train_metaqa.slurm (L=6 row for H = 1..6,
# plus (2,12) off-row and the non-recurrent control (1,1)).
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
RUN=metaqa_retrieved1_H${H}_L${L}

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
DATA_DIR=$SCRATCH/${VARIANT}/std_prepared    # from scripts/prepare_metaqa.sh ${VARIANT}
CKPT_DIR=$SCRATCH/${VARIANT}/ckpts/${RUN}

echo "[${RUN}] H_cycles=${H} L_cycles=${L}  data=${DATA_DIR}  ckpt=${CKPT_DIR}"

# ---- Launch: single GPU, size B, from scratch -----------------------
# Use the venv python to run torchrun (module torchrun -> module python, which
# lacks the venv-only deps like coolname). `python -m torch.distributed.run`
# == torchrun, but under the venv interpreter, so workers inherit it too.
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
  --config-name cfg_metaqa \
  arch/size@arch=B \
  data.path=${DATA_DIR} \
  arch.H_cycles=${H} \
  arch.L_cycles=${L} \
  project_name=HRM-MetaQA-Retrieved1-Ablation \
  run_name=${RUN} \
  checkpoint_path=${CKPT_DIR}
