#!/bin/bash
# =====================================================================
# TRM full-backprop arm on GraphQA (Jean Zay, A100 partition).
#
# Mirrors slurm/train_graphqa_fullbp.slurm cell for cell so HRM and TRM stay
# comparable. TRM is the informative control here: under truncation it shows NO
# cliff at H=3 -> H=4 (0.395 -> 0.395/0.408) whereas HRM collapses (0.400 -> 0.062),
# despite both losing L_bp at the same point. If full backprop leaves TRM flat and
# lifts HRM, the cliff is an HRM-specific interaction with truncation, not depth.
#
# Submit with:  sbatch slurm/train_graphqa_trm_fullbp.slurm
# =====================================================================
#SBATCH --job-name=trm_graphqa_fullbp
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
#SBATCH --array=0-3

set -euo pipefail

# ---- (H,L) grid: identical to the HRM full-backprop arm --------------
H_LIST=(2 3 4 6)
L_LIST=(6 6 6 6)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=graphqa_trm_H${H}_L${L}_fullbp

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
CKPT_DIR=$SCRATCH/graphqa/ckpts/${RUN}

echo "[${RUN}] H_cycles=${H} L_cycles=${L}  full_backprop=True  data=${DATA_DIR}  ckpt=${CKPT_DIR}"

# ---- Launch: single GPU, size B, TRM net, from scratch --------------
# full_backprop=True overrides H_bp_steps/L_bp_steps from config/arch/net/trm.yaml.
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
  --config-name cfg_graphqa \
  arch/net@arch=trm \
  arch/size@arch=B \
  +arch.half_layers=True \
  data.path=${DATA_DIR} \
  arch.H_cycles=${H} \
  arch.L_cycles=${L} \
  arch.full_backprop=True \
  arch.grad_checkpointing=True \
  run_name=${RUN} \
  checkpoint_path=${CKPT_DIR}
