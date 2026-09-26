#!/bin/bash
# =====================================================================
# TRM baseline L/H recursion ablation on GraphQA (Jean Zay, A100 partition).
#
# Head-to-head with the HRM ablation: same (H_cycles, L_cycles) grid, and
# half_layers=True so each config is FLOP-matched to the HRM runs (HRM splits
# 12 layers across two 6-layer blocks; TRM shares ONE 6-layer block for both
# the H and L updates -- see models/baselines/trm_nocarry.py). TRM therefore
# runs on the SAME A100 FlexAttention path as HRM (no code change needed).
#
# One SLURM job per (H_cycles, L_cycles) config via a job array. Each job
# trains a size-B TRM from scratch on graphqa-standard on a single A100.
#
# Prerequisites (run once, on a LOGIN node):
#   - venv built with scripts/setup_venv_jeanzay.sh (torch FlexAttention, no FA3).
#   - tokenizer downloaded + graphqa tokenized to $SCRATCH/graphqa/std_prepared
#     via scripts/prepare_graphqa.sh.
#
# Submit with:  sbatch slurm/train_graphqa_trm.slurm
# =====================================================================
#SBATCH --job-name=trm_graphqa
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
#SBATCH --qos=qos_gpu_a100-dev       # production QoS; the deepest config (6,12) needs >2h
#SBATCH --array=0-15%8             # 16 (H,L) configs (see grid below); %10 = Jean Zay parallel cap

set -euo pipefail

# ---- (H,L) grid: array index -> config ------------------------------
# [v1] Axis 1  (fix ratio L/H = 1.5) :  (2,3) (4,6) (6,9) (8,12)
# [v1] Axis 2  (fix H*L ~= 12)       :  (1,12) (2,6) (3,4) (4,3) (6,2)
# [v1] H_LIST=(2 4 6 8 1 2 3 4 6)
# [v1] L_LIST=(3 6 9 12 12 6 4 3 2)
# [v1] HBP_LIST=(2 4 4 4 1 2 3 4 4)
# [v1] LBP_LIST=(3 1 1 1 4 3 2 1 1)
#
# v2 2-D (H,L) surface -- see train_graphqa.slurm for the full index table and the
# rationale. Mirrors the HRM grid exactly so the two models stay comparable cell by cell.
# [v1] H_LIST=(2 4 6 8 1 2 3 4 6)
# [v1] L_LIST=(3 6 9 12 12 6 4 3 2)
H_LIST=(1 3 6 1 3 5 6 2 3 4 6 1 2 4 1 2)
L_LIST=(3 3 3 6 6 6 6 12 12 12 12 1 1 1 24 24)
# H_LIST=(1 3 6 1 3 5 6 2 3 4 6 1 2 4 1 2 2 4 6 8 1 2 3 4 6) # [v1 + v2]
# L_LIST=(3 3 3 6 6 6 6 12 12 12 12 1 1 1 24 24 3 6 9 12 12 6 4 3 2) # [v1 + v2]
# TRM has no bp warmup: H_bp_steps/L_bp_steps are fixed per config. Mirror HRM's
# steady state (total bp budget 5): H_bp = min(H_cycles, 4), L_bp = 5 - H_bp.
# v2 additionally clamps L_bp to L_cycles, which v1 never needed (it had no L=1 cell).
HBP_LIST=(1 3 4 1 3 4 4 2 3 4 4 1 2 4 1 2)
LBP_LIST=(3 2 1 4 2 1 1 3 2 1 1 1 1 1 4 3)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
HBP=${HBP_LIST[$SLURM_ARRAY_TASK_ID]}
LBP=${LBP_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=graphqa_trm_H${H}_L${L}

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

echo "[${RUN}] H_cycles=${H} L_cycles=${L}  H_bp=${HBP} L_bp=${LBP}  data=${DATA_DIR}  ckpt=${CKPT_DIR}"

# ---- Launch: single GPU, size B, TRM net, from scratch --------------
# Swap the net to TRM (arch/net@arch=trm) and halve layers so each (H,L) is
# FLOP-matched to the HRM runs. TRM's bp is fixed (no warmup) -> set H/L bp
# steps explicitly per config. Everything else mirrors train_graphqa.slurm.
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
  --config-name cfg_graphqa \
  arch/net@arch=trm \
  arch/size@arch=B \
  +arch.half_layers=True \
  data.path=${DATA_DIR} \
  arch.H_cycles=${H} \
  arch.L_cycles=${L} \
  arch.H_bp_steps=${HBP} \
  arch.L_bp_steps=${LBP} \
  run_name=${RUN} \
  checkpoint_path=${CKPT_DIR}
