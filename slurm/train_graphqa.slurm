#!/bin/bash
# =====================================================================
# HRM-Text L/H recursion ablation on GraphQA (Jean Zay, A100 partition).
#
# One SLURM job per (H_cycles, L_cycles) config via a job array. Each job
# trains a size-B HRM from scratch on graphqa-standard on a single A100.
#
# Prerequisites (run once, on a LOGIN node):
#   - venv built with scripts/setup_venv_jeanzay.sh (torch FlexAttention, no FA3).
#   - tokenizer downloaded + graphqa tokenized to $SCRATCH/graphqa/std_prepared
#     via scripts/prepare_graphqa.sh.
#
# Submit with:  sbatch slurm/train_graphqa.slurm
# =====================================================================
#SBATCH --job-name=hrm_graphqa
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=axa@a100          # <-- ADJUST: your A100 allocation
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8           # A100 node: ~8 CPU cores per GPU
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3       # production QoS; the deepest config (6,12) needs >2h
#SBATCH --array=0-15%10             # 16 (H,L) configs (see grid below); %10 = Jean Zay parallel cap

set -euo pipefail

# ---- (H,L) grid: array index -> config ------------------------------
# [v1] Axis 1  (fix ratio L/H = 1.5) :  (2,3) (4,6) (6,9) (8,12)
# [v1] Axis 2  (fix H*L ~= 12)       :  (1,12) (2,6) (3,4) (4,3) (6,2)
# [v1] H_LIST=(2 4 6 8 1 2 3 4 6)
# [v1] L_LIST=(3 6 9 12 12 6 4 3 2)
#
# v2 replaces the two v1 axes with a 2-D (H,L) surface: each v1 axis had a single
# degree of freedom (at fixed H*L=D the ratio is forced to D/H^2), so a ratio effect
# was not separable from an H effect. Core is a balanced 5x3 rectangle,
# H in {1,2,3,4,6} x L in {3,6,12}; the 5 cells v1 already trained -- (2,3) (4,3)
# (2,6) (4,6) (1,12) -- are reused, not re-run, so this array holds the 10 missing
# rectangle cells plus 6 probes: H=5 at L=6 (resolve the H=3 -> H=4 cliff), L=1 at
# H=1,2,4 (non-recurrent control (1,1) + pure-H-recursion anchors), L=24 at H=1,2
# (push past the v1 corner optimum (1,12)).
#
#  idx   H    L   H*L    L/H   H*(L+1)  |  idx   H    L   H*L    L/H   H*(L+1)
#    0   1    3     3   3.00       4    |    8   3   12    36   4.00      39
#    1   3    3     9   1.00      12    |    9   4   12    48   3.00      52
#    2   6    3    18   0.50      24    |   10   6   12    72   2.00      78
#    3   1    6     6   6.00       7    |   11   1    1     1   1.00       2
#    4   3    6    18   2.00      21    |   12   2    1     2   0.50       4
#    5   5    6    30   1.20      35    |   13   4    1     4   0.25       8
#    6   6    6    36   1.00      42    |   14   1   24    24  24.00      25
#    7   2   12    24   6.00      26    |   15   2   24    48  12.00      50
#
# H*(L+1) is the block-application count (L L-block applications plus one H-block
# per cycle); H*L alone undercounts the high-level work by H. Deepest new config is
# (6,12) = 78, below the already-completed v1 (8,12) = 104, so 20h still holds.
H_LIST=(1 3 6 1 3 5 6 2 3 4 6 1 2 4 1 2)
L_LIST=(3 3 3 6 6 6 6 12 12 12 12 1 1 1 24 24)
H=${H_LIST[$SLURM_ARRAY_TASK_ID]}
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
RUN=graphqa_H${H}_L${L}

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

echo "[${RUN}] H_cycles=${H} L_cycles=${L}  data=${DATA_DIR}  ckpt=${CKPT_DIR}"

# ---- Launch: single GPU, size B, from scratch -----------------------
# Use the venv python to run torchrun (module torchrun -> module python, which
# lacks the venv-only deps like coolname). `python -m torch.distributed.run`
# == torchrun, but under the venv interpreter, so workers inherit it too.
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
  --config-name cfg_graphqa \
  arch/size@arch=B \
  data.path=${DATA_DIR} \
  arch.H_cycles=${H} \
  arch.L_cycles=${L} \
  run_name=${RUN} \
  checkpoint_path=${CKPT_DIR}
