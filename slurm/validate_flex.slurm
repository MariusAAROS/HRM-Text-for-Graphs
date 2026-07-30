#!/bin/bash
# =====================================================================
# Validate the FlexAttention prefixLM port on an A100 (forward + gradient
# parity vs a masked-softmax reference). Run this once before training.
#
# Submit with:  sbatch slurm/validate_flex.slurm
# Check result: tail logs/hrm_validate_flex_<JOBID>.out  (expect "ALL PASS")
# =====================================================================
#SBATCH --job-name=hrm_validate_flex
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=axa@a100          # <-- ADJUST: your A100 allocation
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --qos=qos_gpu_a100-dev

set -euo pipefail

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

srun python scripts/validate_flex_attention.py
