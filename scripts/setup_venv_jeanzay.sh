#!/bin/bash
# =====================================================================
# Reproducible venv setup for HRM-Text on Jean Zay (A100 partition).
#
# Builds a Python venv layered on top of the A100 `pytorch-gpu` module
# (torch comes from the module, NOT pip). The attention path uses torch
# FlexAttention, so there is NO FlashAttention-3 build step (FA3 is Hopper
# only). Run on a login node -- everything here is pure-Python / downloads.
#
# GPU compute nodes are offline, so all downloads must happen here,
# never inside the GPU job.
# =====================================================================
set -euo pipefail

# ---- Config (keep in sync with slurm/*.slurm) -----------------------
PYTORCH_MODULE="pytorch-gpu/py3/2.8.0"   # <-- must match slurm/train_graphqa.slurm + slurm/eval_graphqa.slurm
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"  # lives in the repo -> matches slurm 'source .../.venv'

echo "==> Repo:          $REPO_DIR"
echo "==> Venv:          $VENV_DIR"
echo "==> Torch module:  $PYTORCH_MODULE"

# ---- 1. Load the SAME modules the slurm jobs load (A100 first!) ------
module purge
module load arch/a100
module load "$PYTORCH_MODULE"

# ---- 2. Create venv inheriting the module's A100 torch + CUDA --------
if [[ ! -d "$VENV_DIR" ]]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# ---- 3. Pure-Python deps only ---------------------------------------
# Exclude packages that must NOT be pip-installed on Jean Zay:
#   torch          -> provided (A100-optimized) by the module
#   vllm / lm-eval -> pull conflicting torch; only for vllm benchmarks
REQ_FILTERED="$(mktemp)"
grep -vE '^(torch|vllm|lm-eval\[.*\])$' "$REPO_DIR/requirements.txt" > "$REQ_FILTERED"
echo "==> Installing filtered requirements:"
cat "$REQ_FILTERED"
pip install --no-cache-dir -r "$REQ_FILTERED"
rm -f "$REQ_FILTERED"

# ---- 4. Verify (FlexAttention, no FA3 build needed) -----------------
python -c "import torch; from torch.nn.attention.flex_attention import flex_attention; print('torch', torch.__version__, '| FlexAttention import OK')"

echo ""
echo "==> Done. Activate later with:"
echo "      module purge && module load arch/a100 && module load $PYTORCH_MODULE"
echo "      source $VENV_DIR/bin/activate"
