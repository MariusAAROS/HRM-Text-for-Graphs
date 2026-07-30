#!/bin/bash
# =====================================================================
# Reproducible venv setup for HRM-Text on Jean Zay (H100 partition).
#
# Builds a Python venv layered on top of the H100 `pytorch-gpu` module
# (torch comes from the module, NOT pip) and compiles FlashAttention 3
# (Hopper / sm90a only) from source.
#
# The FA3 compile is memory-hungry and WILL be OOM-killed (nvcc exit 255)
# on a login node. Run it on the `compil` partition, which has internet
# AND enough RAM:
#   srun --partition=compil --time=02:00:00 --cpus-per-task=8 --hint=nomultithread --pty bash
#   cd $WORK/HRM-Text && bash scripts/setup_venv_jeanzay.sh
# (`prepost` also works; a login node only works with MAX_JOBS=1.)
#
# GPU compute nodes are offline, so all downloads + the FA3 compile must
# happen on compil/prepost/login -- never inside the GPU job.
# =====================================================================
set -euo pipefail

# ---- Config (keep in sync with slurm/*.slurm) -----------------------
PYTORCH_MODULE="pytorch-gpu/py3/2.8.0"   # <-- must match slurm/train_graphqa.slurm + slurm/eval_graphqa.slurm
VENV_DIR="${VENV_DIR:-$WORK/HRM-Text/.venv}"
FA_SRC="${FA_SRC:-$WORK/flash-attention}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Repo:          $REPO_DIR"
echo "==> Venv:          $VENV_DIR"
echo "==> FA3 source:    $FA_SRC"
echo "==> Torch module:  $PYTORCH_MODULE"

# ---- 1. Load the SAME modules the slurm jobs load (H100 first!) ------
module purge
module load arch/h100
module load "$PYTORCH_MODULE"

# ---- 2. Create venv inheriting the module's H100 torch + CUDA --------
if [[ ! -d "$VENV_DIR" ]]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# ---- 3. Pure-Python deps only ---------------------------------------
# Exclude packages that must NOT be pip-installed on Jean Zay:
#   torch          -> provided (H100-optimized) by the module
#   flash_attn_3   -> not on PyPI; compiled from source below
#   vllm / lm-eval -> pull conflicting torch; only for vllm benchmarks
REQ_FILTERED="$(mktemp)"
grep -vE '^(flash_attn_3|torch|vllm|lm-eval\[.*\])$' "$REPO_DIR/requirements.txt" > "$REQ_FILTERED"
echo "==> Installing filtered requirements:"
cat "$REQ_FILTERED"
pip install --no-cache-dir -r "$REQ_FILTERED"
rm -f "$REQ_FILTERED"

# ---- 4. Compile FlashAttention 3 for Hopper (H100-critical) ---------
# NOTE: FA3's hopper/setup.py hardcodes sm90a and IGNORES TORCH_CUDA_ARCH_LIST.
# By default it ALSO builds the sm80 (A100) kernels -> doubles compile time and
# memory and is the source of the `*_sm80.o` failures on H100. Disable them:
export FLASH_ATTENTION_DISABLE_SM80=TRUE   # H100-only build; skip A100 kernels
export MAX_JOBS="${MAX_JOBS:-4}"           # lower to 1 if building on a login node
export NVCC_THREADS="${NVCC_THREADS:-2}"

if [[ ! -d "$FA_SRC" ]]; then
    git clone https://github.com/Dao-AILab/flash-attention.git "$FA_SRC"
fi
pip install --no-build-isolation "$FA_SRC/hopper"

# ---- 5. Verify -------------------------------------------------------
python -c "import flash_attn_interface, torch; print('torch', torch.__version__, '| FA3 import OK')"

echo ""
echo "==> Done. Activate later with:"
echo "      module purge && module load arch/h100 && module load $PYTORCH_MODULE"
echo "      source $VENV_DIR/bin/activate"
