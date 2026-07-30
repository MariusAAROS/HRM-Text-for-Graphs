#!/bin/bash
# =====================================================================
# One-time GraphQA data preparation for the L/H ablation.
# Run this on a Jean Zay LOGIN node (needs internet for the tokenizer),
# NOT inside a compute job. Everything downstream runs offline.
# =====================================================================
set -euo pipefail

cd $WORK/HRM-Text-for-Graphs

# EPOCHS **must** equal `epochs` in config/cfg_graphqa.yaml.
EPOCHS=20
DATA_DIR=$SCRATCH/graphqa/std_prepared
TOKENIZER=$WORK/HRM-Text-for-Graphs/tokenizer.json

# ---- 1. Download the HRM-Text-1B tokenizer (tokenizer.json only) -----
export HF_HOME=$WORK/hf_cache
python - "$TOKENIZER" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
src = hf_hub_download("sapientinc/HRM-Text-1B", "tokenizer.json")
dst = sys.argv[1]
shutil.copyfile(src, dst)
print("tokenizer ->", dst)
PY

# ---- 2. Tokenize graphqa-standard TRAIN into the V1Dataset layout ----
python scripts/prepare_sft_data.py \
  --train data/graphqa/hrm-text/standard/train.jsonl \
  --tokenizer "$TOKENIZER" \
  --output "$DATA_DIR" \
  --epochs "$EPOCHS"

echo "Prepared data at $DATA_DIR"
echo "val/test JSONL are used raw by scripts/eval_graphqa.py (no tokenization needed)."
