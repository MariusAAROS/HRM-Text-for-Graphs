#!/bin/bash
# =====================================================================
# One-time MetaQA data preparation for the L/H ablation.
# Run this on a Jean Zay LOGIN node (needs internet for the tokenizer),
# NOT inside a compute job. Everything downstream runs offline.
# =====================================================================
set -euo pipefail

cd $WORK/HRM-Text-for-Graphs

# EPOCHS **must** equal `epochs` in config/cfg_metaqa.yaml.
EPOCHS=6
DATA_DIR=$SCRATCH/metaqa/std_prepared
TOKENIZER=$WORK/HRM-Text-for-Graphs/tokenizer.json

# ---- 1. Download the HRM-Text-1B tokenizer (tokenizer.json only) -----
export HF_HOME=$WORK/hf_cache
if [[ ! -s "$TOKENIZER" ]]; then
  python - "$TOKENIZER" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
src = hf_hub_download("sapientinc/HRM-Text-1B", "tokenizer.json")
dst = sys.argv[1]
shutil.copyfile(src, dst)
print("tokenizer ->", dst)
PY
else
  echo "tokenizer already present -> $TOKENIZER"
fi

# ---- 2. Convert the baseline JSON into HRM-Text JSONL ---------------
# Point at baseline/ (NOT data/metaqa/) so kb.txt's sibling JSON files at the
# dataset root are not picked up by the converter's recursive *.json glob.
python scripts/format_graphqa_to_hrm.py \
  --input-dir data/metaqa/baseline \
  --output-dir data/metaqa/hrm-text

# ---- 3. Tokenize metaqa-standard TRAIN into the V1Dataset layout ----
# Longest measured sample is 1367 tokens, so the default 4097 context fits.
python scripts/prepare_sft_data.py \
  --train data/metaqa/hrm-text/standard/train.jsonl \
  --tokenizer "$TOKENIZER" \
  --output "$DATA_DIR" \
  --epochs "$EPOCHS"

echo "Prepared data at $DATA_DIR"
echo "val/test JSONL are used raw by scripts/eval_graphqa.py (no tokenization needed)."
