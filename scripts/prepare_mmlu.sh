#!/bin/bash
# =====================================================================
# One-time MMLU data preparation for the L/H ablation.
# Run this on a Jean Zay LOGIN node (needs internet), NOT inside a compute
# job. GPU nodes run with HF_HUB_OFFLINE=1, so the dev (few-shot) and test
# splits must be cached here too -- evaluation/benchmarks.py loads both in
# MMLU.__init__.
# =====================================================================
set -euo pipefail

cd $WORK/HRM-Text-for-Graphs

# EPOCHS **must** equal `epochs` in config/cfg_mmlu.yaml.
EPOCHS=1
DATA_DIR=$SCRATCH/mmlu/std_prepared
TRAIN_JSONL=data/mmlu/hrm-text/train.jsonl
TOKENIZER=$WORK/HRM-Text-for-Graphs/tokenizer.json
# Optionally cap auxiliary_train (99842 samples / 40.7M tokens) to shorten the
# job, e.g. LIMIT="--limit 30000". Keep cfg_mmlu.yaml `epochs` in sync.
LIMIT=""

export HF_HOME=$WORK/hf_cache

# ---- 1. Download the HRM-Text-1B tokenizer (tokenizer.json only) -----
if [[ ! -s "$TOKENIZER" ]]; then
  python - "$TOKENIZER" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
src = hf_hub_download("sapientinc/HRM-Text-1B", "tokenizer.json")
shutil.copyfile(src, sys.argv[1])
print("tokenizer ->", sys.argv[1])
PY
else
  echo "tokenizer already present -> $TOKENIZER"
fi

# ---- 2. Prefetch train + few-shot + eval splits ---------------------
python - <<'PY'
from datasets import load_dataset
for split in ("auxiliary_train", "dev", "test"):
    ds = load_dataset("cais/mmlu", "all", split=split)
    assert len(ds) > 0, f"empty mmlu {split}"
    print(f"cached cais/mmlu[{split}] = {len(ds)} rows")
PY

# ---- 3. Build the direct-answer train JSONL -------------------------
python scripts/prepare_benchmark_data.py --benchmark mmlu --output "$TRAIN_JSONL" $LIMIT

# ---- 4. Tokenize into the V1Dataset layout --------------------------
python scripts/prepare_sft_data.py \
  --train "$TRAIN_JSONL" \
  --tokenizer "$TOKENIZER" \
  --output "$DATA_DIR" \
  --epochs "$EPOCHS"

echo "Prepared data at $DATA_DIR"
echo "Eval runs through evaluation/main.py against the cached HF test split."
