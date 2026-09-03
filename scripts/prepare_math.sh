#!/bin/bash
# =====================================================================
# One-time MATH (hendrycks_math) data preparation for the L/H ablation.
# Run this on a Jean Zay LOGIN node (needs internet), NOT inside a compute
# job. GPU nodes run with HF_HUB_OFFLINE=1, so the eval split must be
# cached here too -- evaluation/benchmarks.py calls load_dataset() in
# MATH.__init__ (once per subject config).
# =====================================================================
set -euo pipefail

cd $WORK/HRM-Text-for-Graphs

# EPOCHS **must** equal `epochs` in config/cfg_math.yaml.
EPOCHS=30
DATA_DIR=$SCRATCH/math/std_prepared
TRAIN_JSONL=data/math/hrm-text/train.jsonl
TOKENIZER=$WORK/HRM-Text-for-Graphs/tokenizer.json

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

# ---- 2. Prefetch every subject config, both splits ------------------
python - <<'PY'
from datasets import load_dataset, get_dataset_config_names
subsets = get_dataset_config_names("EleutherAI/hendrycks_math")
print(f"{len(subsets)} subsets: {subsets}")
for subset in subsets:
    for split in ("train", "test"):
        ds = load_dataset("EleutherAI/hendrycks_math", subset, split=split)
        assert len(ds) > 0, f"empty hendrycks_math/{subset}[{split}]"
        print(f"cached hendrycks_math/{subset}[{split}] = {len(ds)} rows")
PY

# ---- 3. Build the direct-answer train JSONL -------------------------
python scripts/prepare_benchmark_data.py --benchmark math --output "$TRAIN_JSONL"

# ---- 4. Tokenize into the V1Dataset layout --------------------------
python scripts/prepare_sft_data.py \
  --train "$TRAIN_JSONL" \
  --tokenizer "$TOKENIZER" \
  --output "$DATA_DIR" \
  --epochs "$EPOCHS"

echo "Prepared data at $DATA_DIR"
echo "Eval runs through evaluation/main.py against the cached HF test splits."
