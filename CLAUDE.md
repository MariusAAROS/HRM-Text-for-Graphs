# CLAUDE.md

## Response style

- Give factual, straight-to-the-point answers. No filler, no padding, no restating the question.
- State results and numbers directly; flag uncertainty only when it matters.

## Environment

- Use the conda env **`hrm-text-graph-pt`** for everything (python, tests, scripts).
  - Run commands with `conda run -n hrm-text-graph-pt ...` or the interpreter directly:
    `$HOME/miniforge3/envs/hrm-text-graph-pt/bin/python` (the local run scripts use this via `PY`).
- Local machine: GPUs, no SLURM. `slurm/` targets Jean Zay (A100, `.a100_venv`, offline W&B) — not used locally.
- Attention runs on PyTorch FlexAttention here (no FlashAttention 3); FlexAttention has no CPU backward, so GPU-dependent checks need a GPU.

## What this repo is

Fork of HRM-Text (hierarchical recurrent LM, H/L modules) used to study recursion depth on graph reasoning:
GraphQA (main), plus KGQA (MetaQA, KQA Pro), GSM8k, MATH, MMLU. Models are trained **from scratch** at size B
(`config/cfg_graphqa.yaml`), not fine-tuned from the pretrained 1B.

Main experiments:
- **H/L grid ablation** — `(arch.H_cycles, arch.L_cycles)` sweep, truncated BPTT (`slurm/train_graphqa.slurm`).
- **Full-backprop vs truncated** — `arch.full_backprop=True compile_scope=block` (`scripts/run_graphqa_fullbp_local.sh`, W&B project `HRM-GraphQA-FullBP-Local`).
- **Scaled data** — ~1M samples, `config/cfg_graphqa_scaled.yaml` (`scripts/run_graphqa_scaled.sh`, W&B `HRM-GraphQA-Scaled`).

## Key files

- `pretrain.py` — Hydra/FSDP2 training entrypoint (also used for SFT/from-scratch task training). No in-training eval.
- `models/baselines/hrm_nocarry_bp_warmup.py` — HRM model; BPTT budget (`bp_warmup_ratio`, `bp_max_steps`, `full_backprop`, `H_bp_steps`/`L_bp_steps`).
- `models/baselines/trm_nocarry.py`, `rins_nocarry.py`, `ut_nocarry.py`, `transformer_wrapper.py` — baselines.
- `models/transformer.py` — blocks; `grad_checkpointing` requires compiled blocks (`compile_scope=block`) — eager FlexAttention gives wrong grads under recompute.
- `config/arch/net/*.yaml`, `config/arch/size/*.yaml` — architecture and size presets.
- `scripts/prepare_*.sh`, `scripts/prepare_sft_data.py` — tokenize JSONL → packed data. `--epochs` must equal `epochs` in the training config.
- `scripts/eval_graphqa.py` — greedy exact-match eval; reads H/L from the checkpoint's `all_config.yaml`.
- `report/` — paper figures, tables, notebooks (`paper_results.ipynb`).

## Commands

```bash
# Train one GraphQA config from scratch
conda run -n hrm-text-graph-pt python pretrain.py --config-name cfg_graphqa \
  data.path=<prepared_dir> arch.H_cycles=2 arch.L_cycles=3 run_name=graphqa_H2_L3 checkpoint_path=<ckpt_dir>

# Evaluate
conda run -n hrm-text-graph-pt python scripts/eval_graphqa.py \
  --ckpt_path <ckpt_dir> --data data/graphqa/hrm-text/standard/test.jsonl --use_ema --out <ckpt_dir>/eval_test.json

# Checks
conda run -n hrm-text-graph-pt python scripts/validate_bp_steps.py     # BPTT budget correctness (GPU)
conda run -n hrm-text-graph-pt python scripts/test_prepare_sft_filter.py  # data prep filter (CPU)
```

## Gotchas

- Keep `WANDB_MODE` online/offline, never `disabled`: `all_config.yaml` (required by eval) is only written with a W&B run.
- `resume_from` restarts the LR and bp-warmup schedules at step 0 — interrupted runs must be retrained from scratch.
- `global_batch_size` = packed tokens per step (Flex sequence length); keep it a multiple of 128.
- Truncated BPTT budget shrinks with H (H_bp = min(H, bp-1), L_bp = bp - H_bp), which confounds depth with gradient signal; the full-BP arm exists to de-confound it.
- Always set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for training.
