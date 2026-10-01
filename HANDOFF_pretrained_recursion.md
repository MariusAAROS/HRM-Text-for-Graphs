# Handoff — does a pretrained HRM use its recursion? (12 h window, 2 GPUs)

You are running a fixed batch of experiments on a machine with 2× RTX PRO 6000 (100 GB). The machine
is reclaimed after ~12 h. Everything must be logged online to W&B; local files under `/work` survive,
but treat W&B as the record. **Your job is to launch, monitor and report. Do not change the code or the
experiment design.** If something is ambiguous, stop and ask the user.

## Context (why these runs exist)
- This repo (HRM-Text-for-Graphs) studies recursion depth (H/L cycles) of HRM on text-serialized graph tasks
  (GraphQA). Models trained **from scratch** (size B, ~0.3B) show no recursion effect: accuracy is flat in L
  and drops with H. Their trained L block ignores its incoming state (gain ≈ 0.005), and they sit at the
  answer prior on the edge-lookup tasks (see `report/paper_results.ipynb`, RQ2.3, and `data/results/answer-prior-*.csv`).
- The pretrained **HRM-Text-1B** (H2L3, 16 layers per module) fine-tuned on GraphQA scores far higher
  (e.g. ConnectedNodes 1.00 vs ≤ 0.30).
- The paper's RQ3 becomes "Can recursion depth substitute for pretraining?" and needs one thing these runs
  provide: **a measurement of recursion inside the pretrained model**.

| id | question | what runs |
|---|---|---|
| b | Does the fine-tuned 1B depend on its depth? | each fine-tuned 1B ckpt (GraphQA, MetaQA Gold-1/5/10, Retrieved-1, KQA-Pro) evaluated at 9 (H,L), no training |
| c | Is its recurrent state alive? | gain probe on the raw 1B, a random-init twin, each fine-tuned ckpt, each d/e ckpt |
| d | What is the best depth when fine-tuning from pretrained? | GraphQA fine-tunes at H1L1, H1L6, H2L1, H2L6, H3L3 (H2L3 = existing reference) |
| e | Same, but adapting to the task first, then to the new depth | continue the GraphQA H2L3 fine-tune at H1L1, H2L6, H3L3 |

(a), the answer-prior analysis, is already done (W&B run `a-answer-prior`).

## Where things are
- Code: `/work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM` (GRL), branch **`cross-repo`**,
  commit **`36c9e81`** or later (`git log --oneline -1`; every W&B run records it as `grl_commit`).
  All scripts: `GRL/scripts/pretrained_recursion/`. Training entry `runner.py`, depth eval `depth_runner.py`,
  probe `scripts/pretrained_recursion/probe_gain_hf.py`. Settings and job definitions: `common.sh`.
- Outputs: `ROOT=/work/dfm/marius-ortega/pretrained_recursion`:
  `status/<job>` (one line per job), `logs/<job>.log`, `logs/queue_gpu{0,1}.log`, `done/<job>` (markers),
  `ckpts/<job>/last.ckpt`, `results/probe_hf.csv`, `deadline`, `graphqa_ckpt.txt`.
- W&B: project **`HRM-Pretrained-Recursion`** (entity `m-ortega-p-le-l-onard-de-vinci`), groups `b`, `c`, `d`, `e`,
  `d-test`, `e-test`. Smoke runs go to `HRM-Pretrained-Recursion-smoke`.
- Python: the `graph-hrm` conda env (`$HOME/miniforge3` or `/work/dfm/.home/miniforge3`; `common.sh` picks it).

## Steps

All commands from the GRL root:
```bash
cd /work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM
git branch --show-current          # must print: cross-repo
git log --oneline -1               # 36c9e81 or later
```

**1. GraphQA reference checkpoint.** The user fine-tuned HRM-Text-1B on GraphQA (H2L3, clip 1.0) before the
window: W&B run `8dfnfqn3` (`graphqa-hrm-text-clip1-id`, 9 epochs), checkpoint
`Meta-ICL/8dfnfqn3/checkpoints/last.ckpt`, already recorded in `$ROOT/graphqa_ckpt.txt`. Preflight checks it
exists. If preflight reports anything else, ask the user before passing another checkpoint with
`GRAPHQA_CKPT=/…/last.ckpt bash scripts/pretrained_recursion/preflight.sh`.

**2. Preflight** (~1 min). Must end with `PREFLIGHT OK`:
```bash
bash scripts/pretrained_recursion/preflight.sh
```
| failure | fix |
|---|---|
| W&B not logged in | ask the user to run `$PYTHON -m wandb login` (or export `WANDB_API_KEY`). Never set `WANDB_MODE=disabled` or `offline`. |
| no interpreter / import error | check `/work/dfm/.home/miniforge3/envs/graph-hrm`; ask the user — do not pip install into the env |
| HF cache | `export HF_HOME=/work/dfm/.home/.cache/huggingface` and rerun |
| < 2 GPUs or GPUs busy | stop and ask the user |

**3. Smoke test** (~10 min, GPU0 only, separate W&B project; `SMOKE_BATCH=8` also checks that a batch-8
fine-tune fits in memory). Must end with `SMOKE OK`:
```bash
GPU=0 SMOKE_BATCH=8 bash scripts/pretrained_recursion/smoke.sh 2>&1 | tail -30
```
If a job fails, read `$ROOT/smoke/logs/<job>.log`, report it to the user and stop — do not patch code yourself.

**4. Launch both queues** (detached; deadline = now + 11 h):
```bash
bash scripts/pretrained_recursion/launch.sh
```
Launch **once**. If the window started late, use `HOURS=<hours left − 1>`.

| GPU0 (`queue_gpu0.sh`) | GPU1 (`queue_gpu1.sh`) |
|---|---|
| c: 8 probes (~2 min each) | e: to-H1L1 → eval |
| b: 6 checkpoints × 9 depths (~1 h total) | e: to-H2L6 → eval |
| d: H1L1 → eval | e: to-H3L3 → eval |
| d: H2L6 → eval | d: H3L3 → eval |
| d: H2L1 → eval | *stretch* d: H1L1 repeat → eval |
| d: H1L6 → eval | |
| *stretch* d: H4L3 → eval | |

Every training job stops itself (Lightning `max_time`) 25 min before the deadline so its eval still runs.
A job is skipped if fewer than its minimum minutes remain (`status/<job>` says so). After each training job,
`<job>-eval` scores `last.ckpt` on the GraphQA test split at its trained depth, at L=1 and at H2L3, then probes it.
Estimated total: < 7 h per GPU; the stretch jobs only run if time allows.

**5. Monitor** every 30–60 min:
```bash
R=/work/dfm/marius-ortega/pretrained_recursion
for f in $R/status/*; do printf "%-28s %s\n" "$(basename $f)" "$(cat $f)"; done
tail -5 $R/logs/queue_gpu0.log $R/logs/queue_gpu1.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```
Things to check once, after the first training job has run for ~15 min:
- the W&B run appears and `train/loss` decreases;
- the epoch time (Lightning progress bar in `logs/<job>.log`). The time budget assumed 5–7 min per epoch
  at H2L3 (cost scales with H·(L+1)/8). If an epoch is > 2× that, tell the user: stretch jobs will be skipped,
  core jobs are still bounded by `max_time`.

**6. Failure rules.**
- A failed job does not stop its queue; `status/<job>` says `FAILED`, details in `logs/<job>.log`.
- Every job has a wall-clock cap (probes 20 min, each b checkpoint 75 min, evals 45 min, fine-tunes: the
  deadline) and is killed with its whole process group when it overruns: `status/<job>` says `KILLED`.
  That means it hung (e.g. a W&B connection reset) — report it; the queue has already moved on.
- CUDA OOM in a fine-tune is retried automatically once with batch 4 × accumulation 8 (same effective batch).
  Report it; do nothing else.
- Loss NaN / divergence: record it, do not relaunch.
- Never relaunch a job marked done, never delete `ckpts/` or `done/`, never edit code or configs mid-window,
  never push to git.
- If a queue process died (no `queue_gpuN.sh` in `pgrep -af queue_gpu` and its log does not end with
  `finished`), relaunch only that queue with the same deadline:
  `nohup setsid bash scripts/pretrained_recursion/queue_gpu<N>.sh >> $R/logs/queue_gpu<N>.log 2>&1 < /dev/null &`
  (done jobs are skipped; the deadline is read from `$R/deadline`). A job that was running when it died will
  restart from scratch — only do this with ≥ 2 h left.

**7. Before the machine goes away** (last 20 min), report to the user:
- the status table (step 5),
- the W&B project link and the list of finished runs per group,
- anything that failed or was skipped, with the reason,
- the sanity values below.
`results/probe_hf.csv` and the checkpoints stay on `/work`.

## Sanity values
- b at (2,3) on each checkpoint ≈ that run's reported score (reported on val, b uses test): GraphQA ~0.85 (reference run `8dfnfqn3`, val EM 0.852 at its last epoch),
  Gold-1 ~0.98, KQA-Pro ~0.71.
- c: raw pretrained 1B has L gain ≈ 0.38 (measured in the smoke test; from-scratch size B: 0.004–0.009);
  the random-init twin ≈ 0.5.
- e: the step-0 validation (logged before the first training step) equals the zero-shot score at the new depth.

## Reference
- Plan: `/home/ucloud/.claude/plans/purring-wobbling-anchor.md` (may not exist on this machine).
- Recipe of all fine-tunes = the GraphQA reference: `configs/hrm_text.yaml` + `dataset.name=graphqa
  dataset.dataset_config=baseline dataset.test_type=standard trainer.gradient_clip_val=1.0`
  (constant LR 5e-5, batch 8 × 4, early stop on val loss, patience 5 epochs). d caps at 15 epochs, e at 10.
- Depth override: `+model.hrm_overrides={H_cycles:H,L_cycles:L,L_bp_cycles:[0,…,0,L]}` (pretraining BPTT
  rule: only the last H cycle's L steps get gradient).
