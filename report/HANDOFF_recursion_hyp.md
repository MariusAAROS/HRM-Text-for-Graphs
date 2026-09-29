# Handoff: recursion-hypothesis experiments (RQ2.3) and the H≥4 rows of the full-BP figure

State as of 2026-09-30 00:30. Written for a fresh Claude Code session that will **not** have
access to the GPU machine where the runs happened. Everything needed is either in this file, in the
repo (branch `new-ab`), or on W&B (entity `m-ortega-p-le-l-onard-de-vinci`).

## 1. Goal

The GraphQA H/L grid is flat in L, and full backprop through the recursion changes nothing. There
were three hypotheses for why recursion does not help:

- **H1 (collapse).** With pre-norm blocks, the residual of a recurrent module grows to RMS ~300–500
  before the final norm. The incoming state (RMS ~1) then barely affects the output: the "gain"
  ‖f(h+δ)−f(h)‖/‖δ‖ of a trained L block is ~0.005.
- **H3 (task).** GraphQA tasks are either solved by every config or by none, so the average is flat.
- **TRM gap.** Our TRM is not the paper's recipe.

The second goal was to fill the empty H≥4 rows of `report/figures/ablation_hl_grid_fullbp.pdf`.

## 2. Code added or changed (uncommitted on the GPU machine unless it was pushed; check `git log`)

- `models/transformer.py`
  - `norm_type: "peri"` (Peri-LN): `x = x + N(sublayer(N(x)))`, final norm kept.
  - Records `zstat/<tag>/prenorm_rms` when a probe is active.
- `models/baselines/hrm_nocarry_bp_warmup.py`, `models/baselines/trm_nocarry.py`
  - `inject_x` flag (default False): L steps get `z_H + x`.
  - Gain probe `zstat/<tag>/gain`: one extra no-grad block forward per recursion step on probe steps.
- `utils/instrumentation.py`: `log_gain`, `gain_eps=1.0`, `log_prenorm`, `record_gain`, `record_prenorm`,
  `suspend_probe()`. The summaries `probe/summary/{H,L}/gain_mean` and `prenorm_rms_mean` come for free.
  - `gain_eps=1.0` because of bf16 rounding: at ε=0.05, bf16 reads 0.04 for a true 0.005. At 1.0, bf16 matches fp32.
- `simple_inference_engine.py`: `inference_load_checkpoint(..., arch_overrides=)`.
- `scripts/eval_graphqa.py`: `--H_cycles` and `--L_cycles` override the trained values at inference.
  - The report stores `trained_H_cycles` and `trained_L_cycles`.
  - Override evals are saved as `eval_<set>_Lov1_<split>_ep<E>.json`.
- `scripts/log_graphqa_evals_to_wandb.py`: now handles `<set>_Lov1` and label-less sets (pointer).
- `scripts/lib_local_queue.sh` (new): the shared 2-GPU queue.
  - `scripts/run_graphqa_fullbp_local.sh` now sources it, and its RUN_TABLE gained the H≥4 rows.
- `scripts/run_recursion_hyp_local.sh` (new): run tables for PHASES b (norms), c (2-layer modules and TRM-matched), and p (pointer v2).
- `scripts/probe_gain_ckpt.py` (new): offline gain probe of a checkpoint, fp32-capable.
- `scripts/summarize_per_group.py` (new): per-task or per-k accuracy from eval JSONs; `--per-query` for pointer v2.
- `scripts/pull_wandb_curves.py` (new): exports W&B training curves to a CSV.
- `scripts/make_pointer_chasing.py`, `config/cfg_pointer.yaml`, `config/cfg_pointer2.yaml` (new).
- `scripts/summarize_graphqa_fullbp.py`: line-plot ylim lowered to 0, for the ~0 H≥4 cells.
- `report/paper_results.ipynb`: new section after the old last cell (id `09763815`). Its cells, by id:
  - `rq23-intro`
  - `rq23-pertask-md`, `rq23-pertask`, `rq23-pertask-fig`: per-task heatmap → `report/figures/recursion_hyp_per_task.{pdf,png}`
  - `rq23-h4-md`, `rq23-h4-curves`, `rq23-h4-fig`: JZ loss curves → `report/figures/recursion_hyp_h4_loss.{pdf,png}`

  **No "Reading" cells have been written yet.** The notebook was edited as JSON (no nbformat in the env).

New CSVs in `data/results/`. `data/` is gitignored, so they exist only on the GPU machine unless force-added:
- `graphqa-fullbp-local-per-group.csv`: per-task accuracy of the 28 original local runs.
- `recursion-hyp-h4-curves.csv`: JZ training curves (H1–H8), from W&B `HRM-GraphQA-Ablation`.
- `recursion-hyp-probe.csv`: gain and prenorm per recursion step.
- `recursion-hyp-pointer2-per-k.csv`

## 3. Results so far

All accuracies are greedy exact match on GraphQA val+test pooled (770 samples, 1 SE ≈ 1.8 pts;
seed-to-seed spread up to ~4 pts). Every run is 4,358 steps from scratch with the cfg_graphqa recipe,
seed 0, full backprop, unless noted.

### 3.1 Gain probe (fp32, ε=1.0, 12 val samples; `probe_gain_ckpt.py`)

| checkpoint | gain_L | gain_H | residual RMS before final norm (L / H) | fresh-init gain_L |
|---|---|---|---|---|
| pre H1L6 full-BP | 0.0039 | 0.20 | 485 / 21 | 0.51 |
| pre H1L6 1-step | 0.0049 | 0.22 | 567 / 20 | 0.52 |
| pre H1L24 full-BP | 0.0083 | 0.21 | 265 / 19 | 0.50 |
| pre H3L6 full-BP | 0.0092 | 0.16 | 349 / 39 | 0.48 |
| **peri H1L6** | **0.085** | 0.27 | 12 / 5.6 | 0.45 |
| **post H1L6** | **0.0000** | 0.027 | 1 / 1 | 0.84 |

Training-time probes (W&B `HRM-RecursionHyp-Local-smoke`) at steps 100–1,200:
- peri H1L6: gain_L 0.13–0.24, residual RMS 6–10;
- TRM-matched H3L6: gain_L falls from 0.28 to 0.07.

### 3.2 GraphQA accuracy per arm (val+test pooled; "Lov1" = trained model run with L_cycles=1 at inference)

| arm | H1L1 | H1L6 | H3L6 | Lov1 of H1L6 / H3L6 |
|---|---|---|---|---|
| pre, 12 layers (ref, `graphqa_fullbp`) | 0.534 / 0.530 (s0/s1) | 0.523 / 0.519 | 0.445 | 0.519 / 0.444 → **no drop** |
| peri, 12 layers | 0.538 | 0.526 | – | 0.521 → no drop |
| post, 12 layers | 0.399 | 0.457 | – | 0.457 → no drop (identical) |
| **peri, 2-layer modules** (`arch.n_layers=4`, 2 per module) | **0.543** | 0.523 | 0.534 | **0.468 / 0.479 → −5.5 pts** |
| TRM matched to the paper (1 net of 2 layers, post-norm, `inject_x`, grad through last cycle) | 0.523 | 0.475 | 0.319 | 0.460 / 0.334 |

Also, pre H1L24 full-BP scores 0.516, and 0.519 with L=1 at inference.

### 3.3 H≥4 rows (full backprop, no grad checkpointing, local)

- H5L6 full-BP = **0.000**; H6L6 full-BP = **0.000** (val and test).
- Jean Zay references:
  - 1-step: H4L1 0.374, H4L3 0.062, H4L6 0.055, H5L6/H6L2/H6L6/H8L12 0.000.
  - full-BP with grad checkpointing: H4L6 0.016, H6L6 0.000.
- JZ training curves (`recursion-hyp-h4-curves.csv`, figure `recursion_hyp_h4_loss`):
  - H≥5 runs, and full-BP H4L6/H6L6, drop to loss 0.3–0.7 by step ~200, where the LR warmup ends.
  - They then jump to a flat plateau at ~2.5–2.6, the collapsed level, with spikes.
  - So the collapse is a training instability at peak LR (3e-4), not a depth limit or a truncation effect.

### 3.4 Per-task (H3; `graphqa-fullbp-local-per-group.csv`, figure `recursion_hyp_per_task`)

- **Ceiling everywhere:** Reachability 1.00, CycleCheck ~0.96.
- **Floor everywhere:** MaximumFlow ~0.09 (capacities are missing in this "old" split), DisconnectedNodes ~0.13, ConnectedNodes ~0.23.
- **Flat within noise:** EdgeExistence ~0.72, ShortestPath ~0.52, NodeDegree ~0.36.
- **The only tasks that move** are NodeCount, EdgeCount and TriangleCounting, and they get **worse** with H (e.g. NodeCount 0.88 at H1L1 → 0.27–0.47 at H3L12).
- Per-task n ≈ 70 per cell (SE ≈ 6 pts).

### 3.5 Pointer chasing (positive control): inconclusive

- **v1:** N=32, 1 query per sample, `cfg_pointer`.
  - Every model outputs one constant node, at chance (1/32) for every k, including k=1.
  - Runs: pre H1L6, peri 2-layer H3L6.
- **v2:** N=16, k∈{1,2,3,4,6,8}, 4 queries per graph, 30k samples, `cfg_pointer2`.
  - Per-answer accuracy is 0.05–0.07 for every k (chance 0.0625).
  - Runs: pre H1L1, peri H1L6.
- A 12-layer model trained from scratch on ~20M tokens does not learn edge-list retrieval in this format, even for a single hop.
- The remaining pointer arms were cancelled. Report the control as inconclusive, not as evidence.

### 3.6 Interpretation (draft for the notebook summary)

- **H1 (collapse): real but not the bottleneck.**
  - Pre-norm washes the state out (gain 0.004–0.009). Peri-norm fixes that (0.085, ~20×), yet accuracy is unchanged and L=1 at inference costs nothing.
  - Post-norm washes it out completely (gain 0.000) and is the worst arm.
- **H3 (task): supported.**
  - With 2-layer modules the model *uses* its L steps (−5.5 pts when they are removed), but the recurrent configs only match the no-recursion baseline.
  - 4 layers without recursion (0.543) ≈ 12 layers (0.53) ≈ 12 layers with 6 L-steps. Extra depth of any kind (layers or recursion) buys nothing on GraphQA at this scale.
  - Per task, the average is set by tasks that sit at ceiling or floor in every config.
- **TRM gap: refuted as an explanation.** The paper-matched TRM gets worse with recursion (0.52 → 0.48 → 0.32).
- **H≥4 collapse:** an optimisation instability (divergence right after LR warmup), present with or without truncation and checkpointing. It is not a depth limit.
- **Caveats:** single seed for every new arm; the positive control failed; no deep supervision (Phase D was not run).

## 4. Runs still in flight at handoff (all expected done by ~02:15; nothing needs to be relaunched)

All are GraphQA, cfg_graphqa, seed 0, run names `graphqa_H<H>_L<L>_{fullbp,trunc}_local_s0`:

| run | where results appear |
|---|---|
| H4L6 full-BP (eval running), H4L3 full-BP | trained offline, auto-synced. Train: W&B `HRM-GraphQA-FullBP-Local`. Eval: W&B `HRM-GraphQA-FullBP-Local-eval`, runs `<run>_old_{val,test}_ep20`, key `eval/accuracy` |
| H6L2 full-BP (eval), H4L1 full-BP, H4L6 1-step, H6L6 1-step | online, same two projects |

The Phase C and norm-test training curves (with the gain probe) are in `HRM-RecursionHyp-Local`, and their
evals in `HRM-RecursionHyp-Local-eval` (tag `recursion-hyp`).

## 5. What remains to do

1. **Collect the final H≥4 numbers** from W&B `HRM-GraphQA-FullBP-Local-eval`, by filtering run names on
   `graphqa_H[4-6]_L*_local_s0_old_{val,test}_ep20`.
   - On the GPU machine, `scripts/summarize_graphqa_fullbp.py` does this from the checkpoints instead.
   - Append the **test** rows to `data/results/graphqa-fullbp-local.csv` (columns
     `run,H,L,seed,bp,split,accuracy,correct,n`; bp is `Full backprop` or `1-step gradient`).
     Notebook cell 37 reads that CSV.
2. **Re-render the full-BP grid:** run notebook cells 37–38 (ids `676f0499`, `8170f6e7`; cell 36 is `15dd5186`, cell 39 is `09763815`, all after the
   "RQ2.1''" markdown) to regenerate `report/figures/ablation_hl_grid_fullbp.{pdf,png}` with the H4–H6 rows.
   - Update the cell-36 markdown: it still says "11 cells with H ∈ {1,2,3}".
   - Update the cell-39 "Reading": full BP does not rescue H≥5 (0.000), plus whatever H4 shows.
3. **Write the rest of RQ2.3 in the notebook**, following the existing pattern (markdown intro →
   code cell reading a CSV → figure/table → markdown **Reading.**):
   - Readings for 2.3.1 (per-task) and 2.3.2 (H≥4 curves). The content is in §3.3–3.4.
   - 2.3.3 Collapse probe: a gain/prenorm table from `recursion-hyp-probe.csv` (§3.1), plus the
     trained-L vs Lov1 table (§3.2).
   - 2.3.4 Small modules and matched TRM: the §3.2 table. Put the numbers in a new
     `data/results/recursion-hyp-local.csv` (one row per run with acc and Lov1 acc).
   - 2.3.5 Pointer chasing: a short markdown note that it is inconclusive (§3.5), no figure.
   - 2.3.6 Summary: each hypothesis marked supported, refuted or open, with the numbers from §3.6.
4. **Check the notebook runs.** The env has no nbconvert/nbformat: execute the code cells as a script
   under the Agg backend, or open the notebook in VS Code.
5. **Optional**, needs the GPU machine: `probe_gain_ckpt.py` on the Phase C checkpoints (2-layer peri,
   TRM-matched) under `/work/dfm/marius-ortega/recursion_hyp/ckpts`, for gain values of those arms too.

## 6. Paths on the GPU machine (for reference)

- Checkpoints and eval JSONs:
  - `/work/dfm/marius-ortega/graphqa_fullbp/ckpts/` (full-BP grid, including H≥4)
  - `/work/dfm/marius-ortega/recursion_hyp/ckpts/` (norm test, Phase C, pointer)
- Logs: `.../graphqa_fullbp/logs/`, `.../recursion_hyp/logs/`
- The B1 runs were launched from a worktree snapshot at `/work/dfm/marius-ortega/hrm_b1` (commit `1b0429e`).
