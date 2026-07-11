# NLPSpring2026 — Experimental Progress Tracker

**Last updated:** 2026-07-11  
**Cluster:** local (8× 49 GB GPUs), tmux sessions, no SLURM  
**Env:** `nlp2026` conda (Python 3.11, torch 2.7.1, transformers 4.53.3, peft 0.17.1)  
**Weights dir:** `/data/nathanlabiosa/nlp2026_weights/`  
**HF cache:** `/data/nathanlabiosa/hf_cache/`

---

## ⚠️ Pending Actions

All GPU experiments complete. Two paper-correctness issues flagged below require author action before submission.

### Scatter rerun in progress — do not use old scatter figures

**Old scatter PDFs are contaminated (old-harness acc_delta). Regeneration pipeline written; run the commands below.**

#### Stage 1: GSM8K eval on existing mmlu checkpoints (Phi-3.5, Qwen — no training)

Run 6 windows in parallel (one GPU each). Each window runs seeds 42/43/44 sequentially (~25 min/seed):

```bash
# Phi-3.5 (6 windows × 3 seeds = 18 eval runs, ~2.5 hr wall clock on 6 GPUs)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_scatter_eval.sh phi35 L00-04
CUDA_VISIBLE_DEVICES=1 bash scripts/run_scatter_eval.sh phi35 L05-09
CUDA_VISIBLE_DEVICES=2 bash scripts/run_scatter_eval.sh phi35 L10-14
CUDA_VISIBLE_DEVICES=3 bash scripts/run_scatter_eval.sh phi35 L15-19
CUDA_VISIBLE_DEVICES=4 bash scripts/run_scatter_eval.sh phi35 L20-24
CUDA_VISIBLE_DEVICES=5 bash scripts/run_scatter_eval.sh phi35 L27-31

# Qwen (6 windows × 3 seeds = 18 eval runs)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_scatter_eval.sh qwen L00-04
CUDA_VISIBLE_DEVICES=1 bash scripts/run_scatter_eval.sh qwen L05-09
CUDA_VISIBLE_DEVICES=2 bash scripts/run_scatter_eval.sh qwen L08-11
CUDA_VISIBLE_DEVICES=3 bash scripts/run_scatter_eval.sh qwen L15-19
CUDA_VISIBLE_DEVICES=4 bash scripts/run_scatter_eval.sh qwen L20-23
CUDA_VISIBLE_DEVICES=5 bash scripts/run_scatter_eval.sh qwen L24-27
```

#### Stage 2: Train + eval for Llama-3 and Mistral (6 windows × 3 seeds × 2 models)

Add HF_TOKEN to run_scatter_train.sh if gated. Each window: ~2 hr train + ~1.5 hr eval × 3 seeds = ~10 hr/window; with 6 parallel GPUs → ~10 hr wall clock per model.

```bash
# Llama-3
for W in L00-04 L05-09 L10-14 L15-19 L20-24 L27-31; do
  CUDA_VISIBLE_DEVICES=<GPU> bash scripts/run_scatter_train.sh llama $W &
done

# Mistral
for W in L00-04 L05-09 L10-14 L15-19 L20-24 L27-31; do
  CUDA_VISIBLE_DEVICES=<GPU> bash scripts/run_scatter_train.sh mistral $W &
done
```

#### Stage 3: Disruption measurement (one GPU per model, ~30 min each)

Run AFTER checkpoints exist (after stages 1/2). Can run phi35/qwen immediately after stage 1.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_scatter_disruption.sh phi35
CUDA_VISIBLE_DEVICES=1 bash scripts/run_scatter_disruption.sh qwen
CUDA_VISIBLE_DEVICES=2 bash scripts/run_scatter_disruption.sh llama
CUDA_VISIBLE_DEVICES=3 bash scripts/run_scatter_disruption.sh mistral
```

#### Stage 4: Merge acc_delta into disruption cache, regenerate figures

```bash
python scripts/update_scatter_deltas.py

LEROBOT_PY=/home/nathanlabiosa/miniforge3/envs/lerobot_smol/bin/python
for m in phi35 qwen llama3 mistral; do
  CUDA_VISIBLE_DEVICES="" $LEROBOT_PY src/experiments/expF_clean_disruption.py \
    --model $m --reuse_cache
done
```

Output: `figures/disruption_{model}_scatter.pdf` (4 valid scatter figures)

---

### STOP: Do not use any old scatter figures in the paper

All five `disruption_*_scatter.pdf` figures have old-harness acc_delta values hardcoded in `MODEL_CFGS["sweep_deltas"]` — they are **not** from the v1 fixed harness:

| Figure | sweep_deltas source | Key tell |
|--------|-------------------|---------|
| `disruption_phi35_scatter.pdf` | `p2_sweep_L*` old checkpoints | All **positive** (+4.53 to +7.27); v1 fixed harness gives −5.3 pp at L10-14 |
| `disruption_llama3_scatter.pdf` | `llama_sweep_L*` old checkpoints | L05-09: −3.67 hardcoded vs −6.41 (v1); L15-19: 0.00 vs −2.70 |
| `disruption_mistral_scatter.pdf` | `mistral_sweep_L*` old checkpoints | L05-09: −9.43 vs −17.69 (v1); L20-24: +0.13 vs −8.46 |
| `disruption_qwen_scatter.pdf` | `qwen_sweep_L*` old checkpoints | All positive (+2.83 to +11.50), L24-27 = +11.5 ≈ retracted +11.6 |
| `disruption_gemma_scatter.pdf` | `gemma2_9b_sweep_L*` old sweeps | Slope +5.4 contradicts paper claim |

To fix: update `MODEL_CFGS["sweep_deltas"]` in `expF_clean_disruption.py` with the v1 fixed-harness means from `results/fixed_harness/v1/layer_sweep/*.json`, then regenerate with `--reuse_cache` (no GPU needed). The non-scatter disruption curve figures (`disruption_phi35.pdf`, `disruption_llama3.pdf`, `disruption_mistral.pdf`, `disruption_qwen.pdf`, `disruption_gemma.pdf`) are unaffected — they show per-layer LRD curves, not acc_delta.

### Width-5 std: Table 4 ±0.9 vs aggregate script ±1.1 — statistics convention mismatch

The v1 data for Phi-3.5 L10-14 is exactly −4.0, −6.0, −5.9 pp (seeds 42/43/44):
- **Paper (Table 4) ±0.9** = population std (divide by n=3): √(2.54/3) = 0.92
- **Aggregate script / ablation table ±1.1** = sample std (divide by n−1=2): √(2.54/2) = 1.13

Same data, different formula. The correct convention for n=3 in a paper is sample std (±1.1). Fix Table 4 to report ±1.1, or add a footnote that all stds are population stds — but the tables must be consistent with each other and with the aggregate script output.

---

## Quick Status

| Task | Status | Notes |
|------|--------|-------|
| Task 1: Window-length ablation | ✅ Complete | All 12 runs done (4 windows × 3 seeds) |
| Task 2: All-layer LoRA | ✅ Complete | All 9 seeds done (3 models × 3 seeds) |
| Task 3: MMLU sweep | ✅ Complete | All 36 runs done (Phi-3.5 + Qwen, 6 windows × 3 seeds each) |
| Task 4: Code release | 🔄 In progress | Smoke test ✅; file editing remaining |

---

## Active Sessions

| Session | GPU | Chain (sequential) | Status |
|---------|-----|--------------------|--------|
| s_scatter_0 | 0 | phi35 L00-04→L05-09 → qwen L00-04→L05-09 → llama L00-04→L05-09 → mistral L00-04→L05-09 | 🔄 Running |
| s_scatter_3 | 3 | phi35 L10-14→L15-19 → qwen L08-11→L15-19 → llama L10-14→L15-19 → mistral L10-14→L15-19 | 🔄 Running |
| s_scatter_4 | 4 | phi35 L20-24 → qwen L20-23 → llama L20-24 → mistral L20-24 | 🔄 Running |
| s_scatter_7 | 7 | phi35 L27-31 → qwen L24-27 → llama L27-31 → mistral L27-31 | 🔄 Running |

**Estimated completion:** ~36 hours (sessions 0 and 3 run 2 windows per model; sessions 4 and 7 run 1 each).
**After sessions complete:** run disruption + update_scatter_deltas.py + reuse_cache (see Stage 3/4 commands above).

---

## Task 1: Window-Length Ablation (Phi-3.5)

**Paper value (width 5, existing):** mid L10-14 = −5.3 ± 0.9 pp, late L27-31 = +0.2 ± 0.3 pp  
**Result file:** `results/fixed_harness/v2/layer_sweep/phi35_<WIN>_seed<S>.json`  
**Success criterion:** sign structure width-robust (mid ≤ 0, late ≈ 0 at widths 3 and 7)

### Width 3 (center 12 → L11–13; center 29 → L28–30)

| Window | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|--------|---------|---------|---------|------------|
| L11-13 (mid, w3) | −2.43 | −2.50 | −3.43 | **−2.79 ± 0.55** ✅ |
| L28-30 (late, w3) | −0.07 | +0.07 | −0.13 | **−0.04 ± 0.10** ✅ |

### Width 5 (existing results from v1, max_new_tokens=512)

| Window | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|--------|---------|---------|---------|------------|
| L10-14 (mid, w5) | −4.0 | — | — | −5.3 ± 0.9 (paper) |
| L27-31 (late, w5) | +0.47 | — | — | +0.2 ± 0.3 (paper) |

> Note: width-5 is from v1 (512 tokens). New v2 (768 tokens) may differ slightly.

### Width 7 (center 12 → L09–15; center 29 → L25–31)

| Window | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|--------|---------|---------|---------|------------|
| L09-15 (mid, w7) | −7.60 | −7.50 | −8.20 | **−7.77 ± 0.38** ✅ |
| L25-31 (late, w7) | −0.63 | −1.03 | −0.13 | **−0.60 ± 0.45** ✅ |

> L25-31 is clamped (L26-32 exceeds 32-layer model)

### Width trend (mid window)

| Width | Window | Mean Δ (3 seeds) |
|-------|--------|-----------------|
| w3 | L11-13 | **−2.79 ± 0.55 pp** |
| w5 | L10-14 | −5.3 ± 0.9 pp (paper, v1 512-token) |
| w7 | L09-15 | **−7.77 ± 0.38 pp** |

Sign is robustly negative across all widths. Effect amplitude scales monotonically with window width — wider LoRA captures more representation-stabilizing layers. Consistent with paper claim.

---

## Task 2: All-Layer LoRA Baseline

**Result file:** `results/fixed_harness/v2/all_layer/<model>_all_seed<S>.json`

| Model | Seed | Status | Mean perturbed Δ | Clean |
|-------|------|--------|-----------------|-------|
| Phi-3.5-mini | 42 | ✅ | −8.23 | 86.4 |
| Phi-3.5-mini | 43 | ✅ | −8.23 | 86.4 |
| Phi-3.5-mini | 44 | ✅ | −9.43 | 86.4 |
| Qwen2.5-7B | 42 | ✅ | −13.57 | 90.6 |
| Qwen2.5-7B | 43 | ✅ | −13.57 | 90.6 |
| Qwen2.5-7B | 44 | ✅ | −13.70 | 90.6 |
| Llama-3-8B | 42 | ✅ | −9.60 | 79.6 |
| Llama-3-8B | 43 | ✅ | −11.03 | 79.6 |
| Llama-3-8B | 44 | ✅ | −9.13 | 79.6 |

> **Final means:** Phi-3.5 −8.6 ± 0.7 pp | Qwen −13.6 ± 0.1 pp | Llama −9.9 ± 1.0 pp. All 9 seeds complete.
>
> **Qwen ±0.1 std verified (2026-07-11):** Checkpoints are distinct — per-condition adapter accuracies differ by up to 3 pp (e.g. whitespace: 71.6/74.6/72.4). Seeds 42 and 43 share the same mean (−13.5667) by arithmetic coincidence (their condition deltas differ by up to 3 pp but sum identically to −81.4 at 0.2 pp resolution; seed 44 sums to −82.2). Tight std is real: all-layer LoRA saturates the adaptation signal across all 28 layers, leaving no seed-dependent window choice. Note: the `"seed"` field in the result JSONs records the eval sampling seed (always 42), not the training seed — training seed is only in the checkpoint path.

---

## Task 3: MMLU Layer Sweep ✅ COMPLETE

**Result file:** `results/fixed_harness/v2/mmlu/<model>_<WIN>_seed<S>.json`  
**Eval subset IDs:** `results/mmlu_eval_item_ids.json`  
**max_new_tokens:** 32

### Phi-3.5 MMLU — Summary (all 18 runs done)

| Window | Mean Δ ± Std | Interpretation |
|--------|-------------|---------------|
| L00-04 | +0.37 ± 0.18 pp | noise |
| L05-09 | +0.39 ± 0.14 pp | noise |
| L10-14 | **−1.83 ± 0.38 pp** | only meaningful signal |
| L15-19 | −0.38 ± 0.26 pp | noise |
| L20-24 | −0.26 ± 0.14 pp | noise |
| L27-31 | −0.01 ± 0.10 pp | noise |

Baseline clean acc: 69.2%

### Qwen2.5-7B MMLU — Summary (all 18 runs done)

| Window | Mean Δ ± Std | Interpretation |
|--------|-------------|---------------|
| L00-04 | +0.00 ± 0.09 pp | noise |
| L05-09 | +0.08 ± 0.02 pp | noise |
| L08-11 | −0.02 ± 0.11 pp | noise |
| L15-19 | +0.01 ± 0.14 pp | noise |
| L20-23 | +0.06 ± 0.13 pp | noise |
| L24-27 | −0.11 ± 0.05 pp | noise |

Baseline clean acc: 73.8%

**Key finding:** MMLU shows near-zero perturbation robustness improvement (<0.4 pp absolute for most windows vs ~5 pp for GSM8K mid-layer windows). LRD stabilizer effect is task-specific: strong for CoT reasoning (GSM8K), negligible for multiple-choice classification (MMLU). L10-14 is the only window with a signal for Phi-3.5 (−1.83 pp), consistent with that being the mid-layer maximum for GSM8K too.

---

## Task 4: Code Release

**Target:** anonymous.4open.science
**Deadline:** before next ARR submission

- [ ] Remove absolute paths (`/home1/labiosa/`, `/scratch1/labiosa/`) — **not yet done in this checkout**; will happen in the separate anonymized export (this repo stays as the real working copy, per author decision 2026-07-11)
- [x] Remove cluster-specific SLURM scripts (or move to `slurm/` archive note) — decision: drop `slurm/` from the anonymized export entirely, replace with a short archive note; `scripts/` is the current reproduction path
- [ ] Consolidate `perturbations.py` to single canonical location (`src/lib/perturbations.py`) — not yet done
- [ ] Remove author names/emails/institution strings from comments — deferred to the anonymized export (see above); this repo keeps real identity
- [x] Genericize README: map every Table/Figure → exact reproduce command — done 2026-07-11; see README.md "Reproducing every table and figure" (built directly from the paper's actual `\label`s, cross-checked against script source, not guessed)
- [x] Pin `requirements.txt` with exact versions from `nlp2026` env — done 2026-07-11 (torch 2.7.1+cu126, transformers 4.53.3, peft 0.17.1, datasets 4.1.1, accelerate 1.14.0, numpy 2.4.4, scipy 1.17.1, sentencepiece 0.2.1, matplotlib 3.10.8). Note: this **does not match** the paper's appendix `tab:supp_repro` (torch 2.10.0+cu128, transformers 4.44.2, peft 0.11.0, SLURM+A40/A100/V100) — that table describes the *original* panel's environment; Task 1-3 addenda ran on the new cluster with the versions above. Both are real; the appendix table needs a sentence distinguishing them (author to edit `paper.tex` directly — not done here).
- [x] Add hardware notes (GPU type, hours per cell) to README — done 2026-07-11, see README.md "Hardware and compute"
- [x] Smoke test: LRD profile for Phi-3.5 on typos + 1 LoRA window, 1 seed, n=50
  - Script: `src/eval/eval_lrd_adapter.py` (fixed `trust_remote_code=False`, `attn_impl=sdpa`)
  - Checkpoint: `phase2_phi35_L11-13_seed42/lora_final`
  - Output: `results/smoke_test/phi35_L11-13_seed42_lrd.json`
  - Result: pipeline runs clean; typos LRD peaks layer 0 (0.87 baseline), decays through network as expected; adapter reduces terminal-layer LRD by ~0.017 (L11-13 LoRA does not target layers 0–4, so early-window summary = 0.0% by design)
- [ ] Fresh git init (squash history), upload to anonymous host — in progress, separate anonymized export (this repo's history is untouched, per author decision 2026-07-11)
- [ ] Verify link works logged-out — pending the export above

### Code bugs found and fixed while building the reproduction mapping (2026-07-11)

Found while verifying every table/figure command actually runs, not by inspection alone:

1. `models/gemma2/lrd_run.py` — `PHI_DIR` pointed at `<root>/Phi3.5` (pre-reorg path); the reorg moved it to `<root>/models/phi3.5`. Import of `lrd_diagnostics` would `ModuleNotFoundError` on any fresh run. **Fixed.**
2. `src/plotting/generate_workstream_c_figures.py`'s `figure_c3()` wrote to the same `figures/layer_sweep.pdf` as the correct `generate_layer_sweep.py`, but reads long-gone `stabilizer_weights/p2_sweep_L*/eval_results.json` — running the former after the latter would silently clobber Fig. 6 with an empty/broken plot. Renamed its output to `figures/legacy_p2_layer_sweep.pdf`. **Fixed.**
3. `src/experiments/expA_cohens_d.py`'s `MODEL_DATA` dict was missing a `"Mistral"` entry (the `PATCHING_WINDOW`/`LORA_WINDOW` dicts already had one) — `figures/cohens_d_mistral.pdf` could not be regenerated. Added the entry (data already exists at `models/mistral/lrd_results/mistral_gsm8k/raw_gsm8k.json`). **Fixed.**
4. `scripts/run_scatter_disruption.sh` accepted `llama` (matching every other script's convention) but forwarded it unchanged to `expF_clean_disruption.py --model`, whose `choices` require `llama3` — Stage 3 of the scatter-figure regen (still running as of this writing) would have crashed on the Llama cell. **Fixed** (shell arg stays `llama`; translated to `llama3` only for the Python call).

### Paper-side issue found, not fixed here (needs author edit to `paper.tex`)

Main-body `tab:layer_sweep_panel` reports Phi-3.5's mid window (L10-14, width 5) as **±0.9**; appendix `tab:width_ablation` reports the identical underlying cell as **±1.1** (sample std, n=3 — the correct convention, matching `aggregate_results.py`'s output). This is the population-vs-sample-std bug flagged above (Task 3 section) — it looks like the appendix got corrected but the main table wasn't updated to match. Fix `tab:layer_sweep_panel`'s Phi-3.5 Early-mid cell to ±1.1 before submission.

---

## Technical Notes

### trust_remote_code fix (2026-07-08)
`trust_remote_code=True` caused all scripts to load Phi-3.5 from the HF cache's `modeling_phi3.py`,
which calls `_prepare_4d_causal_attention_mask` even for SDPA. In transformers 4.53.3 this function
has changed behavior, causing a tensor size mismatch during generation. Fix: set `trust_remote_code=False`
in all `from_pretrained` calls (model only — transformers 4.53.3 natively supports all three models).
Changed in: `train_lrd_lora_v14.py`, `eval_fixed_harness.py`, `eval_fixed_harness_mmlu.py`, `train_lrd_stabilizer.py`.

Note: Phi-3.5 `config.json` has `auto_map` pointing to cached files, but `trust_remote_code=False`
correctly overrides this to use the installed transformers implementation (verified with `inspect.getfile`).

### lerobot_smol fallback (2026-07-09)
The run_task*.sh scripts fall back to `/home/nathanlabiosa/miniforge3/envs/lerobot_smol/bin/python`
(Python 3.10) if `nlp2026` env torch import fails. lerobot_smol has old transformers that don't
natively support Phi-3.5 — the HF-cached model loads and crashes. The original t1_w3_late session
was created before nlp2026 was ready, triggered this fallback, and failed. Relaunched 2026-07-09
after verifying nlp2026 env is healthy.

### Result File Schema

```json
{
  "harness_version": "v2",
  "base_model": "microsoft/Phi-3.5-mini-instruct",
  "checkpoint": "/data/nathanlabiosa/nlp2026_weights/phase2_phi35_L11-13_seed42/lora_final",
  "seed": 42,
  "n_samples": 500,
  "max_new_tokens": 768,
  "clean_baseline": 86.4,
  "mean_perturbed_delta": -2.43,
  "results": { "per_condition": "..." }
}
```

Run `python scripts/aggregate_results.py task1|task2|task3|all` to print summary + LaTeX.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not started |
| 🔄 | Running or queued |
| ✅ | Complete — fill in the result value |
| ❌ | Failed — note error |
