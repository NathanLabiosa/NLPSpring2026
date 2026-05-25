# Phase 1 Launcher

**Not auto-submitted.** Inspect, then submit by hand.

## What this does

Layer-sweep retraining + fixed-harness eval, CE-only (`λ_stab=0`), across 5
models, with 3 seeds per window. Total: **81 array tasks** = 81 LoRA
adapters trained + 81 fixed-harness eval JSONs.

| Model | Slurm script | Array size | Windows | Walltime each | Mem | Concurrency |
|---|---|---|---|---|---|---|
| Phi-3.5-mini-instruct | `submit_phase1_phi35.slurm` | 18 | L00-04, L05-09, L10-14, L15-19, L20-24, L27-31 | 10h | 48GB | 6 |
| Llama-3-8B-Instruct | `submit_phase1_llama3.slurm` | 15 | L00-04, L05-09, L15-19, L20-24, L27-31 | 10h | 48GB | 6 |
| Mistral-7B-Instruct-v0.3 | `submit_phase1_mistral.slurm` | 15 | L00-04, L05-09, L15-19, L20-24, L27-31 | 10h | 48GB | 6 |
| Qwen2.5-7B-Instruct | `submit_phase1_qwen.slurm` | 18 | L00-04, L05-09, L08-11, L15-19, L20-23, L24-27 | 10h | 48GB | 6 |
| Gemma-2-9B (base) | `submit_phase1_gemma2.slurm` | 15 | L00-04, L06-11, L15-20, L25-30, L35-40 | 14h | 80GB | 4 |

Concurrency is enforced by the `%N` suffix on `--array=1-K%N`. Adjust if
the queue is empty / busy.

## Pre-flight checklist (do BEFORE submitting any Phase 1 job)

1. **Phase 0 baseline audit JSONs landed.** Verify
   `results/fixed_harness/v1/baseline_audit/*.json` exist for Llama, Mistral,
   Gemma and that clean baselines are within 5 pp of published numbers.
   Update HARNESS.md §2 with the actual numbers.
2. **Regression test passed.** `sbatch submit_regression_test.slurm` must
   exit zero before any Phase 1 batch.
3. **HF tokens cached** — `/home1/labiosa/.cache/huggingface/token` must
   exist; Llama-3-8B and Mistral-7B-v0.3 require access grants.

## Submit

Submit one model at a time so you can sanity-check the first eval JSON
before launching the rest:

```bash
cd /home1/labiosa/NLPSpring2026

# Start with one model (recommend Qwen — best-understood baselines)
sbatch submit_phase1_qwen.slurm

# When tasks 1-3 (one seed at the first window) finish and the JSONs look
# sane, kick off the rest:
sbatch submit_phase1_phi35.slurm
sbatch submit_phase1_llama3.slurm
sbatch submit_phase1_mistral.slurm
sbatch submit_phase1_gemma2.slurm
```

To submit a single array slot only (e.g. dry-run task 1 of Phi-3.5):

```bash
sbatch --array=1 submit_phase1_phi35.slurm
```

## Array → (window, seed) mapping

The scripts compute `(SEED_IDX, WIN_IDX)` from the array task ID as:

```
SEED_IDX = (TASK - 1) / N_WINDOWS
WIN_IDX  = (TASK - 1) % N_WINDOWS
```

So for Phi-3.5 (N_WINDOWS = 6):

| Task | Seed | Window |
|---|---|---|
| 1 | 42 | L00-04 |
| 2 | 42 | L05-09 |
| 3 | 42 | L10-14 |
| 4 | 42 | L15-19 |
| 5 | 42 | L20-24 |
| 6 | 42 | L27-31 |
| 7 | 43 | L00-04 |
| ... | ... | ... |
| 18 | 44 | L27-31 |

Llama, Mistral (N_WINDOWS=5): tasks 1-5 seed 42, 6-10 seed 43, 11-15 seed 44.
Qwen (N_WINDOWS=6): tasks 1-6 seed 42, etc.
Gemma-2 (N_WINDOWS=5): tasks 1-5 seed 42, etc.

## Outputs

Each task produces:

```
stabilizer_weights/phase1_{model_slug}_L{ss}-{ee}_seed{S}/lora_final/
results/fixed_harness/v1/layer_sweep/{model_slug}_L{ss}-{ee}_seed{S}.json
```

Model slugs: `phi35`, `llama3_8b`, `mistral_7b_v03`, `qwen2.5_7b`, `gemma2_9b`.

## Training config (identical across models, per `experiment_plan.md` §1)

| Param | Value | Notes |
|---|---|---|
| `lora_rank` | 4 | per plan |
| `lora_alpha` | 8 | per plan |
| `lora_dropout` | 0.05 | matches exp2 reference |
| `target_modules` | `q_proj v_proj` (or `qkv_proj o_proj` for Phi-3.5) | Phi-3.5 uses fused qkv |
| `lambda_acc` | 1.0 | CE-only |
| `lambda_stab` | 0.0 | **CE-only — no stability loss (plan §1)** |
| `stab_layer` | window end | unused at λ=0, placeholder only |
| `n_per_condition` | 857 | matches exp2 |
| `clean_fraction` | 0.25 | matches exp2 |
| `epochs` | 30 | bounded by `max_steps` |
| `max_steps` | 300 | matches exp2 |
| `batch_size` | 4 (Gemma: 2) | Gemma-9B doesn't fit BS=4 on A40 |
| `grad_accum_steps` | 4 (Gemma: 8) | effective batch 16 throughout |
| `lr` | 5e-5 | matches exp2 |
| `warmup_ratio` | 0.05 | matches exp2 |
| `max_seq_len` | 512 | matches exp2 |
| `early_stop_delta` | 5.0 | stop if clean acc drops > 5 pp from baseline |
| `--no_eval_after_training` | (set) | inline eval uses `max_new_tokens=300`; we route to `eval_fixed_harness.py` instead |
| Eval seed | 42 | fixed across all evals (the run seed varies the *training*, not the eval split) |

## Decision rule (end of Week 1)

Per `experiment_plan.md` §Phase 1:

- **Ranking holds on ≥ 3 of 5 models:** proceed with diagnostic-validation framing.
- **Ranking holds on ≤ 2 models:** stop and reassess — placement-prediction claim may not survive.
