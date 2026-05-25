# Experiment Status — 2026-03-31

## Completed Experiments

### Phi-3.5 (microsoft/Phi-3.5-mini-instruct, 32 layers)

**P1 Baselines (all done):**
- Clean-only, random-layer (3 seeds), data-aug, no-stab, cosine-stab
- Results in `stabilizer_weights/p1_*/eval_results.json`

**P2 Layer Sweep (done):**
- 6 windows (0–4, 5–9, 10–14, 15–19, 20–24, 27–31)
- Best: L15–19. Results in `stabilizer_weights/p2_sweep_L*/eval_results.json`

**2×2 Ablation — Original Rates (done, with bootstrap CIs):**
- Cosine v2 L15–19 stab=24 λ=3.0: **+7.3% avg perturbed** (headline)
- MSE L15–19 stab=24 λ=3.0: +5.9% avg perturbed
- Bootstrap CIs in `bootstrap_results/cosv2_L15_stab24_high/` and `mse_L15_stab24_high/`

**2×2 Ablation — Matched Rate (done, with bootstrap CIs):**
- All perturbations at 10%. Cosine: +6.9%, MSE: +6.1% — CIs overlap heavily
- Taxonomy interaction **disappears** at matched rates (p=0.76 vs p=0.14)
- Results in `bootstrap_results/phi_matched_cosine_stab24/` and `phi_matched_mse_stab24/`

### Llama-3 8B (meta-llama/Meta-Llama-3-8B-Instruct, 32 layers)

**Layer Sweep (done, 1000 steps):**
- 6 windows. Best: L20–24 (+1.2% perturbed, +1.0% clean)
- Early layers (0–14) all hurt. Late peak consistent with late-accumulation model.
- Results in `stabilizer_weights/llama_sweep_L*/eval_results.json`

**2×2 at L20–24, 3000 steps (partially done):**
- Cosine close (stab=29): clean +0.0%, perturbed -0.9%
- MSE close (stab=29): clean -1.2%, perturbed -2.6%
- Cosine far (stab=31): **check** `stabilizer_weights/llama_2x2_cos_far/eval_results.json`
- MSE far (stab=31): **check** `stabilizer_weights/llama_2x2_mse_far/eval_results.json`
- Job 7662701 task 4 was last seen running. Check if done.

### Workstream A — Statistical Analyses (all done)

**A1 Interaction Test:** `experiments/workstream_a/interaction_test/interaction_test_results.json`
- Original rates: per-example interaction +5.6%, CI [+2.5%, +8.7%] — **SIGNIFICANT**
- Matched rates: interaction +0.6%, CI [-2.5%, +3.5%] — NOT significant
- Speech is the only individually significant condition in both regimes

**A2 Whitespace Analysis:** `experiments/workstream_a/whitespace_analysis/whitespace_analysis_results.json`
- Effect sizes overlap between directional/uniform groups (d=0.403 vs 0.371)
- Taxonomy is a continuum, not a binary split. Case is the only clearly low one.

**A3 Power Analysis:** `experiments/workstream_a/power_analysis/power_analysis_results.json`
- No individual condition is powered at n=500. Speech closest (n=637 needed).

### Figures (all generated)

All in `experiments/workstream_c/figures/`:
- `c1_lrd_heatmaps.pdf` — Phi + Llama (Qwen data empty)
- `c2_patching_recovery.pdf` — Llama recovery curves
- `c3_layer_sweep.pdf` — Phi inverted-U
- `c4_2x2_original.pdf` — original rates money figure
- `c5_taxonomy_split_original.pdf` — per-condition original rates
- `c5b_taxonomy_split_matched.pdf` — per-condition matched rates
- `c6_effect_size_ranking.pdf` — Cohen's d ranking
- `c7_original_vs_matched_rate.pdf` — key comparison figure
- `c8_interaction_forest.pdf` — interaction forest plot

---

## TODO: Mistral Layer Sweep

**Not started.** No Mistral infrastructure exists yet.

### What model
- Likely `mistralai/Mistral-7B-Instruct-v0.3` (used in diagnostic section of paper)
- 32 layers, same architecture family as Llama (MistralForCausalLM)
- LoRA target modules: `q_proj v_proj` (same as Llama)
- May need gated access / HF token

### How to set it up
The training script `train_lrd_lora_v14.py` already supports arbitrary models via `--model`. No code changes needed — just a SLURM script.

**Template: copy `submit_llama_layer_sweep.slurm` and change:**
1. `--model` to `mistralai/Mistral-7B-Instruct-v0.3`
2. `--target_modules q_proj v_proj` (same as Llama)
3. `--batch_size 2 --grad_accum_steps 8` (7B model, same as Llama)
4. `--mem=64GB` (same as Llama)
5. Output dirs to `stabilizer_weights/mistral_sweep_L*`
6. Log names to `logs/mistral_layer_sweep_*`

**Layer windows (same 6 as Phi/Llama):**
| Task | Layers | stab_layer |
|------|--------|------------|
| 1 | 0–4 | 9 |
| 2 | 5–9 | 14 |
| 3 | 10–14 | 19 |
| 4 | 15–19 | 24 |
| 5 | 20–24 | 29 |
| 6 | 27–31 | λ_stab=0 (can't go beyond 31) |

Use 1000 steps for the sweep (same as Llama sweep). λ_stab=3.0, cosine, lr=2e-5.

### Key files to reference
- `train_lrd_lora_v14.py` — main training script (supports `--model`)
- `submit_llama_layer_sweep.slurm` — template to copy
- `train_lrd_stabilizer.py` — shared infrastructure (perturbation engine, eval conditions, dataset builders)
- `Phi3.5/perturbations.py` — the PerturbationEngine class (imported via `_import_perturbation_engine()`)

### After sweep
Pick best window, then run 2×2 (cosine vs MSE × close vs far) at 3000 steps, same as Llama 2×2. Template: `submit_llama_2x2.slurm`.

---

## Key Config Reference

All experiments use:
- `--lambda_acc 0.7 --lambda_stab 3.0` (never 0.3 — that was the scaling bug)
- `--stab_cosine` for cosine/LRD loss variant
- `--n_per_condition 857 --clean_fraction 0.25`
- `--epochs 30 --max_seq_len 512`
- Eval: `--eval_n_samples 500`

**Phi-3.5:** `--target_modules qkv_proj o_proj`, batch_size=4, grad_accum=4, lr=5e-5
**Llama-3 / Mistral:** `--target_modules q_proj v_proj`, batch_size=2, grad_accum=8, lr=2e-5, mem=64GB
