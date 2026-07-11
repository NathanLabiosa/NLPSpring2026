# Coding Session: New Experiments & Code Release Instructions

## Context (read this first)

You are supporting the revision of an ARR/EMNLP submission (scores 1.5/2/2.5) for resubmission next ARR cycle. The paper studies layer-wise perturbation robustness in LLMs via three per-layer signals — LRD (Layer-Wise Representation Divergence: 1 − cosine between mean-pooled clean vs. perturbed hidden states at each layer), activation patching recovery, and localized LoRA effectiveness — and shows they dissociate. A key result is a **fixed-harness layer sweep**: LoRA windows placed at diagnostic-flagged early/mid layers uniformly hurt perturbed accuracy; only the deepest windows are non-negative.

The user has an existing experimental pipeline (LoRA training, perturbation generation, patching, LRD extraction, GSM8K evaluation harness). Your job is to extend it for four deliverables, in priority order below. A parallel writing session is revising the paper and has left `% TODO[EXPT]` placeholders for these results.

## Existing experimental configuration (match exactly unless a task says otherwise)

- **Models:** Phi-3.5-mini-instruct (3.8B, 32 layers), Qwen2.5-7B-Instruct, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9B base. Adjudicable models for placement claims: **Phi-3.5, Qwen2.5-7B, Llama-3** (Mistral excluded for an lr artifact; Gemma adapter-insensitive).
- **LoRA:** r=4, α=8, on attention `q_proj`/`v_proj`; Phi-3.5 has fused attention, so use `qkv_proj`/`o_proj` there. Base model frozen. CE-only loss (λ_stab=0) on answer positions, trained on perturbed GSM8K with clean supervision targets.
- **Windows:** non-overlapping nominal 5-layer windows (Phi-3.5: L00–04, L05–09, L10–14, L15–19, L20–24, L27–31; Qwen: …, L08–11, …, L20–23, L24–27 — replicate the partitions in the paper's Table 4 exactly).
- **Seeds:** 3 per cell. Report mean ± std across seeds.
- **Learning rate:** shared grid used 5e-5. Known issue: Mistral at 5e-5 is uniformly strongly negative; a 1e-5 single-cell audit recovered to +0.13pp. Any new Mistral runs should use 1e-5, but Mistral is low priority.
- **Evaluation harness (CRITICAL):** the fixed harness with `max_new_tokens=768`, n=500 per condition. The paper's central negative result is that `max_new_tokens=100` truncated chain-of-thought and scored it as empty, producing fake +7–11pp "gains." **Every new evaluation must use the fixed harness. Any surprising positive result must be re-checked for generation-length sensitivity before being believed.**
- **Perturbations:** six types — typos (5%), OCR (5%), whitespace (10%), case swap (10%), speech-like (10%), homophones (20% per-word default; 40%/50% variants exist). Reported perturbed Δ = mean change in perturbed accuracy over the six types vs. no-adapter baseline.
- **Key baselines from the paper (for sanity checks):** clean GSM8K accuracy — Phi-3.5 85.4%, Mistral 59.6%, Llama-3 78.4%, Gemma 64.8%, Qwen2.5-7B 89.0%.

## Task 1 (highest priority): Window-length ablation

**Reviewer ask (KR2F):** "The length of the LoRA adapter window is fixed at 5. Different window lengths might yield different results. The paper has not verified whether shorter or longer windows could lead to better recovery performance."

**Design:** Phi-3.5 only (cheapest model). Two window positions:
- The worst mid window (L10–14, paper value −5.3 ± 0.9 pp at width 5)
- The best late window (L27–31, paper value +0.2 ± 0.3 pp at width 5)

For each position, run widths **3 and 7**, centered on the same center layer as the width-5 window (L10–14 → center 12 → widths 3: L11–13, 7: L09–15; L27–31 → center 29 → width 3: L28–30, width 7: L26–32 — clamp to L25–31 if layer 32 doesn't exist; Phi-3.5-mini has 32 decoder layers indexed 0–31, so clamp and note it). 3 seeds each.

**Budget:** 2 positions × 2 widths × 3 seeds = 12 LoRA trainings + fixed-harness evals (6 perturbations + clean, n=500) on a 3.8B model.

**Deliverable:** a table — rows = window position, columns = width 3/5/7 (width-5 column copied from existing results), cells = mean perturbed Δ ± std. Success criterion for the paper: the *sign structure* is width-robust (mid negative, late ≥ 0 at all widths). If not robust, that's still reportable — just flag it loudly.

## Task 2: All-layer LoRA baseline

**Motivation:** conspicuous missing baseline; a fresh reviewer will ask "what if you just adapt every layer?" Also sharpens the paper either way: if all-layer LoRA also regresses → cascade disruption is even more general; if it helps → "deep-only matches/beats all-layer at a fraction of the trainable parameters" becomes a small positive recipe.

**Design:** LoRA on q_proj/v_proj (qkv/o for Phi) at **every** layer, same r=4/α=8, CE-only, 3 seeds, on the three adjudicable models (Phi-3.5, Qwen2.5-7B, Llama-3-8B). Same training data/steps as the sweep cells. Evaluate under the fixed harness: clean Δ and mean perturbed Δ.

**Watch out:** all-layer LoRA has ~6× the trainable parameters of a 5-layer window; if training steps were tuned for the window setting, keep them identical anyway (the comparison is placement at matched budget), but record training loss curves so over/under-training is diagnosable. Optionally add one lr=1e-5 seed per model if 5e-5 looks unstable.

**Budget:** 3 models × 3 seeds = 9 trainings + evals.

**Deliverable:** one row per model appended to the layer-sweep table: "All layers" column with mean ± std, plus clean Δ.

## Task 3: Fixed-harness layer sweep on MMLU (the one big compute item)

**Reviewer ask (KR2F):** "The Layer Sweep is only performed on GSM8K, which undermines the generalizability across different tasks." The paper's own Discussion admits results "may not transfer to non-CoT settings."

**Why MMLU:** multiple-choice → answers are single tokens/letters → generation-length harness artifacts are structurally impossible. This converts the paper's own caveat into a strength.

**Design:**
- Models: **Phi-3.5 (spike-and-suppress) and Qwen2.5-7B (late-accumulation)** — one per regime. If compute runs short, Phi-3.5 alone is still a meaningful result.
- Training: LoRA per window (same partitions as GSM8K sweep), CE-only, trained on **perturbed MMLU prompts with clean supervision targets** (mirror the GSM8K protocol: perturb the question text, supervise the correct answer letter/token). Use the same six perturbation types at the same rates. 3 seeds per window.
- Eval: n=500 MMLU items (fixed subset, stratified across subjects; save the item IDs), same six perturbations + clean. Report mean perturbed Δ per window vs. no-adapter MMLU baseline. `max_new_tokens` can be small here since answers are short — that's fine and is the point; still record it.
- Also compute **LRD profiles on MMLU** for both models if not already stored (the paper reports cross-task LRD stability ρ > 0.95 for Phi-3.5 and Mistral on GSM8K/MMLU/BBH — extending to Qwen strengthens this, and it's cheap: forward passes only).

**Success criterion:** the mid-layer wall replicates (early/mid windows ≤ 0, deepest windows least regressive) on a task where truncation artifacts cannot exist. Partial replication or a flat profile is still reportable — report whatever happens honestly.

**Budget:** ~2 models × ~6 windows × 3 seeds = ~36 trainings + evals. This is comparable to a substantial fraction of the original sweep; start it early and let it run while Tasks 1–2 complete.

## Task 4: Anonymous code release (zero GPU, high score impact)

**Reviewer damage:** one reviewer gave Datasets: 1, Software: 1, Reproducibility: 2 because code was promised only "at camera-ready."

**Steps:**
1. Clean the repo: remove dead code, absolute paths, API keys, cluster-specific job scripts (or genericize them), author names/emails/institution strings in comments, git history (fresh repo, single squashed commit).
2. Structure: `perturbations/` (the six generators + the three held-out types char_insert/char_swap/qwerty), `lrd/` (extraction + slope/recovery-rate summary stats), `patching/`, `lora_sweep/` (training + fixed harness eval), `analysis/` (block-bootstrap correlations, block size 5, 10k resamples; Cohen's d; plots), `configs/` (one YAML per paper table/figure cell).
3. A README mapping **every table and figure in the paper to the exact command that reproduces it**, plus hardware notes (GPU type/hours per cell so reviewers can scope partial re-runs — the paper's ethics section promises this).
4. Include per-model evaluation logs or per-example prediction dumps where licensing permits (the perturbation outputs on GSM8K/MMLU are derivative of MIT-licensed data — fine to include; check BBH Apache-2.0 similarly).
5. Pin the environment: `requirements.txt` with exact versions + torch/transformers/peft versions used; note bf16 (the paper reports reproduction within a bf16 numerical floor, max abs diff ≤ 0.0025).
6. Upload to an anonymized host (e.g., anonymous.4open.science) and verify the link works logged-out.
7. Smoke test: from a fresh clone, reproduce one cheap cell end-to-end (e.g., LRD profile for Phi-3.5 on typos, and one LoRA window with 1 seed at reduced n) and confirm numbers land near the paper's within seed noise.

## Cross-cutting rules

- **Harness discipline:** every accuracy number goes through the fixed harness config committed in the repo. For any GSM8K eval, `max_new_tokens=768`. If any new result looks surprisingly positive, rerun that cell's eval at a longer budget before reporting it — this paper's credibility now rests on harness hygiene.
- **Determinism/logging:** log seed, lr, steps, window layers, adapter target modules, harness config, and dataset item IDs with every run; write results to structured JSON/CSV so the writing session can pull exact numbers.
- **No cherry-picking:** all seeds get reported; failed/unstable runs get documented, not silently rerun (the paper's honest-reporting posture is a selling point — Mistral's lr artifact and the harness reversal are *featured*, not hidden).
- **Statistical reporting:** for new tables, mean ± std over 3 seeds matches the paper's convention. If asked for significance on layer-wise correlations, use the existing moving-block bootstrap (block size 5, 10k resamples) — do not use naive N across layers, since layer-wise signals autocorrelate.
- **Output format:** for each task, produce (a) the results table as CSV + a LaTeX-formatted table snippet matching the paper's booktabs style, (b) a 3–5 sentence plain-language summary of what the result means for the paper's claims, flagged for the writing session's matching `% TODO[EXPT]` placeholder.

## Priority and scheduling

1. Kick off **Task 3** training jobs first (longest wall-clock), then do Tasks 1–2 in the gaps.
2. Task 4 in parallel on CPU time.
3. If total compute is constrained, the cut order is: drop Qwen from Task 3 (keep Phi-3.5), then drop width-7 from Task 1 (keep width-3), never drop Task 4.

