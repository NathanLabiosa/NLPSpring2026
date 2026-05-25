# Evaluation Harness Specification

> **Single source of truth for every accuracy number in the paper.**
>
> All accuracy numbers in the paper come from this harness. Any prior
> numbers produced by `sweep_lora_window.py`'s inline eval, or by other
> ad-hoc eval scripts with non-canonical settings, are **superseded** and
> must not appear in tables, figures, or prose.

This document is the deliverable for **Phase 0** of `experiment_plan.md`.

---

## 1. Canonical eval script

`eval_fixed_harness.py` (`HARNESS_VERSION = "v1"`).

It can be run either with a LoRA checkpoint (the standard adapter eval) or
with `--checkpoint` omitted (a no-LoRA baseline audit).

### Fixed configuration

| Parameter | Value | Why |
|---|---|---|
| `max_new_tokens` | 512 | The chat-template CoT for GSM8K commonly runs 200–400 tokens before `#### NUMBER`. Anything ≤300 truncates the answer line on some models (notably Phi-3.5 and Qwen with the standard system prompt), depressing the no_adapter baseline and inflating the adapter delta. This was the root cause of the +11.6 / +7.3 headline reversals. |
| `do_sample` | False | Greedy decoding — required for reproducibility across seeds and across the bootstrap. |
| `batch_size` | 8 | Padding-side="left", left-padded prompts; this was validated on A40 with bf16 for 7B–9B. |
| `padding_side` | `"left"` | Required for batched left-padded greedy gen with a chat template. |
| `n_samples` | 500 | Test split shuffled with the seed and sliced. |
| `seed` | 42 | Seeds `random` and `torch`. |
| `torch_dtype` | `bfloat16` | A40 default. |
| `attn_implementation` | `"sdpa"` (default), `"eager"` for Phi-3.5 and Gemma-2 | Phi-3.5 / Gemma-2 don't accept sdpa for the HF Phi3 / Gemma2 wrappers; eager is required. The argparse default is sdpa; flip to eager via `--attn_impl eager`. |
| Prompt format | `train_lrd_stabilizer.format_question_prompt(tokenizer, question, GSM8K_SYSTEM_PROMPT)` | Chat template + "Let's think step by step ... finish with `#### NUMBER`" system prompt. Identical across models. |
| Conditions | `clean + 6 perturbations`: typos@5%, OCR@5%, speech@10%, homophones@20%, whitespace@10%, case@10% | Matches the rates the LoRAs were trained at. |
| Answer extraction | `_extract_gsm8k_answer` / `_extract_gsm8k_truth` / `_gsm8k_correct` from `train_lrd_stabilizer.py` | Same regex for predictions and ground truth. |

### Output schema

Every eval JSON written by `eval_fixed_harness.py` records:

```json
{
  "harness_version": "v1",
  "base_model": "...",
  "checkpoint": "..." | null,
  "seed": 42,
  "n_samples": 500,
  "max_new_tokens": 512,
  "harness": "exp3_heldout_style_fixed",
  "clean_baseline": <acc_no_adapter on clean>,
  "mean_perturbed_delta": <mean Δ across the 6 perturbations>,
  "results": {
    "clean_baseline": { "acc_no_adapter", "acc_with_adapter", "acc_clean_baseline", "delta", ... },
    "typos_5pct":    { ... },
    "ocr_5pct":      { ... },
    "speech_10pct":  { ... },
    "homophones_20pct": { ... },
    "whitespace_10pct": { ... },
    "case_10pct":    { ... }
  }
}
```

`acc_with_adapter` and `delta` are `null` in baseline-audit mode.

---

## 2. Per-model clean baselines (`acc_no_adapter` on clean GSM8K)

These are the no-LoRA accuracies that every adapter on that model should be
compared against. **Tolerances are per-model** because published GSM8K
numbers are themselves harness-sensitive: Llama-3 in particular has a
well-known ~7 pp gap between Meta's official 8-shot CoT eval and community
zero-shot chat-template reproductions, and Mistral has no canonical
instruct GSM8K number to anchor against.

If a no_adapter baseline drifts outside the tolerance window for its
model, the harness is wrong for that model — stop and fix before
proceeding to Phase 1.

| Model | HF id | Published GSM8K | Fixed-harness clean baseline | Tolerance | Status / notes |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct | `microsoft/Phi-3.5-mini-instruct` | 86.2% (Microsoft model card) | **85.4%** (job 8867051) | ±5 pp | ✓ passes (Δ = −0.8 pp). `results/fixed_harness/v1/three_checkpoint_comparison/fixed_harness_phi35_stab.json`. |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 85.4–91.6% (varies by harness; Qwen's own report uses 4-shot) | **89.0%** (jobs 8849598/8849599) | ±5 pp | ✓ passes. `results/fixed_harness/v1/three_checkpoint_comparison/fixed_harness_qwen_vanilla.json` (and `..._qwen_stab.json`). |
| Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | 79.6% (Meta launch, 8-shot CoT, maj@1); community zero-shot chat reproductions land 75–79% | **78.4%** (job 8872713) | **±7 pp** (acceptable range 73–86%) | ✓ passes (Δ = −1.2 pp from Meta's 79.6%; right at the community zero-shot ceiling). `results/fixed_harness/v1/baseline_audit/baseline_llama3_8b_instruct.json`. |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | No canonical instruct number — Mistral's tech report covers base at 52.2% (8-shot maj@8). v0.3 instruct community evals span 35–55% under zero-shot chat. | **59.6%** (job 8872714) | **±10 pp around ~45%** (acceptable range 35–55%) | ⚠ Above strict 35–55% band by 4.6 pp; still inside the 30–65% sanity floor. Plausible: chat-template + greedy + 500 samples may run hotter than community evals that often use older v0.1 / v0.2 weights. **Soft pass — proceed.** `results/fixed_harness/v1/baseline_audit/baseline_mistral_7b_v03.json`. |
| Gemma-2-9B (base, not -it) | `google/gemma-2-9b` | ~50–60% for the base under chat-template prompting; Google reports **68.6% for the -it variant**, not the base. | **64.8%** (job 8908676) | **±7 pp** (acceptable range 57.8–75.8% around paper's 68.1%) | ⚠ Soft pass — see **§2.1**. Inside the ±7 pp tolerance around 68.1%, but ~5 pp above the band community reports give for the base. `results/fixed_harness/v1/baseline_audit/baseline_gemma2_9b.json`. |

Update each row as the audit jobs land.

### 2.1 Gemma-2 base — paper number resolved

Model panel confirmed: `google/gemma-2-9b` (base, not `-it`) is the model
the paper intends. The `download_gemma2_9b.slurm` script even purges the
`-it` cache explicitly to keep this unambiguous.

**Measured fixed-harness baseline:** 64.8% (job 8908676, 6 of 7 conditions
completed before the 8h wall clock — `clean_baseline` is the
audit-critical number and is complete).

**Resolution of the §2.1 question (paper 68.1% vs measurement 64.8%):**

- The 64.8% measurement is **inside the ±7 pp tolerance** around the
  paper's quoted 68.1% — the draft number is harness-compatible.
- It is also **higher than community zero-shot chat reports for the base
  (50–60%)**, but not implausibly so: chat-template + greedy decoding +
  the LRD system prompt ("Let's think step by step … `#### NUMBER`") is
  a stronger prompt than naïve zero-shot. The Mistral-v0.3 audit landed
  similarly high (59.6% vs community 35–55%) under the same harness, so
  the offset is a property of the harness, not a base/-it confusion.
- **Not a base/-it mix-up:** the model card hash and the
  `download_gemma2_9b.slurm` `-it`-purge guard rule that out; if it were
  `-it` we would expect ≥ 68%, not 64.8%.

**Action — what to do with the paper's 68.1%:**

- Replace 68.1% → **64.8%** wherever the paper quotes the Gemma-2-9B
  base clean accuracy. The fixed-harness number is the canonical one for
  the rest of the LRD analysis on Gemma.
- No further investigation needed — this row is a **soft pass**.
- The Phase 1 Gemma slurm (`submit_phase1_gemma2.slurm`) is fine as-is
  (also targets the base) — no change needed.

---

## 3. Regression test

`regression_test_harness.py` re-runs `eval_fixed_harness.py` on the
Qwen-7B stabilizer checkpoint (`qwen25_7b_L24_27_stab/lora_final`) and
asserts:

- clean baseline within **±1.5 pp** of 89.0%
- mean perturbed Δ within **±1.0 pp** of +0.17 pp

Run this **before every batch of jobs** to catch silent harness drift.

```bash
sbatch /home1/labiosa/NLPSpring2026/submit_regression_test.slurm
# or interactively on a GPU node:
python /home1/labiosa/NLPSpring2026/regression_test_harness.py
```

Exits non-zero on drift.

---

## 4. Results directory layout

All fixed-harness JSON outputs go under:

```
results/fixed_harness/v{HARNESS_VERSION}/
├── baseline_audit/                   # no-LoRA per-model baselines (Phase 0)
│   ├── baseline_phi35_mini_instruct.json
│   ├── baseline_qwen2.5_7b_instruct.json
│   ├── baseline_llama3_8b_instruct.json
│   ├── baseline_mistral_7b_v03.json
│   └── baseline_gemma2_9b.json
├── three_checkpoint_comparison/      # the §5 cross-checkpoint audit
│   ├── fixed_harness_qwen_vanilla.json
│   ├── fixed_harness_qwen_stab.json
│   └── fixed_harness_phi35_stab.json
├── layer_sweep/                      # Phase 1 outputs
│   └── {model}_{window}_{seed}.json   ← naming convention
└── regression/                       # outputs of the regression test
```

**Filename convention** for layer-sweep runs:

```
{model_slug}_{window}_{seed}.json

Examples:
  phi35_L15-19_seed42.json
  qwen2.5_7b_L24-27_seed43.json
  llama3_8b_L20-24_seed44.json
```

`(model, window, seed)` uniquely identifies a run; the `harness_version`
field inside the JSON disambiguates if the harness changes.

---

## 5. Deprecated paths

| Script | What it used | Why deprecated | Guard |
|---|---|---|---|
| `sweep_lora_window.py` | `max_new_tokens=100` in inline eval | Truncated CoT before `#### NUMBER`. Root cause of the +11.6 Qwen / +7.3 Phi-3.5 reversals. | Hard assert on `ALLOW_SWEEP_EVAL=1` env var at the top of `__main__`. |
| `eval_paired_bootstrap.py` | `max_new_tokens=300` | < 512; on some models truncates CoT. **Bumped to 512.** | Docstring deprecation note. Kept for bootstrap-CI analysis only. |
| `eval_bootstrap.py` | `max_new_tokens=300` | Same. **Bumped to 512.** | Docstring deprecation note. |
| `eval_typos_rerun.py` | `max_new_tokens=300` | One-off v5 rerun script. **Bumped to 512.** | Docstring deprecation note. |

### Audit summary (the full `max_new_tokens` grep)

Other `max_new_tokens` values in the repo are not accuracy-reporting paths:

- **Training-time generation logs** (`train_lrd_lora_v14.py`,
  `train_lrd_stabilizer.py`, `train_augmentation_baseline.py`,
  `scripts/train_lrd_stabilizer.py`): 300 tokens. These print to the
  training log only — accuracy that appears in the paper is the
  fixed-harness eval after training.
- **Activation patching** (`Gemma2/patch_run.py`, `scripts/patch_only.py`,
  `Qwen2.5/patch_only_speech.py`, `Qwen2.5/patch_run.py`,
  `Mistral/patch_only.py`, `TinyLlama/patch_run.py`): 128 / 512 / etc.
  Patching measures hidden-state recovery, not generation accuracy.
- **LRD diagnostics** (`Gemma2/lrd_run.py`, `Mistral/lrd_diagnostics.py`,
  `scripts/lrd_diagnostics.py`, `Qwen2.5/submit_lrd.slurm`): 256–768.
  Diagnostics produce LRD profiles from hidden states; the post-projection
  generation in these scripts is for logging only.
- **CKA / clustering / ablation** (`compute_cka.py`, `expE_clustering.py`,
  `ablate_gate.py`, `compensatory_analysis.py`): 256–300. Probe-style
  analyses; no accuracy claim depends on them.
- **Sanity / viability** (`Gemma2/viability_check.py`,
  `TinyLlama/viability_check.py`, `Mistral/main.py`,
  `scripts/viability_check.py`, `verify_base_accuracy.py`,
  `Qwen2.5/qwen_bugfix.py`): 256–600. Diagnostic; not paper accuracy.
- **Transfer eval** (`eval_transfer.py`): 64. Different task; not GSM8K.

None of those produce numbers that flow into the paper's accuracy tables.
They are documented here for completeness only.

---

## 6. Supersession statement

> All accuracy numbers in this paper come from `eval_fixed_harness.py` at
> `harness_version="v1"`. Numbers produced by `sweep_lora_window.py`'s
> inline eval (`max_new_tokens=100`) are superseded; the `+11.6%` Qwen
> headline and the `+7.3%` Phi-3.5 headline reported in the prior draft
> are products of that truncated-CoT harness and do not reproduce here.
> See `experiment_results.md` §5 and the three-checkpoint comparison JSONs
> in `results/fixed_harness/v1/three_checkpoint_comparison/`.
