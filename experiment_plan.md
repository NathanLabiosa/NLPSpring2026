# Experiment Plan: Diagnostic-Only Paper Rewrite

**Target:** ARR / ACL 2027 / EACL (NOT EMNLP — the 5-day deadline forces a rushed submission of work with known harness issues).
**Budget:** ~1 month of GPU access.
**Scope change:** The stability loss is dropped as a contribution. The paper is now a **diagnostic + mechanism + placement-validation** paper. The thesis sentence the whole paper must serve:

> We introduce LRD, a per-layer cosine-divergence metric, and use it alongside activation patching and LoRA adapter sweeps to measure three dissociable per-layer properties of transformer robustness — where perturbations cause representational change, where they cause failures, and where adapters can repair them. We show these dissociate, identify two propagation regimes that organize the dissociation pattern, and explain via cascade disruption why the layers patching identifies are the wrong layers for adapter placement.

If a sentence, table, figure, or experiment doesn't serve this thesis, it goes to supplementary or gets cut.

---

## What's being dropped

- The stability-loss contribution (old contribution #3, "regime-conditional intervention design").
- The Phi-3.5 +7.3% headline result.
- The Qwen +11.6% vanilla result and the "stability loss suppresses late-accumulation gains" framing.
- All claims of the form "apply stability loss on spike-and-suppress models only."
- Section 6.2 (`sec:regime_dependent`) in its current form.
- The "stabilizer helps/hurts" column in Table 6.

## What's being kept

- **LRD metric definition** (Section 3.1).
- **Three-signal dissociation** (sensitivity / causality / compensatory capacity) — old contribution #1.
- **Two propagation regimes** (spike-and-suppress vs. late-accumulation), with the family/regime confound stated honestly.
- **LRD × patching anti-correlation** ($\rho = -0.72$ to $-0.88$ on confirmed-patching models).
- **Cascade disruption mechanism** (Section 5.6) — this is the most novel mechanistic finding. Hidden-state measurements are harness-independent and survive.
- **Layer sweep** — reframed as **validating the diagnostic** (placement ranking tracks LRD/patching prediction), NOT as a stability-loss intervention.
- **Qwen scale variation** (1.5B → 7B → 14B) — currently underexploited; promote into the main paper as within-family validation.
- **14B activation patching** (84/100 pairs, full 48-layer sweep) — direct support for the late-accumulation regime at a new scale.

---

## Phase 0 — Codebase prep (Day 1, before queueing anything)

**Goal:** Make the fixed harness the only eval path. The harness divergence is the root cause of every numerical problem in the current paper; eliminate the possibility of regressing.

### Tasks

1. **Designate `eval_fixed_harness.py` as the canonical eval.** Verified config (from §5 of experiment_results.md):
   - chat-template prompt via `train_lrd_stabilizer.format_question_prompt`
   - standard GSM8K system prompt
   - `max_new_tokens=512`
   - `do_sample=False`
   - `batch_size=8`
   - `padding_side="left"`
   - `n_samples=500`
   - `seed=42`
   - GSM8K test split shuffled with that seed
2. **Add a hard guard to `sweep_lora_window.py`** so it cannot be run as a primary eval. Make it import from `eval_fixed_harness.py` for all accuracy reporting, OR add an assertion at the top: `assert os.environ.get("ALLOW_SWEEP_EVAL") == "1", "Use eval_fixed_harness.py — sweep harness is deprecated for accuracy reporting (see experiment_results.md §5)."`
3. **Audit every `max_new_tokens` value** in the repo. Anything not 512 in an eval path is a bug. Truncation at 100 was the root cause of the Phi-3.5 / Qwen baseline depression.
4. **Audit eval baselines.** For every model in the panel, run `eval_fixed_harness.py` with **no adapter** and confirm the clean GSM8K baseline matches published numbers:
   - Phi-3.5-mini-instruct: ~85% (saw 85.4% in §5)
   - Qwen2.5-7B-Instruct: ~88-89% (saw 89.0% in §5)
   - Llama-3-8B-Instruct: check against Meta's published GSM8K
   - Mistral-7B-Instruct-v0.3: check against published
   - Gemma-2-9B base: check against published
   If any baseline is >5pp off published, the harness is still wrong for that model — fix before proceeding.
5. **Create a regression test:** a tiny script that loads a known LoRA checkpoint (Qwen stab, which reproduces) and asserts its mean perturbed Δ is within ±1pp of the recorded value. Run before every batch of jobs to catch silent harness drift.
6. **Centralize result schema.** Every eval JSON must record: `harness_version`, `max_new_tokens`, `n_samples`, `seed`, `clean_baseline`, per-perturbation `no_adapter` and `with_adapter` accuracies, `delta` per condition, `mean_perturbed_delta`. No exceptions. This becomes the single source of truth for tables.
7. **Set up a `results/` directory structure** keyed by `(model, window, seed, harness_version)` so reruns don't overwrite and cross-experiment comparisons are unambiguous.

### Deliverable

A `HARNESS.md` in the repo root documenting:
- Which eval script is canonical
- What each parameter is and why
- The clean baseline for each model in the panel
- The regression-test command
- A statement: "All accuracy numbers in the paper come from this harness; any prior numbers from `sweep_lora_window.py` eval are superseded."

---

## Phase 1 — Re-baseline the layer sweep under the fixed harness (Week 1)

**Goal:** Rebuild the layer-sweep evidence (currently §6.2, Table 7) under the fixed harness. The claim being defended is now **diagnostic validation**: the placement *ranking* across windows tracks the LRD/patching prediction, regardless of whether absolute Δ is positive or negative.

This is the highest-priority week. Everything downstream depends on these numbers.

### Why this matters

The layer-sweep table is what connects the diagnostic (Sections 3–5) to a downstream prediction. Even if every adapter shows null or negative absolute Δ under the fixed harness, the *relative* ordering of windows is the prediction: windows the diagnostic flags as good should beat windows it flags as bad. That's a much weaker, more honest, and more defensible claim than "+7.3% on Phi-3.5."

### Experiments to queue

**Eval harness:** `eval_fixed_harness.py`, fixed config from Phase 0.
**Training harness:** the canonical λ>0 trainer `train_lrd_lora_v14.py` (per experiment_results.md exp1 root-cause: this is the trainer that has been validated; do NOT use `sweep_lora_window.py`'s training path for stability runs).
**Loss:** CE-only (`λ_stab=0`). The stability loss is dropped — every experiment from here on is CE-only LoRA, training on perturbed inputs with clean targets. This is the data-augmentation baseline from exp2.
**LoRA config:** rank 4, α=8, attention `q_proj`/`v_proj` (match the paper).

For each model, run the windows below × 3 seeds (42, 43, 44). Each run produces a fixed-harness eval JSON.

#### Phi-3.5-mini-instruct (32 layers)

Windows: `L00-04`, `L05-09`, `L10-14`, `L15-19`, `L20-24`, `L27-31`. (6 windows × 3 seeds = 18 runs.)

#### Llama-3-8B-Instruct (32 layers)

Windows: `L00-04`, `L05-09`, `L15-19`, `L20-24`, `L27-31`. (5 windows × 3 seeds = 15 runs.)

#### Mistral-7B-Instruct-v0.3 (32 layers)

Same 5 windows as Llama-3 × 3 seeds = 15 runs.

#### Qwen2.5-7B-Instruct (28 layers)

Windows: `L00-04`, `L05-09`, `L08-11`, `L15-19`, `L20-23`, `L24-27`. (`L08-11` is included because it was the C3/C4 prediction that failed; keeping it falsifies the post-hoc-metric claim.) (6 windows × 3 seeds = 18 runs.)

#### Gemma-2-9B base (42 layers)

Windows: `L00-04`, `L06-11`, `L15-20`, `L25-30`, `L35-40`. (5 windows × 3 seeds = 15 runs.) Note: Gemma was flat in the original sweep; this confirms or refutes flatness under the corrected harness.

**Total: ~81 training+eval runs.** Each is ~1h for 7B-class training + ~5h eval (per job 8867051). With parallelism across the queue, plan for ~4-5 days wall-clock if 4-6 jobs run in parallel.

### What to report

For each model, a table of `(window, mean perturbed Δ, std across seeds, per-condition Δ)`. Then a **ranking** column: which window ranks best, 2nd, ..., worst on mean perturbed Δ within that model.

The headline claim is: **does the within-model ranking track the LRD/patching prediction?**
- Phi-3.5 (S&S, suppressed-LRD prediction): L15-19 should rank above L00-04 and L27-31.
- Llama-3, Mistral (Late, accumulating-LRD prediction): L20-24 should rank above L00-04.
- Qwen (Late): L24-27 should rank above L08-11 (the failed C3/C4 prediction) and above L00-04.

If the ranking holds even with negative absolute Δ values, the diagnostic is validated as a *placement guide*, not as a guarantee of accuracy gains. That's the honest claim.

### Decision rule at end of Week 1

- **If ranking holds on ≥3 of 5 models:** Proceed with diagnostic-validation framing as planned.
- **If ranking holds on ≤2 models:** Stop and reassess — the diagnostic may not predict placement reliably. (This would shrink the paper further, possibly to a pure measurement paper without the placement-prediction claim.)

---

## Phase 2 — Strengthen the diagnostic claims (Week 2)

**Goal:** Promote and harden the parts of the diagnostic story that already exist but are underweighted in the current draft.

### 2.1 — Qwen scale variation (promote to main paper)

Already complete (experiment_results.md §1, §2, §6). Three data points: Qwen2.5-1.5B, 7B, 14B, all late-accumulation, with late/early ratio rising monotonically (1.59 → 2.90 → 4.10). Plus 14B activation patching at 84/100 pairs cleared the 80% identity ceiling.

**Action:** Build a dedicated subsection in the main paper (currently buried). New main-paper figure: late/early ratio vs model scale, three Qwen points, showing monotonic intensification. New main-paper table: 14B per-region patching recovery (already in §6 of results: early 73.1%, mid 71.1%, late 42.3%, peak L23 → 82.1%).

This directly addresses the biggest weakness of the current draft — the n=5 / family-regime confound. Three within-family points on Qwen is now a load-bearing piece of evidence. No new compute required, just rewriting.

### 2.2 — Llama-3.2-1B LRD profiling (if HF access comes through)

Currently blocked (experiment_results.md §3). **Re-request access on day 1 of the month** — the HF approval is usually 1-3 days. If granted:
- Run the existing `Llama/submit_llama32_1b_lrd.slurm` — should run end-to-end.
- This gives a second family with within-family scale variation (Llama-3.2-1B vs Llama-3-8B).
- A second family substantially strengthens the regime-as-general-property claim.

**Action:** Submit access request day 1; if granted by day 3, run the 1B profiling. If denied or pending by day 5, drop and note in limitations.

### 2.3 — Cascade disruption verification

Hidden-state cosines (LRD between frozen and adapted models on clean inputs) are harness-independent — they're not affected by `max_new_tokens`. But run a sanity check anyway: pick one adapted checkpoint per model, recompute `frozen × adapted` clean-input LRD, confirm it matches the numbers in the current Figure 5 (disruption_phi35.pdf and the supplementary disruption figures).

**Action:** One small script, one afternoon. If numbers match: cascade disruption section is clean and stays as-is. If they don't match: investigate.

### 2.4 — Held-out perturbation evaluation of the diagnostic

The diagnostic claim is that LRD profiles are stable across tasks ($\rho > 0.95$, Section 5.7). Extend to perturbation types: pick one model per regime (Phi-3.5 and Llama-3), compute LRD profiles on perturbations not used in the main analysis (e.g., random-character insertion, character swap). The claim "regime classification transfers across perturbation types" becomes stronger.

**Action:** ~2-3 days, mostly compute. Can run in parallel with Phase 1.

### Deliverable end of Week 2

- New main-paper subsection on Qwen scale variation, with figure and table.
- Llama scale data if access cleared, otherwise a noted limitation.
- Confirmed cascade-disruption numbers.
- Held-out perturbation LRD profiles for at least 2 models.

---

## Phase 3 — Writing (Week 3)

The experiments support a rewrite, not patches. Plan to rewrite the abstract, intro, contributions, Section 6, discussion, and conclusion from scratch against the new thesis. Methods (Section 3) and most of Diagnostic Results (Sections 4–5) survive with minor edits.

### 3.1 — Title

Drop "Regime-Conditional Approach" — it promises an intervention prescription you no longer claim. Working titles to choose from:
- "Sensitivity, Causality, and Repair Dissociate: A Layer-Wise Analysis of Transformer Robustness"
- "Where Perturbations Live, Fail, and Can Be Repaired in Transformer LMs"
- "Three Layer Maps of Surface-Perturbation Robustness in Language Models"

### 3.2 — Abstract rewrite

Drop the stabilizer numbers. New abstract emphasizes: (1) the LRD metric, (2) the three-signal dissociation, (3) the two regimes including the scale-variation Qwen evidence, (4) the LRD × patching anti-correlation, (5) the cascade disruption mechanism, (6) the layer-sweep validation that placement ranking tracks the diagnostic.

### 3.3 — Contributions rewrite

Three contributions, all diagnostic/mechanistic:

1. **A three-signal dissociation across five models.** (Same as current contribution #1, lightly edited. Drop the LRD×LoRA significance claim — directional pattern only.)
2. **Cascade disruption as the mechanism.** Promote from current contribution #2's secondary clause. Early-layer LoRA breaks clean-input computation; this explains why the patching window is the wrong placement target. Mechanistically interesting and harness-independent.
3. **Diagnostic-guided placement validated by layer sweep.** Reframe current contribution #3. The diagnostic predicts within-model placement ranking; the sweep confirms it on N of 5 models (fill in from Phase 1 results). No claims about absolute accuracy gains, no claims about stability losses.

### 3.4 — Section 6 rewrite

Old §6 (Stabilization Results) becomes new §6 (Diagnostic-Guided Placement). Contents:
- Brief framing: testing whether the diagnostic predicts useful placement (the *ranking* claim).
- Layer-sweep results under fixed harness, table per model with seed std.
- Ranking analysis: does the LRD/patching-predicted window rank above non-predicted windows?
- Honest reporting of absolute Δ — if negative across the board, say so. The relative ranking is what matters for the diagnostic claim.

**Section 6.2 (regime-dependent stability loss) is deleted.** The +5 offset implementation detail, the cosine vs MSE controls, all the loss-design machinery — supplementary or cut.

### 3.5 — Discussion rewrite

Drop "implications for future loss design." The new discussion:
- What the dissociation reveals about transformer perturbation processing.
- Why cascade disruption matters for any future adapter-placement work.
- The architecture/regime confound, honestly addressed, with the Qwen scale data as the partial response.
- A short, honest paragraph on the negative stabilizer result and what it suggests for future intervention work — frame as a constraint, not a contribution.

### 3.6 — Limitations rewrite

Already strong in the current draft. Update:
- Panel size: keep, mention Qwen scale data partially mitigates.
- Drop the "stabilizer positive gains demonstrated only on Phi-3.5/GSM8K" paragraph entirely.
- Add: "We do not provide a positive intervention recipe; the diagnostic's predictions are validated as relative-ranking claims under a fixed eval harness, and absolute accuracy gains are not claimed."
- Patching ceiling effects: keep.

---

## Phase 4 — Polish + final experiment slot (Week 4)

### 4.1 — Pick one of these for the last compute slot

Use remaining compute on whichever is most accessible and most strengthens the paper:

**Option A — Sixth model in the panel.** Adding any model (a Phi variant, a smaller Llama, a TinyLlama) shrinks the n=5 problem. Needs LRD profiling + at least 2-3 layer-sweep windows under the fixed harness. ~3-4 days of compute.

**Option B — Cross-task layer sweep on a non-GSM8K dataset.** Current cross-task evidence (§5.7) is LRD profiles only. Showing the *placement ranking* also transfers (e.g., L15-19 wins on Phi-3.5 on MMLU too) is a much stronger transfer claim. ~3-4 days.

**Option C — A second within-family scale point** if Llama-3.2-1B came through in Phase 2. Bringing the Llama family to 2-3 scales mirrors the Qwen evidence and substantially de-confounds the panel.

Recommend Option C if Llama access cleared, otherwise Option B.

### 4.2 — Polish

- Rebuild all figures from fixed-harness data. The PDFs in the current draft (`lrd_heatmaps.pdf`, `patching_recovery.pdf`, `three_map_phi35.pdf`, `disruption_phi35.pdf`, `cross_task_phi35.pdf`, `layer_sweep.pdf`) all need verification or regeneration.
- Supplementary materials: per-model versions of each main-paper figure, per-perturbation breakdowns, full per-window sweep tables, hyperparameter tables, harness specification.
- Reproducibility: the `HARNESS.md` from Phase 0, training scripts, eval scripts, seeds, checkpoint hashes.
- Bib pass.

### 4.3 — Pre-submission checks

- Every accuracy number in the paper traces to a fixed-harness eval JSON. Grep the LaTeX for any number not in the canonical results directory.
- Every claim of significance is backed by the block-bootstrap procedure already documented in §3.6, and the effective-N caveat is preserved.
- No claim of the form "the stability loss helps" or "regime-conditional intervention" survives anywhere — abstract, intro, contributions, body, discussion, conclusion, limitations.
- A friendly reader test: ask one collaborator to read the paper cold and identify the contributions. If they can recite the three diagnostic contributions back, the framing is clean.

---

## Quick decision log (questions for you, the human, to answer before Claude Code starts)

These can be added as comments/decisions at the top of the repo:

1. **Confirm Llama-3.2-1B access request goes out on day 1?** (Recommended.)
2. **Confirm the target venue is ARR/ACL/EACL, not EMNLP?** (Recommended given timeline.)
3. **OK to delete the stability-loss training scripts from the main repo path?** (They can stay in `archive/` for reproducibility of the original draft; just out of the main training flow.)
4. **Three seeds per window in Phase 1, or five?** Three is the plan above; five buys tighter variance estimates at the cost of ~50% more compute. With ~81 runs at 3 seeds vs ~135 at 5, three is probably the right tradeoff for a month.
5. **Phase 4 final-slot choice (A/B/C above)?** Defer until end of Week 2 — depends on whether Llama access cleared.

---

## Risk register

- **The placement ranking fails on too many models in Phase 1.** Mitigation: the paper still has a measurement contribution (LRD, three-signal dissociation, cascade disruption, regime taxonomy, Qwen scale variation) without the placement-prediction claim. Worst case: drop contribution #3 too, paper becomes pure measurement + mechanism.
- **HF access stays blocked.** Mitigation: Qwen scale evidence already mitigates the confound; note honestly in limitations.
- **A new harness bug surfaces during Phase 1 reruns.** Mitigation: the regression test from Phase 0 catches drift; the day-1 baseline audit catches static bugs. If something does surface, address it as Phase 0 work — do not paper over it.
- **Compute budget runs out before Phase 4.** Mitigation: Phase 4 is intentionally optional polish; the paper is submittable with Phase 1-3 alone. Prioritize Phase 1 reruns; everything else is upside.
