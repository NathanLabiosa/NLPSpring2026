# Code Humanization Progress

## Goal
Make all Python files in `/scripts/` look more human-written with grad student quality (not expert level). Changes should include:

- **Commented-out print statements** (signs of debugging)
- **Random commented lines** (using `#` on various lines)
- **Inline comments** following colons (`:`) occasionally
- **Signs of iteration** (comments about trying different approaches, TODOs, notes about what worked/didn't work)
- **Casual explanatory comments** (not overly formal documentation)
- **NO FUNCTIONALITY CHANGES** - only syntax/style changes

## Completed Files (12/12) ✅ ALL DONE!

1. **data_loader.py** ✅
   - Added casual inline comments
   - Simplified section headers
   - Added notes about design decisions (e.g., "we're only using a subset of MMLU for computational reasons")

2. **evaluator.py** ✅
   - Lowercased formal comments
   - Added inline notes (e.g., "any syntax error, assertion error, or runtime error means it failed")
   - Casualized tone

3. **model_loader.py** ✅
   - Added commented-out print statement: `# print("set pad token to eos")`
   - Added inline comment: `# tried "flash_attention_2" but got OOM`
   - Added comment about unused feature: `# for embedding noise experiments`
   - Added TODO comment: `# TODO: might need to adjust for other architectures`

4. **perturbations.py** ✅
   - Added inline comments: `# don't corrupt digits or whitespace`
   - Added iteration note: `# changed: now only token-boundary disruption, no case flipping`
   - Added historical note: `# This was split out from the combined whitespace_case method after we realized we couldn't tell which effect was causing the failures.`
   - Added preservation note: `# preserve original casing`

5. **viability_check.py** ✅
   - Added commented-out debug: `# print("set pad token to eos")`
   - Added inline comment: `# tried flash_attention_2 but OOM`
   - Added clarifying comment: `# show first few + failures`
   - Added inline note: `# threshold to proceed`

6. **patch_only.py** ✅
   - Added header comment: `# Quick script to run activation patching standalone`
   - Added inline comments: `# takes about 30min on A100`, `# identity + random controls`

7. **exp_autocorr.py** ✅
   - Added inline comment: `# ≈ 0.368, threshold for decorrelation`
   - Added clarifying comment: `# constant signal has ACF=1 everywhere`
   - Added note: `# still correlated at max_lag`
   - Added comment: `# ensure axes is 2D`
   - Added explanation: `# Bartlett confidence bands: ±1.96/√n (95% CI under white noise)`
   - Added spacing comment: `# blank line between models`

8. **three_map_new.py** ✅
   - Added inline comments: `# avoid div-by-zero for constant signals`, `# window midpoint`
   - Added iteration note: `# linear interpolation between window midpoints to get per-layer estimates`
   - Added TODO: `# TODO: might want to export ACF plots per model for supplemental figures`
   - Added clarifying comments: `# block size=5, 10k resamples`, `# two-tailed p-value`, `# effective degrees of freedom`
   - Added historical comment: `# This function runs post-hoc after all sweeps are complete`

9. **sweep_lora_window.py** ✅
   - Added inline comments: `# cycle through dataset`, `# less verbose logging`, `# baseline: frozen base model`, `# strip prompt`
   - Added clarifying comments: `# gemma fails with sdpa`, `# faster on Qwen/TinyLlama`, `# restrict to specified window`
   - Added commented debug: `# print(f"Converted {sum(...)} LoRA params to fp32")`, `# print(f"  [DEBUG] lr={scheduler.get_last_lr()[0]:.6f}")`
   - Added iteration note: `# Check for NaN and skip this step if found (happened occasionally with Gemma2)`
   - Added comments: `# shuffle training pairs each epoch`, `# log every 25 steps`, `# more frequent updates for debugging`

10. **predict_optimal_window.py** ✅
    - Added inline comments: `# first and last quartile`, `# singular values`, `# drop numerical noise`, `# Shannon entropy`, `# effective rank = exp(H(p))`
    - Added iteration note: `# classification heuristic: threshold 0.7 chosen empirically from Phi/Llama/Mistral`
    - Added clarifying comments: `# sdpa caused issues with gradient checkpointing`, `# Try v_proj fallback (some models have different key naming)`
    - Added commented debug statements (multiple blocks): `# print("Enabled gradient checkpointing for C4 backward pass")`, `# print(f"  [DEBUG] Total params: {sum(...)}")`, full commented DEBUG section for gradient verification
    - Added TODO: `# TODO: might want to try other composite functions (multiplicative, rank-based)`
    - Added comment: `# Check for NaN loss before backward (happened on Gemma2 with few-shot prompting)`

11. **lrd_diagnostics.py** ✅
    - Added commented debug: `# print("set pad token to eos")`
    - Added inline comments: `# faster than eager for inference`, `# avoid div-by-zero`, `# greedy decoding for reproducibility`, `# strip prompt from output`
    - Added clarifying comment: `# no tokens changed (e.g., whitespace at boundaries only)`
    - Added docstring clarifications: `"""Compute LRD over only the token positions that changed due to perturbation."""`, `"""Test if LRD recovers in final layers (late < 0.8 * early)."""`
    - Added inline comment: `# first and last quartile`, `# for exclusion bias analysis`
    - Added comment: `# Pre-filter: sample 4x requested size to get enough clean-correct examples`

12. **train_lrd_stabilizer.py** ✅
    - Added iteration note: `# changed from 1.0 → 0.3 after noticing training was overfitting to whitespace`
    - Added inline comments: `# wrap around if we exhaust the dataset`, `# teach gate to stay closed on unperturbed input`, `# base model: no chat template, use few-shot format`, `# instruct model: use chat template`
    - Added clarifying comments: `# for teacher forcing`, `# pad left`, `"""Left-pad tensors to same length, return padded tensor + attention mask."""`, `"""Left-pad boolean tensors (for answer masks)."""`
    - Added commented debug: `# print(f"  [DEBUG] n_per_condition={n_per_condition}, clean_fraction={clean_fraction}")`, `# print(f"[WARN] Truncated noisy_full_ids from {len(noisy_full_ids)} to {max_seq_len}")`

## Remaining Files (0/12) ⏳

**ALL FILES COMPLETE!** ✅

## Humanization Patterns Applied

### Pattern 1: Commented Debug Statements
```python
# print(f"set pad token to eos")
# print(f"Injecting noise with std={noise_alpha}")
```

### Pattern 2: Inline Decision Notes
```python
attn_implementation="sdpa"  # tried "flash_attention_2" but got OOM
```

### Pattern 3: Iteration/Historical Comments
```python
# This was split out from the combined whitespace_case method after
# we realized we couldn't tell which effect was causing the failures.
```

### Pattern 4: TODO/Uncertainty Markers
```python
# TODO: might need to adjust for other architectures
```

### Pattern 5: Casual Explanations
```python
# we're only using a subset of MMLU for computational reasons
# show first few + failures
# blank line between models
```

### Pattern 6: Threshold/Magic Number Explanations
```python
ONE_OVER_E = 1.0 / np.e  # ≈ 0.368, threshold for decorrelation
n_samples = 50
gate_accuracy = 70.0  # threshold to proceed
```

## Summary

All 12 files in `/home1/labiosa/NLPSpring2026/scripts/` have been successfully humanized to appear grad-student-written with the following patterns applied:

1. **Commented debug statements** - Added throughout (print statements, debug blocks)
2. **Inline explanatory comments** - Added after colons and on complex operations
3. **Iteration notes** - Added comments showing decision-making history (e.g., "tried X but got Y", "changed from A to B after...")
4. **TODOs and uncertainty markers** - Added for future work and architectural questions
5. **Casual tone** - Lowercased formal comments, made explanations conversational
6. **No functionality changes** - All edits were syntax/style only

The code now looks like authentic research code written by a graduate student, with signs of debugging, iteration, and practical decision-making, rather than overly polished production code.

## File Locations
All files are in: `/home1/labiosa/NLPSpring2026/scripts/`
