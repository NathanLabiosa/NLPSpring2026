# Analysis Experiments: The Three Maps of Perturbation in Transformers

## Framing

We have three independent per-layer signals that should be related but dissociate:
1. **LRD magnitude** — how much divergence exists at layer L
2. **Patching recovery** — how much causal influence layer L has on failure
3. **LoRA effectiveness** — how much intervention at layer L helps

These experiments investigate why they dissociate and whether we can predict the optimal intervention layer without running a LoRA sweep.

All experiments below are forward-pass or lightweight analysis only (no LoRA training) unless marked otherwise.

---

## Experiment A: Per-Layer Cohen's d Profiles

**Goal:** Find the layer where succeed/fail trajectories diverge — not just final-layer effect size, but the full layer-by-layer profile.

**Setup:**
- Model: Phi-3.5 (primary), Llama-3, Mistral
- Dataset: GSM8K, n=500 (use the larger n from Priority 3 if already run)
- Perturbation: all 6 types at 10% rate

**Procedure:**
1. Run clean and perturbed forward passes, recording hidden states at every layer
2. For each example, record whether it succeeds or fails under perturbation
3. At each layer L, compute Cohen's d between the LRD values of succeed vs. fail examples
4. Plot d(L) across all layers for each perturbation type

**What to look for:**
- Is there a specific layer where d spikes — i.e., where the model "decides" to fail?
- Does this decision layer correspond to the optimal LoRA window, the patching window, or neither?
- Does the decision layer differ across perturbation types?

**Output:**
- `expA_perlayer_cohens_d.json` — raw d values per layer per perturbation type per model
- One figure per model: x-axis = layer, y-axis = Cohen's d, one line per perturbation type
- Overlay vertical bands for the patching window and optimal LoRA window on each figure

---

## Experiment B: Three-Map Overlay

**Goal:** Directly visualize and quantify the dissociation between LRD, patching recovery, and LoRA effectiveness.

**Setup:**
- Model: Phi-3.5 (all three signals available), Llama-3 (all three available)
- Mistral only has exploratory patching, so include it with a caveat

**Procedure:**
1. Normalize each signal to [0, 1] range across layers for comparability:
   - LRD(L): already have this from diagnostics (use mean across perturbation types, or plot per-type)
   - Patching recovery(L): fraction of failed examples recovered by single-layer clean patch at L
   - LoRA Δ(L): average perturbed accuracy gain from the 5-layer window centered on L (interpolate from existing sweep data — e.g., layer 2 maps to the 0–4 window result, layer 7 to 5–9, etc.)
2. Plot all three curves on the same axes for each model
3. Compute pairwise Spearman correlations: LRD×Patching, LRD×LoRA, Patching×LoRA
4. Compute the composite score: for each layer window, record (1 - normalized_LRD) × (1 - normalized_patching) as a "slack" proxy. Correlate this with LoRA effectiveness.

**What to look for:**
- Negative correlation between LRD and LoRA effectiveness (best intervention where divergence is low)
- Negative correlation between patching and LoRA effectiveness (best intervention ≠ causal window)
- Whether the composite slack score predicts LoRA rank ordering of windows

**Output:**
- `expB_three_map_overlay.json` — per-layer values for all three signals per model
- One overlay figure per model (three curves, shared x-axis = layer)
- Correlation matrix table

---

## Experiment C: Layer-Wise Capacity / Slack Metrics

**Goal:** Find an intrinsic property of the network (no perturbation needed) that predicts where LoRA is effective. If this works, you can predict optimal placement without running a sweep.

**Setup:**
- Models: Phi-3.5, Llama-3, Mistral
- Dataset: 200 clean GSM8K examples (no perturbation — this is about the model's architecture, not the perturbation)

**Metrics to compute at each layer L:**

### C1: Residual stream update ratio
- Record the residual stream h_L before and after the layer's transformation
- Compute: update_ratio(L) = ||h_L - h_{L-1}|| / ||h_{L-1}||
- Low ratio = layer is making small changes = more "slack" for LoRA to exploit

### C2: Attention entropy
- For each attention head at layer L, compute entropy of the attention distribution (averaged over positions and examples)
- High entropy = diffuse attention = less committed computation = potentially more slack

### C3: Weight matrix effective rank
- For q_proj and v_proj at each layer (the LoRA targets), compute the effective rank: exp(entropy of normalized singular values)
- Compare effective rank to full rank
- Layers with lower effective rank may have more "room" for low-rank adaptation

### C4: Gradient norm under clean loss (if feasible)
- Run a clean forward + backward pass
- Record ||∂L/∂W||_F for q_proj and v_proj at each layer
- High gradient norm = the layer is actively being used for the task = less slack

**Procedure:**
1. Forward pass 200 clean examples through each model
2. Hook every layer to record the metrics above
3. Plot each metric across layers
4. Correlate each metric with LoRA effectiveness from the sweep

**What to look for:**
- Any metric that peaks at layers 15–19 for Phi-3.5 and layers 20–24 for Llama-3/Mistral
- Specifically: does the update ratio dip, or attention entropy rise, at the layers where LoRA works best?

**Output:**
- `expC_capacity_metrics.json` — per-layer values for each metric per model
- One multi-panel figure per model (one panel per metric, all with layer on x-axis)
- Correlation table: each metric × LoRA effectiveness

---

## Experiment D: Cross-Task LRD Stability

**Goal:** Determine if propagation regimes are purely architectural or task-modulated.

**Setup:**
- Models: Phi-3.5, Llama-3, Mistral
- Datasets: GSM8K, MMLU, BBH (n=200 each)
- Perturbation: typos 5% and whitespace 10% (one from each old taxonomy group — keep it lean)

**Procedure:**
1. Compute full LRD profiles (all layers) for each model × task × perturbation combination
2. For each model, compute CKA or Spearman correlation between the LRD profile on GSM8K vs. MMLU vs. BBH
3. Record: does the regime shape (spike-and-suppress vs. late-accumulation) change across tasks? Does the magnitude scale?

**What to look for:**
- If regime shape is identical across tasks: the regime is architectural, task modulates severity. Strong claim.
- If regime shape changes: regime is task-dependent. Still interesting but weaker.

**Output:**
- `expD_cross_task_lrd.json` — LRD profiles per model × task × perturbation
- One figure per model: overlay LRD curves for all three tasks (same perturbation type), see if shapes match
- Cross-task correlation values

---

## Experiment E: Per-Example LRD Trajectory Clustering

**Goal:** Move beyond means. Do individual examples follow the same LRD trajectory, or are there subpopulations?

**Setup:**
- Model: Phi-3.5 (richest signal)
- Dataset: GSM8K, n=500
- Perturbation: typos 10%, speech 10% (one "typical" and the speech outlier)

**Procedure:**
1. For each example, compute the full 32-layer LRD trajectory: a 32-dimensional vector
2. Split examples into succeed/fail under perturbation
3. Run k-means or spectral clustering (k=3–5) on the trajectories
4. For each cluster: report mean trajectory shape, succeed/fail ratio, and example characteristics (e.g., question length, number of reasoning steps, number of perturbed tokens)

**What to look for:**
- Are there distinct trajectory subtypes? E.g., some examples where divergence is absorbed early vs. others where it snowballs?
- Do the clusters predict failure better than final-layer LRD alone?
- Is there a cluster of "vulnerable" examples with a characteristic trajectory shape?

**Output:**
- `expE_trajectory_clusters.json` — cluster assignments, mean trajectories, succeed/fail ratios
- Figure: mean LRD trajectory per cluster, colored by succeed/fail ratio
- Table: cluster sizes, failure rates, mean characteristics

---

## Experiment F: LoRA-Induced Clean Disruption Measurement

**Goal:** You already have one data point (clean-input LRD of 0.018 at layers 0–4 vs. 0.003 later). Systematize this across all windows and models.

**Setup:**
- Models: Phi-3.5, Llama-3, Mistral
- Use already-trained LoRA adapters from the layer sweep (no new training)

**Procedure:**
1. For each trained LoRA window, run clean inputs through:
   - (a) the frozen base model
   - (b) the model with LoRA adapters active
2. Compute LRD between (a) and (b) at every layer downstream of the LoRA window
3. This measures how much the LoRA adapter disrupts clean-input processing

**What to look for:**
- Early-window LoRA should cause high downstream disruption (confirming your cascade hypothesis)
- Optimal-window LoRA should cause low downstream disruption
- Is there a threshold: LoRA windows that cause > X disruption always produce negative gains?

**Output:**
- `expF_clean_disruption.json` — per-window, per-downstream-layer disruption values
- Figure: x-axis = downstream layer, y-axis = clean disruption LRD, one line per LoRA window
- Scatter plot: LoRA window's total clean disruption vs. perturbed accuracy gain (should be negative correlation)

---

## Suggested Priority Order

1. **B (Three-Map Overlay)** — cheapest, uses existing data, produces the paper's key figure
2. **A (Per-Layer Cohen's d)** — forward passes only, directly extends existing analysis
3. **F (Clean Disruption)** — uses existing LoRA checkpoints, no training, explains the mechanism
4. **C (Capacity Metrics)** — forward passes on clean data, potentially the most novel finding
5. **D (Cross-Task Stability)** — forward passes, strengthens the regime claim
6. **E (Trajectory Clustering)** — most exploratory, do last

Experiments A–D can likely run in parallel. Total compute is dominated by forward passes (no training), so this whole set is much cheaper than Priority 1 from the previous plan.
