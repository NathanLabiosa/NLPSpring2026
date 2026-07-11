# Layer-Wise Representation Divergence

This repository contains the experimental code and results behind an ARR submission on layer-wise representation divergence (LRD): evaluating language model robustness under input perturbations, training lightweight adaptive stabilizer layers, and running LRD diagnostics and LoRA window experiments across six model families (Llama, Mistral, Phi-3.5, Qwen2.5, Gemma2, TinyLlama).

These experiments are computationally expensive. Most workflows load 1B-9B parameter models, generate over full benchmark splits or hundreds of examples, and may run multiple perturbation conditions per model. Use a CUDA GPU whenever possible.

## Repository Map

```
evaluation/                  Baseline perturbation-robustness eval harness (see §1)
adaptive_layer_experiments/  Notebooks for training/evaluating adaptive stabilizer layers (see §2)
lrd_experiments/             Model-agnostic LRD diagnostics, patching, LoRA window tooling (see §3)

models/<name>/                Per-model driver scripts, raw LRD/patching results, and figures
  llama/ mistral/ phi3.5/ qwen2.5/ gemma2/ tinyllama/

src/                          Shared analysis, plotting, and eval code (post-hoc, cross-model)
  analysis/                   Bootstrap resampling, power/interaction analysis, window prediction
  diagnostics/                Batched/LoRA diagnostic runners
  eval/                       Bootstrap, transfer, held-out, and regression eval harnesses
  experiments/                Named paper experiments (expA-expF, three-map, CKA, ablations)
  lib/                        Shared library code (stabilizer module, figure style, LoRA training)
  plotting/                   Figure-generation scripts for the paper
  training/                   LoRA/augmentation-baseline training scripts

slurm/<family>/               SLURM submission scripts, grouped by model family (+ shared/)

results/                      JSON/TeX result artifacts from analysis and eval scripts
  bootstrap/                  Paired bootstrap significance results
  predictions/                Predicted optimal LoRA windows
  workstream_a/                Interaction/power/whitespace statistical analyses

figures/                      Final PDF figures referenced by the paper
```

## Paper

This code backs an ACL Rolling Review (ARR) submission, *"Sensitivity, Causality, and Repair Dissociate: A Layer-Wise Analysis of Perturbation Robustness and Its Scaling."* `figures/` contains every figure cited in the paper; `results/` contains the underlying numeric results; `models/<name>/` contains the per-model scripts and raw data used to produce them; `src/` holds the shared analysis/plotting/eval code, including the code behind the revision's new experiments (window-width ablation, all-layer LoRA baseline, MMLU layer sweep).

### Reproducing every table and figure

Every command below was verified against the actual script source (argparse flags, hardcoded paths, on-disk JSON schemas) rather than inferred from filenames. Run from the repo root with the pinned environment below unless noted.

Two release-scope notes up front:

- **"Per-cell config"** = the run's own result JSON, which is self-describing (model, checkpoint, seed, harness settings — see [Result File Schema](#result-file-schema)). There is no separate `configs/` directory duplicating that information; the JSON *is* the config record.
- **"Evaluation logs"** = the per-perturbation-condition breakdown embedded in every fixed-harness result JSON (`acc_no_adapter`, `acc_with_adapter`, `delta` per condition — see schema below), not raw SLURM/tmux stdout, which is reproducible from scratch and not committed (`.gitignore`'d as noise).

#### Main paper

| Ref | What it shows | Command | Reads |
|---|---|---|---|
| **Table 1** (regime classification) | 5-model recovery %/slope | Per model: `cd models/<name> && python lrd_diagnostics.py --dataset gsm8k --model <hf_id> --n_samples 500 --output_dir lrd_results/<run>` (see the [per-model grid](#lrd-diagnostic-runs) below); Gemma-2 via `python lrd_run.py` (no flags). Table itself is hand-tabulated from each `stats_gsm8k.json`'s `"Typos_5%"` → `pct_recovered`/`mean_cascade_slope`; no aggregator script. | `stats_gsm8k.json` per model |
| **Table 2** (scaling ratio) | Late/early LRD ratio, Qwen 1.5B/7B/14B + Llama 1B/8B | `python src/plotting/generate_scale_table.py` (no args) | `models/qwen2.5/{lrd_results/qwen_gsm8k,scale_experiments/lrd_results_{1.5B,14B}}/raw_gsm8k.json`, `models/llama/{lrd_results,lrd_results_3.2_1B}/raw_gsm8k.json` → writes `results/scale_table.json` + `results/scale_table.tex` |
| **Table 3** (layer-sweep panel) | Mean perturbed Δ, 5 models × windows + all-layer | v1 windows (per model/window/seed): `python src/training/train_lrd_lora_v14.py ...` then `python src/eval/eval_fixed_harness.py --base_model <hf_id> --checkpoint <ckpt>/lora_final --n_samples 500 --seed {42,43,44} --output_file results/fixed_harness/v1/layer_sweep/<model>_<WIN>_seed<S>.json` (window/target-module/attn-impl grid [below](#lrd-diagnostic-runs)). All-layer column: `bash scripts/run_task2.sh {phi35\|qwen\|llama}` → `python scripts/aggregate_results.py task2`. **No single script builds the full panel** — mean±std per (model, window) is computed by hand from the 78 result JSONs in `results/fixed_harness/v1/layer_sweep/`. | `results/fixed_harness/v1/layer_sweep/*.json`, `results/fixed_harness/v2/all_layer/*.json` |
| **Fig. 1** (`qwen_scaling_lrd`) | Scaling plot | `python src/plotting/generate_qwen_scaling.py` (run Table 2's command first — reads `results/scale_table.json`) | `results/scale_table.json` |
| **Fig. 2** (`cohens_d_phi35`) | Per-layer Cohen's d | `python src/experiments/expA_cohens_d.py` (no args; one run also regenerates the appendix's Llama-3/Mistral/Qwen/Gemma panels) | `models/phi3.5/lrd_results/phi_hardening2/raw_gsm8k.json` |
| **Fig. 3** (`patching_recovery`) | Patching recovery, Phi-3.5 + Llama-3 | `python src/plotting/generate_workstream_c_figures.py` (runs `figure_c1()`–`figure_c8()`; `figure_c2()` produces this one) | `models/{phi3.5/lrd_results/phi_patching_v3,llama/lrd_results}/patching_gsm8k_typos.json` |
| **Fig. 4** (`three_map_{phi35,llama3}`) | Three-map overlay | `python src/experiments/expB_three_map.py` (no args; also writes `three_map_mistral.pdf`) | per-model LRD/patching/LoRA-sweep JSONs |
| **Fig. 5** (`disruption_phi35`) | Cascade disruption curves | `python src/experiments/expF_clean_disruption.py --model phi35` (add `--reuse_cache` to replot from the cached `results/expF_clean_disruption_phi35.json` without reloading models) | GSM8K + per-window LoRA checkpoints |
| **Fig. 6** (`layer_sweep`) | Sweep summary plot | `python src/plotting/generate_layer_sweep.py` (no args) | `results/fixed_harness/v1/layer_sweep/*.json` |
| **Fig. 7** (`harness_artifact`) | Old vs. fixed harness | `python src/plotting/generate_supplement_figures.py` (`figure_harness_artifact()`; same run also produces the appendix's `layer_sweep_per_model_*.pdf` and `qwen14b_patching_per_layer.pdf`) | See [Known Gaps](#known-gaps) #3 — two of the four plotted deltas are cross-checked hardcoded literals, not live-computed |

#### Appendix

| Ref | Command | Notes |
|---|---|---|
| Fig. `heatmap` (Phi-3.5 + Llama-3) | `python src/plotting/generate_workstream_c_figures.py` (`figure_c1()`) | |
| Fig. `supp_lrd` (Gemma/Qwen/Mistral heatmaps) | `python src/plotting/generate_lrd_heatmaps.py` | script hardcodes an old absolute `fig_dir` — edit the path before running |
| Fig. `supp_scaling_heatmaps` (Qwen 1.5B/14B) | Byte-identical copies of `models/qwen2.5/scale_experiments/lrd_results_{1.5B,14B}/fig3_heatmap_gsm8k.pdf`, a side effect of the Table 2 LRD runs | copied into `figures/` by hand — no script performs the copy |
| Fig. `supp_qwen14b_perlayer` | `cd models/qwen2.5/scale_experiments && python ../lrd_diagnostics.py --dataset gsm8k --model Qwen/Qwen2.5-14B-Instruct --n_samples 200 --max_new_tokens 512 --patching --n_pairs 100 --skip_sanity --output_dir patch_results_14B`, then `python src/plotting/generate_supplement_figures.py` | |
| Table `supp_recovery` | hand-tabulated from each model's `stats_gsm8k.json` (same source as Table 1, all six perturbation types) | |
| Table `correlations` | `python src/analysis/bootstrap_block.py` (block=5, 10k resamples, matches paper) | reads `results/expB_three_map_overlay.json` — only Phi-3.5/Llama-3/Mistral live there; see [Known Gaps](#known-gaps) #4 |
| Table `intrinsic` (C1–C4) | `python src/experiments/expC_capacity_metrics.py --model {phi35\|llama3\|mistral} --n_samples 200` | writes its JSON to the repo root; move into `results/` by hand |
| Table `supp_c3c4` | `python src/analysis/predict_optimal_window.py --model {TinyLlama\|Gemma2\|Qwen25} --compute_c3c4` then `--check_prediction` | see [Known Gaps](#known-gaps) #5 before trusting a fresh `--check_prediction` run |
| Fig. `supp_three_map` (Mistral/Qwen/Gemma) | Mistral: same command as Fig. 4. Qwen/Gemma: see [Known Gaps](#known-gaps) #2 | |
| Fig. `supp_cohens_d` (Llama/Mistral/Qwen/Gemma) | same command as Fig. 2 (one run produces all five) | |
| Fig. `supp_capacity` (Llama-3/Mistral) | `python src/experiments/expC_capacity_metrics.py --model {llama3\|mistral}` | |
| Fig. `supp_disruption` (Llama/Mistral/Qwen/Gemma) | `python src/experiments/expF_clean_disruption.py --model {llama3\|mistral\|qwen\|gemma}` | same script as Fig. 5 |
| Fig. `supp_disruption_scatter` (Phi/Llama/Mistral) | see [Known Gaps](#known-gaps) #1 — **do not use the currently-committed PDFs** | |
| Fig. `supp_crosstask` (Phi/Mistral) | `python src/experiments/expD_cross_task.py` (no args) | |
| Fig. `supp_layer_sweep_per_model` (5 models) | `python src/plotting/generate_supplement_figures.py` (same run as Fig. 7) | reads `results/fixed_harness/v1/layer_sweep/*.json` |
| Table `width_ablation` (Task 1) | `CUDA_VISIBLE_DEVICES=<gpu> bash scripts/run_task1.sh {w3_mid\|w3_late\|w7_mid\|w7_late}` | 3 seeds looped internally → `results/fixed_harness/v2/layer_sweep/` |
| Table `all_layer` (Task 2) | `CUDA_VISIBLE_DEVICES=<gpu> bash scripts/run_task2.sh {phi35\|qwen\|llama}` | → `results/fixed_harness/v2/all_layer/` |
| Table `mmlu_sweep` (Task 3) | `CUDA_VISIBLE_DEVICES=<gpu> bash scripts/run_task3.sh {phi35\|qwen} {0\|1}` | eval via `src/eval/eval_fixed_harness_mmlu.py --item_ids_file results/mmlu_eval_item_ids.json` → `results/fixed_harness/v2/mmlu/` |
| Table `supp_repro` | this README + `requirements.txt` | |

For any of the three Task tables: `python scripts/aggregate_results.py {task1\|task2\|task3}` prints the summary and a booktabs-formatted LaTeX snippet.

#### LRD diagnostic runs

Per-model invocations underlying Table 1 and the LRD-derived panels (all `--dataset gsm8k`, `--n_samples 500` unless noted):

| Model | Command (run from `models/<name>/`) |
|---|---|
| Phi-3.5 | `python lrd_diagnostics.py --model microsoft/Phi-3.5-mini-instruct --max_new_tokens 256 --run_exclusion_analysis --run_taxonomy_variance --output_dir lrd_results/exp3_phi_n500` |
| Llama-3-8B | `python lrd_diagnostics.py --model meta-llama/Meta-Llama-3-8B-Instruct --max_new_tokens 256 --run_exclusion_analysis --run_taxonomy_variance --output_dir lrd_results/exp3_llama_n500` |
| Mistral-7B-v0.3 | `python lrd_diagnostics.py --model mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 256 --run_exclusion_analysis --run_taxonomy_variance --output_dir lrd_results/exp3_mistral_n500` |
| Qwen2.5-7B | `python lrd_diagnostics.py --model Qwen/Qwen2.5-7B-Instruct --max_new_tokens 768 --run_taxonomy_variance --output_dir lrd_results/qwen_gsm8k` |
| Gemma-2-9B | `python lrd_run.py` (no flags; hardcodes `n_samples=500`, `google/gemma-2-9b`, `output_dir=lrd_results/gemma2_9b_gsm8k`) |

Table 3's v1 layer-sweep window grid (per model, all width-5 nominal windows, LoRA `r=4`/`α=8`, CE-only):

| Model | Windows | Target modules | Attn impl |
|---|---|---|---|
| Phi-3.5 (32 layers) | L00-04, L05-09, L10-14, L15-19, L20-24, L27-31 | `qkv_proj o_proj` | eager |
| Llama-3-8B / Mistral-7B-v0.3 | L00-04, L05-09, L15-19, L20-24, L27-31 | `q_proj v_proj` | sdpa |
| Qwen2.5-7B | L00-04, L05-09, L08-11, L15-19, L20-23, L24-27 | `q_proj v_proj` | sdpa |
| Gemma-2-9B (42 layers) | L00-04, L06-11, L15-20, L25-30, L35-40 | `q_proj v_proj` | eager |

Each cell above is trained + evaluated at seeds 42/43/44 by re-running with `--seed`.

### Known gaps

Stated plainly rather than papered over, since that's the point of a reproducibility statement:

1. **`figures/disruption_{phi35,llama3,mistral}_scatter.pdf` are mid-regeneration.** The committed PDFs are known-wrong (hardcoded old-harness `acc_delta`); the fix pipeline (`scripts/run_scatter_{eval,train,disruption}.sh` → `scripts/update_scatter_deltas.py`) is running now. Do not treat the current PDFs as reproducing the appendix figure until that finishes.
2. **`figures/three_map_{qwen,gemma}.pdf`** have no confirmed reproduction command in the current script layout — `expB_three_map.py` covers Phi-3.5/Llama-3/Mistral only; the older `src/experiments/three_map_new.py` targets a different model set and different output paths/filenames. Needs a short follow-up to wire these two models into `expB_three_map.py`.
3. **`figures/harness_artifact.pdf`**: of its four plotted deltas, the fixed-harness Qwen value (-7.5pp) is an exact match to `models/qwen2.5/experiments/results/fixed_harness_qwen_vanilla.json`; the fixed-harness Phi-3.5 value (-3.5pp) is close but not exact against the nearest on-disk checkpoint eval (-2.87pp); the two pre-fix-harness values (+7.3pp/+11.6pp) have no surviving raw output in this repo. All four are archival/hardcoded literals in `generate_supplement_figures.py`, not live-computed from checkpoints on each run.
4. **Appendix `tab:correlations`**: `results/expB_three_map_overlay.json` (read by `bootstrap_block.py`) covers Phi-3.5/Llama-3/Mistral only; the paper's Gemma-2/Qwen-2.5 rows were computed on a path not currently wired into that script.
5. **Appendix `tab:supp_c3c4`**: `predict_optimal_window.py --check_prediction`'s "actual" column reads pre-fix-harness sweep deltas by default. The numbers in the paper are the corrected fixed-harness values — re-point the script (or read `results/fixed_harness/` by hand) rather than trusting a fresh `--check_prediction` run as-is.

Items 2, 4, and 5 are wiring/documentation gaps in otherwise-correct analysis code, not open experimental questions — the underlying data each needs already exists on disk. Item 1 is an active, running fix. Item 3's two archival values predate the fixed harness and cannot be regenerated from this codebase; they are reported as-is in the paper.

### Result File Schema

Every fixed-harness result JSON (`results/fixed_harness/v1/`, `results/fixed_harness/v2/`) is self-describing and doubles as that cell's config record and evaluation log:

```json
{
  "harness_version": "v2",
  "base_model": "microsoft/Phi-3.5-mini-instruct",
  "checkpoint": "<path>/phase2_phi35_L11-13_seed42/lora_final",
  "seed": 42,
  "n_samples": 500,
  "max_new_tokens": 768,
  "clean_baseline": 86.4,
  "mean_perturbed_delta": -2.43,
  "results": {
    "typos_5pct": {
      "method": "typos", "rate": 0.05, "n_samples": 500,
      "acc_no_adapter": 60.0, "acc_with_adapter": 58.2,
      "acc_clean_baseline": 69.2, "delta": -1.8
    }
    /* ... one entry per perturbation condition plus clean_baseline ... */
  }
}
```

The MMLU fixed evaluation subset (500 stratified items, seed 42) used by Task 3 is `results/mmlu_eval_item_ids.json`.

### Hardware and compute

The original diagnostic/patching/layer-sweep panel (Tables 1–3, Figs. 1–7 main results) ran on single-GPU SLURM jobs (NVIDIA A40 46GB / A100 80GB / V100 32GB), ≈900 GPU-hours total (reported experiments only, excludes exploratory/failed runs). The revision's Task 1–3 addenda (window-width ablation, all-layer LoRA baseline, MMLU layer sweep) ran on a second machine (8×RTX 6000 Ada 47GB, no SLURM — plain `CUDA_VISIBLE_DEVICES` GPU pinning) with the `requirements.txt` versions pinned below; per-window-sweep wall clock is a few hours per seed on a single GPU for the 3.8B–8B models.

## Environment Setup

Clone the repository and create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The base `requirements.txt` installs the core packages:

- `torch`
- `transformers`
- `datasets`
- `tqdm`
- `accelerate`
- `sentencepiece`

Some LRD and LoRA scripts also use additional packages that may need to be installed manually:

```bash
pip install peft matplotlib scipy numpy
```

If you use gated Hugging Face models such as Llama or Gemma, authenticate first:

```bash
huggingface-cli login
```

### Colab Recommendations

Google Colab is the easiest place to run the notebooks, but the free tier is usually not enough for the full experiments.

Recommended runtime:

- Use a GPU runtime.
- Prefer A100, L4, or V100 when available.
- Use high-RAM mode for Mistral-7B, Qwen2.5-7B, Gemma-2-9B, and full LRD diagnostics.
- Mount Google Drive if you want checkpoints and result JSON files to persist after the runtime disconnects.
- Keep `batch_size=1` or `batch_size=4` for training workflows unless you have enough GPU memory.
- Start with small `n_samples`, `n_eval`, or `max_steps` values to verify the pipeline before launching a full run.

In Colab, install the project dependencies at the top of a notebook:

```python
%pip install torch transformers datasets tqdm accelerate sentencepiece peft matplotlib scipy numpy
```

Large models and datasets are downloaded from Hugging Face. The first run can spend substantial time downloading model weights and benchmark data.

## 1. Baseline Evaluation Scripts

The baseline evaluation workflow is in `evaluation/`. It evaluates an unmodified model on clean prompts and several perturbation conditions.

### Files

- `evaluation/main.py`: command-line entry point.
- `evaluation/model_loader.py`: maps short model names to Hugging Face model IDs and creates a text-generation pipeline.
- `evaluation/data_loader.py`: loads and formats datasets with model chat templates.
- `evaluation/perturbations.py`: applies OCR, typo, whitespace, homophone, speech-filler, case, and internal-noise perturbations.
- `evaluation/evaluator.py`: task-specific scoring logic.

### Supported Models

Use the short names accepted by `evaluation/model_loader.py`:

| Name | Hugging Face model |
| --- | --- |
| `phi` | `microsoft/Phi-3.5-mini-instruct` |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `llama` | `meta-llama/Llama-2-7b-instruct-v0.1` |

### Supported Datasets

Use the dataset names accepted by `evaluation/data_loader.py`:

| Name | Dataset |
| --- | --- |
| `humaneval` | `openai_humaneval` test split |
| `gsm8k` | `openai/gsm8k`, `main`, test split |
| `mmlu` | select `cais/mmlu` test subsets |
| `bbh` | `lukaemon/bbh`, `logical_deduction_seven_objects` |
| `arc` | `allenai/ai2_arc`, `ARC-Challenge` |
| `squad` | `rajpurkar/squad_v2` validation split |

### What Each Run Measures

`evaluation/main.py` evaluates these conditions:

- `Baseline`: clean prompt, no perturbation.
- `OCR_5%`: OCR-like character substitutions.
- `Typos_5%`: insertions, deletions, substitutions, and transpositions.
- `Whitespace_10%`: whitespace changes.
- `Case_10%`: case changes.
- `Homophones_20%`: homophone substitutions.
- `Speech_10%`: speech filler insertions.
- `Gaussian_0.05`: Gaussian noise injected into token embeddings.

Each condition reloads and formats the dataset, runs greedy decoding, scores outputs with the appropriate evaluator, and appends results to a text file.

### Running One Baseline

Run commands from inside the `evaluation/` directory so local imports resolve cleanly:

```bash
cd evaluation
python main.py --model mistral --dataset gsm8k
```

The output file is written in the current directory:

```text
results_mistral_gsm8k.txt
```

Each line has this format:

```text
[gsm8k] Baseline: 123/1319 (9.33%)
[gsm8k] OCR_5%: 100/1319 (7.58%)
```

### Running Multiple Models and Datasets

Run one command per model and dataset:

```bash
cd evaluation

python main.py --model phi --dataset gsm8k
python main.py --model phi --dataset mmlu
python main.py --model mistral --dataset gsm8k
python main.py --model mistral --dataset arc
```

For a full baseline grid, repeat over the supported model and dataset names. Results append to `results_<model>_<dataset>.txt`, so delete or rename old files if you want a fresh report.

### Practical Notes

- The script uses greedy decoding with `do_sample=False` for reproducibility.
- `BATCH_SIZE` is hard-coded in `evaluation/main.py`. Lower it if you hit out-of-memory errors.
- `max_new_tokens=600` is generous and can make full benchmark runs slow.
- MMLU combines ten selected subsets, so it is much smaller than all of MMLU but still expensive.
- HumanEval executes generated Python in a separate process with a timeout.

## 2. Adaptive Layer Experiment Notebooks

The adaptive layer experiments are in `adaptive_layer_experiments/`. These notebooks add trainable stabilizer modules to selected Mistral layers while freezing the base model.

### Files

- `adaptive_layer_experiments/mistral_expermiments.ipynb`: trains stabilizer layers.
- `adaptive_layer_experiments/evaluate.ipynb`: loads stabilizer weights and evaluates the wrapped model.

The training notebook currently targets:

```text
mistralai/Mistral-7B-Instruct-v0.3
```

### Training Workflow

Open `adaptive_layer_experiments/mistral_expermiments.ipynb` in Colab or Jupyter and run cells in order.

The notebook does the following:

1. Installs dependencies.
2. Loads Mistral-7B-Instruct with `torch.bfloat16` and `device_map="auto"`.
3. Defines a `Stabilizer` module: RMSNorm, bottleneck down-projection, SiLU, up-projection, dropout, and a learned gate.
4. Defines `WrappedMistralLayer`, which inserts the stabilizer after self-attention and optionally after the MLP.
5. Wraps the first 8 Mistral transformer layers.
6. Freezes the base model and leaves only parameters containing `"stabilizer"` trainable.
7. Builds supervised fine-tuning data from GSM8K with a clean/noisy mix.
8. Trains with masked labels so loss is computed on target tokens, not the prompt.
9. Saves only the stabilizer weights.

The default training setup in the notebook uses GSM8K with OCR perturbations:

```python
train_hf = builder.build_dataset(
    dataset_name="gsm8k",
    perturbation_name="ocr",
    rate=0.10,
    clean_mix_prob=0.5,
    split="train",
)
```

The default training call uses:

```python
model = train_stabilizers(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch_size=1,
    grad_accum_steps=16,
    num_epochs=1,
    lr=2e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    log_every=10,
    eval_every=50,
    num_workers=2,
)
```

The notebook saves the trained stabilizer-only state dict as:

```text
mistral_stabilizers_gsm8k_ocr.pt
```

Save this file somewhere persistent if you are using Colab.

### Changing the Training Experiment

Useful knobs in the training notebook:

- Change `perturbation_name` to train against a different corruption type, such as `typos`, `ocr`, `qwerty`, `whitespace`, `homophones`, or `speech`.
- Change `rate` to control perturbation severity.
- Change `clean_mix_prob` to alter the mix of clean and perturbed examples.
- Change `range(8)` in the layer-wrapping cell to train stabilizers on more or fewer layers.
- Change `bottleneck=256` to adjust stabilizer capacity.
- Increase `num_epochs` for longer training.
- Reduce `max_length`, `batch_size`, or the number of wrapped layers if you hit GPU memory limits.

### Evaluating a Trained Adaptive Layer Model

Open `adaptive_layer_experiments/evaluate.ipynb` and run cells in order.

The evaluation notebook:

1. Loads the same base Mistral model.
2. Recreates the same `Stabilizer` and `WrappedMistralLayer` definitions.
3. Loads `mistral_stabilizers_gsm8k_ocr.pt`.
4. Defines perturbation and evaluation helpers.
5. Builds a Hugging Face `text-generation` pipeline from the already-wrapped model.
6. Evaluates a selected dataset and perturbation condition.
7. Writes a result text file.

The notebook includes a sanity check:

```python
assert pipe.model is model
assert pipe.tokenizer is tokenizer
assert isinstance(pipe.model.model.layers[0], WrappedMistralLayer)
```

This confirms the pipeline is using the wrapped model with stabilizers rather than reloading a clean base model.

To evaluate a different condition, edit:

```python
exp = experiments[1]
p_func = lambda text: perturber.apply(text, exp["type"], exp["rate"])
dataset_name = "gsm8k"
```

Set `p_func = None` to evaluate the clean baseline. The notebook writes files such as:

```text
gsm8k_OCR_5%.txt
```

### Notebook Resource Notes

- Mistral-7B plus stabilizer training generally requires a high-memory GPU.
- `batch_size=1` with gradient accumulation is intentional.
- Keep the base model frozen. Training all Mistral parameters is not the intended experiment and will require far more memory.
- If the runtime disconnects, reload the notebook, rerun the model/stabilizer definition cells, and load the saved `mistral_stabilizers_gsm8k_ocr.pt` file before evaluation.

## 3. LRD Experiments

The LRD experiments are in `lrd_experiments/`. They measure how perturbations change hidden representations across layers, optionally run activation patching, predict useful LoRA intervention windows, and train/evaluate LoRA windows.

Run these commands from inside `lrd_experiments/`:

```bash
cd lrd_experiments
```

### Main LRD Diagnostics

Use `lrd_diagnostics.py` to compute layer-wise representation divergence profiles.

Example on GSM8K with Mistral:

```bash
python lrd_diagnostics.py \
  --dataset gsm8k \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --n_samples 200 \
  --output_dir lrd_results/mistral_gsm8k \
  --max_new_tokens 256
```

The diagnostics script:

1. Loads the model and tokenizer.
2. Pre-filters examples to examples the clean model answers correctly.
3. Applies perturbations such as typos, OCR, whitespace, case, homophones, and speech fillers.
4. Extracts hidden states for clean and perturbed prompts.
5. Computes cosine-distance LRD profiles across layers.
6. Generates perturbed outputs and scores whether the model remains correct.
7. Saves raw records, summary statistics, and PDF figures.

Expected outputs include:

```text
lrd_results/mistral_gsm8k/raw_gsm8k.json
lrd_results/mistral_gsm8k/stats_gsm8k.json
lrd_results/mistral_gsm8k/fig1_lrd_curves_gsm8k.pdf
lrd_results/mistral_gsm8k/fig2_lrd_vs_accuracy_gsm8k.pdf
lrd_results/mistral_gsm8k/fig3_heatmap_gsm8k.pdf
lrd_results/mistral_gsm8k/fig5_per_token_lrd_gsm8k.pdf
```

Optional analysis flags:

```bash
python lrd_diagnostics.py \
  --dataset gsm8k \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --n_samples 200 \
  --output_dir lrd_results/mistral_gsm8k \
  --run_exclusion_analysis \
  --run_taxonomy_variance \
  --homophones_rates 0.40 0.50
```

### Activation Patching

Activation patching can be run through `lrd_diagnostics.py`:

```bash
python lrd_diagnostics.py \
  --dataset gsm8k \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n_samples 200 \
  --output_dir lrd_results/qwen_patching \
  --patching \
  --patch_perturb typos \
  --n_pairs 100 \
  --max_new_tokens 512
```

This searches for clean-success/noisy-failure pairs, patches clean hidden states into noisy runs layer by layer, and saves:

```text
lrd_results/qwen_patching/patching_gsm8k_typos.json
lrd_results/qwen_patching/fig4_patching_gsm8k_typos.pdf
lrd_results/qwen_patching/fig4b_sanity_gsm8k_typos.pdf
```

There is also a quick standalone script:

```bash
python patch_only.py
```

Edit `patch_only.py` if you want to change the model, perturbation type, perturbation rate, output directory, or number of pairs.

### Viability Check

`viability_check.py` is a small Qwen2.5 GSM8K clean-accuracy gate:

```bash
python viability_check.py
```

It evaluates 50 GSM8K test examples and writes:

```text
lrd_results/qwen_viability.json
```

Use this before larger Qwen experiments to confirm the base model and evaluator are behaving as expected.

### LoRA Window Sweeps

Use `sweep_lora_window.py` to train a LoRA adapter on a specific contiguous layer window and evaluate the adapter across perturbations.

Example:

```bash
python sweep_lora_window.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --layer_start 0 \
  --layer_end 3 \
  --n_steps 300 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --target_modules q_proj v_proj \
  --output_dir stabilizer_weights/qwen_sweep_L00_03 \
  --n_eval 500 \
  --n_per_condition 150 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --grad_accum_steps 4
```

The script trains on GSM8K clean/noisy pairs and saves:

```text
stabilizer_weights/qwen_sweep_L00_03/lora_final/
stabilizer_weights/qwen_sweep_L00_03/training_logs.json
stabilizer_weights/qwen_sweep_L00_03/eval_results.json
```

To sweep a model, repeat the command over all layer windows. The repository's post-processing scripts expect names like:

```text
qwen_sweep_L00_03
qwen_sweep_L04_07
qwen_sweep_L08_11
qwen_sweep_L12_15
qwen_sweep_L16_19
qwen_sweep_L20_23
qwen_sweep_L24_27
```

For TinyLlama and Gemma, see the hard-coded window lists in `predict_optimal_window.py` and `three_map_new.py`.

### Predicting an Optimal LoRA Window

`predict_optimal_window.py` uses existing LRD results plus C3/C4 metrics to predict which LoRA window should work best.

Example:

```bash
python predict_optimal_window.py \
  --model Qwen25 \
  --compute_c3c4 \
  --n_samples 50
```

After LoRA sweep results exist, compare the prediction to actual sweep deltas:

```bash
python predict_optimal_window.py \
  --model Qwen25 \
  --check_prediction
```

Supported model keys are:

```text
TinyLlama
Gemma2
Qwen25
```

Predictions are saved under:

```text
results/predictions/
```

### Three-Map and Autocorrelation Post-Processing

`three_map_new.py` overlays three layer-wise signals:

- LRD profile.
- Activation patching recovery.
- LoRA sweep effectiveness.

Run it after the corresponding LRD, patching, and LoRA sweep output files exist:

```bash
python three_map_new.py
```

It writes model-specific `*_three_map.json`, `*_three_map.pdf`, `*_autocorrelation.json`, and a combined overlay JSON.

`exp_autocorr.py` computes autocorrelation summaries from a combined overlay file:

```bash
python exp_autocorr.py
```

It expects `expB_three_map_overlay.json` in the current directory and writes:

```text
exp_autocorr_results.json
exp_autocorr.pdf
```

### LRD Resource Notes

- `lrd_diagnostics.py` is expensive because it runs generation, hidden-state extraction, and scoring over multiple perturbations.
- Activation patching is slower than ordinary evaluation because it performs layer-by-layer patched generation over clean-success/noisy-failure pairs.
- LoRA sweeps multiply training cost by the number of layer windows.
- Start with `--n_samples 20`, `--n_pairs 10`, `--n_eval 50`, or `--n_steps 20` for a smoke test.
- Full sweeps for 7B-9B models are best run on A100-class GPUs or equivalent high-memory hardware.
- The LRD scripts contain some hard-coded expected output paths for cross-model post-processing; keep output directory names consistent with the script configs or update the config dictionaries before running post-processing.

## License

MIT — see [LICENSE](LICENSE).
