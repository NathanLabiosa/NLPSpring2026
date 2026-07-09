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

This code backs an ACL Rolling Review (ARR) submission. `figures/` contains every figure cited in the paper; `results/` contains the underlying numeric results; `models/<name>/` contains the per-model scripts and raw data used to produce them.

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
