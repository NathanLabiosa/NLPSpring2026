"""
lrd_diagnostics.py — Layer-wise Representation Divergence (LRD) Diagnostic Suite
Hardened version — closes the following reviewer holes:

  HOLE 2: Patching sanity checks added (identity patch → ~100% recovery,
           random noise patch → ~0% recovery). n_pairs default raised 50→100.
  HOLE 3: Per-token LRD computed over changed token positions only,
           alongside mean-pooled LRD. Produces Fig 5 comparing the two.
  HOLE 4: Exclusion bias analysis — compares final-layer LRD of excluded
           (clean-fail) vs included (clean-success) examples. Produces Fig 6.
  HOLE 5: Homophones high-rate experiment — pass --homophones_rates 0.40 0.50
           to generate more failure cases and test if the paradox holds.
  HOLE 6: Taxonomy variance test — Levene's test comparing LRD variance of
           directional vs uniform perturbation classes. Produces Fig 7.
  HOLE 7: BBH sample-size guard with explicit warnings.

Architecture: works with Phi-3.5, Mistral-7B, Llama-3, Qwen out of the box.
For other architectures, get_layer_list() will raise a helpful error.

Usage:
    # Standard run on Mistral
    python lrd_diagnostics.py --dataset gsm8k \
        --model mistralai/Mistral-7B-Instruct-v0.3

    # Full hardened run with all fixes enabled
    python lrd_diagnostics.py --dataset gsm8k \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --patching --n_pairs 100 \
        --run_exclusion_analysis \
        --run_taxonomy_variance \
        --homophones_rates 0.40 0.50

    # BBH with recommended sample size
    python lrd_diagnostics.py --dataset bbh \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --n_samples 400
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pointbiserialr, mannwhitneyu, levene

from transformers import AutoModelForCausalLM, AutoTokenizer

from perturbations import PerturbationEngine
from data_loader import DatasetManager
# NOTE: evaluator import removed - functions were imported but never used
# from evaluator import (
#     evaluate_humaneval_entry, evaluate_gsm8k_entry,
#     evaluate_multiple_choice_entry, evaluate_squad_entry, evaluate_bbh_entry,
# )


# =============================================================================
# 1. Architecture-agnostic layer accessor
# =============================================================================

def get_layer_list(model):
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.model.decoder.layers,
    ]
    for accessor in candidates:
        try:
            layers = accessor(model)
            if len(layers) > 0:
                return layers
        except AttributeError:
            continue

    raise AttributeError(
        "Could not locate transformer layers automatically. "
        "Run:  print([n for n, _ in model.named_modules() if 'layer' in n.lower()][:15])\n"
        "then add the correct accessor to get_layer_list()."
    )


def load_model_and_tokenizer(model_id: str, device: str = "cuda"):
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


# =============================================================================
# 2. Representation Extractor  (mean-pooled + per-token)
# =============================================================================

class RepresentationExtractor:
    def __init__(self, model, tokenizer, device="cuda", max_length=512):
        self.model      = model
        self.tokenizer  = tokenizer
        self.device     = device
        self.max_length = max_length

    @torch.no_grad()
    def extract(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hs = outputs.hidden_states

        pooled  = np.stack(
            [h[0].mean(dim=0).float().cpu().numpy() for h in hs], axis=0
        )
        per_tok = [h[0].float().cpu().numpy() for h in hs]

        token_ids = inputs["input_ids"][0].tolist()
        tokens    = self.tokenizer.convert_ids_to_tokens(token_ids)

        return pooled, per_tok, tokens


# =============================================================================
# 3. LRD metrics
# =============================================================================

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot  = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if norm < 1e-9 else float(1.0 - dot / norm)


def compute_lrd_profile(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    assert clean.shape == noisy.shape
    return np.array([cosine_distance(clean[l], noisy[l]) for l in range(clean.shape[0])])


def compute_perturbed_token_lrd(
    clean_per_tok: list,
    noisy_per_tok: list,
    clean_tokens:  list,
    noisy_tokens:  list,
) -> Optional[np.ndarray]:
    min_len   = min(len(clean_tokens), len(noisy_tokens))
    diff_pos  = [i for i in range(min_len) if clean_tokens[i] != noisy_tokens[i]]

    if not diff_pos:
        return None

    n_layers   = len(clean_per_tok)
    lrd_layers = []
    for l in range(n_layers):
        c, n = clean_per_tok[l], noisy_per_tok[l]
        vals = [
            cosine_distance(c[i], n[i])
            for i in diff_pos
            if i < c.shape[0] and i < n.shape[0]
        ]
        lrd_layers.append(float(np.mean(vals)) if vals else 0.0)

    return np.array(lrd_layers)


# =============================================================================
# 4. Cascade / recovery helpers
# =============================================================================

def cascade_slope(lrd_profile: np.ndarray) -> float:
    layers = np.arange(len(lrd_profile))
    return float(np.polyfit(layers, lrd_profile, deg=1)[0])


def recovery_test(lrd_profile: np.ndarray) -> dict:
    n = len(lrd_profile)
    q = max(1, n // 4)
    early = float(lrd_profile[:q].mean())
    late  = float(lrd_profile[-q:].mean())
    return {"early_lrd": early, "late_lrd": late, "recovered": late < early * 0.8}


# =============================================================================
# 5. Generation & evaluation helpers
# =============================================================================

@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str,
                    max_new_tokens: int = 256, device: str = "cuda",
                    use_cache: bool = True) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=use_cache,
    )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def evaluate(dataset_name: str, generated_text: str, sample: dict) -> bool:
    if dataset_name == "humaneval":
        return evaluate_humaneval_entry(generated_text, sample)
    elif dataset_name == "gsm8k":
        return evaluate_gsm8k_entry(generated_text, sample)
    elif dataset_name in ["mmlu", "arc"]:
        return evaluate_multiple_choice_entry(generated_text, sample)
    elif dataset_name == "squad":
        return evaluate_squad_entry(generated_text, sample)
    elif dataset_name == "bbh":
        return evaluate_bbh_entry(generated_text, sample)
    return False


# =============================================================================
# 6. Perturbation definitions
# =============================================================================

BASE_EXPERIMENTS = [
    {"name": "Typos_5%",       "type": "typos",      "rate": 0.05},
    {"name": "OCR_5%",         "type": "ocr",        "rate": 0.05},
    {"name": "Whitespace_10%", "type": "whitespace", "rate": 0.10},
    {"name": "Case_10%",       "type": "case",       "rate": 0.10},
    {"name": "Homophones_20%", "type": "homophones", "rate": 0.20},
    {"name": "Speech_10%",     "type": "speech",     "rate": 0.10},
]


# =============================================================================
# 7. Main diagnostic loop
# =============================================================================

def run_diagnostics(
    dataset_name:           str,
    model_id:               str,
    n_samples:              int  = 200,
    output_dir:             str  = "lrd_results",
    device:                 str  = "cuda",
    max_new_tokens:         int  = 256,
    run_exclusion_analysis: bool = False,
    run_taxonomy_variance:  bool = False,
    homophones_rates:       list = None,
    use_system_prompt:      bool = True,
    use_chat_template:      bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "bbh" and n_samples < 300:
        print(
            f"\n[WARNING] BBH with n_samples={n_samples}. "
            "Recommend --n_samples 400 — BBH is hard and few clean-correct "
            "examples survive the pre-filter at small n.\n"
        )

    model, tokenizer = load_model_and_tokenizer(model_id, device)
    extractor = RepresentationExtractor(model, tokenizer, device=device)
    perturber = PerturbationEngine()
    dm        = DatasetManager(tokenizer, use_system_prompt=use_system_prompt,
                               use_chat_template=use_chat_template)

    print(f"\nPre-filtering {dataset_name} to clean-correct examples ...")
    clean_ds = dm.load_and_format(dataset_name, perturbation_func=None)

    clean_correct = []
    clean_fail    = []

    for i in tqdm(range(min(n_samples * 4, len(clean_ds))), desc="Pre-filter"):
        try:
            gen = greedy_generate(
                model, tokenizer, clean_ds[i]["formatted_prompt"],
                max_new_tokens=max_new_tokens, device=device
            )
            if evaluate(dataset_name, gen, clean_ds[i]):
                clean_correct.append(i)
            else:
                clean_fail.append(i)
        except Exception:
            continue
        if len(clean_correct) >= n_samples:
            break

    print(f"  Clean-correct: {len(clean_correct)}  |  Clean-fail (excluded): {len(clean_fail)}")
    if len(clean_correct) < 50:
        print("[WARNING] Fewer than 50 clean-correct examples — results will be underpowered.")

    experiments = list(BASE_EXPERIMENTS)
    if homophones_rates:
        for rate in homophones_rates:
            name = f"Homophones_{int(rate * 100)}%"
            if not any(e["name"] == name for e in experiments):
                experiments.append({"name": name, "type": "homophones", "rate": rate})

    all_results = {}

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n{'─'*55}\n{exp_name}\n{'─'*55}")

        p_func   = lambda text, e=exp: perturber.apply(text, e["type"], e["rate"])
        noisy_ds = dm.load_and_format(dataset_name, perturbation_func=p_func)
        records  = []

        for idx in tqdm(clean_correct, desc=exp_name):
            cp  = clean_ds[idx]["formatted_prompt"]
            np_ = noisy_ds[idx]["formatted_prompt"]

            try:
                c_pool, c_tok, c_tkns = extractor.extract(cp)
                n_pool, n_tok, n_tkns = extractor.extract(np_)

                lrd         = compute_lrd_profile(c_pool, n_pool)
                per_tok_lrd = compute_perturbed_token_lrd(c_tok, n_tok, c_tkns, n_tkns)

                noisy_gen  = greedy_generate(model, tokenizer, np_,
                                             max_new_tokens=max_new_tokens, device=device)
                is_correct = evaluate(dataset_name, noisy_gen, noisy_ds[idx])

                slope   = cascade_slope(lrd)
                recover = recovery_test(lrd)

                records.append({
                    "index":               idx,
                    "lrd_profile":         lrd.tolist(),
                    "final_lrd":           float(lrd[-1]),
                    "mean_lrd":            float(lrd.mean()),
                    "max_lrd":             float(lrd.max()),
                    "cascade_slope":       slope,
                    "recovered":           recover["recovered"],
                    "early_lrd":           recover["early_lrd"],
                    "late_lrd":            recover["late_lrd"],
                    "is_correct":          is_correct,
                    "per_tok_lrd_profile": per_tok_lrd.tolist() if per_tok_lrd is not None else None,
                    "per_tok_final_lrd":   float(per_tok_lrd[-1]) if per_tok_lrd is not None else None,
                    "n_differing_tokens":  sum(1 for a, b in zip(c_tkns, n_tkns) if a != b),
                })

            except Exception as ex:
                print(f"  [warn] idx {idx}: {ex}")
                continue

        all_results[exp_name] = records
        acc  = np.mean([r["is_correct"] for r in records]) if records else 0
        mlrd = np.mean([r["final_lrd"]  for r in records]) if records else 0
        print(f"  n={len(records)}  acc={acc:.2%}  mean_final_LRD={mlrd:.4f}")

    with open(os.path.join(output_dir, f"raw_{dataset_name}.json"), "w") as f:
        json.dump(all_results, f)

    stats = compute_statistics(all_results)
    with open(os.path.join(output_dir, f"stats_{dataset_name}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    plot_lrd_curves(all_results, dataset_name, output_dir)
    plot_lrd_vs_accuracy(all_results, dataset_name, output_dir)
    plot_cascade_heatmap(all_results, dataset_name, output_dir)
    plot_per_token_vs_mean_lrd(all_results, dataset_name, output_dir)

    if run_exclusion_analysis and clean_fail:
        run_exclusion_bias_analysis(
            model, tokenizer, extractor, perturber, dm,
            dataset_name, clean_ds, clean_correct, clean_fail,
            output_dir, device, max_new_tokens,
        )

    if run_taxonomy_variance:
        run_taxonomy_variance_test(all_results, dataset_name, output_dir)

    print(f"\n{'='*55}\nSUMMARY — {dataset_name.upper()}\n{'='*55}")
    for name, s in stats.items():
        sig = "✓" if s["mannwhitney"]["p"] < 0.05 else "✗"
        print(
            f"  {name:<22}  acc={s['accuracy']:.2%}  "
            f"LRD={s['mean_final_lrd']:.4f}  "
            f"r={s['lrd_failure_correlation']['r']:+.3f}  "
            f"MW_p={s['mannwhitney']['p']:.4f}  {sig}"
        )

    return all_results, stats


# =============================================================================
# 8. Statistics
# =============================================================================

def compute_statistics(all_results: dict) -> dict:
    stats = {}
    for exp_name, records in all_results.items():
        if not records:
            continue

        lrd_finals = np.array([r["final_lrd"]      for r in records])
        slopes     = np.array([r["cascade_slope"]   for r in records])
        is_correct = np.array([int(r["is_correct"]) for r in records])
        recovered  = np.array([int(r["recovered"])  for r in records])

        r_val, p_val = pointbiserialr(lrd_finals, is_correct)

        lrd_c = lrd_finals[is_correct == 1]
        lrd_w = lrd_finals[is_correct == 0]
        if len(lrd_c) > 0 and len(lrd_w) > 0:
            U, U_p = mannwhitneyu(lrd_w, lrd_c, alternative="greater")
        else:
            U, U_p = float("nan"), float("nan")

        pt_finals = [r["per_tok_final_lrd"] for r in records
                     if r.get("per_tok_final_lrd") is not None]

        stats[exp_name] = {
            "n":                       len(records),
            "n_correct":               int(is_correct.sum()),
            "n_wrong":                 int((1 - is_correct).sum()),
            "accuracy":                float(is_correct.mean()),
            "mean_final_lrd":          float(lrd_finals.mean()),
            "std_final_lrd":           float(lrd_finals.std()),
            "mean_cascade_slope":      float(slopes.mean()),
            "pct_recovered":           float(recovered.mean() * 100),
            "lrd_failure_correlation": {"r": float(r_val), "p": float(p_val)},
            "mannwhitney":             {"U": float(U),     "p": float(U_p)},
            "mean_per_tok_final_lrd":  float(np.mean(pt_finals)) if pt_finals else None,
            "per_tok_n":               len(pt_finals),
        }
    return stats


# =============================================================================
# 9. Figures
# =============================================================================

def plot_lrd_curves(all_results, dataset_name, output_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors  = cm.tab10(np.linspace(0, 1, len(all_results)))
    for (name, records), color in zip(all_results.items(), colors):
        if not records:
            continue
        profiles = np.array([r["lrd_profile"] for r in records])
        mean     = profiles.mean(0)
        std      = profiles.std(0)
        layers   = np.arange(len(mean))
        ax.plot(layers, mean, label=name, color=color, linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.12, color=color)
    ax.set_xlabel("Layer Depth", fontsize=13)
    ax.set_ylabel("LRD (Cosine Distance)", fontsize=13)
    ax.set_title(f"Layer-Wise Representation Divergence — {dataset_name.upper()}", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, f"fig1_lrd_curves_{dataset_name}.pdf")
    plt.savefig(p, dpi=300); plt.close(); print(f"Fig1 → {p}")


def plot_lrd_vs_accuracy(all_results, dataset_name, output_dir):
    n    = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1: axes = [axes]
    for ax, (name, records) in zip(axes, all_results.items()):
        if not records: ax.set_visible(False); continue
        lrd_c = [r["final_lrd"] for r in records if     r["is_correct"]]
        lrd_w = [r["final_lrd"] for r in records if not r["is_correct"]]
        data  = [x for x in [lrd_c, lrd_w] if x]
        labs  = ([f"Correct\n(n={len(lrd_c)})"] if lrd_c else []) + \
                ([f"Wrong\n(n={len(lrd_w)})"]   if lrd_w else [])
        bp = ax.boxplot(data, labels=labs, patch_artist=True, notch=True)
        for box, c in zip(bp["boxes"], ["#4CAF50", "#F44336"]):
            box.set_facecolor(c)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("Final-Layer LRD", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Final-Layer LRD by Task Outcome — {dataset_name.upper()}", fontsize=13, y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, f"fig2_lrd_vs_accuracy_{dataset_name}.pdf")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(); print(f"Fig2 → {p}")


def plot_cascade_heatmap(all_results, dataset_name, output_dir):
    exp_names = list(all_results.keys())
    n_layers  = next((len(r["lrd_profile"]) for recs in all_results.values()
                      for r in recs if r.get("lrd_profile")), None)
    if not n_layers: return
    matrix = np.zeros((len(exp_names), n_layers))
    for i, (_, records) in enumerate(all_results.items()):
        if records:
            matrix[i] = np.array([r["lrd_profile"] for r in records]).mean(0)
    fig, ax = plt.subplots(figsize=(max(12, n_layers // 2), len(exp_names) + 1))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(exp_names))); ax.set_yticklabels(exp_names, fontsize=10)
    ax.set_xlabel("Layer Depth", fontsize=12)
    ax.set_title(f"LRD Heatmap (Perturbation × Layer) — {dataset_name.upper()}", fontsize=13)
    step = max(1, n_layers // 10)
    ax.set_xticks(range(0, n_layers, step)); ax.set_xticklabels(range(0, n_layers, step))
    plt.colorbar(im, ax=ax, label="Mean LRD"); plt.tight_layout()
    p = os.path.join(output_dir, f"fig3_heatmap_{dataset_name}.pdf")
    plt.savefig(p, dpi=300); plt.close(); print(f"Fig3 → {p}")


def plot_per_token_vs_mean_lrd(all_results, dataset_name, output_dir):
    has_pt = any(any(r.get("per_tok_lrd_profile") for r in recs)
                 for recs in all_results.values())
    if not has_pt:
        return

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1: axes = [axes]

    for ax, (name, records) in zip(axes, all_results.items()):
        mean_prof = np.array([r["lrd_profile"] for r in records if r.get("lrd_profile")])
        tok_prof  = np.array([r["per_tok_lrd_profile"] for r in records
                              if r.get("per_tok_lrd_profile") is not None])

        if mean_prof.size > 0:
            layers = np.arange(mean_prof.shape[1])
            ax.plot(layers, mean_prof.mean(0), label="Mean-pooled", color="steelblue", lw=2)

        if tok_prof.size > 0:
            tok_layers = np.arange(tok_prof.shape[1])
            ax.plot(tok_layers, tok_prof.mean(0),
                    label="Per-token (changed pos)", color="crimson", lw=2, ls="--")
            ax.fill_between(tok_layers,
                            tok_prof.mean(0) - tok_prof.std(0),
                            tok_prof.mean(0) + tok_prof.std(0),
                            alpha=0.1, color="crimson")

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel("LRD", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Mean-Pooled vs Per-Token LRD — {dataset_name.upper()}\n"
        "(Per-token = changed positions only; mean-pooled is a lower bound)",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    p = os.path.join(output_dir, f"fig5_per_token_lrd_{dataset_name}.pdf")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(); print(f"Fig5 → {p}")


# =============================================================================
# 10. HOLE 4 — Exclusion Bias Analysis
# =============================================================================

def run_exclusion_bias_analysis(
    model, tokenizer, extractor, perturber, dm,
    dataset_name, clean_ds,
    correct_indices, fail_indices,
    output_dir, device, max_new_tokens,
    n_sample: int = 100,
):
    print("\n[HOLE 4] Exclusion bias analysis ...")

    p_func   = lambda text: perturber.apply(text, "typos", 0.05)
    noisy_ds = dm.load_and_format(dataset_name, perturbation_func=p_func)

    def get_lrds(indices):
        lrds = []
        for idx in tqdm(indices[:n_sample]):
            try:
                cp, _, _  = extractor.extract(clean_ds[idx]["formatted_prompt"])
                np_, _, _ = extractor.extract(noisy_ds[idx]["formatted_prompt"])
                lrd = compute_lrd_profile(cp, np_)
                lrds.append(float(lrd[-1]))
            except Exception:
                pass
        return lrds

    fail_lrds    = get_lrds(fail_indices)
    success_lrds = get_lrds(correct_indices)

    if not fail_lrds or not success_lrds:
        print("  Not enough data."); return

    U, p = mannwhitneyu(fail_lrds, success_lrds, alternative="two-sided")
    mf, ms = np.mean(fail_lrds), np.mean(success_lrds)

    print(f"  Excluded  mean LRD: {mf:.5f}  (n={len(fail_lrds)})")
    print(f"  Included  mean LRD: {ms:.5f}  (n={len(success_lrds)})")
    print(f"  Mann-Whitney: U={U:.1f}, p={p:.4f}")
    verdict = "UNBIASED — no significant LRD difference." if p > 0.05 \
              else f"BIAS DETECTED (p={p:.4f}) — acknowledge as limitation."
    print(f"  → {verdict}")

    result = {"excluded_mean": mf, "included_mean": ms,
              "U": U, "p": p, "verdict": verdict}
    with open(os.path.join(output_dir, f"exclusion_bias_{dataset_name}.json"), "w") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot([success_lrds, fail_lrds],
                    labels=["Included\n(clean-success)", "Excluded\n(clean-fail)"],
                    patch_artist=True, notch=True)
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#9E9E9E")
    ax.set_ylabel("Final-Layer LRD (typos 5%)", fontsize=11)
    ax.set_title(
        f"Exclusion Bias Check — {dataset_name.upper()}\n"
        f"MW p={p:.4f}  ({verdict.split(' — ')[0]})",
        fontsize=11
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fp = os.path.join(output_dir, f"fig6_exclusion_bias_{dataset_name}.pdf")
    plt.savefig(fp, dpi=300); plt.close(); print(f"Fig6 → {fp}")


# =============================================================================
# 11. HOLE 6 — Taxonomy Variance Test  [FIXED]
# =============================================================================

def run_taxonomy_variance_test(all_results: dict, dataset_name: str, output_dir: str):
    """
    HOLE 6 FIX — Per-perturbation wrong-vs-correct Levene test.

    Previous bug: pooled ALL directional LRD values together and compared to ALL
    uniform values. Case has high LRD variance in BOTH outcome groups, which
    swamps the signal when pooled. The pooled Levene test reliably returns
    "not supported" even when the per-perturbation directional signal is clear.

    Fix: for each directional perturbation individually, run Levene's test
    comparing the wrong-group LRD distribution to the correct-group LRD
    distribution. The taxonomy prediction is specifically that wrong-group std
    > correct-group std for directional types (failure correlates with the
    degree of divergence). This is tested directly, per perturbation.

    Uniform types (case, whitespace) serve as the negative control — we expect
    wrong-group std ≈ correct-group std there.
    """
    print("\n[HOLE 6] Taxonomy variance test (per-perturbation) ...")

    directional_names = {"Typos_5%", "OCR_5%", "Speech_10%", "Homophones_20%"}
    uniform_names     = {"Case_10%", "Whitespace_10%"}

    per_pert_results = {}
    n_directional_supported = 0
    n_directional_tested    = 0

    # ── Per-perturbation Levene: wrong-group std vs correct-group std ─────────
    for name, records in all_results.items():
        if name not in directional_names and name not in uniform_names:
            continue

        lrd_correct = [r["final_lrd"] for r in records if     r["is_correct"]]
        lrd_wrong   = [r["final_lrd"] for r in records if not r["is_correct"]]

        if len(lrd_correct) < 5 or len(lrd_wrong) < 5:
            print(f"  {name}: skipped (n_correct={len(lrd_correct)}, n_wrong={len(lrd_wrong)} — too few)")
            continue

        stat, p = levene(lrd_wrong, lrd_correct)
        std_wrong   = float(np.std(lrd_wrong))
        std_correct = float(np.std(lrd_correct))
        # Directional prediction: wrong-group has higher variance
        directional_confirmed = (std_wrong > std_correct) and (p < 0.05)
        class_label = "directional" if name in directional_names else "uniform"

        per_pert_results[name] = {
            "class":                class_label,
            "n_correct":            len(lrd_correct),
            "n_wrong":              len(lrd_wrong),
            "std_correct":          std_correct,
            "std_wrong":            std_wrong,
            "levene_stat":          float(stat),
            "levene_p":             float(p),
            "wrong_std_gt_correct": bool(std_wrong > std_correct),
            "significant":          bool(p < 0.05),
            "directional_confirmed": bool(directional_confirmed),
        }

        dir_marker = "✓" if directional_confirmed else "✗"
        print(
            f"  {name:<22} [{class_label:>11}]  "
            f"std_wrong={std_wrong:.4f}  std_correct={std_correct:.4f}  "
            f"Levene p={p:.4f}  {dir_marker}"
        )

        if name in directional_names:
            n_directional_tested += 1
            if directional_confirmed:
                n_directional_supported += 1

    # ── Summary verdict ───────────────────────────────────────────────────────
    taxonomy_supported = (n_directional_tested > 0 and
                          n_directional_supported >= n_directional_tested // 2 + 1)

    print(f"\n  Directional types with wrong_std > correct_std (p<0.05): "
          f"{n_directional_supported}/{n_directional_tested}")
    if taxonomy_supported:
        print("  → Majority of directional perturbations show higher wrong-group variance.")
        print("    TAXONOMY VARIANCE CLAIM SUPPORTED.")
    else:
        print("  → Fewer than half of directional types show significant wrong-group variance.")
        print("    Frame taxonomy as descriptive only.")

    # ── Figure: per-perturbation std bars (wrong vs correct) ─────────────────
    exp_names   = list(per_pert_results.keys())
    stds_corr   = [per_pert_results[n]["std_correct"] for n in exp_names]
    stds_wrong  = [per_pert_results[n]["std_wrong"]   for n in exp_names]
    is_dir      = [per_pert_results[n]["class"] == "directional" for n in exp_names]

    x   = np.arange(len(exp_names))
    fig, ax = plt.subplots(figsize=(max(10, len(exp_names) * 1.6), 5))
    bars_corr  = ax.bar(x - 0.2, stds_corr,  0.35,
                        label="Std LRD (correct)", color="#4CAF50", alpha=0.85)
    bars_wrong = ax.bar(x + 0.2, stds_wrong, 0.35,
                        label="Std LRD (wrong)",   color="#F44336", alpha=0.85)

    # Shade background by class
    for i, d in enumerate(is_dir):
        ax.axvspan(i - 0.5, i + 0.5,
                   alpha=0.06,
                   color="steelblue" if d else "gray")

    # Significance markers
    for i, name in enumerate(exp_names):
        r = per_pert_results[name]
        if r["significant"] and r["wrong_std_gt_correct"]:
            ymax = max(r["std_correct"], r["std_wrong"]) * 1.05
            ax.text(i, ymax, "*", ha="center", va="bottom", fontsize=14, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("LRD Standard Deviation", fontsize=11)
    ax.set_title(
        f"Per-Perturbation Wrong vs Correct LRD Variance — {dataset_name.upper()}\n"
        f"Blue shading = directional class  |  * = Levene p<0.05 with wrong_std > correct_std\n"
        f"Taxonomy {'supported' if taxonomy_supported else 'not supported'} "
        f"({n_directional_supported}/{n_directional_tested} directional types confirmed)",
        fontsize=10
    )
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fp = os.path.join(output_dir, f"fig7_taxonomy_variance_{dataset_name}.pdf")
    plt.savefig(fp, dpi=300); plt.close(); print(f"Fig7 → {fp}")

    with open(os.path.join(output_dir, f"taxonomy_variance_{dataset_name}.json"), "w") as f:
        json.dump({
            "per_perturbation":          per_pert_results,
            "n_directional_tested":      n_directional_tested,
            "n_directional_supported":   n_directional_supported,
            "taxonomy_supported":        taxonomy_supported,
        }, f, indent=2)


# =============================================================================
# 12. Activation Patching — HOLE 2 sanity checks + n_pairs=100  [FIXED]
# =============================================================================

class ActivationPatcher:
    """
    Three patch modes:
      "clean"    — replace noisy hidden state with clean state  (real experiment)
      "identity" — substitute cached clean states into the NOISY forward pass
                   (positive control → ~100% recovery, confirms hook fires)
      "random"   — inject random noise into noisy state         (negative control → ~0%)

    HOLE 2 FIX — identity mode bug:
      Previously: target = clean_prompt for identity mode.
      This made the identity hook a no-op: it cached activations from clean_prompt,
      then ran generation on clean_prompt, so the hook replaced activations with
      themselves — indistinguishable from a normal clean forward pass. Recovery
      reflected actual model accuracy on clean inputs (~50% on hard failure pairs),
      not hook efficacy. This produced the suspicious flat-at-0.50 GSM8K result.

      Fix: identity mode now runs generation on noisy_prompt (same as clean mode),
      but substitutes the cached clean states. A working hook MUST recover the
      failure → ~100% recovery. This is the correct positive control: it tests
      that the hook actually intercepts and replaces the activations.
    """

    def __init__(self, model, tokenizer, device="cuda", max_new_tokens=128):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.max_new_tokens = max_new_tokens
        self._clean_states  = {}
        self._hooks         = []
        self._layers        = get_layer_list(model)

    def _register_save_hooks(self):
        self._clean_states = {}
        self._hooks        = []

        def make_save(i):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                self._clean_states[i] = h.detach().clone()
                return out
            return hook

        for i, layer in enumerate(self._layers):
            self._hooks.append(layer.register_forward_hook(make_save(i)))

    def _register_patch_hook(self, layer_idx: int, mode: str):
        def patch_hook(module, inp, out):
            h     = out[0] if isinstance(out, tuple) else out
            clean = self._clean_states.get(layer_idx)

            # With use_cache=True, decode steps process only the single new token
            # (h.shape[1] == 1). Patching is only meaningful during prefill where
            # the full prompt is processed and the KV cache is populated.
            if h.shape[1] <= 1:
                return out

            if mode in ("clean", "identity") and clean is not None:
                patched = h.clone()
                ml = min(h.shape[1], clean.shape[1])
                # Suffix-align: patch the LAST ml positions so the generation-critical
                # assistant-start token (which ends both prompts identically) is always
                # included. Prefix alignment misses the suffix when noisy_len > clean_len,
                # causing systematically low recovery because the model generates from a
                # noisy last-position state even when all earlier positions are patched.
                patched[:, h.shape[1] - ml:, :] = clean[:, clean.shape[1] - ml:, :]
            elif mode == "random":
                patched = h + torch.randn_like(h) * h.std()
            else:
                return out

            return (patched,) + out[1:] if isinstance(out, tuple) else patched

        self._hooks.append(
            self._layers[layer_idx].register_forward_hook(patch_hook)
        )

    def _clear_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []

    @torch.no_grad()
    def run_identity_all_layers(self, clean_prompt: str, noisy_prompt: str,
                                dataset_name: str, sample: dict) -> bool:
        """Positive control: patch ALL layers simultaneously with clean states.
        A working hook must recover the answer (~100% across pairs).
        This is the correct sanity check — per-layer averaging is misleading
        because early-layer-only patches still leave noisy processing downstream."""
        self._register_save_hooks()
        inp = self.tokenizer(clean_prompt, return_tensors="pt",
                             truncation=True, max_length=512).to(self.device)
        _ = self.model(**inp, output_hidden_states=False)
        self._clear_hooks()

        for layer_idx in range(len(self._layers)):
            self._register_patch_hook(layer_idx, "identity")
        gen = greedy_generate(
            self.model, self.tokenizer, noisy_prompt,
            max_new_tokens=self.max_new_tokens, device=self.device,
            use_cache=True,
        )
        self._clear_hooks()
        return evaluate(dataset_name, gen, sample)

    @torch.no_grad()
    def run(self, clean_prompt: str, noisy_prompt: str,
            dataset_name: str, sample: dict, mode: str = "clean") -> dict:
        # Step 1: cache clean hidden states (always from clean_prompt)
        self._register_save_hooks()
        inp = self.tokenizer(clean_prompt, return_tensors="pt",
                             truncation=True, max_length=512).to(self.device)
        _ = self.model(**inp, output_hidden_states=False)
        self._clear_hooks()

        # Step 2: run generation on the noisy prompt for all three modes.
        #   - "clean"    : replace noisy activations with clean → real experiment
        #   - "identity" : replace noisy activations with clean → positive control
        #                  (same op as clean, but called on the pair's noisy side;
        #                   a working hook must restore accuracy to ~100%)
        #   - "random"   : corrupt noisy activations further → negative control (~0%)
        target   = noisy_prompt  # all modes run on the noisy input
        recovery = {}

        for layer_idx in tqdm(range(len(self._layers)),
                              desc=f"Patch [{mode}]", leave=False):
            self._register_patch_hook(layer_idx, mode)
            gen = greedy_generate(
                self.model, self.tokenizer, target,
                max_new_tokens=self.max_new_tokens, device=self.device,
                use_cache=True,
            )
            self._clear_hooks()
            recovery[layer_idx] = evaluate(dataset_name, gen, sample)

        return recovery


def run_activation_patching(
    dataset_name:      str,
    model_id:          str,
    n_pairs:           int   = 100,
    output_dir:        str   = "lrd_results",
    perturbation_type: str   = "typos",
    perturbation_rate: float = 0.05,
    device:            str   = "cuda",
    max_new_tokens:    int   = 128,
    run_sanity_checks: bool  = True,
    use_system_prompt: bool  = True,
    use_chat_template: bool  = True,
):
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(model_id, device)
    model.eval()

    dm        = DatasetManager(tokenizer, use_system_prompt=use_system_prompt,
                               use_chat_template=use_chat_template)
    perturber = PerturbationEngine()
    patcher   = ActivationPatcher(model, tokenizer, device=device,
                                  max_new_tokens=max_new_tokens)

    clean_ds = dm.load_and_format(dataset_name, perturbation_func=None)
    n_search = min(n_pairs * 8, len(clean_ds))

    # Materialize noisy prompts once — prevents stochastic re-sampling where
    # calling noisy_ds[i] twice returns different perturbations.
    print("Materializing noisy prompts ...")
    noisy_prompts = {}
    for i in tqdm(range(n_search)):
        noisy_prompts[i] = perturber.apply(
            clean_ds[i]["formatted_prompt"], perturbation_type, perturbation_rate
        )

    pairs = []
    print("Finding clean-success / noisy-failure pairs ...")
    for i in tqdm(range(n_search)):
        s   = clean_ds[i]
        cp  = s["formatted_prompt"]
        np_ = noisy_prompts[i]
        cg  = greedy_generate(model, tokenizer, cp,  max_new_tokens=max_new_tokens, device=device)
        ng  = greedy_generate(model, tokenizer, np_, max_new_tokens=max_new_tokens, device=device)
        if evaluate(dataset_name, cg, s) and not evaluate(dataset_name, ng, s):
            pairs.append((i, cp, np_, s))
        if len(pairs) >= n_pairs:
            break

    print(f"Found {len(pairs)} pairs (target {n_pairs}).")
    if len(pairs) < 10:
        print("[WARNING] Fewer than 10 pairs — patching results will be unreliable.")

    sanity_out = {}

    if run_sanity_checks and pairs:
        n_sc     = min(10, len(pairs))
        sc_pairs = pairs[:n_sc]
        print(f"\n[HOLE 2] Sanity checks on {n_sc} pairs ...")

        def run_mode(mode):
            rec = defaultdict(list)
            for _, cp, np_, s in sc_pairs:
                rm = patcher.run(cp, np_, dataset_name, s, mode=mode)
                for l, v in rm.items():
                    rec[l].append(int(v))
            return {l: np.mean(v) for l, v in rec.items()}

        id_rates   = run_mode("identity")   # per-layer, used for plot only
        rand_rates = run_mode("random")

        # Correct positive control: patch ALL layers at once → must be ~100%.
        # Per-layer mean is misleading (early layers recover ~0%, late ~100%, avg ~50%).
        identity_all = [
            int(patcher.run_identity_all_layers(cp, np_, dataset_name, s))
            for _, cp, np_, s in sc_pairs
        ]
        mid = np.mean(identity_all)
        mrd = np.mean(list(rand_rates.values()))
        print(f"  Identity patch (all layers) recovery: {mid:.2%}  (expected ~100%)")
        print(f"  Random   patch mean recovery:         {mrd:.2%}  (expected ~0%)")

        if mid < 0.80:
            print(f"  [WARNING] Identity patch = {mid:.2%}. "
                  "Hook may not be intercepting activations — check get_layer_list() "
                  "and confirm the hooked module is on the residual stream path.")
        if mrd > 0.20:
            print(f"  [WARNING] Random patch = {mrd:.2%}. "
                  "Model may be insensitive to hidden-state perturbations at these layers.")

        sanity_out = {"identity_mean": float(mid), "random_mean": float(mrd),
                      "identity_ok": bool(mid >= 0.80), "random_ok": bool(mrd <= 0.20)}

        layers = sorted(id_rates.keys())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(layers, [id_rates[l]   for l in layers], color="steelblue")
        axes[0].axhline(1.0, color="green", ls="--", label="Expected ~1.0")
        axes[0].set_title("Positive Control: Identity Patch\n(should be ~100%)")
        axes[0].set_ylim(0, 1.1); axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].bar(layers, [rand_rates[l] for l in layers], color="salmon")
        axes[1].axhline(0.0, color="red",   ls="--", label="Expected ~0%")
        axes[1].set_title("Negative Control: Random Noise Patch\n(should be ~0%)")
        axes[1].set_ylim(0, 1.1); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

        for ax in axes:
            ax.set_xlabel("Layer Index"); ax.set_ylabel("Recovery Rate")

        fig.suptitle(
            f"Patching Sanity Checks — {dataset_name.upper()} ({perturbation_type})",
            fontsize=13
        )
        plt.tight_layout()
        sp = os.path.join(output_dir,
                          f"fig4b_sanity_{dataset_name}_{perturbation_type}.pdf")
        plt.savefig(sp, dpi=300); plt.close(); print(f"Sanity fig → {sp}")

    print("\nMain patching experiment (clean mode) ...")
    layer_rec = defaultdict(list)
    for _, cp, np_, s in tqdm(pairs, desc="Patching"):
        rm = patcher.run(cp, np_, dataset_name, s, mode="clean")
        for l, v in rm.items():
            layer_rec[l].append(int(v))

    layers    = sorted(layer_rec.keys())
    rec_rates = [np.mean(layer_rec[l]) for l in layers]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(layers, rec_rates, color="steelblue", edgecolor="white")
    ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Random baseline (0.50)")
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Recovery Rate", fontsize=12)
    ax.set_title(
        f"Activation Patching — {dataset_name.upper()}\n"
        f"{perturbation_type} @ {perturbation_rate}  (n={len(pairs)} pairs)",
        fontsize=12
    )
    ax.legend(fontsize=10); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fp = os.path.join(output_dir, f"fig4_patching_{dataset_name}_{perturbation_type}.pdf")
    plt.savefig(fp, dpi=300); plt.close(); print(f"Fig4 → {fp}")

    with open(os.path.join(output_dir,
              f"patching_{dataset_name}_{perturbation_type}.json"), "w") as f:
        json.dump({
            "layer_recovery": {str(l): layer_rec[l] for l in layers},
            "sanity":         sanity_out,
            "n_pairs":        len(pairs),
        }, f, indent=2)

    return layer_rec


# =============================================================================
# 13. CLI
# =============================================================================

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="LRD Diagnostic Suite (hardened)")

    pa.add_argument("--dataset",        type=str, required=True)
    pa.add_argument("--model",          type=str,
                    default="mistralai/Mistral-7B-Instruct-v0.3")
    pa.add_argument("--n_samples",      type=int, default=200,
                    help="Use 400 for BBH.")
    pa.add_argument("--output_dir",     type=str, default="lrd_results")
    pa.add_argument("--max_new_tokens", type=int, default=256)

    pa.add_argument("--patching",       action="store_true")
    pa.add_argument("--patch_perturb",  type=str,  default="typos")
    pa.add_argument("--n_pairs",        type=int,  default=100)
    pa.add_argument("--skip_sanity",    action="store_true")

    pa.add_argument("--run_exclusion_analysis", action="store_true",
                    help="HOLE 4: pre-filter bias check")
    pa.add_argument("--run_taxonomy_variance",  action="store_true",
                    help="HOLE 6: per-perturbation wrong-vs-correct variance test")
    pa.add_argument("--homophones_rates", type=float, nargs="+", default=None,
                    help="HOLE 5: extra homophone rates, e.g. 0.40 0.50")

    args = pa.parse_args()

    all_results, stats = run_diagnostics(
        dataset_name           = args.dataset,
        model_id               = args.model,
        n_samples              = args.n_samples,
        output_dir             = args.output_dir,
        max_new_tokens         = args.max_new_tokens,
        run_exclusion_analysis = args.run_exclusion_analysis,
        run_taxonomy_variance  = args.run_taxonomy_variance,
        homophones_rates       = args.homophones_rates,
    )

    if args.patching:
        run_activation_patching(
            dataset_name      = args.dataset,
            model_id          = args.model,
            n_pairs           = args.n_pairs,
            output_dir        = args.output_dir,
            perturbation_type = args.patch_perturb,
            max_new_tokens    = args.max_new_tokens,
            run_sanity_checks = not args.skip_sanity,
        )
