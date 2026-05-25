# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
compensatory_analysis.py — Compensatory computation analysis (Priority 7).

Tests how the v17 LoRA adapter achieves robustness:
  1. Post-hoc LRD comparison: LRD at layers 0–4 with and without adapter
  2. Linear probe accuracy: probe on layer-4 hidden states to predict correctness,
     with and without adapter active

Usage:
    python compensatory_analysis.py \
        --checkpoint_dir ./stabilizer_weights/lora_v17/lora_final \
        --n_samples 200 \
        --output ./compensatory_results
"""

import os, sys, json, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lrd_stabilizer import (
    EVAL_CONDITIONS, format_question_prompt, _import_perturbation_engine,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
    GSM8K_SYSTEM_PROMPT,
)

PerturbationEngine = _import_perturbation_engine()


def mean_cosine_dist(h1, h2, mask):
    """Mean cosine distance between h1 and h2 at positions where mask is True."""
    if not mask.any():
        return 0.0
    h1_m = h1[mask].float()
    h2_m = h2[mask].float()
    cos_sim = F.cosine_similarity(h1_m, h2_m, dim=-1)
    return float((1 - cos_sim).mean().item())


@torch.no_grad()
def extract_hidden_states(model, tokenizer, prompt, device, target_layers=None):
    """Extract hidden states at specified layers."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=512).to(device)
    out = model(**enc, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states  # tuple of (1, T, D) for each layer

    if target_layers is not None:
        return {l: hs[l][0].float().cpu() for l in target_layers}
    return {l: hs[l][0].float().cpu() for l in range(len(hs))}


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, device):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=512).to(device)
    out = model.generate(**enc, max_new_tokens=300, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--causal_layers", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Causal window layers to analyze")
    parser.add_argument("--probe_layer", type=int, default=5,
                        help="Layer index for linear probe (hidden_states[i])")
    parser.add_argument("--output", type=str, default="./compensatory_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model with adapter
    print(f"Loading {args.model} + adapter from {args.checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": 0}, torch_dtype=torch.float16,
        trust_remote_code=True, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    model.eval()

    perturber = PerturbationEngine()

    # Load test data
    from datasets import load_dataset
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(test_ds)
    random.shuffle(test_items)
    test_items = test_items[:args.n_samples]

    # Target layers for hidden state extraction (causal window + probe layer)
    target_layers = sorted(set(args.causal_layers + [args.probe_layer]))

    results = {}

    for cond_name, cond_type, cond_rate in EVAL_CONDITIONS:
        print(f"\n{'─'*55}\n{cond_name}\n{'─'*55}")

        # Per-layer LRD with and without adapter
        lrd_base = {l: [] for l in target_layers}    # perturbed vs clean, no adapter
        lrd_adapted = {l: [] for l in target_layers}  # perturbed vs clean, with adapter

        # Linear probe data
        probe_features_base = []
        probe_features_adapted = []
        probe_labels = []

        for item in tqdm(test_items, desc=cond_name):
            q = item["question"]
            truth = _extract_gsm8k_truth(item["answer"])
            if truth is None:
                continue

            clean_prompt = format_question_prompt(tokenizer, q, GSM8K_SYSTEM_PROMPT)
            if cond_type == "none":
                noisy_prompt = clean_prompt
            else:
                noisy_q = perturber.apply(q, cond_type, cond_rate)
                noisy_prompt = format_question_prompt(tokenizer, noisy_q, GSM8K_SYSTEM_PROMPT)

            try:
                # Clean hidden states (no adapter)
                with model.disable_adapter():
                    hs_clean = extract_hidden_states(model, tokenizer, clean_prompt,
                                                     device, target_layers)

                # Perturbed hidden states WITHOUT adapter
                with model.disable_adapter():
                    hs_noisy_base = extract_hidden_states(model, tokenizer, noisy_prompt,
                                                          device, target_layers)

                # Perturbed hidden states WITH adapter
                hs_noisy_adapted = extract_hidden_states(model, tokenizer, noisy_prompt,
                                                          device, target_layers)

                # Compute LRD at each target layer
                seq_len = min(hs_clean[target_layers[0]].shape[0],
                              hs_noisy_base[target_layers[0]].shape[0],
                              hs_noisy_adapted[target_layers[0]].shape[0])
                mask = torch.ones(seq_len, dtype=torch.bool)

                for l in target_layers:
                    lrd_base[l].append(mean_cosine_dist(
                        hs_clean[l][:seq_len], hs_noisy_base[l][:seq_len], mask))
                    lrd_adapted[l].append(mean_cosine_dist(
                        hs_clean[l][:seq_len], hs_noisy_adapted[l][:seq_len], mask))

                # Probe features: mean-pooled hidden states at probe_layer
                probe_features_base.append(
                    hs_noisy_base[args.probe_layer][:seq_len].mean(dim=0).numpy())
                probe_features_adapted.append(
                    hs_noisy_adapted[args.probe_layer][:seq_len].mean(dim=0).numpy())

                # Generate with adapter to get correctness label
                gen = generate_answer(model, tokenizer, noisy_prompt, device)
                is_correct = _gsm8k_correct(gen, truth)
                probe_labels.append(int(is_correct))

            except Exception as ex:
                print(f"  [warn]: {ex}")
                continue

        # Summarize LRD comparison
        lrd_comparison = {}
        for l in target_layers:
            if lrd_base[l]:
                mean_base = float(np.mean(lrd_base[l]))
                mean_adapted = float(np.mean(lrd_adapted[l]))
                reduction_pct = ((mean_base - mean_adapted) / (mean_base + 1e-9)) * 100
                lrd_comparison[f"layer_{l}"] = {
                    "lrd_base": mean_base,
                    "lrd_adapted": mean_adapted,
                    "reduction_pct": reduction_pct,
                }
                in_window = "←" if l in args.causal_layers else ""
                print(f"  Layer {l:2d}: base={mean_base:.6f}  adapted={mean_adapted:.6f}  "
                      f"Δ={reduction_pct:+.2f}% {in_window}")

        # Linear probe comparison
        probe_result = {"n": len(probe_labels)}
        if len(probe_labels) >= 20 and len(set(probe_labels)) > 1:
            X_base = np.stack(probe_features_base)
            X_adapted = np.stack(probe_features_adapted)
            y = np.array(probe_labels)

            clf_base = LogisticRegression(max_iter=1000, C=1.0)
            scores_base = cross_val_score(clf_base, X_base, y, cv=5, scoring="accuracy")

            clf_adapted = LogisticRegression(max_iter=1000, C=1.0)
            scores_adapted = cross_val_score(clf_adapted, X_adapted, y, cv=5, scoring="accuracy")

            probe_result["base_probe_acc"] = float(scores_base.mean())
            probe_result["adapted_probe_acc"] = float(scores_adapted.mean())
            probe_result["probe_acc_delta"] = float(scores_adapted.mean() - scores_base.mean())

            print(f"  Probe (layer {args.probe_layer}): base={scores_base.mean():.3f}  "
                  f"adapted={scores_adapted.mean():.3f}  Δ={scores_adapted.mean()-scores_base.mean():+.3f}")

        results[cond_name] = {
            "lrd_comparison": lrd_comparison,
            "probe": probe_result,
        }

    out_path = os.path.join(args.output, "compensatory_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
