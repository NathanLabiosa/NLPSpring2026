# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
eval_adapter_divergence.py — Measure how much each LoRA adapter rewrites
the base model's hidden states on CLEAN inputs.

For each adapter, computes per-layer cosine distance between:
  - base model hidden states (adapter disabled) on clean input
  - adapted model hidden states (adapter enabled) on the SAME clean input

High divergence = the adapter aggressively rewrites clean representations.
Low divergence  = the adapter preserves clean behavior (L_stab's job).

This distinguishes "the adapter learned robustness" from "the adapter
rewrote the model so much that everything changed, including perturbed
input handling."

Usage:
    python eval_adapter_divergence.py \
        --adapters v17=./stabilizer_weights/lora_v17/lora_final \
                   clean_only=./stabilizer_weights/p1_clean_only/lora_final \
                   rand_s1=./stabilizer_weights/p1_random_layers_seed1/lora_final \
        --n_samples 200 \
        --output ./adapter_divergence_results.json
"""

import os, sys, json, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lrd_stabilizer import (
    format_question_prompt, GSM8K_SYSTEM_PROMPT,
    _extract_gsm8k_truth, _gsm8k_correct,
)


@torch.no_grad()
def measure_adapter_divergence(model, tokenizer, prompts, device):
    """For each prompt, compute per-layer cosine distance between
    adapter-enabled and adapter-disabled hidden states."""

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    all_divergences = []

    for prompt in tqdm(prompts, desc="Measuring divergence"):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512).to(device)

        # Forward with adapter
        out_adapted = model(**enc, output_hidden_states=True, return_dict=True)
        hs_adapted = [h[0].float().cpu() for h in out_adapted.hidden_states]

        # Forward without adapter
        with model.disable_adapter():
            out_base = model(**enc, output_hidden_states=True, return_dict=True)
        hs_base = [h[0].float().cpu() for h in out_base.hidden_states]

        # Per-layer cosine distance (mean over token positions)
        per_layer = []
        for l in range(len(hs_adapted)):
            cos_sim = F.cosine_similarity(hs_adapted[l], hs_base[l], dim=-1)
            dist = (1 - cos_sim).mean().item()
            per_layer.append(dist)

        all_divergences.append(per_layer)

    return np.array(all_divergences)  # (n_samples, n_layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--adapters", type=str, nargs="+", required=True,
                        help="name=path pairs, e.g. v17=./stabilizer_weights/lora_v17/lora_final")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="./adapter_divergence_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse adapter specs
    adapter_specs = {}
    for spec in args.adapters:
        name, path = spec.split("=", 1)
        adapter_specs[name] = path

    # Load clean GSM8K prompts
    from datasets import load_dataset
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(test_ds)
    random.seed(42)
    random.shuffle(test_items)
    test_items = test_items[:args.n_samples]

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Format clean prompts
    prompts = [format_question_prompt(tokenizer, item["question"], GSM8K_SYSTEM_PROMPT)
               for item in test_items]

    results = {}

    for adapter_name, adapter_path in adapter_specs.items():
        print(f"\n{'='*60}")
        print(f"Adapter: {adapter_name} ({adapter_path})")
        print(f"{'='*60}")

        # Load fresh base model + this adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map={"": 0}, torch_dtype=torch.float16,
            trust_remote_code=True, attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        divergences = measure_adapter_divergence(model, tokenizer, prompts, device)
        # divergences: (n_samples, n_layers)

        mean_per_layer = divergences.mean(axis=0).tolist()
        std_per_layer = divergences.std(axis=0).tolist()

        # Summary stats
        n_layers = len(mean_per_layer)
        causal_window = mean_per_layer[1:6]  # layers 0-4 (hidden_states[1] = after layer 0)
        overall_mean = float(np.mean(mean_per_layer[1:]))  # skip embedding

        print(f"  Layers: {n_layers}")
        print(f"  Overall mean divergence (excl embedding): {overall_mean:.6f}")
        print(f"  Causal window (layers 0-4) mean: {np.mean(causal_window):.6f}")
        print(f"  Max divergence: {max(mean_per_layer[1:]):.6f} at layer {np.argmax(mean_per_layer[1:])+1}")
        print()
        print(f"  Per-layer divergence:")
        for l in range(n_layers):
            tag = ""
            if 1 <= l <= 5:
                tag = " ← causal window"
            print(f"    layer {l:2d}: {mean_per_layer[l]:.6f} ± {std_per_layer[l]:.6f}{tag}")

        results[adapter_name] = {
            "adapter_path": adapter_path,
            "n_samples": len(prompts),
            "mean_per_layer": mean_per_layer,
            "std_per_layer": std_per_layer,
            "overall_mean_divergence": overall_mean,
            "causal_window_mean": float(np.mean(causal_window)),
            "max_divergence_layer": int(np.argmax(mean_per_layer[1:])),
            "max_divergence_value": float(max(mean_per_layer[1:])),
        }

        # Free GPU memory
        del model, base_model
        torch.cuda.empty_cache()

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: Clean-input divergence (adapter vs base)")
    print(f"{'='*70}")
    print(f"{'Adapter':<20} {'Overall':>10} {'Causal(0-4)':>12} {'Max':>10} {'MaxLayer':>10}")
    print(f"{'-'*62}")
    for name, r in results.items():
        print(f"{name:<20} {r['overall_mean_divergence']:>10.6f} "
              f"{r['causal_window_mean']:>12.6f} "
              f"{r['max_divergence_value']:>10.6f} "
              f"{r['max_divergence_layer']:>10d}")


if __name__ == "__main__":
    main()
