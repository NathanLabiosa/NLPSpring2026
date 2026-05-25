"""
compute_cka.py — CKA comparison for LRD validation (Priority 4).

Computes linear CKA between clean and perturbed representations at each layer,
alongside standard LRD (cosine distance). Compares rank-ordering of layers
to validate LRD as a simpler proxy.

Also computes per-position LRD (not mean-pooled) for the homophones paradox.

Usage:
    python compute_cka.py \
        --model microsoft/Phi-3.5-mini-instruct \
        --dataset gsm8k \
        --n_samples 100 \
        --output ./cka_results
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Phi3.5"))
from perturbations import PerturbationEngine
from data_loader import DatasetManager
from lrd_diagnostics import (
    load_model_and_tokenizer, RepresentationExtractor,
    cosine_distance, greedy_generate, evaluate,
    BASE_EXPERIMENTS,
)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices X, Y of shape (n, d).

    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic_xy = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro")
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro")

    denom = hsic_xx * hsic_yy
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def per_position_lrd(clean_per_tok, noisy_per_tok, n_layers):
    """Compute LRD at each token position across layers.

    Returns: array of shape (n_layers, min_seq_len)
    """
    min_len = min(clean_per_tok[0].shape[0], noisy_per_tok[0].shape[0])
    result = np.zeros((n_layers, min_len))
    for l in range(n_layers):
        c = clean_per_tok[l][:min_len]
        n = noisy_per_tok[l][:min_len]
        for pos in range(min_len):
            result[l, pos] = cosine_distance(c[pos], n[pos])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default="./cka_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    extractor = RepresentationExtractor(model, tokenizer, device=device)
    perturber = PerturbationEngine()
    dm = DatasetManager(tokenizer)

    # Pre-filter to clean-correct examples
    print(f"Pre-filtering {args.dataset} to clean-correct examples...")
    clean_ds = dm.load_and_format(args.dataset, perturbation_func=None)
    clean_correct = []
    for i in tqdm(range(min(args.n_samples * 4, len(clean_ds))), desc="Pre-filter"):
        try:
            gen = greedy_generate(model, tokenizer, clean_ds[i]["formatted_prompt"],
                                  max_new_tokens=args.max_new_tokens, device=device)
            if evaluate(args.dataset, gen, clean_ds[i]):
                clean_correct.append(i)
        except Exception:
            continue
        if len(clean_correct) >= args.n_samples:
            break
    print(f"Clean-correct: {len(clean_correct)}")

    all_results = {}

    for exp in BASE_EXPERIMENTS:
        exp_name = exp["name"]
        print(f"\n{'─'*55}\n{exp_name}\n{'─'*55}")

        p_func = lambda text, e=exp: perturber.apply(text, e["type"], e["rate"])
        noisy_ds = dm.load_and_format(args.dataset, perturbation_func=p_func)

        # Collect per-layer representations for CKA
        layer_clean_reps = {}  # layer -> list of mean-pooled vectors
        layer_noisy_reps = {}
        lrd_profiles = []
        per_pos_lrd_samples = []

        for idx in tqdm(clean_correct, desc=exp_name):
            cp = clean_ds[idx]["formatted_prompt"]
            np_ = noisy_ds[idx]["formatted_prompt"]

            try:
                c_pool, c_tok, c_tkns = extractor.extract(cp)
                n_pool, n_tok, n_tkns = extractor.extract(np_)

                n_layers = c_pool.shape[0]

                # Mean-pooled LRD
                lrd = np.array([cosine_distance(c_pool[l], n_pool[l]) for l in range(n_layers)])
                lrd_profiles.append(lrd)

                # Per-position LRD
                pp_lrd = per_position_lrd(c_tok, n_tok, n_layers)
                per_pos_lrd_samples.append(pp_lrd)

                # Collect for CKA (per-layer)
                for l in range(n_layers):
                    layer_clean_reps.setdefault(l, []).append(c_pool[l])
                    layer_noisy_reps.setdefault(l, []).append(n_pool[l])

            except Exception as ex:
                print(f"  [warn] idx {idx}: {ex}")
                continue

        # Compute CKA per layer
        n_layers = len(layer_clean_reps)
        cka_per_layer = []
        lrd_mean_per_layer = []

        for l in range(n_layers):
            X = np.stack(layer_clean_reps[l])  # (n, d)
            Y = np.stack(layer_noisy_reps[l])  # (n, d)
            cka_val = linear_cka(X, Y)
            cka_per_layer.append(cka_val)
            lrd_mean_per_layer.append(float(np.mean([p[l] for p in lrd_profiles])))

        # Rank correlation between CKA (1 - CKA for divergence) and LRD
        cka_divergence = [1.0 - c for c in cka_per_layer]
        rho, p_val = spearmanr(cka_divergence, lrd_mean_per_layer)

        # Per-position LRD: average across examples, report changed vs unchanged positions
        mean_per_pos_lrd = np.mean(per_pos_lrd_samples, axis=0) if per_pos_lrd_samples else None

        print(f"  Spearman ρ (CKA-divergence vs LRD): {rho:.4f} (p={p_val:.4f})")
        print(f"  CKA range: [{min(cka_per_layer):.4f}, {max(cka_per_layer):.4f}]")
        print(f"  LRD range: [{min(lrd_mean_per_layer):.4f}, {max(lrd_mean_per_layer):.4f}]")

        all_results[exp_name] = {
            "n_examples": len(lrd_profiles),
            "cka_per_layer": cka_per_layer,
            "lrd_per_layer": lrd_mean_per_layer,
            "cka_lrd_spearman_rho": float(rho),
            "cka_lrd_spearman_p": float(p_val),
            "mean_per_position_lrd": mean_per_pos_lrd.tolist() if mean_per_pos_lrd is not None else None,
        }

    out_path = os.path.join(args.output, f"cka_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
