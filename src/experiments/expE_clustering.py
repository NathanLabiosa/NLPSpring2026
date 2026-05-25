"""
expE_clustering.py — Per-Example LRD Trajectory Clustering

Moves beyond mean LRD to ask: do individual examples follow the same
32-layer trajectory, or are there subpopulations?

Model: Phi-3.5-mini-instruct (richest signal)
Dataset: GSM8K, n=500
Perturbations: typos 10%, speech 10%

Procedure:
  1. For each example, compute the full per-layer LRD trajectory
     (32-dimensional vector: cosine distance at each layer)
  2. Split examples into succeed / fail under perturbation
  3. Run k-means clustering (k=3–5) on the trajectories
  4. For each cluster: mean trajectory, succeed/fail ratio,
     example characteristics (question length, n_perturbed_tokens)

Output:
  expE_trajectory_clusters.json
  expE_trajectories_typos.pdf
  expE_trajectories_speech.pdf
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in [ROOT, os.path.join(ROOT, "Phi3.5")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from perturbations import PerturbationEngine

MODEL_ID      = "microsoft/Phi-3.5-mini-instruct"
SYSTEM_PROMPT = "Solve the math problem step by step. The last line must be '#### ANSWER'."

PERTURBATIONS = [
    ("typos",   0.10),
    ("speech",  0.10),
]


def format_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot  = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if norm < 1e-9 else float(1.0 - dot / norm)


@torch.no_grad()
def compute_lrd_trajectory(model, tokenizer, clean_prompt: str,
                            noisy_prompt: str, device: str) -> np.ndarray:
    """
    Returns per-layer LRD as a 1-D array of length n_layers+1.
    Uses mean-pooled hidden states.
    """
    def encode_and_extract(prompt):
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=512, padding=False
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        hs  = out.hidden_states   # tuple of [1, T, D]
        return np.stack([h[0].float().mean(dim=0).cpu().numpy()
                         for h in hs], axis=0)   # [n_layers+1, D]

    h_clean = encode_and_extract(clean_prompt)
    h_noisy = encode_and_extract(noisy_prompt)
    n_layers = min(h_clean.shape[0], h_noisy.shape[0])
    return np.array([cosine_distance(h_clean[l], h_noisy[l])
                     for l in range(n_layers)])


def is_correct_gsm8k(prediction: str, gold_answer: str) -> bool:
    """Extract #### ANSWER from generation and compare."""
    import re
    def extract(text):
        m = re.search(r"####\s*([^\n]+)", text)
        return m.group(1).strip().replace(",", "") if m else None

    pred_num = extract(prediction)
    gold_num = extract(gold_answer)
    if pred_num is None or gold_num is None:
        return False
    try:
        return abs(float(pred_num) - float(gold_num)) < 1e-3
    except ValueError:
        return pred_num.strip().lower() == gold_num.strip().lower()


def cluster_and_plot(
    trajectories: list,   # list of 1-D arrays (same length)
    labels: list,         # list of bool (True = correct)
    characteristics: list,# list of dicts with 'q_len', 'n_perturbed'
    k_range: range,
    cond_name: str,
    out_prefix: str,
) -> dict:
    """Run k-means for k in k_range; pick best k by silhouette, return results."""
    from sklearn.metrics import silhouette_score

    X = np.array(trajectories)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, best_score, best_km = None, -1, None
    for k in k_range:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        if len(set(km.labels_)) < 2:
            continue
        sc = silhouette_score(X_scaled, km.labels_)
        print(f"  k={k}: silhouette={sc:.3f}")
        if sc > best_score:
            best_score = sc
            best_k     = k
            best_km    = km

    if best_km is None:
        print("  Could not find valid clustering — using k=3")
        best_k  = min(3, len(X))
        best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        best_km.fit(X_scaled)

    cluster_labels = best_km.labels_
    print(f"  Best k={best_k} (silhouette={best_score:.3f})")

    results = {}
    layers  = np.arange(X.shape[1])
    colors  = plt.cm.tab10(np.linspace(0, 1, best_k))

    fig, ax = plt.subplots(figsize=(10, 5))

    for c in range(best_k):
        mask        = cluster_labels == c
        members     = X[mask]
        correct_arr = np.array(labels)[mask]
        fail_rate   = 1 - correct_arr.mean()
        mean_traj   = members.mean(axis=0)
        std_traj    = members.std(axis=0)

        char_arr = [characteristics[i] for i in range(len(labels)) if mask[i]]
        mean_q_len   = float(np.mean([ch["q_len"]      for ch in char_arr]))
        mean_n_pert  = float(np.mean([ch["n_perturbed"] for ch in char_arr]))

        print(f"  Cluster {c}: n={mask.sum()}, fail_rate={fail_rate:.2%}, "
              f"mean_q_len={mean_q_len:.0f}, mean_n_perturbed={mean_n_pert:.1f}")

        results[str(c)] = {
            "n":              int(mask.sum()),
            "fail_rate":      round(float(fail_rate), 4),
            "mean_trajectory": mean_traj.tolist(),
            "std_trajectory":  std_traj.tolist(),
            "mean_q_len":     round(mean_q_len, 1),
            "mean_n_perturbed": round(mean_n_pert, 2),
        }

        label = (f"Cluster {c} (n={mask.sum()}, "
                 f"fail={fail_rate:.0%})")
        ax.plot(layers, mean_traj, label=label, color=colors[c], linewidth=2.0)
        ax.fill_between(layers, mean_traj - std_traj, mean_traj + std_traj,
                        alpha=0.15, color=colors[c])

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean LRD", fontsize=11)
    ax.set_title(f"LRD Trajectory Clusters — Phi3.5 / {cond_name}\n"
                 f"(k={best_k}, silhouette={best_score:.3f})", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_path = f"{out_prefix}_{cond_name.replace(' ', '_').lower()}.pdf"
    plt.savefig(fig_path)
    plt.close()
    print(f"  → figure: {fig_path}")

    return {
        "best_k":     best_k,
        "silhouette": round(best_score, 4),
        "clusters":   results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="LRD trajectory computation is done one-by-one")
    parser.add_argument("--k_min",      type=int, default=3)
    parser.add_argument("--k_max",      type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    engine = PerturbationEngine(seed=args.seed)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading GSM8K test set...")
    ds    = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds)
    random.shuffle(items)
    # Keep only examples that are correct on clean input (standard exclusion filter)
    print("Pre-filtering to clean-correct examples (evaluating clean accuracy)...")
    correct_items = []
    for it in items:
        prompt = format_prompt(tokenizer, it["question"])
        enc    = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            out  = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        pred = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        if is_correct_gsm8k(pred, it["answer"]):
            correct_items.append(it)
        if len(correct_items) >= args.n_samples:
            break

    print(f"  Clean-correct examples collected: {len(correct_items)}")
    items = correct_items[: args.n_samples]

    # ── Compute trajectories for each perturbation ────────────────────────────
    all_out = {}
    out_prefix = os.path.join(ROOT, "expE_trajectories")

    for method, rate in PERTURBATIONS:
        cond_name = f"{method}_{int(rate * 100)}pct"
        print(f"\n=== {cond_name} ===")

        trajectories    = []
        is_correct_list = []
        char_list       = []

        for idx, it in enumerate(items):
            clean_q  = it["question"]
            noisy_q  = engine.apply(clean_q, method, rate)
            n_pert   = sum(a != b for a, b in zip(clean_q, noisy_q))

            clean_prompt = format_prompt(tokenizer, clean_q)
            noisy_prompt = format_prompt(tokenizer, noisy_q)

            traj = compute_lrd_trajectory(model, tokenizer, clean_prompt,
                                          noisy_prompt, device)
            trajectories.append(traj)
            char_list.append({
                "q_len":      len(clean_q),
                "n_perturbed": n_pert,
            })

            # Evaluate noisy correctness
            enc = tokenizer(noisy_prompt, return_tensors="pt",
                            truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out_gen = model.generate(
                    **enc, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            pred = tokenizer.decode(
                out_gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
            )
            correct = is_correct_gsm8k(pred, it["answer"])
            is_correct_list.append(correct)

            if (idx + 1) % 50 == 0:
                acc = sum(is_correct_list) / len(is_correct_list)
                print(f"  {idx+1}/{len(items)}  running_acc={acc:.2%}")

            del enc, out_gen
            torch.cuda.empty_cache()

        total_fail_rate = 1 - sum(is_correct_list) / len(is_correct_list)
        print(f"  Failure rate under {cond_name}: {total_fail_rate:.2%}")

        # ── Clustering ────────────────────────────────────────────────────────
        cluster_res = cluster_and_plot(
            trajectories, is_correct_list, char_list,
            k_range=range(args.k_min, args.k_max + 1),
            cond_name=cond_name,
            out_prefix=out_prefix,
        )

        all_out[cond_name] = {
            "n_examples":    len(items),
            "failure_rate":  round(total_fail_rate, 4),
            "clustering":    cluster_res,
            "trajectories": [t.tolist() for t in trajectories],
            "is_correct":   is_correct_list,
            "characteristics": char_list,
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(ROOT, "expE_trajectory_clusters.json")
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
