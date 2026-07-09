"""
Diagnostic script to investigate Mistral's low identity-patch recovery.

Hypothesis: token length mismatch between clean and noisy prompts causes
suffix-alignment to leave early noisy positions unpatched, degrading recovery.

This script:
  1. Finds pairs and logs token length differences (clean_len vs noisy_len)
  2. Splits sanity pairs into "matched" (|diff| <= max_len_diff) and "mismatched"
  3. Reports identity recovery separately for each group
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from lrd_diagnostics import (
    load_model_and_tokenizer, greedy_generate, evaluate,
    ActivationPatcher, get_layer_list,
)
from data_loader import DatasetManager
from perturbations import PerturbationEngine

os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

MODEL_ID          = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET           = "gsm8k"
PERTURBATION_TYPE = "typos"
PERTURBATION_RATE = 0.05
N_PAIRS           = 100
MAX_NEW_TOKENS    = 128
MAX_LEN_DIFF      = 3   # tokens; pairs with |diff| > this are "mismatched"
DEVICE            = "cuda"

print(f"Loading {MODEL_ID} ...")
model, tokenizer = load_model_and_tokenizer(MODEL_ID, DEVICE)
model.eval()

dm        = DatasetManager(tokenizer)
perturber = PerturbationEngine()
patcher   = ActivationPatcher(model, tokenizer, device=DEVICE,
                               max_new_tokens=MAX_NEW_TOKENS)

clean_ds = dm.load_and_format(DATASET, perturbation_func=None)
n_search = len(clean_ds)

print("Materializing noisy prompts ...")
noisy_prompts = {}
for i in tqdm(range(n_search)):
    noisy_prompts[i] = perturber.apply(
        clean_ds[i]["formatted_prompt"], PERTURBATION_TYPE, PERTURBATION_RATE
    )

pairs      = []
len_diffs  = []   # track token length differences

print("Finding pairs and logging token lengths ...")
for i in tqdm(range(n_search)):
    cp  = clean_ds[i]["formatted_prompt"]
    np_ = noisy_prompts[i]
    cg  = greedy_generate(model, tokenizer, cp,  max_new_tokens=MAX_NEW_TOKENS, device=DEVICE)
    ng  = greedy_generate(model, tokenizer, np_, max_new_tokens=MAX_NEW_TOKENS, device=DEVICE)
    if evaluate(DATASET, cg, clean_ds[i]) and not evaluate(DATASET, ng, clean_ds[i]):
        cl = tokenizer(cp,  return_tensors="pt", truncation=True, max_length=1024)["input_ids"].shape[1]
        nl = tokenizer(np_, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].shape[1]
        pairs.append((i, cp, np_, clean_ds[i], cl, nl))
        len_diffs.append(nl - cl)
    if len(pairs) >= N_PAIRS:
        break

print(f"\nFound {len(pairs)} pairs (target {N_PAIRS})")
len_diffs = np.array(len_diffs)
print(f"Token length diffs (noisy - clean):")
print(f"  mean={len_diffs.mean():.2f}  std={len_diffs.std():.2f}  "
      f"min={len_diffs.min()}  max={len_diffs.max()}")
print(f"  exact match (diff=0): {(len_diffs==0).sum()}/{len(len_diffs)}")
print(f"  |diff| <= {MAX_LEN_DIFF}: {(np.abs(len_diffs)<=MAX_LEN_DIFF).sum()}/{len(len_diffs)}")

# Sanity check split by length match
n_sc     = min(20, len(pairs))
sc_pairs = pairs[:n_sc]

matched   = [(i, cp, np_, s, cl, nl) for i, cp, np_, s, cl, nl in sc_pairs if abs(nl-cl) <= MAX_LEN_DIFF]
mismatched= [(i, cp, np_, s, cl, nl) for i, cp, np_, s, cl, nl in sc_pairs if abs(nl-cl) >  MAX_LEN_DIFF]

print(f"\nSanity pairs: {n_sc} total | matched (|diff|<={MAX_LEN_DIFF}): {len(matched)} | mismatched: {len(mismatched)}")

def check_identity(pair_list, label):
    if not pair_list:
        print(f"  {label}: no pairs")
        return
    results = [
        int(patcher.run_identity_all_layers(cp, np_, DATASET, s))
        for _, cp, np_, s, cl, nl in tqdm(pair_list, desc=label)
    ]
    print(f"  {label}: identity recovery = {np.mean(results):.2%}  ({sum(results)}/{len(results)})")
    for (i, cp, np_, s, cl, nl), r in zip(pair_list, results):
        print(f"    idx={i}  clean_len={cl}  noisy_len={nl}  diff={nl-cl}  recovered={r}")

print("\n--- Identity recovery by length match ---")
check_identity(matched,    f"Matched   (|diff|<={MAX_LEN_DIFF})")
check_identity(mismatched, f"Mismatched (|diff|>{MAX_LEN_DIFF})")
