# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
eval_typos_rerun.py
Quick re-evaluation of v5 checkpoint on typos at n=200 to check whether
the -2% result is noise or a real regression from v4's +6%.

NOTE (Phase 0 harness audit): The canonical accuracy path is
``eval_fixed_harness.py`` (max_new_tokens=512). This standalone v5 rerun is
historical; max_new_tokens bumped to 512 to match the canonical harness.
"""
import sys, os, random, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from stabilizer_system import MultiLayerStabilizerSystem
from train_lrd_stabilizer import (
    format_question_prompt, GSM8K_SYSTEM_PROMPT,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
)

def _import_perturber():
    from perturbations import PerturbationEngine
    return PerturbationEngine()

MODEL_PATH  = "microsoft/Phi-3.5-mini-instruct"
CKPT_PATH   = "./stabilizer_weights/phi35_gsm8k_mlp_v5/stabilizer_final.pt"
N_SAMPLES   = 200
SEED        = 42

random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
print(f"Loading {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map={"": 0}, torch_dtype=torch.float16,
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()

# Load stabilizer
print(f"Loading stabilizer from {CKPT_PATH} ...")
system = MultiLayerStabilizerSystem.load(model, CKPT_PATH, device=device)

# Load data
print(f"Loading GSM8K test split, n={N_SAMPLES} ...")
ds = list(load_dataset("openai/gsm8k", "main", split="test"))
random.shuffle(ds)
items = ds[:N_SAMPLES]

perturber = _import_perturber()
gen_kwargs = dict(max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id)

def generate(prompts):
    outputs = []
    for i in range(0, len(prompts), 4):
        batch = prompts[i:i+4]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        new = out[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    return outputs

def accuracy(preds, truths):
    return sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
               for p, t in zip(preds, truths)) / len(truths) * 100

truths = [_extract_gsm8k_truth(it["answer"]) for it in items]

for method, rate, label in [
    ("none",  0.00, "clean_baseline"),
    ("typos", 0.05, "typos_5pct"),
    ("ocr",   0.05, "ocr_5pct"),        # included for free since we're already running
]:
    print(f"\n--- {label} (n={N_SAMPLES}) ---")
    perturbed = [
        format_question_prompt(tokenizer, perturber.apply(it["question"], method, rate)
                               if rate > 0 else it["question"], GSM8K_SYSTEM_PROMPT)
        for it in items
    ]

    # Without stabilizer
    preds_base = generate(perturbed)
    acc_base   = accuracy(preds_base, truths)

    # With stabilizer
    handles = system.register_inference_hooks()
    try:
        preds_stab = generate(perturbed)
    finally:
        for h in handles: h.remove()
    acc_stab = accuracy(preds_stab, truths)

    print(f"  No stabilizer:   {acc_base:.1f}%")
    print(f"  With stabilizer: {acc_stab:.1f}%  (Δ = {acc_stab - acc_base:+.1f}%)")
