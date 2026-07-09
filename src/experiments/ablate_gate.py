# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
ablate_gate.py — Gate ablation for the trained LRD stabilizer.

Compares three modes on the same GSM8K test examples:
  1. No stabilizer         (pure perturbed baseline)
  2. Learned gate          (standard inference — alpha from gate_net)
  3. Gate forced to 1      (correction always fully applied, gate ablated)

Mode 3 tells us: did the correction MLP actually learn something useful,
independent of the gate controller's failures?

Usage:
    python ablate_gate.py
"""

import os, sys, re, json, random, torch
sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from stabilizer_system import MultiLayerStabilizerSystem, StabilizerConfig

# PerturbationEngine lives in models/phi3.5/perturbations.py
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "Phi3.5"))
from perturbations import PerturbationEngine

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID     = "microsoft/Phi-3.5-mini-instruct"
WEIGHTS_PATH = "./stabilizer_weights/phi35_gsm8k_lam50_v3/stabilizer_final.pt"
N_SAMPLES    = 100
DEVICE       = "cuda"
SEED         = 42

EVAL_CONDITIONS = [
    (None,          0.00),   # clean baseline
    ("typos",       0.05),
    ("ocr",         0.05),
    ("whitespace",  0.10),
    ("case",        0.10),
    ("speech",      0.10),
    ("homophones",  0.20),
]

GSM8K_SYSTEM_PROMPT = (
    "You are a math expert. Solve the problem step by step, "
    "then write your final answer after ####."
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def extract_truth(answer_str: str):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_str)
    return m.group(1).replace(",", "") if m else None


def extract_pred(text: str):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def is_correct(pred, truth):
    if pred is None or truth is None:
        return False
    try:
        return float(pred) == float(truth)
    except ValueError:
        return False


def run_generation(model, tokenizer, prompts, device, max_new_tokens=300):
    outputs = []
    for i in range(0, len(prompts), 4):
        batch = prompts[i:i+4]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = out_ids[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
    return outputs


def accuracy(preds, truths):
    return sum(is_correct(extract_pred(p), t)
               for p, t in zip(preds, truths)) / len(truths) * 100


# ── Gate-forced-to-1 hooks ────────────────────────────────────────────────────
def register_forced_gate_hooks(system) -> list:
    """
    Same as register_inference_hooks but alpha is clamped to 1.0,
    so the correction MLP is fully applied regardless of the gate controller.
    """
    system.stabilizers.eval()
    handles = []

    try:
        backbone_dtype = next(
            system._layers[system.config.inject_layers[0]].parameters()
        ).dtype
    except StopIteration:
        backbone_dtype = torch.float32

    for L in system.config.inject_layers:
        stab_inf = system.stabilizers[str(L)].to(backbone_dtype)

        def _make_hook(s):
            def _hook(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    h_norm = s.layer_norm(hs)
                    correction = s.mlp(h_norm)          # correction MLP only
                    h_corrected = hs + correction        # alpha = 1 always
                if isinstance(output, tuple):
                    return (h_corrected,) + output[1:]
                return h_corrected
            return _hook

        handles.append(system._layers[L].register_forward_hook(_make_hook(stab_inf)))

    return handles


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    print(f"Loading stabilizer weights from {WEIGHTS_PATH} ...")
    system = MultiLayerStabilizerSystem.load(model, WEIGHTS_PATH)
    system.stabilizers.eval()

    perturber = PerturbationEngine()

    print("Loading GSM8K test split ...")
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(test_ds)
    random.shuffle(items)
    items = items[:N_SAMPLES]
    truths = [extract_truth(x["answer"]) for x in items]

    results = {}

    print("\n" + "="*60)
    print("GATE ABLATION EVALUATION")
    print("="*60)
    header = f"{'Condition':<20} {'No stab':>8} {'Learned g':>10} {'Gate=1':>8}"
    print(header)
    print("-" * len(header))

    for method, rate in EVAL_CONDITIONS:
        cond = f"{method}_{int(rate*100)}pct" if rate else "clean_baseline"
        print(f"\n[{cond}] Building prompts ...", flush=True)

        # Build prompts
        prompts = []
        for item in items:
            q = perturber.apply(item["question"], method, rate) if rate else item["question"]
            prompts.append(format_prompt(tokenizer, q))

        # 1. No stabilizer
        print(f"[{cond}] Running: no stabilizer ...", flush=True)
        preds_base = run_generation(model, tokenizer, prompts, DEVICE)
        acc_base = accuracy(preds_base, truths)
        print(f"[{cond}]   no_stab = {acc_base:.1f}%", flush=True)

        # 2. Learned gate
        print(f"[{cond}] Running: learned gate ...", flush=True)
        hooks_learned = system.register_inference_hooks()
        preds_learned = run_generation(model, tokenizer, prompts, DEVICE)
        for h in hooks_learned: h.remove()
        acc_learned = accuracy(preds_learned, truths)
        print(f"[{cond}]   learned_gate = {acc_learned:.1f}%", flush=True)

        # 3. Gate forced to 1
        print(f"[{cond}] Running: gate forced=1 ...", flush=True)
        hooks_forced = register_forced_gate_hooks(system)
        preds_forced = run_generation(model, tokenizer, prompts, DEVICE)
        for h in hooks_forced: h.remove()
        acc_forced = accuracy(preds_forced, truths)
        print(f"[{cond}]   gate_forced_1 = {acc_forced:.1f}%", flush=True)

        results[cond] = {
            "no_stab":       acc_base,
            "learned_gate":  acc_learned,
            "gate_forced_1": acc_forced,
            "delta_learned": acc_learned - acc_base,
            "delta_forced":  acc_forced  - acc_base,
        }

        # Write partial results after each condition so progress is never lost
        out_path = os.path.join(os.path.dirname(WEIGHTS_PATH), "gate_ablation_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"{cond:<20} {acc_base:>7.1f}%  {acc_learned:>8.1f}%  {acc_forced:>6.1f}%", flush=True)

    out_path = os.path.join(os.path.dirname(WEIGHTS_PATH), "gate_ablation_results.json")
    print(f"\nAll done. Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
