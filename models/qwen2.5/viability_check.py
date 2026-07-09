"""
viability_check.py — Qwen2.5-7B-Instruct viability validation (Step B0).

Runs 50 clean GSM8K examples using the same pipeline as main.py to confirm accuracy >= 70%.
"""

import sys
import os
import json
import re
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import evaluate_gsm8k_entry


def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    n_samples = 50
    gate_accuracy = 70.0

    print(f"=== Qwen2.5-7B-Instruct Viability Check (B0) ===")
    print(f"Samples: {n_samples}, Gate: {gate_accuracy}%\n")

    # Load model exactly as main.py / model_loader.py does
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Generation settings matching main.py
    gen_kwargs = {
        "max_new_tokens": 600,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "return_full_text": False,
    }

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds)[:n_samples]

    correct = 0
    results = []

    for i, item in enumerate(items):
        # Same prompt format as data_loader.py _setup_gsm8k
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step. The last line must be '#### ANSWER'."},
            {"role": "user", "content": item["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        out = pipe(prompt, **gen_kwargs)
        generated_text = out[0]["generated_text"]

        # Use the same evaluator as main.py
        is_correct = evaluate_gsm8k_entry(generated_text, item)
        correct += int(is_correct)

        # Extract for logging
        pred_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", generated_text)
        extracted = pred_match.group(1).replace(",", "") if pred_match else "(no ####)"
        truth_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", item["answer"])
        truth = truth_match.group(1).replace(",", "") if truth_match else ""

        results.append({
            "idx": i, "truth": truth, "extracted": extracted,
            "correct": is_correct,
        })

        if i < 5 or not is_correct:
            status = "CORRECT" if is_correct else "WRONG"
            print(f"  [{i:2d}] {status}  truth={truth}  extracted={extracted}")

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n_samples}  running acc={correct/(i+1)*100:.1f}%")

    acc = correct / n_samples * 100
    passed = acc >= gate_accuracy

    print(f"\n{'='*50}")
    print(f"RESULT: {correct}/{n_samples} = {acc:.1f}%")
    print(f"GATE ({gate_accuracy}%): {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*50}")

    os.makedirs("lrd_results", exist_ok=True)
    out = {
        "model": model_id,
        "step": "B0_viability",
        "n_samples": n_samples,
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "gate_threshold": gate_accuracy,
        "passed": passed,
        "timestamp": datetime.now().isoformat(),
        "details": results,
    }
    out_path = "lrd_results/qwen_viability.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")

    if not passed:
        vpath = "lrd_results/qwen25_7b_viability.txt"
        with open(vpath, "w") as f:
            f.write(f"Model: {model_id}\n")
            f.write(f"Clean accuracy: {acc:.1f}% ({correct}/{n_samples})\n")
            f.write(f"Gate: {gate_accuracy}% — FAILED\n")
            f.write(f"Reason: Accuracy below threshold after bug fix attempt.\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        print(f"Viability report: {vpath}")


if __name__ == "__main__":
    main()
