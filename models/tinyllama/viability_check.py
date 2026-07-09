"""
viability_check.py — TinyLlama-1.1B-Chat viability check (Step C1).

Gate: >= 40% accuracy on 50 clean GSM8K examples to proceed.
      < 40%: STOP. Report and move on.
"""

import sys
import os
import json
import re
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_gsm8k_answer(text: str) -> str:
    """Try multiple answer extraction patterns."""
    patterns = [
        r"####\s*([\-\d,\.]+)",
        r"[Tt]he answer is\s*([\-\d,\.]+)",
        r"\\boxed\{([\-\d,\.]+)\}",
        r"=\s*([\-\d,\.]+)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\-\d,]+\.?\d*", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def extract_truth(answer_text: str) -> str:
    m = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    return m.group(1).replace(",", "").strip() if m else ""


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    n_samples = 50
    gate_accuracy = 40.0

    print(f"=== TinyLlama-1.1B-Chat Viability Check (C1) ===")
    print(f"Samples: {n_samples}, Gate: >={gate_accuracy}%\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds)[:n_samples]

    correct = 0
    results = []

    for i, item in enumerate(items):
        # TinyLlama-Chat uses standard chat template
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step. End with #### followed by the numerical answer."},
            {"role": "user", "content": item["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=1024).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_ids = out_ids[0, enc["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        extracted = extract_gsm8k_answer(response)
        truth = extract_truth(item["answer"])
        is_correct = (extracted == truth)
        correct += int(is_correct)

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
    print(f"GATE ({gate_accuracy}%): {'PASSED — proceed with pipeline' if passed else 'FAILED — STOP, model not viable'}")
    print(f"{'='*50}")

    os.makedirs("lrd_results", exist_ok=True)
    out = {
        "model": model_id,
        "step": "C1_viability",
        "n_samples": n_samples,
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "gate_threshold": gate_accuracy,
        "passed": passed,
        "timestamp": datetime.now().isoformat(),
        "details": results,
    }
    out_path = "lrd_results/tinyllama_viability.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")

    if not passed:
        vpath = "lrd_results/tinyllama_1b_viability.txt"
        with open(vpath, "w") as f:
            f.write(f"Model: {model_id}\n")
            f.write(f"Clean accuracy: {acc:.1f}% ({correct}/{n_samples})\n")
            f.write(f"Gate: {gate_accuracy}% — FAILED\n")
            f.write(f"Reason: Model accuracy too low for GSM8K pipeline.\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        print(f"Viability report: {vpath}")


if __name__ == "__main__":
    main()
