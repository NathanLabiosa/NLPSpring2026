"""
viability_check.py — Gemma-2-9B base viability check (Step A1).

This is a BASE model (not instruction-tuned), so we use few-shot prompting
with 4 GSM8K exemplars with chain-of-thought instead of chat templates.

Gate: >= 50% accuracy on 50 clean GSM8K examples to proceed.
      40-50%: proceed with caution (need >= 150 clean-correct from n=500).
      < 40%: stop and report.
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

# 4-shot exemplars for GSM8K with chain-of-thought
FEW_SHOT_EXAMPLES = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.
#### 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
#### 5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
#### 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.
#### 8

"""


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
    # Fallback: last number
    nums = re.findall(r"[\-\d,]+\.?\d*", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def extract_truth(answer_text: str) -> str:
    m = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    return m.group(1).replace(",", "").strip() if m else ""


def main():
    model_id = "google/gemma-2-9b"
    n_samples = 50
    gate_high = 50.0
    gate_low = 40.0

    print(f"=== Gemma-2-9B Base Viability Check (A1) ===")
    print(f"Samples: {n_samples}, Gate: >={gate_high}% proceed, <{gate_low}% stop\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
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
        # Few-shot prompt (no chat template for base model)
        prompt = FEW_SHOT_EXAMPLES + f"Q: {item['question']}\nA:"

        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=2048).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_ids = out_ids[0, enc["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)

        # Stop at next "Q:" (few-shot model may continue generating)
        if "Q:" in response:
            response = response[:response.index("Q:")]

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
            if i < 3:
                print(f"       Response (last 150): ...{response[-150:]}")

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n_samples}  running acc={correct/(i+1)*100:.1f}%")

    acc = correct / n_samples * 100

    if acc >= gate_high:
        decision = "PROCEED"
    elif acc >= gate_low:
        expected_clean = int(500 * acc / 100)
        decision = f"CAUTION — expect ~{expected_clean} clean-correct from n=500 (need >=150)"
    else:
        decision = "STOP — model not viable for GSM8K"

    print(f"\n{'='*60}")
    print(f"RESULT: {correct}/{n_samples} = {acc:.1f}%")
    print(f"DECISION: {decision}")
    print(f"{'='*60}")

    os.makedirs("lrd_results", exist_ok=True)
    out = {
        "model": model_id,
        "step": "A1_viability",
        "n_samples": n_samples,
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "gate_high": gate_high,
        "gate_low": gate_low,
        "decision": decision,
        "timestamp": datetime.now().isoformat(),
        "prompt_type": "few-shot (4 exemplars, no chat template)",
        "details": results,
    }
    out_path = "lrd_results/gemma2_9b_viability.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")

    if acc < gate_low:
        vpath = "lrd_results/gemma2_9b_viability.txt"
        with open(vpath, "w") as f:
            f.write(f"Model: {model_id}\n")
            f.write(f"Clean accuracy: {acc:.1f}% ({correct}/{n_samples})\n")
            f.write(f"Gate: {gate_low}% — FAILED\n")
            f.write(f"Reason: Base model accuracy too low for GSM8K pipeline.\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        print(f"Viability report: {vpath}")


if __name__ == "__main__":
    main()
