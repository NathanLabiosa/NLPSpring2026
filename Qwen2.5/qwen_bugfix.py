"""
qwen_bugfix.py — Diagnose and fix Qwen2.5-7B-Instruct low accuracy (19% → expected ~80%).

Priority 0 from experiment_plan.md. Checks:
  1. Chat template (ChatML format)
  2. Answer extraction regex
  3. System prompt handling
  4. Generation parameters

Runs 10 examples with verbose output so you can inspect what's happening.
"""

import sys
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Phi3.5"))

from perturbations import PerturbationEngine


def extract_gsm8k_answer(text: str) -> str:
    """Try multiple answer extraction patterns for GSM8K."""
    patterns = [
        r"####\s*([\-\d,\.]+)",                    # Standard: #### 42
        r"[Tt]he answer is\s*([\-\d,\.]+)",        # "The answer is 42"
        r"\\boxed\{([\-\d,\.]+)\}",                # LaTeX boxed
        r"=\s*([\-\d,\.]+)\s*$",                   # Trailing "= 42"
        r"([\-\d,\.]+)\s*$",                        # Last number in text
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(",", "").strip()
    return ""


def extract_truth(answer_text: str) -> str:
    """Extract ground truth from GSM8K answer field."""
    m = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if m:
        return m.group(1).replace(",", "").strip()
    return ""


def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_id}...")

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
    items = list(ds)[:10]

    # ── Check 1: Inspect chat template ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 1: Chat template inspection")
    print("=" * 70)
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Has chat_template: {tokenizer.chat_template is not None}")
    if tokenizer.chat_template:
        print(f"Chat template (first 200 chars): {tokenizer.chat_template[:200]}")

    # ── Check 2: Test different prompt formats ────────────────────────────────
    test_q = items[0]["question"]
    truth = extract_truth(items[0]["answer"])
    print(f"\nTest question: {test_q[:100]}...")
    print(f"Ground truth: {truth}")

    formats = {}

    # Format A: ChatML with system prompt
    messages_a = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step. End your answer with #### followed by the numerical answer."},
        {"role": "user", "content": test_q},
    ]
    formats["ChatML+system"] = tokenizer.apply_chat_template(
        messages_a, tokenize=False, add_generation_prompt=True)

    # Format B: ChatML without system prompt
    messages_b = [
        {"role": "user", "content": test_q},
    ]
    formats["ChatML_no_system"] = tokenizer.apply_chat_template(
        messages_b, tokenize=False, add_generation_prompt=True)

    # Format C: ChatML with GSM8K few-shot style prompt
    messages_c = [
        {"role": "system", "content": "Solve the math problem. Show your work, then give the final answer after ####."},
        {"role": "user", "content": test_q},
    ]
    formats["ChatML+math_system"] = tokenizer.apply_chat_template(
        messages_c, tokenize=False, add_generation_prompt=True)

    # Format D: Raw text (no template)
    formats["raw_text"] = f"Q: {test_q}\nA: Let's solve this step by step.\n"

    for fmt_name, prompt in formats.items():
        print(f"\n--- Format: {fmt_name} ---")
        print(f"Prompt (first 300 chars):\n{prompt[:300]}")

    # ── Check 3: Generate with each format ────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 3: Generation with different formats and params")
    print("=" * 70)

    gen_configs = [
        {"max_new_tokens": 512, "do_sample": False, "temperature": 1.0},
        {"max_new_tokens": 512, "do_sample": True, "temperature": 0.7, "top_p": 0.9},
        {"max_new_tokens": 1024, "do_sample": False, "temperature": 1.0},
    ]

    best_format = None
    best_config = None
    best_acc = 0

    for fmt_name, prompt_template in [("ChatML+system", messages_a),
                                       ("ChatML_no_system", messages_b),
                                       ("ChatML+math_system", messages_c)]:
        for ci, gen_kw in enumerate(gen_configs):
            print(f"\n--- {fmt_name} / config {ci} ({gen_kw}) ---")
            correct = 0
            total = min(10, len(items))

            for idx in range(total):
                q = items[idx]["question"]
                t = extract_truth(items[idx]["answer"])

                if fmt_name == "ChatML+system":
                    msgs = [
                        {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step. End your answer with #### followed by the numerical answer."},
                        {"role": "user", "content": q},
                    ]
                elif fmt_name == "ChatML_no_system":
                    msgs = [{"role": "user", "content": q}]
                else:
                    msgs = [
                        {"role": "system", "content": "Solve the math problem. Show your work, then give the final answer after ####."},
                        {"role": "user", "content": q},
                    ]

                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)

                enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=1024).to(model.device)

                with torch.no_grad():
                    out_ids = model.generate(
                        **enc,
                        pad_token_id=tokenizer.pad_token_id,
                        **gen_kw,
                    )

                new_ids = out_ids[0, enc["input_ids"].shape[1]:]
                response = tokenizer.decode(new_ids, skip_special_tokens=True)
                extracted = extract_gsm8k_answer(response)
                is_correct = (extracted == t)
                correct += int(is_correct)

                if idx < 3:  # Print first 3 for inspection
                    print(f"\n  Example {idx}: truth={t}, extracted={extracted}, correct={is_correct}")
                    print(f"  Response (last 200 chars): ...{response[-200:]}")

            acc = correct / total * 100
            print(f"\n  Accuracy: {correct}/{total} = {acc:.0f}%")

            if acc > best_acc:
                best_acc = acc
                best_format = fmt_name
                best_config = ci

    print("\n" + "=" * 70)
    print(f"BEST: format={best_format}, config={best_config}, acc={best_acc:.0f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
