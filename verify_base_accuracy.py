"""
verify_base_accuracy.py — Sanity check: load Phi-3.5 with NO adapter and eval on
GSM8K clean test set. Expected: ~79–82%. Confirms the base model and eval
pipeline are intact before launching v15.
"""
import random, sys, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lrd_stabilizer import (
    GSM8K_SYSTEM_PROMPT, format_question_prompt,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
)

MODEL = "microsoft/Phi-3.5-mini-instruct"
N     = 200
SEED  = 42

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading {MODEL} (no adapter) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Total params: {total_params:,}")

    print(f"\nLoading GSM8K test split ...")
    ds    = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds)
    rng   = random.Random(SEED)
    rng.shuffle(items)
    items = items[:N]
    print(f"Evaluating on {N} examples (seed={SEED})")

    gen_kwargs = dict(
        max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )

    correct = 0
    for i in range(0, len(items), 4):
        batch_items = items[i : i + 4]
        prompts = [format_question_prompt(tokenizer, it["question"]) for it in batch_items]
        truths  = [_extract_gsm8k_truth(it["answer"]) for it in batch_items]

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)
        new_ids = out_ids[:, enc["input_ids"].shape[1]:]
        preds   = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        for p, t in zip(preds, truths):
            correct += _gsm8k_correct(_extract_gsm8k_answer(p), t)

        if (i // 4 + 1) % 10 == 0:
            print(f"  {i+len(batch_items)}/{N} done, running acc: {correct/(i+len(batch_items))*100:.1f}%")

    acc = correct / N * 100
    print(f"\n{'='*40}")
    print(f"Base model clean accuracy: {acc:.1f}%  (n={N})")
    print(f"Expected: 79–82%")
    print(f"Status: {'OK' if acc >= 77 else 'BELOW EXPECTED — investigate before v15'}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
