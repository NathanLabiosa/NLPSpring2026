"""Replicate compute_disruption_curves logic exactly, with logging per layer.

If the diff at layers 25-28 is non-zero in a single-example test (it was, in
diag_qwen_lora.py) but zero in the production cascade-disruption script,
the bug must be in the batched path. Test n_samples=4, batch_size=4, with
padding/masking on, exact chat-template format used by the script.
"""
import os, sys, torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import random

ROOT = '/home1/labiosa/NLPSpring2026'
model_id = 'Qwen/Qwen2.5-7B-Instruct'
adapter = os.path.join(ROOT, 'stabilizer_weights/qwen_sweep_L24_27/lora_final')
sys_prompt = "Solve the math problem step by step. The last line must be '#### ANSWER'."

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map={"": 0}, torch_dtype=torch.float32,
    trust_remote_code=True, attn_implementation='eager',
)
base_model.eval()
peft_model = PeftModel.from_pretrained(base_model, adapter)
peft_model.eval()

random.seed(42)
ds = load_dataset("openai/gsm8k", "main", split="test")
items = list(ds)
random.shuffle(items)
questions = [it["question"] for it in items[:4]]

def fmt(q):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_prompt},
         {"role": "user",   "content": q}],
        tokenize=False, add_generation_prompt=True
    )

prompts = [fmt(q) for q in questions]
enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
mask = enc['attention_mask']
print(f'batch input_ids shape: {enc["input_ids"].shape}  mask sum per row: {mask.sum(dim=1).tolist()}')

with torch.no_grad():
    with peft_model.disable_adapter():
        out_base = peft_model(**enc, output_hidden_states=True)
    h_base = [hs.detach().clone() for hs in out_base.hidden_states]
    del out_base
    out_lora = peft_model(**enc, output_hidden_states=True)
    h_lora = [hs.detach().clone() for hs in out_lora.hidden_states]
    del out_lora

print(f'\nnum hidden states: {len(h_base)}')
print(f'{"layer":>5}  {"max_abs_diff":>14}  {"bit_identical":>14}  {"cos_sim_mean":>14}  {"dist_after_nan_guard_mean":>30}')
for l in range(len(h_base)):
    if l not in [0, 5, 10, 20, 23, 24, 25, 26, 27, 28]:
        continue
    hb = h_base[l].float()
    hl = h_lora[l].float()
    T = min(hb.shape[1], hl.shape[1], mask.shape[1])
    hb = hb[:, :T, :]
    hl = hl[:, :T, :]
    m  = mask[:, :T].float()
    diff = (hb - hl).abs().max().item()
    bitsame = torch.equal(hb, hl)
    cos = F.cosine_similarity(hb, hl, dim=-1)
    dist = 1.0 - cos
    dist_guarded = torch.where(torch.isnan(dist), torch.zeros_like(dist), dist)
    cos_mean = cos.float().mean().item()
    denom = m.sum().item()
    if denom > 1e-8:
        per_layer_sum = ((dist_guarded * m).sum() / denom).item()
    else:
        per_layer_sum = float('nan')
    print(f'  {l:3d}   {diff:14.4e}  {str(bitsame):>14}  {cos_mean:14.6f}  {per_layer_sum:30.6e}')
