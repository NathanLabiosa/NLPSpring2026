"""Diagnose whether the Qwen LoRA adapter actually perturbs hidden states.

Three independent tests:
  A) Direct comparison: load base; load PeftModel; compare hidden states with
     adapter disabled vs enabled, on the same input.
  B) Module-level: walk peft_model and check that LoRA submodules were
     attached, that their lora_A/lora_B weights are non-zero, and that the
     wrapped Linear modules are reachable via the expected name path.
  C) Forward-with-merge: merge the adapter into the base weights
     (peft_model.merge_and_unload()) and compare to the disabled-adapter pass.
"""
import os, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = '/home1/labiosa/NLPSpring2026'
model_id = 'Qwen/Qwen2.5-7B-Instruct'
adapter = os.path.join(ROOT, 'stabilizer_weights/qwen_sweep_L24_27/lora_final')

print(f'PEFT version: ', end=''); import peft; print(peft.__version__)
print(f'transformers version: ', end=''); import transformers; print(transformers.__version__)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map={"": 0}, torch_dtype=torch.float32,
    trust_remote_code=True, attn_implementation='eager',
)
base_model.eval()

# Sanity: check that base_model contains a layer 24 q_proj module.
q24 = base_model.model.layers[24].self_attn.q_proj
print(f'\nbase model layer 24 q_proj: {type(q24).__name__}  weight shape={list(q24.weight.shape)}  dtype={q24.weight.dtype}')

peft_model = PeftModel.from_pretrained(base_model, adapter)
peft_model.eval()

# === Test B: module-level inspection ===
print('\n=== Test B: LoRA modules attached? ===')
lora_modules = []
for n, m in peft_model.named_modules():
    if 'lora_A' in n or 'lora_B' in n:
        lora_modules.append((n, type(m).__name__))
print(f'  found {len(lora_modules)} lora_A/lora_B submodules. Sample:')
for n, t in lora_modules[:4]:
    print(f'    {n}  ({t})')

# Inspect the actual wrapped q_proj at layer 24
q24w = peft_model.base_model.model.model.layers[24].self_attn.q_proj
print(f'  wrapped layer 24 q_proj: {type(q24w).__name__}  has_lora={"lora_A" in dir(q24w)}')
if hasattr(q24w, 'lora_A'):
    for ad_name, ad_mod in q24w.lora_A.items():
        print(f'    adapter "{ad_name}" lora_A.weight max_abs={ad_mod.weight.abs().max().item():.4e}')
    for ad_name, ad_mod in q24w.lora_B.items():
        print(f'    adapter "{ad_name}" lora_B.weight max_abs={ad_mod.weight.abs().max().item():.4e}')
    print(f'    active_adapter: {q24w.active_adapter}')

# === Test A: forward pass diff ===
print('\n=== Test A: hidden state delta with adapter ON vs OFF ===')
messages = [
    {"role": "system", "content": "Solve the math problem step by step."},
    {"role": "user", "content": "Janet has 24 ducks. They lay 16 eggs per day. She eats 3 for breakfast. How many remain?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tokenizer(prompt, return_tensors='pt', padding=True).to('cuda')

with torch.no_grad():
    with peft_model.disable_adapter():
        out_base = peft_model(**enc, output_hidden_states=True)
    h_base = [hs.detach().clone() for hs in out_base.hidden_states]
    out_lora = peft_model(**enc, output_hidden_states=True)
    h_lora = [hs.detach().clone() for hs in out_lora.hidden_states]

print(f'  num hidden states: {len(h_base)}')
for l in [0, 5, 10, 20, 23, 24, 25, 26, 27, 28, len(h_base)-1]:
    if l >= len(h_base): continue
    hb = h_base[l].float()
    hl = h_lora[l].float()
    diff = (hb - hl).abs().max().item()
    bitsame = torch.equal(hb, hl)
    print(f'    layer {l:2d}: max_abs_diff={diff:.4e}  bit-identical={bitsame}')

# === Test C: merge then compare ===
print('\n=== Test C: merge_and_unload diff vs disabled-adapter pass ===')
# Need a fresh base_model since merge modifies in-place.
del peft_model
torch.cuda.empty_cache()
base2 = AutoModelForCausalLM.from_pretrained(
    model_id, device_map={"": 0}, torch_dtype=torch.float32,
    trust_remote_code=True, attn_implementation='eager',
)
base2.eval()
pm2 = PeftModel.from_pretrained(base2, adapter)
pm2.eval()
merged = pm2.merge_and_unload()
merged.eval()

with torch.no_grad():
    out_merged = merged(**enc, output_hidden_states=True)
    h_merged = [hs.detach().clone() for hs in out_merged.hidden_states]

for l in [0, 5, 23, 24, 25, 26, 27, 28, len(h_base)-1]:
    if l >= len(h_merged): continue
    hb = h_base[l].float()
    hm = h_merged[l].float()
    diff = (hb - hm).abs().max().item()
    print(f'    layer {l:2d}: max_abs_diff (base vs merged) = {diff:.4e}')
