import torch
import random
import string
import re
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

# --- Configuration ---
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_FILE = "results_humaneval_all_perturbations.txt"

# --- 1. Define Perturbation Functions ---

def add_typos(text, error_rate=0.05):
    """5% chance per character to swap, delete, insert, or replace."""
    if error_rate == 0: return text
    chars = list(text)
    for i in range(len(chars) - 1, -1, -1):
        if chars[i].isdigit() or chars[i] in string.whitespace:
            continue
        if random.random() < error_rate:
            r = random.random()
            if r < 0.25 and i < len(chars)-1 and not chars[i+1].isdigit():
                chars[i], chars[i+1] = chars[i+1], chars[i] 
            elif r < 0.50:
                del chars[i] 
            elif r < 0.75:
                chars[i] = random.choice(string.ascii_lowercase)
            else:
                chars.insert(i, random.choice(string.ascii_lowercase))
    return "".join(chars)

def add_homophone_swaps(text, error_rate=0.2): # Renamed arg to error_rate for consistency
    """Replaces words with homophones (e.g., their -> there)."""
    HOMOPHONES = {
        "their": ["there", "they're"], "there": ["their", "they're"], "they're": ["their", "there"],
        "your": ["you're"], "you're": ["your"],
        "its": ["it's"], "it's": ["its"],
        "to": ["too", "two"], "too": ["to", "two"], "two": ["to", "too"],
        "then": ["than"], "than": ["then"],
        "weather": ["whether"], "whether": ["weather"],
        "write": ["right"], "right": ["write"],
        "read": ["red"], "red": ["read"],
        "for": ["four"], "four": ["for"],
        "sun": ["son"], "son": ["sun"]
    }

    def replace_match(m):
        word = m.group(0)
        lower = word.lower()
        if lower not in HOMOPHONES: return word
        
        # Use error_rate as the probability to swap
        if random.random() >= error_rate: return word

        choice = random.choice(HOMOPHONES[lower])
        if word.isupper(): return choice.upper()
        if word[0].isupper(): return choice.capitalize()
        return choice

    return re.sub(r"\b[A-Za-z']+\b", replace_match, text)

def add_ocr_errors(text, error_rate=0.10):
    """Replaces characters with visual lookalikes (e.g., 0 -> O, 1 -> l)."""
    OCR_MAPPINGS = {
        'l': '1', '1': 'l', 'I': '1',
        'O': '0', '0': 'O',
        'S': '5', '5': 'S',
        'B': '8', '8': 'B',
        'Z': '2', '2': 'Z',
        'c': 'e', 'e': 'c',
        'o': 'a', 'a': 'o',
        'i': 'j', 'j': 'i',
        'm': 'n', 'n': 'm',
        'v': 'u', 'u': 'v',
        'F': 'P', 'P': 'F'
    }
    chars = list(text)
    for i in range(len(chars)):
        if chars[i] in OCR_MAPPINGS and random.random() < error_rate:
            chars[i] = OCR_MAPPINGS[chars[i]]
    return "".join(chars)

# --- 2. Main Logic ---

def main():
    
    # Force device map to GPU 0 for 4-bit stability
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Create Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id
    )

    # --- DEFINE EXPERIMENTS HERE ---
    PERTURBATIONS = [
        #{"name": "Baseline",   "func": None,                "rate": 0.0},
        #{"name": "Typos",      "func": add_typos,           "rate": 0.05},
        #{"name": "Homophones", "func": add_homophone_swaps, "rate": 0.20}, # 20% of keywords swapped
        {"name": "OCR",        "func": add_ocr_errors,      "rate": 0.05}  # 10% of characters swapped
    ]

    results_summary = []

    # Loop through each experiment
    for config in PERTURBATIONS:
        print(f"\n--- Running Experiment: {config['name']} (Rate: {config['rate']}) ---")
        
        # 1. Reload Dataset (Clean slate every time)
        dataset = load_dataset("openai_humaneval", split="test")

        # 2. Define Formatter for this specific config
        def format_sample(sample):
            prompt_text = sample['prompt']
            
            # Apply perturbation if function exists
            if config["func"]:
                prompt_text = config["func"](prompt_text, error_rate=config["rate"])

            messages = [
                {"role": "system", "content": "You are a helpful coding assistant. You will be provided with a function signature and docstring. Complete the Python function. Output ONLY the code within markdown code blocks. Do not add explanations or tests."},
                {"role": "user", "content": prompt_text}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return {"formatted_prompt": full_prompt}

        # 3. Apply Mapping
        dataset = dataset.map(format_sample)

        # 4. Run Inference
        gen_kwargs = {
            "max_new_tokens": 600, # Reduced slightly for speed, usually enough for HumanEval
            "do_sample": False,
            "temperature": 0.0,
            "return_full_text": False
        }

        correct = 0
        total = 0
        BATCH_SIZE = 8

        # Using tqdm with mininterval to avoid log spam on Slurm
        for i, out in enumerate(tqdm(pipe(KeyDataset(dataset, "formatted_prompt"), batch_size=BATCH_SIZE, **gen_kwargs), total=len(dataset), mininterval=10.0)):
            generated_text = out[0]['generated_text']
            
            # Extract Code
            pattern = r"```(?:python)?\n(.*?)```"
            match = re.search(pattern, generated_text, re.DOTALL)
            code_body = match.group(1) if match else generated_text

            # Construct Full Code
            if "def " not in code_body:
                # Use original prompt for reconstruction to avoid syntax errors in signature
                # But note: The model SAW the perturbed prompt, but we EXECUTE with the clean signature + generated body
                full_code = dataset[i]['prompt'] + code_body
            else:
                full_code = code_body

            # Execution Check
            try:
                exec_globals = {}
                exec("import math\nimport re\nfrom typing import List, Dict, Tuple, Optional, Any", exec_globals)
                exec(full_code, exec_globals)
                exec(dataset[i]['test'], exec_globals)
                exec(f"check({dataset[i]['entry_point']})", exec_globals)
                correct += 1
            except Exception:
                pass
            total += 1

        # 5. Record Results
        score = correct / total
        res_str = f"{config['name']}: {correct}/{total} ({score:.2%})"
        print(f"Result -> {res_str}")
        results_summary.append(res_str)

    # --- Final Report ---
    print("\n" + "="*30)
    print("FINAL RESULTS SUMMARY")
    print("="*30)
    with open(OUTPUT_FILE, "w") as f:
        for line in results_summary:
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    main()
