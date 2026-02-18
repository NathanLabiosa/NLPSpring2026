import os
import torch
import argparse
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

# --- Custom Modules ---
from perturbations import PerturbationEngine
from data_loader import DatasetManager
from model_loader import ModelWrapper

# Import all specific evaluators from your new file
from evaluator import (
    evaluate_humaneval_entry,
    evaluate_gsm8k_entry,
    evaluate_multiple_choice_entry, 
    evaluate_squad_entry, 
    evaluate_bbh_entry
)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="humaneval, gsm8k, mmlu, etc.")
    parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    args = parser.parse_args()

    DATASET_NAME = args.dataset
    MODEL_ID = args.model
    OUTPUT_FILE = f"results_{DATASET_NAME}.txt"

    # 1. Initialize Components
    # ------------------------
    model_wrapper = ModelWrapper(MODEL_ID)
    pipe, tokenizer = model_wrapper.load()
    
    data_manager = DatasetManager(tokenizer)
    perturber = PerturbationEngine()

    # 2. Define Experiments
    # ---------------------
    experiments = [
        {"name": "Baseline",       "type": None,          "rate": 0.0},
        
        # Surface / Token Level
        {"name": "OCR_5%",         "type": "ocr",         "rate": 0.05},
        {"name": "Typos_5%",       "type": "typos",       "rate": 0.05},
        {"name": "Whitespace_10%", "type": "whitespace",  "rate": 0.10},
        
        # Semantic Level
        {"name": "Homophones_20%", "type": "homophones",  "rate": 0.20},
        {"name": "Speech_10%",     "type": "speech",      "rate": 0.10},
        
        # Internal Level (Latent Space)
        {"name": "Gaussian_0.05",  "type": "internal",    "rate": 0.05},
    ]

    results = []

    # 3. Experiment Loop
    # ------------------
    for exp in experiments:
        print(f"\n--- Running: {exp['name']} ---")
        
        # A. Handle Perturbation Logic
        p_func = None
        
        if exp['type'] == "internal":
            # CASE 1: Internal Noise (Gaussian)
            # We register the hook on the model wrapper
            model_wrapper.register_embedding_noise(exp['rate'])
            # No text perturbation needed
            p_func = None
            
        else:
            # CASE 2: Text Noise (Typos, OCR, etc.)
            # Ensure internal noise is OFF
            model_wrapper.register_embedding_noise(0.0)
            
            # Create the text perturbation function
            if exp['type']:
                p_func = lambda text: perturber.apply(text, exp['type'], exp['rate'])
            else:
                p_func = None # Baseline

        # B. Load & Format Data
        # We reload the dataset every time to ensure fresh samples
        dataset = data_manager.load_and_format(DATASET_NAME, perturbation_func=p_func)

        # C. Run Inference
        correct = 0
        total = 0
        
        # Generation settings (Greedy decoding for reproducibility)
        gen_kwargs = {
            "max_new_tokens": 600,
            "do_sample": False,
            "temperature": 0.0,
            "return_full_text": False
        }

        BATCH_SIZE = 8
        
        # D. Processing Loop
        for i, out in enumerate(tqdm(pipe(KeyDataset(dataset, "formatted_prompt"), batch_size=BATCH_SIZE, **gen_kwargs), total=len(dataset), mininterval=10.0)):
            generated_text = out[0]['generated_text']
            sample = dataset[i]
            
            # --- Dynamic Evaluator Dispatch ---
            is_correct = False
            
            if DATASET_NAME == "humaneval":
                is_correct = evaluate_humaneval_entry(generated_text, sample)
            
            elif DATASET_NAME == "gsm8k":
                is_correct = evaluate_gsm8k_entry(generated_text, sample)
            
            elif DATASET_NAME in ["mmlu", "arc"]:
                is_correct = evaluate_multiple_choice_entry(generated_text, sample)
            
            elif DATASET_NAME == "squad":
                is_correct = evaluate_squad_entry(generated_text, sample)
            
            elif DATASET_NAME == "bbh":
                is_correct = evaluate_bbh_entry(generated_text, sample)
            
            if is_correct: correct += 1
            total += 1

        # E. Record Results
        score = correct / total
        res_str = f"[{DATASET_NAME}] {exp['name']}: {correct}/{total} ({score:.2%})"
        print(f"Result -> {res_str}")
        results.append(res_str)

    # 4. Save Final Report
    # --------------------
    print("\n" + "="*30)
    print("FINAL RESULTS SUMMARY")
    print("="*30)
    with open(OUTPUT_FILE, "a") as f: # Append mode to not overwrite previous runs if you run multiple datasets
        f.write(f"\n--- Run for {DATASET_NAME} ---\n")
        for line in results:
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    main()
