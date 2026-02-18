import re
import string

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def evaluate_humaneval_entry(generated_text, sample):
    """
    HumanEval Logic: Exec the code and run the test case.
    """
    # 1. Extract Code
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, generated_text, re.DOTALL)
    code_body = match.group(1) if match else generated_text

    # 2. Reconstruct Full Function
    # If the model only output the body, prepend the signature (prompt)
    if "def " not in code_body:
        full_code = sample['prompt'] + code_body
    else:
        full_code = code_body

    # 3. Execution Check
    try:
        exec_globals = {}
        # Pre-import common libraries used in HumanEval solutions
        exec("import math\nimport re\nfrom typing import List, Dict, Tuple, Optional, Any", exec_globals)
        
        # Run the generated solution
        exec(full_code, exec_globals)
        
        # Run the hidden test case provided by the dataset
        exec(sample['test'], exec_globals)
        
        # Verify the entry point
        exec(f"check({sample['entry_point']})", exec_globals)
        return True
    except Exception:
        return False

def evaluate_gsm8k_entry(generated_text, sample):
    """
    GSM8K Logic: Look for '####' token and extract the number immediately following it.
    """
    # 1. Parse the GROUND TRUTH (The dataset stores it as ".... #### 42")
    truth_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", sample['answer'])
    truth = truth_match.group(1).replace(',', '') if truth_match else None

    # 2. Parse the PREDICTION
    # The model might output "#### 42" or just "The answer is 42"
    # We prioritize looking for "####" if the model followed instructions
    pred_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", generated_text)
    if pred_match:
        pred = pred_match.group(1).replace(',', '')
    else:
        # Fallback: Find the LAST number in the text
        numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", generated_text)
        pred = numbers[-1].replace(',', '') if numbers else None

    # 3. Compare (Float comparison to handle 42 vs 42.0)
    if pred and truth:
        try:
            return float(pred) == float(truth)
        except ValueError:
            return False
    return False

def evaluate_multiple_choice_entry(generated_text, sample):
    """
    MMLU / ARC Logic: Look for the final answer letter (A, B, C, D).
    Input sample['answer'] is usually an integer (0-3) or letter (A-D).
    """
    # 1. Parse Ground Truth
    truth = sample['answer']
    # Convert index (0) to Letter (A) if necessary (common in MMLU)
    if isinstance(truth, int):
        truth = ["A", "B", "C", "D"][truth]
    
    # 2. Parse Prediction
    # Look for "Answer: A" or "(A)" or just "A" at the end
    # We scan the LAST few characters for a standalone letter
    match = re.search(r"Answer:\s*([A-D])", generated_text, re.IGNORECASE)
    if match:
        pred = match.group(1).upper()
    else:
        # Fallback: Look for the last capital letter A-D surrounded by spaces/punctuation
        matches = re.findall(r"\b([A-D])\b", generated_text.upper())
        pred = matches[-1] if matches else None

    return pred == truth

def evaluate_squad_entry(generated_text, sample):
    """
    SQuAD Logic: 'Exact Match' (EM) or 'Contains'. 
    We check if the normalized ground truth is IN the normalized prediction.
    """
    # SQuAD 'answers' is a dict: {'text': ['Answer1', 'Answer2'], ...}
    possible_truths = sample['answers']['text'] 
    
    # Is the question unanswerable?
    if not possible_truths:
        # If truth is empty, we check if model refused to answer (heuristic)
        return "unanswerable" in generated_text.lower() or "no answer" in generated_text.lower()

    # Normalization
    pred_norm = normalize_text(generated_text)
    
    # Check if ANY of the acceptable answers are in the prediction
    for truth in possible_truths:
        truth_norm = normalize_text(truth)
        if truth_norm in pred_norm:
            return True
            
    return False

def evaluate_bbh_entry(generated_text, sample):
    """
    BBH Logic: Often requires extracting the final answer from a CoT.
    The target is usually in sample['target'].
    """
    truth = normalize_text(sample['target'])
    pred = normalize_text(generated_text)
    
    # BBH is strict. Usually we check if the exact answer word is at the end.
    # Simple 'contains' is often enough for a robustness check.
    return truth in pred
