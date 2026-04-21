import re
import string
import multiprocessing

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

def _humaneval_worker(full_code, test_code, entry_point, result_queue):
    """Runs the code in an isolated process to allow for hard timeouts."""
    try:
        exec_globals = {}
        exec("import math\nimport re\nfrom typing import List, Dict, Tuple, Optional, Any", exec_globals)
        exec(full_code, exec_globals)
        exec(test_code, exec_globals)
        exec(f"check({entry_point})", exec_globals)
        result_queue.put(True)
    except Exception:
        # any syntax error, assertion error, or runtime error means it failed
        result_queue.put(False)

def evaluate_humaneval_entry(generated_text, sample):
    """Specific evaluation logic for HumanEval with a hard timeout to prevent hangs."""
    # extract code from markdown if present
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, generated_text, re.DOTALL)
    code_body = match.group(1) if match else generated_text

    if "def " not in code_body:
        full_code = sample['prompt'] + code_body
    else:
        full_code = code_body

    # setup isolated process
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_humaneval_worker,
        args=(full_code, sample['test'], sample['entry_point'], result_queue)
    )

    # start and wait with a 5-second timeout
    p.start()
    p.join(timeout=5.0)

    # check if it hung
    if p.is_alive():
        p.terminate()  # kill the infinite loop
        p.join()
        return False  # timeout counts as a failure

    # get result if it finished cleanly
    if not result_queue.empty():
        return result_queue.get()

    return False

def evaluate_gsm8k_entry(generated_text, sample):
    """
    GSM8K Logic: Look for '####' token and extract the number immediately following it.
    """
    # parse the GROUND TRUTH (dataset stores it as ".... #### 42")
    truth_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", sample['answer'])
    truth = truth_match.group(1).replace(',', '') if truth_match else None

    # parse the PREDICTION
    # model might output "#### 42" or just "The answer is 42"
    # we prioritize looking for "####" if the model followed instructions
    pred_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", generated_text)
    if pred_match:
        pred = pred_match.group(1).replace(',', '')
    else:
        # fallback: find the LAST number in the text
        numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", generated_text)
        pred = numbers[-1].replace(',', '') if numbers else None

    # compare (float comparison to handle 42 vs 42.0)
    if pred and truth:
        try:
            return float(pred) == float(truth)
        except ValueError:
            return False
    return False

def evaluate_multiple_choice_entry(generated_text, sample):
    """
    MMLU / ARC Logic: Look for the final answer letter or number.
    Handles both MMLU (sample['answer'] as int) and ARC (sample['answerKey'] as str).
    """
    # parse ground truth based on dataset schema
    if 'answerKey' in sample:
        truth = sample['answerKey']  # ARC format
    else:
        truth = sample['answer']     # MMLU format

    # convert MMLU integer index (0, 1, 2, 3) to letter (A, B, C, D)
    if isinstance(truth, int):
        truth = ["A", "B", "C", "D"][truth]

    # ensure truth is a clean, uppercase string
    truth = str(truth).strip().upper()

    # parse prediction
    # look for "Answer: A" or "Answer: 1"
    match = re.search(r"Answer:\s*([A-D1-4])", generated_text, re.IGNORECASE)
    if match:
        pred = match.group(1).upper()
    else:
        # fallback: look for the last capital letter A-D or number 1-4
        matches = re.findall(r"\b([A-D1-4])\b", generated_text.upper())
        pred = matches[-1] if matches else None

    return pred == truth


def evaluate_squad_entry(generated_text, sample):
    """
    SQuAD Logic: 'Exact Match' (EM) or 'Contains'.
    We check if the normalized ground truth is IN the normalized prediction.
    """
    # SQuAD 'answers' is a dict: {'text': ['Answer1', 'Answer2'], ...}
    possible_truths = sample['answers']['text']

    # is the question unanswerable?
    if not possible_truths:
        # if truth is empty, check if model refused to answer (heuristic)
        return "unanswerable" in generated_text.lower() or "no answer" in generated_text.lower()

    # normalization
    pred_norm = normalize_text(generated_text)

    # check if ANY of the acceptable answers are in the prediction
    for truth in possible_truths:
        truth_norm = normalize_text(truth)
        if truth_norm in pred_norm:
            return True

    return False

def evaluate_bbh_entry(generated_text, sample):
    """
    BBH Logic: Handles multiple choice (A-G) and exact word matches
    by extracting the specific prediction.
    """
    truth = sample['target'].strip()

    # handle multiple choice formats (e.g., "(A)", "(B)")
    truth_letter_match = re.match(r"^\(([A-Z])\)$", truth, re.IGNORECASE)

    if truth_letter_match:
        truth_letter = truth_letter_match.group(1).upper()

        # try to find a formal answer declaration first
        match = re.search(r"Answer:\s*\(?([A-Z])\)?", generated_text, re.IGNORECASE)
        if match:
            pred = match.group(1).upper()
        else:
            # fallback: grab the last standalone capital letter in the text
            matches = re.findall(r"\b([A-Z])\b", generated_text.upper())
            pred = matches[-1] if matches else None

        return pred == truth_letter

    # handle text formats (e.g., "valid", "invalid")
    else:
        truth_clean = truth.lower()

        # extract just the words, discarding punctuation
        words = re.findall(r"\b\w+\b", generated_text.lower())

        # only look at the final 5 words of the output to avoid CoT leakage
        last_few_words = words[-5:] if words else []

        return truth_clean in last_few_words
