from datasets import load_dataset, concatenate_datasets
import random

class DatasetManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load_and_format(self, dataset_name, perturbation_func=None, split="test"):
        """
        Loads a dataset and applies the correct prompt template + perturbations.
        
        Args:
            dataset_name (str): One of 'humaneval', 'gsm8k', 'mmlu', 'bbh', 'arc', 'squad'
            perturbation_func (callable): Function to apply noise to the input text.
            split (str): Dataset split to load (default: 'test' or 'validation' depending on dataset).
        """
        dataset_name = dataset_name.lower()

        if dataset_name == "humaneval":
            return self._setup_humaneval(perturbation_func)
        elif dataset_name == "gsm8k":
            return self._setup_gsm8k(perturbation_func)
        elif dataset_name == "mmlu":
            return self._setup_mmlu(perturbation_func)
        elif dataset_name == "bbh":
            return self._setup_bbh(perturbation_func)
        elif dataset_name == "arc":
            return self._setup_arc(perturbation_func)
        elif dataset_name == "squad":
            return self._setup_squad(perturbation_func)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported.")

    # --- 1. HumanEval (Code Completion) ---
    def _setup_humaneval(self, perturbation_func):
        # HumanEval only has a 'test' split
        dataset = load_dataset("openai_humaneval", split="test")
        
        def format_fn(sample):
            prompt_text = sample['prompt']
            if perturbation_func:
                prompt_text = perturbation_func(prompt_text)
                
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant. Complete the Python function. Output ONLY the code within markdown code blocks."},
                {"role": "user", "content": prompt_text}
            ]
            return {"formatted_prompt": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}
            
        return dataset.map(format_fn)

    # --- 2. GSM8K (Math Chain-of-Thought) ---
    def _setup_gsm8k(self, perturbation_func):
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        
        def format_fn(sample):
            prompt_text = sample['question']
            if perturbation_func:
                prompt_text = perturbation_func(prompt_text)

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step. The last line must be '#### ANSWER'."},
                {"role": "user", "content": prompt_text}
            ]
            return {"formatted_prompt": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}
            
        return dataset.map(format_fn)

    # --- 3. MMLU (Multiple Choice Knowledge) ---
    def _setup_mmlu(self, perturbation_func):
        # The list of specific subsets you requested
        target_subsets = [
            "college_computer_science",
            "college_mathematics",
            "college_physics",
            "electrical_engineering", 
            "abstract_algebra",
            "machine_learning",
            "philosophy",
            "high_school_european_history",
            "professional_law",
            "business_ethics"
        ]
        
        dataset_list = []
        print(f"Loading {len(target_subsets)} MMLU subsets...")
        
        # 1. Load each subset individually
        for sub in target_subsets:
            try:
                # MMLU usually has 'test' split. 
                # We use 'cais/mmlu' (the official Hugging Face path)
                ds = load_dataset("cais/mmlu", sub, split="test")
                dataset_list.append(ds)
            except Exception as e:
                print(f"Warning: Could not load MMLU subset '{sub}': {e}")

        # 2. Combine them into one big dataset
        if not dataset_list:
            raise ValueError("No MMLU subsets were loaded successfully.")
        
        combined_dataset = concatenate_datasets(dataset_list)
        print(f"Combined MMLU Size: {len(combined_dataset)} examples")

        # 3. Define Formatting (Same as before)
        def format_fn(sample):
            question = sample['question']
            choices = sample['choices'] # List of strings
            
            # Apply perturbation to the QUESTION only
            if perturbation_func:
                question = perturbation_func(question)

            # Format: Question + Options
            formatted_input = f"{question}\n"
            options = ["A", "B", "C", "D"]
            for i, choice in enumerate(choices):
                formatted_input += f"{options[i]}. {choice}\n"
            formatted_input += "Answer:"

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Choose the correct answer (A, B, C, or D) for the multiple choice question."},
                {"role": "user", "content": formatted_input}
            ]
            return {"formatted_prompt": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}
            
        return combined_dataset.map(format_fn)

    # --- 4. BBH (Logical Reasoning) ---
    def _setup_bbh(self, perturbation_func):
        # Using 'logical_deduction_seven_objects' as the representative task
        dataset = load_dataset("lukaemon/bbh", "logical_deduction_seven_objects", split="test")
        
        def format_fn(sample):
            input_text = sample['input']
            if perturbation_func:
                input_text = perturbation_func(input_text)
            
            # BBH often works best with a strict CoT instruction or just "Answer:"
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Think step by step and then provide the final answer."},
                {"role": "user", "content": f"{input_text}\nAnswer:"}
            ]
            return {"formatted_prompt": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}
            
        return dataset.map(format_fn)

    # --- 5. ARC-Challenge (Science Reasoning) ---
    def _setup_arc(self, perturbation_func):
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        
        def format_fn(sample):
            question = sample['question']
            choices = sample['choices'] # Dict with 'text' and 'label' lists
            
            if perturbation_func:
                question = perturbation_func(question)

            formatted_input = f"{question}\n"
            for label, text in zip(choices['label'], choices['text']):
                formatted_input += f"{label}. {text}\n"
            formatted_input += "Answer:"

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Choose the correct answer from the options provided."},
                {"role": "user", "content": formatted_input}
            ]
            return {"formatted_prompt": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}

        return dataset.map(format_fn)

    # --- 6. SQuAD v2 (Reading Comprehension) ---
    def _setup_squad(self, perturbation_func):
        # SQuAD uses 'validation' for evaluation usually (test is hidden)
        dataset = load_dataset("rajpurkar/squad_v2", split="validation")
        
        def format_fn(sample):
            context = sample['context']
            question = sample['question']
            
            # NOTE: For SQuAD, you might want to perturb the CONTEXT (to simulate bad documents) 
            # or the QUESTION (to simulate bad user queries). 
            # Here we perturb the QUESTION to stay consistent with other tasks.
            if perturbation_func:
                question = perturbation_func(question)

            prompt_content = f"Context: {context}\n\nQuestion: {question}"

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question based ONLY on the context provided. If the question cannot be answered from the context, respond with 'unanswerable'."},
                {"role": "user", "content": prompt_content}
            ]
            return {"formatted_prompt": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}

        return dataset.map(format_fn)