from datasets import load_dataset, concatenate_datasets
import random

# Few-shot template for base models without chat template (matches train_lrd_stabilizer.py)
GSM8K_FEW_SHOT = """\
Solve each math problem step by step. The final answer must be on the last line as '#### NUMBER'.

Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day at the farmers' market?
Answer: She eats 3 eggs and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left to sell. She earns 9 × $2 = $18.
#### 18

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: White fiber needed is 2 / 2 = 1 bolt. Total is 2 + 1 = 3 bolts.
#### 3

Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Answer: The value increased by 80,000 × 1.5 = $120,000. New value is 80,000 + 120,000 = $200,000. Total cost was 80,000 + 50,000 = $130,000. Profit is 200,000 - 130,000 = $70,000.
#### 70000

Question: {question}
Answer:"""

# BBH has a bunch of subtasks, we're using all of them
BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two",
    "navigate", "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]


class DatasetManager:
    def __init__(self, tokenizer, use_system_prompt=True, use_chat_template=True):
        self.tokenizer = tokenizer
        self.use_system_prompt = use_system_prompt
        self.use_chat_template = use_chat_template

    def _format_prompt(self, messages):
        """Format messages into a prompt string, with fallback for models without chat_template."""
        if getattr(self.tokenizer, "chat_template", None) is not None:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Fallback for base models without chat template
        parts = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant:")
        return "".join(parts)

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

    # HumanEval (Code Completion)
    def _setup_humaneval(self, perturbation_func):
        # HumanEval only has a 'test' split
        dataset = load_dataset("openai_humaneval", split="test")
        # print(f"loaded humaneval: {len(dataset)} examples")

        def format_fn(sample):
            prompt_text = sample['prompt']
            if perturbation_func:
                prompt_text = perturbation_func(prompt_text)

            messages = [
                {"role": "system", "content": "You are a helpful coding assistant. Complete the Python function. Output ONLY the code within markdown code blocks."},
                {"role": "user", "content": prompt_text}
            ]
            return {"formatted_prompt": self._format_prompt(messages)}

        return dataset.map(format_fn)

    # GSM8K (Math Chain-of-Thought)
    def _setup_gsm8k(self, perturbation_func):
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        # print(f"gsm8k: {len(dataset)} examples")

        # Check if model has chat template - if not, use few-shot prompting
        has_chat_template = getattr(self.tokenizer, "chat_template", None) is not None

        def format_fn(sample):
            prompt_text = sample['question']
            if perturbation_func:
                prompt_text = perturbation_func(prompt_text)

            if has_chat_template:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step. The last line must be '#### ANSWER'."},
                    {"role": "user", "content": prompt_text}
                ]
                return {"formatted_prompt": self._format_prompt(messages)}
            else:
                # Use few-shot template for base models
                return {"formatted_prompt": GSM8K_FEW_SHOT.format(question=prompt_text)}

        return dataset.map(format_fn)

    # MMLU (Multiple Choice Knowledge)
    def _setup_mmlu(self, perturbation_func):
        # we're only using a subset of MMLU for computational reasons
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

        # load each subset individually
        for sub in target_subsets:
            try:
                # MMLU usually has 'test' split
                # we use 'cais/mmlu' (the official Hugging Face path)
                ds = load_dataset("cais/mmlu", sub, split="test")
                dataset_list.append(ds)
            except Exception as e:
                print(f"Warning: Could not load MMLU subset '{sub}': {e}")

        # combine them into one big dataset
        if not dataset_list:
            raise ValueError("No MMLU subsets were loaded successfully.")

        combined_dataset = concatenate_datasets(dataset_list)
        print(f"Combined MMLU Size: {len(combined_dataset)} examples")

        # format the multiple choice prompts
        def format_fn(sample):
            question = sample['question']
            choices = sample['choices']  # list of strings

            # apply perturbation to the QUESTION only (not the choices)
            if perturbation_func:
                question = perturbation_func(question)

            # format: Question + Options
            formatted_input = f"{question}\n"
            options = ["A", "B", "C", "D"]
            for i, choice in enumerate(choices):
                formatted_input += f"{options[i]}. {choice}\n"
            formatted_input += "Answer:"

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Choose the correct answer (A, B, C, or D) for the multiple choice question."},
                {"role": "user", "content": formatted_input}
            ]
            return {"formatted_prompt": self._format_prompt(messages)}

        return combined_dataset.map(format_fn)


    # BBH (Logical Reasoning)
    def _setup_bbh(self, perturbation_func):
        from datasets import load_dataset, concatenate_datasets

        subtask_datasets = []
        for task in BBH_TASKS:
            try:
                ds = load_dataset("lukaemon/bbh", task, split="test")
                # add task name so we can track which task each example is from
                ds = ds.map(lambda x, t=task: {**x, "bbh_task": t})
                subtask_datasets.append(ds)
            except Exception as e:
                print(f"  [warn] Could not load BBH subtask {task}: {e}")

        if not subtask_datasets:
            raise RuntimeError("No BBH subtasks could be loaded.")

        # shuffle before pre-filtering so clean-correct examples are drawn
        # proportionally across tasks rather than exhausting one task first
        combined = concatenate_datasets(subtask_datasets)
        combined = combined.shuffle(seed=42)  # fixed seed for reproducibility
        # print(f"bbh combined: {len(combined)} examples")

        def format_fn(sample):
            input_text = sample['input']
            if perturbation_func:
                input_text = perturbation_func(input_text)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Think step by step and then provide the final answer."
                },
                {
                    "role": "user",
                    "content": f"{input_text}\nAnswer:"
                }
            ]
            return {"formatted_prompt": self._format_prompt(messages)}

        return combined.map(format_fn)


    # ARC-Challenge (Science Reasoning)
    def _setup_arc(self, perturbation_func):
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        # print(f"arc: {len(dataset)} examples")

        def format_fn(sample):
            question = sample['question']
            choices = sample['choices']  # dict with 'text' and 'label' lists

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
            return {"formatted_prompt": self._format_prompt(messages)}

        return dataset.map(format_fn)

    # SQuAD v2 (Reading Comprehension)
    def _setup_squad(self, perturbation_func):
        # SQuAD uses 'validation' for evaluation (test is hidden)
        dataset = load_dataset("rajpurkar/squad_v2", split="validation")
        # print(f"squad: {len(dataset)} examples")

        def format_fn(sample):
            context = sample['context']
            question = sample['question']

            # NOTE: we perturb the QUESTION to stay consistent with other tasks
            # (could also perturb context to simulate noisy documents)
            if perturbation_func:
                question = perturbation_func(question)

            prompt_content = f"Context: {context}\n\nQuestion: {question}"

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question based ONLY on the context provided. If the question cannot be answered from the context, respond with 'unanswerable'."},
                {"role": "user", "content": prompt_content}
            ]
            return {"formatted_prompt": self._format_prompt(messages)}

        return dataset.map(format_fn)
