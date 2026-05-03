# Quick script to run activation patching standalone
from lrd_diagnostics import run_activation_patching

# print("starting patching")
run_activation_patching(
    dataset_name = "gsm8k",
    model_id = "Qwen/Qwen2.5-7B-Instruct",
    n_pairs = 100,  # takes about 30min on A100
    output_dir = "lrd_results",
    perturbation_type = "ocr",
    perturbation_rate = 0.05,
    max_new_tokens = 512,
    run_sanity_checks = True,  # identity + random controls
)
# print("done")
