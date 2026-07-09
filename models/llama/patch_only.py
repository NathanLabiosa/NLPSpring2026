from lrd_diagnostics import run_activation_patching

run_activation_patching(
    dataset_name      = "gsm8k",
    model_id          = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_pairs           = 100,
    output_dir        = "lrd_results",
    perturbation_type = "typos",
    perturbation_rate = 0.05,
    max_new_tokens    = 128,
    run_sanity_checks = True,
)
