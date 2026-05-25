from lrd_diagnostics import run_activation_patching

run_activation_patching(
    dataset_name      = "gsm8k",
    model_id          = "mistralai/Mistral-7B-Instruct-v0.3",
    n_pairs           = 100,
    output_dir        = "lrd_results/mistral_patching_nocache_v2",
    perturbation_type = "ocr",
    perturbation_rate = 0.05,
    max_new_tokens    = 128,
    run_sanity_checks = True,
    use_cache         = False,
)
