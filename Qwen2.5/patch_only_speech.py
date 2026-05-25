from lrd_diagnostics import run_activation_patching

# Experiment 4: Rerun Qwen2.5-7B activation patching with speech perturbation.
#
# The original OCR run hit a 60% identity ceiling due to BPE tokenization
# boundary misalignment (OCR substitutions split tokens differently).
# Speech perturbations largely preserve tokenization (phonetic substitutions
# are character-level but don't fragment tokens as severely as OCR glyphs).
#
# If identity recovery exceeds 80%, the causal window result is upgraded from
# exploratory to confirmed, strengthening the dissociability claim (Claim 4).
#
# Target: check identity recovery first. If >80%, run the full layer-wise profile.

run_activation_patching(
    dataset_name      = "gsm8k",
    model_id          = "Qwen/Qwen2.5-7B-Instruct",
    n_pairs           = 100,
    output_dir        = "lrd_results_speech",
    perturbation_type = "speech",
    perturbation_rate = 0.10,
    max_new_tokens    = 512,
    run_sanity_checks = True,
)
