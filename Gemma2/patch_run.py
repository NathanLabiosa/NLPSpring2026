"""Activation patching run script for Gemma-2-9B base (Step A3)."""

import sys
import os

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHI_DIR = os.path.join(ROOT, "Phi3.5")
for p in [PHI_DIR, os.path.dirname(os.path.abspath(__file__))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from lrd_diagnostics import run_activation_patching

run_activation_patching(
    dataset_name      = "gsm8k",
    model_id          = "google/gemma-2-9b",
    n_pairs           = 100,
    output_dir        = "lrd_results/gemma2_9b_patching",
    perturbation_type = "typos",
    perturbation_rate = 0.05,
    max_new_tokens    = 512,
    run_sanity_checks = True,
    use_system_prompt = False,
    use_chat_template = False,
)
