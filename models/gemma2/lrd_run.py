"""LRD Diagnostic run script for Gemma-2-9B base (Step A2)."""

import sys
import os

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHI_DIR = os.path.join(ROOT, "Phi3.5")
for p in [PHI_DIR, os.path.dirname(os.path.abspath(__file__))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from lrd_diagnostics import run_diagnostics

run_diagnostics(
    dataset_name           = "gsm8k",
    model_id               = "google/gemma-2-9b",
    n_samples              = 500,
    output_dir             = "lrd_results/gemma2_9b_gsm8k",
    max_new_tokens         = 512,
    run_exclusion_analysis = False,
    run_taxonomy_variance  = True,
    use_system_prompt      = False,
    use_chat_template      = False,
)
