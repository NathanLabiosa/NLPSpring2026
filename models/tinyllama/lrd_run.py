"""LRD Diagnostic run script for TinyLlama-1.1B-Chat-v1.0."""

import sys
import os

# Pull in shared utilities from Phi3.5 directory (identical code, different model)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHI_DIR = os.path.join(ROOT, "Phi3.5")
for p in [PHI_DIR, os.path.dirname(os.path.abspath(__file__))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from lrd_diagnostics import run_diagnostics

run_diagnostics(
    dataset_name           = "gsm8k",
    model_id               = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_samples              = 500,
    output_dir             = "lrd_results/tinyllama_gsm8k",
    max_new_tokens         = 256,
    run_exclusion_analysis = False,
    run_taxonomy_variance  = True,
)
