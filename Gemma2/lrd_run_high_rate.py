"""LRD Diagnostic run script for Gemma-2-9B base with HIGH perturbation rates.

Doubles the default rates to push the model to more failures:
- Typos: 5% -> 10%
- OCR: 5% -> 10%
- Whitespace: 10% -> 20%
- Case: 10% -> 20%
- Homophones: 20% -> 40%
- Speech: 10% -> 20%
"""

import sys
import os

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHI_DIR = os.path.join(ROOT, "Phi3.5")
for p in [PHI_DIR, os.path.dirname(os.path.abspath(__file__))]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Patch the default experiments BEFORE importing run_diagnostics
import lrd_diagnostics
lrd_diagnostics.DEFAULT_EXPERIMENTS = [
    {"name": "Typos_10%",       "type": "typos",      "rate": 0.10},
    {"name": "OCR_10%",         "type": "ocr",        "rate": 0.10},
    {"name": "Whitespace_20%",  "type": "whitespace", "rate": 0.20},
    {"name": "Case_20%",        "type": "case",       "rate": 0.20},
    {"name": "Homophones_40%",  "type": "homophones", "rate": 0.40},
    {"name": "Speech_20%",      "type": "speech",     "rate": 0.20},
]

from lrd_diagnostics import run_diagnostics

run_diagnostics(
    dataset_name           = "gsm8k",
    model_id               = "google/gemma-2-9b",
    n_samples              = 500,
    output_dir             = "lrd_results/gemma2_9b_gsm8k_high_rate",
    max_new_tokens         = 512,
    run_exclusion_analysis = False,
    run_taxonomy_variance  = True,
    use_system_prompt      = False,
    use_chat_template      = False,
)
