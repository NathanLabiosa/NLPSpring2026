#!/usr/bin/env python3
"""Generate LRD heatmap figures for Gemma-2-9B and Qwen2.5-7B."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Apply publication styling
from figure_style import apply_style, remove_spines, COLORS
apply_style()


def load_lrd_data(json_path):
    """Load raw LRD data and compute mean profiles per perturbation."""
    with open(json_path) as f:
        data = json.load(f)

    profiles = {}
    for pert_name, entries in data.items():
        lrd_arrays = [np.array(e['lrd_profile']) for e in entries]
        profiles[pert_name] = {
            'mean': np.mean(lrd_arrays, axis=0),
            'std': np.std(lrd_arrays, axis=0),
            'n': len(entries)
        }
    return profiles


def plot_lrd_heatmap(profiles, model_name, n_layers, output_path, regime_desc):
    """Generate LRD heatmap with line plot overlay."""

    # Order perturbations consistently
    pert_order = ['Typos_5%', 'OCR_5%', 'Whitespace_10%', 'Case_10%', 'Homophones_20%', 'Speech_10%']
    pert_labels = ['Typos 5%', 'OCR 5%', 'Whitespace 10%', 'Case 10%', 'Homophones 20%', 'Speech 10%']

    # Filter to available perturbations
    available = [p for p in pert_order if p in profiles]
    labels = [pert_labels[pert_order.index(p)] for p in available]

    # Build heatmap matrix
    matrix = np.array([profiles[p]['mean'] for p in available])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1.5])

    # Use regime description as subplot title instead of figure suptitle
    # This makes the regime distinction visible at-a-glance

    # Top plot: Line plot of mean LRD per perturbation
    colors = plt.cm.tab10(np.linspace(0, 1, len(available)))
    for i, (pert, label) in enumerate(zip(available, labels)):
        mean = profiles[pert]['mean']
        ax1.plot(range(len(mean)), mean, label=label, color=colors[i], linewidth=2.5)

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean LRD')
    ax1.set_xlim(0, n_layers - 1)
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    # Subplot title with regime description
    ax1.set_title(f'{model_name}: {regime_desc}')
    remove_spines(ax1)

    # Bottom plot: Heatmap
    im = ax2.imshow(matrix, aspect='auto', cmap='YlOrRd',
                    extent=[0, n_layers, len(available)-0.5, -0.5])

    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Perturbation')
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_title('LRD Heatmap (mean across examples)')

    # Wider colorbar with readable labels
    cbar = fig.colorbar(im, ax=ax2, label='LRD (cosine distance)',
                        fraction=0.08, pad=0.02, shrink=0.9)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Create figures directory
    fig_dir = Path('/home1/labiosa/NLPSpring2026/figures')
    fig_dir.mkdir(exist_ok=True)

    # Gemma-2-9B (42 layers + 1 = 43 total)
    gemma_path = '/home1/labiosa/NLPSpring2026/Gemma2/lrd_results/gemma2_9b_gsm8k/raw_gsm8k.json'
    gemma_profiles = load_lrd_data(gemma_path)
    plot_lrd_heatmap(
        gemma_profiles,
        'Gemma-2-9B',
        n_layers=43,
        output_path=fig_dir / 'lrd_heatmap_gemma.pdf',
        regime_desc='spike-and-suppress'
    )

    # Qwen2.5-7B (28 layers + 1 = 29 total)
    qwen_path = '/home1/labiosa/NLPSpring2026/Qwen2.5/lrd_results/qwen_gsm8k/raw_gsm8k.json'
    qwen_profiles = load_lrd_data(qwen_path)
    plot_lrd_heatmap(
        qwen_profiles,
        'Qwen2.5-7B',
        n_layers=29,
        output_path=fig_dir / 'lrd_heatmap_qwen.pdf',
        regime_desc='late-accumulation'
    )

    # Mistral-7B (32 layers + 1 = 33 total)
    mistral_path = '/home1/labiosa/NLPSpring2026/Mistral/lrd_results/mistral_gsm8k/raw_gsm8k.json'
    mistral_profiles = load_lrd_data(mistral_path)
    plot_lrd_heatmap(
        mistral_profiles,
        'Mistral-7B',
        n_layers=33,
        output_path=fig_dir / 'lrd_heatmap_mistral.pdf',
        regime_desc='late-accumulation'
    )

    print("\nDone! Generated figures in:", fig_dir)


if __name__ == '__main__':
    main()
