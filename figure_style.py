"""
figure_style.py - Shared publication-quality figure styling

Import and call apply_style() at the top of any figure generation script.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication-quality style settings
STYLE_PARAMS = {
    # Font sizes (readable at print scale)
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,

    # Font family (matches LaTeX paper body)
    'font.family': 'serif',
    'mathtext.fontset': 'cm',

    # Line widths (visible at print scale)
    # Primary lines: 2.0pt, reference lines: 1.0-1.2pt
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,

    # Marker sizes
    'lines.markersize': 6,

    # Figure quality
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,

    # Remove box around plots (ACL/ML standard)
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Grid styling
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
}

# Color palette (colorblind-friendly, grayscale-distinguishable)
COLORS = {
    'lrd': '#D62728',         # red - for LRD signals
    'patching': '#FF7F0E',    # orange - for patching recovery
    'lora': '#1F77B4',        # blue - for LoRA effectiveness
    'clean': '#2CA02C',       # green - for clean baselines
    'directional': '#2166AC', # blue - for directional perturbations
    'uniform': '#B2182B',     # red - for uniform perturbations
    'cosine': '#2166AC',      # blue - for cosine loss
    'mse': '#B2182B',         # red - for MSE loss
}

# Line styles for B&W printing and colorblind accessibility
LINE_STYLES = {
    'lrd': '-',        # solid
    'patching': '--',  # dashed
    'lora': ':',       # dotted
}

# Reference line style (thinner, for identity ceiling, random floor, etc.)
REFERENCE_LINE_KW = {
    'linewidth': 1.0,
    'alpha': 0.8,
}


def apply_style():
    """Apply publication-quality style settings to matplotlib."""
    plt.rcParams.update(STYLE_PARAMS)


def remove_spines(ax):
    """Remove top and right spines from an axes object."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def setup_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    """
    Standard setup for a publication-quality axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure
    xlabel, ylabel, title : str, optional
        Labels and title
    grid : bool
        Whether to show grid (y-axis only by default)
    """
    remove_spines(ax)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)


def add_legend_below(ax, ncol=4, y_offset=-0.18):
    """
    Move legend below the plot in a horizontal layout.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ncol : int
        Number of columns in the legend
    y_offset : float
        Vertical offset below the plot (negative = below)
    """
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor='black',
    )


def add_correlation_box(ax, correlations, loc='upper right', fontsize=10):
    """
    Add a clean correlation table as a text box annotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    correlations : dict
        Dictionary of {label: {'rho': float, 'p': float}} or {label: rho_value}
    loc : str
        Location string ('upper right', 'upper left', 'lower right', 'lower left')
    fontsize : int
    """
    lines = []
    for label, stats in correlations.items():
        if isinstance(stats, dict):
            rho = stats.get('rho', 0)
        else:
            rho = stats
        lines.append(f"{label}: $\\rho$={rho:+.2f}")

    text = '\n'.join(lines)

    # Position based on loc
    if 'lower' in loc:
        va, y = 'bottom', 0.02
    else:
        va, y = 'top', 0.98
    if 'left' in loc:
        ha, x = 'left', 0.02
    else:
        ha, x = 'right', 0.98

    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=va, horizontalalignment=ha,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.9))


# Legacy alias for backward compatibility
add_correlation_legend = add_correlation_box


# Perturbation ordering and taxonomy
PERTURBATION_ORDER = ['typos', 'ocr', 'speech', 'homophones', 'whitespace', 'case']
DIRECTIONAL_SET = {'typos', 'ocr', 'speech', 'homophones'}
UNIFORM_SET = {'whitespace', 'case'}


def method_from_cond(cond_name):
    """Extract method name from condition key like 'Typos_5%'."""
    return cond_name.rsplit('_', 1)[0].lower()
