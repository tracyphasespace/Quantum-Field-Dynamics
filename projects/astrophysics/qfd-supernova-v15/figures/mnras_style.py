"""
MNRAS Figure Style Configuration
Standardized matplotlib settings for MNRAS two-column layout
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np

# MNRAS column widths (in points)
MNRAS_SINGLE_COLUMN_PT = 244  # ~84 mm
MNRAS_DOUBLE_COLUMN_PT = 508  # ~178 mm

def setup_mnras_style():
    """
    Configure matplotlib for MNRAS publication quality figures.

    Features:
    - Vector PDF output (editable text)
    - Serif fonts matching newtx/Termes
    - Appropriate sizes for print (7-8 pt)
    - Monochrome-friendly line styles
    """
    mpl.rcParams.update({
        # Font rendering (editable in Illustrator)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Font family (match paper's newtx/Termes)
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times", "DejaVu Serif"],

        # Font sizes (readable at print)
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.titlesize": 8.0,

        # Line widths (appropriate for print)
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 0.8,

        # Grid and spines
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "legend.fancybox": False,

        # Figure
        "figure.dpi": 72,  # Points per inch (for sizing)
        "savefig.dpi": 900,  # High DPI for publication quality
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

def create_figure_single_column(aspect_ratio=1.2):
    """
    Create a single-column MNRAS figure.

    Args:
        aspect_ratio: height/width ratio (default 1.2 for portrait-ish)

    Returns:
        fig: matplotlib Figure object
    """
    width_pt = MNRAS_SINGLE_COLUMN_PT
    height_pt = width_pt / aspect_ratio

    width_in = width_pt / 72
    height_in = height_pt / 72

    fig = plt.figure(figsize=(width_in, height_in))
    return fig

def create_figure_double_column(aspect_ratio=2.0):
    """
    Create a double-column MNRAS figure.

    Args:
        aspect_ratio: height/width ratio (default 2.0 for wide)

    Returns:
        fig: matplotlib Figure object
    """
    width_pt = MNRAS_DOUBLE_COLUMN_PT
    height_pt = width_pt / aspect_ratio

    width_in = width_pt / 72
    height_in = height_pt / 72

    fig = plt.figure(figsize=(width_in, height_in))
    return fig

# Line style definitions (colorful + distinguishable in grayscale)
LINE_STYLES = {
    'qfd': {'linestyle': '-', 'linewidth': 1.0, 'color': '#1f77b4', 'label': 'QFD'},  # Blue
    'lcdm': {'linestyle': '--', 'linewidth': 0.8, 'color': '#ff7f0e', 'label': 'ΛCDM'},  # Orange
    'data': {'marker': 'o', 'markersize': 3, 'markerfacecolor': '#2ca02c',  # Green
             'markeredgecolor': 'black', 'markeredgewidth': 0.4, 'linestyle': 'none'},
    'basis1': {'linestyle': '-', 'linewidth': 0.8, 'color': '#1f77b4'},  # Blue
    'basis2': {'linestyle': '--', 'linewidth': 0.8, 'color': '#ff7f0e'},  # Orange
    'basis3': {'linestyle': ':', 'linewidth': 1.0, 'color': '#2ca02c'},  # Green
}

def save_figure_with_provenance(fig, filename, provenance_data):
    """
    Save figure as PDF with metadata and generate provenance JSON.

    Args:
        fig: matplotlib Figure
        filename: output filename (e.g., 'figure_hubble.pdf')
        provenance_data: dict with keys like 'dataset', 'filters', 'config', etc.
    """
    # Save figure
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.5/72)

    # Generate provenance file
    provenance_file = filename.replace('.pdf', '_provenance.json')

    # Add metadata
    full_provenance = {
        'figure': filename,
        'created': datetime.now().isoformat(),
        'software': {
            'numpy': np.__version__,
            'matplotlib': mpl.__version__,
        },
        **provenance_data
    }

    # Try to add git SHA if available
    try:
        import subprocess
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         stderr=subprocess.DEVNULL).decode().strip()
        full_provenance['git_sha'] = git_sha
    except:
        pass

    with open(provenance_file, 'w') as f:
        json.dump(full_provenance, f, indent=2)

    print(f"✓ Saved: {filename}")
    print(f"✓ Provenance: {provenance_file}")

def compute_data_hash(data_file):
    """Compute SHA256 hash of data file for provenance."""
    sha256 = hashlib.sha256()
    with open(data_file, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def add_panel_label(ax, label, loc='top-left', fontsize=8, fontweight='bold'):
    """
    Add panel label like "(a)" to subplot.

    Args:
        ax: matplotlib Axes
        label: text (e.g., "(a)")
        loc: 'top-left', 'top-right', etc.
        fontsize: font size in points
        fontweight: 'normal' or 'bold'
    """
    if loc == 'top-left':
        x, y = 0.05, 0.95
        ha, va = 'left', 'top'
    elif loc == 'top-right':
        x, y = 0.95, 0.95
        ha, va = 'right', 'top'
    elif loc == 'bottom-left':
        x, y = 0.05, 0.05
        ha, va = 'left', 'bottom'
    elif loc == 'bottom-right':
        x, y = 0.95, 0.05
        ha, va = 'right', 'bottom'
    else:
        x, y = 0.05, 0.95
        ha, va = 'left', 'top'

    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            ha=ha, va=va, bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white',
                                    edgecolor='none',
                                    alpha=0.8))

def equal_count_bins(z, values, nbins=30):
    """
    Create equal-count bins for plotting.

    Args:
        z: array of redshifts
        values: array of values to bin
        nbins: number of bins

    Returns:
        bin_centers: array of bin centers (median z in each bin)
        bin_means: array of mean values in each bin
        bin_errors: array of SEM in each bin
    """
    # Sort by z
    idx = np.argsort(z)
    z_sorted = z[idx]
    values_sorted = values[idx]

    # Create equal-count bins
    bin_edges = np.percentile(z_sorted, np.linspace(0, 100, nbins + 1))

    bin_centers = []
    bin_means = []
    bin_errors = []

    for i in range(nbins):
        mask = (z_sorted >= bin_edges[i]) & (z_sorted < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(np.median(z_sorted[mask]))
            bin_means.append(np.mean(values_sorted[mask]))
            bin_errors.append(np.std(values_sorted[mask]) / np.sqrt(mask.sum()))

    return np.array(bin_centers), np.array(bin_means), np.array(bin_errors)
