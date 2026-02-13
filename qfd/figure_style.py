#!/usr/bin/env python3
"""
QFD Unified Figure Style — Publication-quality defaults for all book figures.

Provides consistent rcParams, color palette, figure sizes, and save helpers
so that every figure in the QFD book (Edition v8.8) looks uniform.

Usage:
    from qfd.figure_style import apply_qfd_style, qfd_savefig, QFD_COLORS, FIGURE_SIZES

    apply_qfd_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    # ... plot ...
    qfd_savefig(fig, 'fig_12_01_golden_loop', output_dir='book_figures')

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Publication rcParams (based on V22 create_manuscript_figures.py)
# =============================================================================

QFD_STYLE = {
    # Fonts
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    # Axes
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.linewidth': 1.0,
    # Ticks
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    # Legend
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    # Figure / save
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    # Grid
    'axes.grid': True,
    'grid.alpha': 0.3,
}


# =============================================================================
# Standard figure sizes (width, height in inches)
# =============================================================================

FIGURE_SIZES = {
    'single': (7, 5),       # One-panel figures
    'wide':   (10, 4),      # Side-by-side 2-panel
    'tall':   (7, 10),      # Stacked 2-panel
    'grid':   (12, 10),     # 2x2 grid
    'large':  (14, 10),     # 2x3 or 3x2 grid
}


# =============================================================================
# Named colour palette
# =============================================================================

QFD_COLORS = {
    # Leptons
    'electron': '#2E86AB',
    'muon':     '#A23B72',
    'tau':      '#F18F01',
    # Models
    'qfd':      '#2E86AB',
    'lcdm':     '#E63946',
    'data':     '#264653',
    # Nuclear fission
    'symmetric_fission':  '#E63946',
    'asymmetric_fission': '#2E86AB',
    # General palette (for multi-line plots)
    'blue':   '#2E86AB',
    'red':    '#E63946',
    'green':  '#2A9D8F',
    'orange': '#F18F01',
    'purple': '#A23B72',
    'grey':   '#6C757D',
}


# =============================================================================
# Helpers
# =============================================================================

def apply_qfd_style():
    """Apply QFD publication rcParams globally.

    Call once at the top of a script or at the start of each figure
    generation function.
    """
    matplotlib.use('Agg')          # non-interactive backend (safe for CI)
    plt.rcParams.update(QFD_STYLE)


def qfd_savefig(fig, name, output_dir='book_figures'):
    """Save *fig* as both PNG (300 DPI) and PDF in *output_dir*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Stem name without extension (e.g. ``'fig_12_01_golden_loop'``).
    output_dir : str or Path
        Target directory (created if missing).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ext in ('png', 'pdf'):
        path = out / f'{name}.{ext}'
        fig.savefig(str(path), dpi=300, bbox_inches='tight', pad_inches=0.05)

    plt.close(fig)
    print(f"  Saved: {out / name}.{{png,pdf}}")


def qfd_textbox(ax, text, loc='upper left', **kwargs):
    """Place a styled annotation box on *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    text : str
    loc : str
        ``'upper left'``, ``'upper right'``, ``'lower left'``, ``'lower right'``.
    """
    positions = {
        'upper left':  (0.05, 0.95, 'top',    'left'),
        'upper right': (0.95, 0.95, 'top',    'right'),
        'lower left':  (0.05, 0.05, 'bottom', 'left'),
        'lower right': (0.95, 0.05, 'bottom', 'right'),
    }
    x, y, va, ha = positions.get(loc, positions['upper left'])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    props.update(kwargs)
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10,
            verticalalignment=va, horizontalalignment=ha, bbox=props)
