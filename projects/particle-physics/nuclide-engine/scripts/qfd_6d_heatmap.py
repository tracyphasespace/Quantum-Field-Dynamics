#!/usr/bin/env python3
"""
QFD 6-Dimensional Nuclide Diagnostic Heat Map
==============================================

Compares the QFD nuclide engine (zero free parameters, all from α)
against NUBASE2020 across 6 simultaneous dimensions:

    Panel 1: Valley stress ε = Z − Z*(A)
    Panel 2: Mode error confusion categories
    Panel 3: Zero-param clock residual Δlog₁₀(t½)
    Panel 4: Survival score S(Z,A)
    Panel 5: Composite outlier hotspot
    Panel 6: Clock coverage by decay mode

Output: qfd_6d_heatmap.png (3×2 grid, 24×16 inches, dark theme)

Imports from model_nuclide_topology.py (read-only — no modifications).
"""

import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Import from engine (read-only) ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_nuclide_topology import (
    ALPHA, BETA, PI, E_NUM,
    A_CRIT, WIDTH, A_ALPHA_ONSET,
    z_star, survival_score, load_nubase, validate_against_nubase,
    _clock_log10t_zero_param, predict_decay,
)


# ═══════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════

def resolve_nubase_path() -> str:
    """Find nubase2020_raw.txt using the same search order as the engine."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, '..', 'data', 'raw', 'nubase2020_raw.txt'),
        os.path.join(script_dir, 'data', 'nubase2020_raw.txt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "nubase2020_raw.txt not found. Searched:\n  " +
        "\n  ".join(candidates)
    )


def classify_confusion(actual: str, predicted: str) -> str:
    """Classify a (actual, predicted) pair into 5 error categories.

    Categories:
        correct          — prediction matches observation
        alpha-beta       — alpha ↔ B+/B- swap (Z~70–100 competition)
        stability-bound  — stable ↔ B±  (stability boundary errors)
        SF-region        — SF ↔ anything (superheavy fission confusion)
        drip-other       — drip-line (p/n) or other mismatch
    """
    if actual == predicted:
        return 'correct'

    alpha_set = {'alpha'}
    beta_set = {'B-', 'B+'}
    stable_set = {'stable'}
    sf_set = {'SF'}
    drip_set = {'p', 'n'}

    # Alpha ↔ beta swap
    if (actual in alpha_set and predicted in beta_set) or \
       (actual in beta_set and predicted in alpha_set):
        return 'alpha-beta'

    # Stability boundary
    if (actual in stable_set and predicted in beta_set) or \
       (actual in beta_set and predicted in stable_set) or \
       (actual in stable_set and predicted in alpha_set) or \
       (actual in alpha_set and predicted in stable_set):
        return 'stability-bound'

    # SF confusion
    if actual in sf_set or predicted in sf_set:
        return 'SF-region'

    # Drip-line or other
    return 'drip-other'


def compute_clock_residual(entry: dict) -> float:
    """Compute Δlog₁₀(t½) = predicted − measured using zero-param clock.

    Returns NaN if:
        - nuclide is stable (t½ = ∞)
        - no clock for this mode (stable, p, n, SF)
        - measured half-life unavailable
    """
    Z, A = entry['Z'], entry['A']
    actual_mode = entry['actual']
    half_life_s = entry['half_life_s']

    # No clock for these modes
    if actual_mode in ('stable', 'p', 'n', 'SF', 'IT', 'unknown'):
        return np.nan
    if not np.isfinite(half_life_s) or half_life_s <= 0:
        return np.nan

    eps = entry['eps']

    # Map actual mode to clock mode label
    clock_mode = actual_mode  # B-, B+, alpha
    log_t_pred = _clock_log10t_zero_param(Z, eps, clock_mode)

    if log_t_pred is None:
        return np.nan

    log_t_meas = math.log10(half_life_s)
    return log_t_pred - log_t_meas


def compute_composite_badness(mode_error: bool, confusion_cat: str,
                              clock_residual: float, eps: float) -> float:
    """Weighted outlier score.  Higher = more problematic nuclide.

    Components:
        - Mode error:      +1.0 if wrong
        - Clock residual:  |Δlog₁₀t| / 5  (capped at 1.0)
        - Extreme stress:  (|ε| - 4)² / 16  if |ε| > 4, else 0
    """
    score = 0.0

    # Mode error contribution
    if mode_error:
        score += 1.0

    # Clock residual contribution (NaN → 0)
    if np.isfinite(clock_residual):
        score += min(abs(clock_residual) / 5.0, 1.0)

    # Extreme stress contribution
    if abs(eps) > 4.0:
        score += (abs(eps) - 4.0) ** 2 / 16.0

    return score


def build_records(nubase_path: str) -> list:
    """Build the full record set for all 6 panels.

    Returns list of dicts with keys:
        Z, N, A, actual, predicted, match, eps, half_life_s,
        confusion, clock_residual, score, badness
    """
    print("Loading NUBASE2020...")
    nubase = load_nubase(nubase_path)
    print(f"  Loaded {len(nubase)} ground-state entries")

    print("Running validation against NUBASE2020...")
    results = validate_against_nubase(nubase)
    entries = results['entries']
    total = results['total']
    correct = results['correct_mode']
    direction = results['correct_direction']
    dir_total = results['direction_total']
    print(f"  {total} nuclides | Mode: {correct}/{total} = "
          f"{100*correct/total:.1f}% | Direction: {direction}/{dir_total} = "
          f"{100*direction/dir_total:.1f}%")

    print("Computing 6 diagnostic dimensions...")
    records = []
    for e in entries:
        confusion = classify_confusion(e['actual'], e['predicted'])
        clock_res = compute_clock_residual(e)
        score = survival_score(e['Z'], e['A'])
        badness = compute_composite_badness(
            not e['match'], confusion, clock_res, e['eps']
        )
        records.append({
            **e,
            'confusion': confusion,
            'clock_residual': clock_res,
            'score': score,
            'badness': badness,
        })

    print(f"  Built {len(records)} records")
    return records, results


# ═══════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════

BG_COLOR  = '#0A0A1A'
GRID_COLOR = '#222233'
SPINE_COLOR = '#444444'
MAGIC_Z = [2, 8, 20, 28, 50, 82]
MAGIC_N = [2, 8, 20, 28, 50, 82, 126]

CONFUSION_COLORS = {
    'correct':         '#44AA44',   # Green
    'alpha-beta':      '#DDAA00',   # Gold
    'stability-bound': '#3366CC',   # Blue
    'SF-region':       '#CC3333',   # Red
    'drip-other':      '#CC66FF',   # Purple
}

CONFUSION_LABELS = {
    'correct':         'Correct',
    'alpha-beta':      r'$\alpha \leftrightarrow \beta$ swap',
    'stability-bound': 'Stability boundary',
    'SF-region':       'SF region',
    'drip-other':      'Drip-line / other',
}


def style_axis(ax, title: str):
    """Apply dark theme styling to an axis."""
    ax.set_facecolor(BG_COLOR)
    ax.set_xlabel('N (neutron number)', color='white', fontsize=10)
    ax.set_ylabel('Z (proton number)', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.set_xlim(-2, 180)
    ax.set_ylim(-2, 122)


def add_valley_line(ax):
    """Overlay the Z*(A) valley curve in (N, Z) space."""
    A_vals = np.arange(4, 295, 1)
    zs_vals = [z_star(A) for A in A_vals]
    ns_vals = [A - zs for A, zs in zip(A_vals, zs_vals)]
    ax.plot(ns_vals, zs_vals, 'w--', linewidth=0.8, alpha=0.5, label='Z*(A)')


def add_magic_lines(ax):
    """Overlay magic number grid lines."""
    for z in MAGIC_Z:
        ax.axhline(z, color='#666666', linestyle=':', linewidth=0.4, alpha=0.5)
    for n in MAGIC_N:
        ax.axvline(n, color='#666666', linestyle=':', linewidth=0.4, alpha=0.5)


# ═══════════════════════════════════════════════════════════════════
# 6 PANEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def plot_panel_epsilon(ax, records):
    """Panel 1: Valley stress ε = Z − Z*(A).  Diverging RdBu_r."""
    N = [r['N'] for r in records]
    Z = [r['Z'] for r in records]
    eps = [r['eps'] for r in records]

    sc = ax.scatter(N, Z, c=eps, cmap='RdBu_r', vmin=-8, vmax=8,
                    s=3, edgecolors='none', alpha=0.85, rasterized=True)
    style_axis(ax, r'Panel 1: Valley Stress $\varepsilon = Z - Z^*(A)$')
    add_valley_line(ax)
    add_magic_lines(ax)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(r'$\varepsilon$  (blue = n-rich, red = p-rich)',
                   color='white', fontsize=9)
    cbar.ax.tick_params(colors='white', labelsize=7)


def plot_panel_mode_error(ax, records):
    """Panel 2: Mode error confusion categories.  5-color categorical."""
    # Plot correct first (background), errors on top
    correct = [r for r in records if r['confusion'] == 'correct']
    errors = [r for r in records if r['confusion'] != 'correct']

    # Correct: dim green background
    if correct:
        ax.scatter([r['N'] for r in correct], [r['Z'] for r in correct],
                   c=CONFUSION_COLORS['correct'], s=3, alpha=0.20,
                   edgecolors='none', rasterized=True, label='Correct')

    # Errors: vivid on top, sorted by category for legend
    for cat in ['alpha-beta', 'stability-bound', 'SF-region', 'drip-other']:
        subset = [r for r in errors if r['confusion'] == cat]
        if subset:
            ax.scatter([r['N'] for r in subset], [r['Z'] for r in subset],
                       c=CONFUSION_COLORS[cat], s=6, alpha=0.85,
                       edgecolors='none', rasterized=True,
                       label=CONFUSION_LABELS[cat])

    style_axis(ax, 'Panel 2: Mode Error Classification')
    add_valley_line(ax)
    add_magic_lines(ax)

    leg = ax.legend(loc='upper left', fontsize=7, framealpha=0.7,
                    facecolor='#1A1A2A', edgecolor='#444444',
                    labelcolor='white', markerscale=2.5)


def plot_panel_clock_residual(ax, records):
    """Panel 3: Zero-param clock residual Δlog₁₀(t½).  coolwarm, ±5 cap."""
    # Split: has clock vs no clock
    has_clock = [r for r in records if np.isfinite(r['clock_residual'])]
    no_clock = [r for r in records if not np.isfinite(r['clock_residual'])]

    # Gray background for no-clock nuclides
    if no_clock:
        ax.scatter([r['N'] for r in no_clock], [r['Z'] for r in no_clock],
                   c='#333333', s=2, alpha=0.3, edgecolors='none',
                   rasterized=True)

    # Colored residuals
    if has_clock:
        residuals = [max(-5, min(5, r['clock_residual'])) for r in has_clock]
        sc = ax.scatter([r['N'] for r in has_clock],
                        [r['Z'] for r in has_clock],
                        c=residuals, cmap='coolwarm', vmin=-5, vmax=5,
                        s=4, edgecolors='none', alpha=0.85, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label(r'$\Delta\log_{10}(t_{1/2})$  (pred $-$ meas)',
                       color='white', fontsize=9)
        cbar.ax.tick_params(colors='white', labelsize=7)

    style_axis(ax, r'Panel 3: Zero-Param Clock Residual $\Delta\log_{10}(t_{1/2})$')
    add_valley_line(ax)
    add_magic_lines(ax)


def plot_panel_survival(ax, records):
    """Panel 4: Survival score S(Z,A).  viridis, 5th–95th percentile."""
    scores = [r['score'] for r in records]
    vlo = np.percentile(scores, 5)
    vhi = np.percentile(scores, 95)

    N = [r['N'] for r in records]
    Z = [r['Z'] for r in records]

    sc = ax.scatter(N, Z, c=scores, cmap='viridis', vmin=vlo, vmax=vhi,
                    s=3, edgecolors='none', alpha=0.85, rasterized=True)
    style_axis(ax, 'Panel 4: Survival Score S(Z, A)')
    add_valley_line(ax)
    add_magic_lines(ax)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('S(Z, A)  (high = more stable)', color='white', fontsize=9)
    cbar.ax.tick_params(colors='white', labelsize=7)


def plot_panel_composite(ax, records):
    """Panel 5: Composite outlier hotspot.  hot_r, dark background."""
    badness = [r['badness'] for r in records]
    vhi = np.percentile(badness, 98)  # cap color at 98th percentile

    # Only show nuclides with nonzero badness visibly
    N = [r['N'] for r in records]
    Z = [r['Z'] for r in records]

    sc = ax.scatter(N, Z, c=badness, cmap='hot_r', vmin=0, vmax=max(vhi, 0.5),
                    s=4, edgecolors='none', alpha=0.85, rasterized=True)
    style_axis(ax, 'Panel 5: Composite Outlier Hotspot')
    add_valley_line(ax)
    add_magic_lines(ax)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Badness  (mode + clock + stress)', color='white', fontsize=9)
    cbar.ax.tick_params(colors='white', labelsize=7)


def plot_panel_clock_coverage(ax, records):
    """Panel 6: Clock coverage by mode.  Color=mode, size=|residual|."""
    mode_colors = {
        'B-':    '#3366CC',   # Blue
        'B+':    '#CC3333',   # Red
        'alpha': '#DDAA00',   # Gold
    }

    # No-clock: gray background
    no_clock = [r for r in records if not np.isfinite(r['clock_residual'])]
    if no_clock:
        ax.scatter([r['N'] for r in no_clock], [r['Z'] for r in no_clock],
                   c='#333333', s=2, alpha=0.3, edgecolors='none',
                   rasterized=True)

    # Plot each clock mode separately
    for mode, color in mode_colors.items():
        subset = [r for r in records
                  if np.isfinite(r['clock_residual']) and r['actual'] == mode]
        if not subset:
            continue
        residuals = [abs(r['clock_residual']) for r in subset]
        # Size: min 2, max 25, scaled by |residual|
        sizes = [max(2, min(25, 2 + 4 * res)) for res in residuals]
        ax.scatter([r['N'] for r in subset], [r['Z'] for r in subset],
                   c=color, s=sizes, alpha=0.7, edgecolors='none',
                   rasterized=True, label=mode)

    style_axis(ax, 'Panel 6: Clock Coverage (color=mode, size=|residual|)')
    add_valley_line(ax)
    add_magic_lines(ax)

    leg = ax.legend(loc='upper left', fontsize=8, framealpha=0.7,
                    facecolor='#1A1A2A', edgecolor='#444444',
                    labelcolor='white', markerscale=1.5)


# ═══════════════════════════════════════════════════════════════════
# FIGURE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════

def create_figure(records, results):
    """Assemble the 3×2 panel figure."""
    total = results['total']
    correct = results['correct_mode']
    direction = results['correct_direction']
    dir_total = results['direction_total']

    fig, axes = plt.subplots(2, 3, figsize=(24, 16),
                             facecolor=BG_COLOR)
    fig.subplots_adjust(hspace=0.28, wspace=0.22,
                        left=0.04, right=0.96, top=0.90, bottom=0.05)

    # Suptitle with summary statistics
    mode_pct = 100 * correct / total
    dir_pct = 100 * direction / dir_total
    fig.suptitle(
        f'QFD 6-Dimensional Nuclide Diagnostic — {total} nuclides vs NUBASE2020\n'
        r'Mode: %.1f%% | Direction: %.1f%% | Zero-param clock '
        r'($\beta^-$: R²=0.67, $\beta^+$: 0.63, $\alpha$: 0.25)'
        % (mode_pct, dir_pct),
        color='white', fontsize=14, fontweight='bold', y=0.96
    )

    # 6 panels
    plot_panel_epsilon(axes[0, 0], records)
    plot_panel_mode_error(axes[0, 1], records)
    plot_panel_clock_residual(axes[0, 2], records)
    plot_panel_survival(axes[1, 0], records)
    plot_panel_composite(axes[1, 1], records)
    plot_panel_clock_coverage(axes[1, 2], records)

    return fig


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    nubase_path = resolve_nubase_path()
    print(f"NUBASE2020: {nubase_path}")

    records, results = build_records(nubase_path)

    print("Generating 6-panel diagnostic figure...")
    fig = create_figure(records, results)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'qfd_6d_heatmap.png')
    fig.savefig(out_path, dpi=200, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Summary stats
    n_errors = sum(1 for r in records if not r['match'])
    n_clock = sum(1 for r in records if np.isfinite(r['clock_residual']))
    clock_rmse = np.sqrt(np.nanmean(
        [r['clock_residual']**2 for r in records
         if np.isfinite(r['clock_residual'])]
    ))
    print(f"\nDiagnostic summary:")
    print(f"  Total nuclides:    {len(records)}")
    print(f"  Mode errors:       {n_errors} ({100*n_errors/len(records):.1f}%)")
    print(f"  Clock coverage:    {n_clock} nuclides")
    print(f"  Clock RMSE:        {clock_rmse:.2f} decades")
    print(f"  Badness > 1.0:     "
          f"{sum(1 for r in records if r['badness'] > 1.0)} nuclides")


if __name__ == '__main__':
    main()
