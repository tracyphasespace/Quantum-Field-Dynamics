#!/usr/bin/env python3
"""
Overflow Unified Test — Rates from stress, modes from geometry
================================================================

The frozen core conjecture says:
  - RATES (half-lives) come from epsilon: how far above capacity
  - MODES (what is shed) come from geometry: peanut neck, core fullness
  - These appear disconnected but are both aspects of the same overflow

This test:
  1. Defines geometric overflow parameters:
     - Core fullness: N / N_max(Z) — how close to the density ceiling
     - Peanut factor: (A - A_CRIT) / WIDTH — how deep into peanut regime
     - Neck thinness: epsilon / sqrt(A) — proton excess normalized by size
  2. Tests whether mode selection correlates with geometry
  3. Tests whether rate within each mode correlates with stress
  4. Tests whether the "disconnection" disappears when both are combined

Provenance: QFD_DERIVED (epsilon, z_star, A_CRIT, WIDTH) + EMPIRICAL_LOOKUP (NUBASE2020)
"""

from __future__ import annotations
import math
import os
import sys

import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_nuclide_topology import (
    ALPHA, BETA, PI, E_NUM,
    z_star, normalize_nubase, load_nubase, predict_decay,
    ELEMENTS, _format_halflife, A_CRIT, WIDTH, A_ALPHA_ONSET,
)
from isomer_clock_analysis import find_nubase, build_dataframe


# ═══════════════════════════════════════════════════════════════
# Geometric overflow parameters
# ═══════════════════════════════════════════════════════════════

# Maximum N for each Z (from diameter ceiling test)
def compute_N_max_table(entries):
    """Build N_max(Z) lookup from observed data."""
    by_Z = defaultdict(list)
    for e in entries:
        by_Z[e['Z']].append(e['A'] - e['Z'])
    return {Z: max(Ns) for Z, Ns in by_Z.items()}


def peanut_factor(A):
    """How deep into peanut regime. 0 at onset, 1 at full alpha zone."""
    return (A - A_CRIT) / WIDTH if A > A_CRIT else 0.0


def core_fullness(N, Z, N_max_table):
    """N / N_max(Z) — how close to the density ceiling. 1.0 = at ceiling."""
    nm = N_max_table.get(Z, N + 10)
    return N / nm if nm > 0 else 0.0


def neck_proxy(eps, A):
    """Proton excess normalized by soliton size — neck thinness proxy."""
    return eps / math.sqrt(A) if A > 0 else 0.0


def main():
    print("=" * 80)
    print("  OVERFLOW UNIFIED TEST")
    print("  Rates from stress, modes from geometry — testing the connection")
    print("=" * 80)

    nubase_path = find_nubase()
    entries = load_nubase(nubase_path, include_isomers=False)
    data = build_dataframe(entries)

    N_max_table = compute_N_max_table(entries)

    # Compute overflow parameters for all nuclides
    n_total = len(data['A'])
    data['peanut_f'] = np.array([peanut_factor(A) for A in data['A']])
    data['core_full'] = np.array([
        core_fullness(data['N'][i], data['Z'][i], N_max_table)
        for i in range(n_total)
    ])
    data['neck_thin'] = np.array([
        neck_proxy(data['eps'][i], data['A'][i])
        for i in range(n_total)
    ])

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Mode selection by geometry
    # For each observed decay mode, what are the geometric conditions?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 1: Geometric Conditions by Decay Mode")
    print("  What geometry selects each mode?")
    print("=" * 80)

    gs_mask = data['state'] == 'gs'
    modes_to_check = ['B-', 'B+', 'alpha', 'n', 'p', 'SF', 'stable']

    print(f"\n  {'Mode':>8s} {'n':>6s} {'mean ε':>8s} {'mean |ε|':>9s} "
          f"{'peanut_f':>9s} {'core_full':>10s} {'neck':>8s} {'mean A':>8s}")
    print(f"  {'-'*72}")

    mode_stats = {}
    for mode in modes_to_check:
        if mode == 'stable':
            mask = gs_mask & data['is_stable']
        else:
            mask = gs_mask & (data['mode'] == mode) & ~data['is_stable']

        if np.sum(mask) < 5:
            continue

        idx = np.where(mask)[0]
        eps_vals = data['eps'][idx]
        abs_eps = data['abs_eps'][idx]
        pf = data['peanut_f'][idx]
        cf = data['core_full'][idx]
        nt = data['neck_thin'][idx]
        A_vals = data['A'][idx]

        mode_stats[mode] = {
            'n': len(idx),
            'eps_mean': np.mean(eps_vals),
            'abs_eps_mean': np.mean(abs_eps),
            'pf_mean': np.mean(pf),
            'cf_mean': np.mean(cf),
            'nt_mean': np.mean(nt),
            'A_mean': np.mean(A_vals),
        }

        print(f"  {mode:>8s} {len(idx):>6d} {np.mean(eps_vals):>+8.2f} {np.mean(abs_eps):>9.2f} "
              f"{np.mean(pf):>9.2f} {np.mean(cf):>10.3f} {np.mean(nt):>+8.3f} {np.mean(A_vals):>8.1f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Mode boundaries — where does alpha start vs neutron vs SF?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 2: Mode Boundaries")
    print("  At what geometric thresholds do modes switch?")
    print("=" * 80)

    # Focus on heavy nuclei where multiple modes compete
    heavy = gs_mask & (data['A'] >= 150) & ~data['is_stable']
    heavy_idx = np.where(heavy)[0]

    if len(heavy_idx) > 10:
        print(f"\n  Heavy nuclei (A >= 150): {len(heavy_idx)} nuclides")

        # Group by mode
        for mode in ['B-', 'B+', 'alpha', 'SF', 'n']:
            mode_mask = data['mode'][heavy_idx] == mode
            if np.sum(mode_mask) < 3:
                continue
            m_idx = heavy_idx[mode_mask]

            eps_range = (np.min(data['eps'][m_idx]), np.max(data['eps'][m_idx]))
            pf_range = (np.min(data['peanut_f'][m_idx]), np.max(data['peanut_f'][m_idx]))
            cf_range = (np.min(data['core_full'][m_idx]), np.max(data['core_full'][m_idx]))

            print(f"\n  {mode:>6s} (n={np.sum(mode_mask)}):")
            print(f"    ε range:       [{eps_range[0]:>+6.2f}, {eps_range[1]:>+6.2f}]")
            print(f"    peanut_f range: [{pf_range[0]:>6.2f}, {pf_range[1]:>6.2f}]")
            print(f"    core_full range:[{cf_range[0]:>6.3f}, {cf_range[1]:>6.3f}]")

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: Rate vs stress WITHIN each mode
    # Does |ε| predict half-life equally well across geometric regimes?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 3: Rate vs Stress Within Each Mode")
    print("  Does overflow amount (|ε|) predict rate uniformly?")
    print("=" * 80)

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        mask = gs_mask & (data['mode'] == mode_key) & data['has_hl'] & (data['A'] >= 3)
        idx = np.where(mask)[0]
        if len(idx) < 20:
            continue

        log_hl = data['log_hl'][idx]
        abs_eps = data['abs_eps'][idx]
        sqrt_eps = data['sqrt_abs_eps'][idx]
        pf = data['peanut_f'][idx]
        cf = data['core_full'][idx]

        # Correlation of log(t½) with |ε|
        r_eps = np.corrcoef(abs_eps, log_hl)[0, 1]
        r_sqrt = np.corrcoef(sqrt_eps, log_hl)[0, 1]

        # Correlation with geometric parameters
        r_pf = np.corrcoef(pf, log_hl)[0, 1] if np.std(pf) > 0.01 else 0
        r_cf = np.corrcoef(cf, log_hl)[0, 1] if np.std(cf) > 0.01 else 0

        print(f"\n  {mode_label} (n={len(idx)}):")
        print(f"    r(√|ε|, log t½)     = {r_sqrt:+.4f}  (stress → rate)")
        print(f"    r(|ε|, log t½)      = {r_eps:+.4f}")
        print(f"    r(peanut_f, log t½) = {r_pf:+.4f}  (geometry → rate)")
        print(f"    r(core_full, log t½)= {r_cf:+.4f}  (fullness → rate)")

        # Split by peanut regime: does stress-rate relation change?
        pre_peanut = pf <= 0
        post_peanut = pf > 1.0

        for label, sub_mask in [("Pre-peanut (A < A_CRIT)", pre_peanut),
                                 ("Post-peanut (peanut_f > 1)", post_peanut)]:
            if np.sum(sub_mask) < 10:
                continue
            sub_idx = idx[sub_mask]
            r_sub = np.corrcoef(data['sqrt_abs_eps'][sub_idx], data['log_hl'][sub_idx])[0, 1]
            print(f"    {label}: r(√|ε|, log t½) = {r_sub:+.4f} (n={np.sum(sub_mask)})")

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: Core fullness at mode transitions
    # Where exactly does N/N_max trigger neutron vs alpha?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 4: Core Fullness at Mode Selection")
    print("  Does N/N_max(Z) determine which overflow channel opens?")
    print("=" * 80)

    # Bin by core fullness
    cf_bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9),
               (0.9, 0.95), (0.95, 0.98), (0.98, 1.0), (1.0, 1.1)]

    print(f"\n  {'Core full':>12s} {'n':>6s} {'%B-':>6s} {'%B+':>6s} {'%α':>6s} "
          f"{'%n':>6s} {'%SF':>6s} {'%stab':>6s} {'mean |ε|':>9s}")
    print(f"  {'-'*72}")

    for lo, hi in cf_bins:
        mask = gs_mask & (data['core_full'] >= lo) & (data['core_full'] < hi)
        n_bin = np.sum(mask)
        if n_bin < 5:
            continue

        idx = np.where(mask)[0]
        modes = data['mode'][idx]
        stable = data['is_stable'][idx]

        pcts = {}
        for m in ['B-', 'B+', 'alpha', 'n', 'SF']:
            pcts[m] = np.sum(modes == m) / n_bin * 100

        pct_stable = np.sum(stable) / n_bin * 100
        mean_eps = np.mean(data['abs_eps'][idx])

        print(f"  {f'{lo:.2f}-{hi:.2f}':>12s} {n_bin:>6d} {pcts['B-']:>5.1f}% {pcts['B+']:>5.1f}% "
              f"{pcts['alpha']:>5.1f}% {pcts['n']:>5.1f}% {pcts['SF']:>5.1f}% "
              f"{pct_stable:>5.1f}% {mean_eps:>9.2f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 5: Peanut factor at alpha vs SF boundary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 5: Peanut Factor — Alpha vs SF Boundary")
    print("  At what peanut factor does SF replace alpha?")
    print("=" * 80)

    pf_bins = [(-1, 0), (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
               (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0)]

    print(f"\n  {'Peanut_f':>12s} {'n':>6s} {'%α':>6s} {'%SF':>6s} {'%B-':>6s} "
          f"{'%B+':>6s} {'mean A':>8s} {'mean ε':>8s}")
    print(f"  {'-'*66}")

    for lo, hi in pf_bins:
        mask = gs_mask & (data['peanut_f'] >= lo) & (data['peanut_f'] < hi) & ~data['is_stable']
        n_bin = np.sum(mask)
        if n_bin < 3:
            continue

        idx = np.where(mask)[0]
        modes = data['mode'][idx]

        pcts = {}
        for m in ['alpha', 'SF', 'B-', 'B+']:
            pcts[m] = np.sum(modes == m) / n_bin * 100

        print(f"  {f'{lo:.1f}-{hi:.1f}':>12s} {n_bin:>6d} {pcts['alpha']:>5.1f}% {pcts['SF']:>5.1f}% "
              f"{pcts['B-']:>5.1f}% {pcts['B+']:>5.1f}% "
              f"{np.mean(data['A'][idx]):>8.1f} {np.mean(data['eps'][idx]):>+8.2f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 6: Unified predictor — stress × geometry
    # Can we combine ε and geometric parameters to beat either alone?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 6: Unified Overflow Predictor")
    print("  Does combining stress (rate) with geometry (mode) improve prediction?")
    print("=" * 80)

    from numpy.linalg import lstsq

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        mask = gs_mask & (data['mode'] == mode_key) & data['has_hl'] & (data['A'] >= 3)
        idx = np.where(mask)[0]
        if len(idx) < 30:
            continue

        y = data['log_hl'][idx]

        # Model A: stress only (√|ε| + const)
        X_A = np.column_stack([data['sqrt_abs_eps'][idx], np.ones(len(idx))])
        coefs_A, _, _, _ = lstsq(X_A, y, rcond=None)
        pred_A = X_A @ coefs_A
        ss_res_A = np.sum((y - pred_A)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_A = 1 - ss_res_A / ss_tot

        # Model B: geometry only (peanut_f + core_full + const)
        X_B = np.column_stack([data['peanut_f'][idx], data['core_full'][idx],
                                data['ln_A'][idx], np.ones(len(idx))])
        coefs_B, _, _, _ = lstsq(X_B, y, rcond=None)
        pred_B = X_B @ coefs_B
        ss_res_B = np.sum((y - pred_B)**2)
        r2_B = 1 - ss_res_B / ss_tot

        # Model C: unified (stress + geometry)
        stress_col = data['abs_eps'][idx] if mode_key == 'alpha' else data['sqrt_abs_eps'][idx]
        X_C = np.column_stack([stress_col, data['peanut_f'][idx],
                                data['core_full'][idx], data['ln_A'][idx],
                                data['is_ee'][idx], np.ones(len(idx))])
        coefs_C, _, _, _ = lstsq(X_C, y, rcond=None)
        pred_C = X_C @ coefs_C
        ss_res_C = np.sum((y - pred_C)**2)
        r2_C = 1 - ss_res_C / ss_tot

        # Model D: full V5-like for comparison
        feats_D = ['sqrt_abs_eps', 'ln_A_over_Z', 'ln_A', 'is_ee', 'is_oo']
        if mode_key == 'alpha':
            feats_D[0] = 'abs_eps'
        X_D = np.column_stack([data[f][idx] for f in feats_D] + [np.ones(len(idx))])
        coefs_D, _, _, _ = lstsq(X_D, y, rcond=None)
        pred_D = X_D @ coefs_D
        ss_res_D = np.sum((y - pred_D)**2)
        r2_D = 1 - ss_res_D / ss_tot

        print(f"\n  {mode_label} (n={len(idx)}):")
        print(f"    Model A (stress only):   R² = {r2_A:.4f}  (2 params)")
        print(f"    Model B (geometry only):  R² = {r2_B:.4f}  (4 params)")
        print(f"    Model C (unified):        R² = {r2_C:.4f}  (6 params)")
        print(f"    Model D (V5 basis):       R² = {r2_D:.4f}  (6 params)")

        if r2_C > r2_A and r2_C > r2_B:
            print(f"    → Unified BEATS both stress-only (+{r2_C-r2_A:.4f}) "
                  f"and geometry-only (+{r2_C-r2_B:.4f})")
        elif r2_C > r2_A:
            print(f"    → Geometry adds {r2_C-r2_A:.4f} to stress alone")

    # ═══════════════════════════════════════════════════════════════
    # TEST 7: The "meniscus" regime — barely overfull vs far overfull
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 7: The Meniscus Regime")
    print("  Barely overfull (small |ε|) vs far overfull (large |ε|)")
    print("  Meniscus: the surface tension is barely strained → long half-life")
    print("=" * 80)

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        mask = gs_mask & (data['mode'] == mode_key) & data['has_hl']
        idx = np.where(mask)[0]
        if len(idx) < 30:
            continue

        abs_eps = data['abs_eps'][idx]
        log_hl = data['log_hl'][idx]

        # Bin by |ε|
        eps_bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 7), (7, 15)]
        print(f"\n  {mode_label}:")
        print(f"  {'|ε| range':>12s} {'n':>5s} {'mean log t½':>12s} {'std':>6s} "
              f"{'mean t½':>14s} {'interpretation':>20s}")
        print(f"  {'-'*75}")

        for lo, hi in eps_bins:
            bin_mask = (abs_eps >= lo) & (abs_eps < hi)
            n_bin = np.sum(bin_mask)
            if n_bin < 3:
                continue

            mean_lh = np.mean(log_hl[bin_mask])
            std_lh = np.std(log_hl[bin_mask])
            mean_t = 10**mean_lh

            if mean_lh > 15:
                interp = "meniscus holds"
            elif mean_lh > 6:
                interp = "slow overflow"
            elif mean_lh > 0:
                interp = "steady overflow"
            else:
                interp = "immediate burst"

            # Format mean half-life
            t_str = _format_halflife(mean_t)

            print(f"  {f'{lo:.1f}-{hi:.1f}':>12s} {n_bin:>5d} {mean_lh:>12.1f} {std_lh:>6.1f} "
                  f"{t_str:>14s} {interp:>20s}")

    # ═══════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_dir = os.path.dirname(os.path.abspath(__file__))

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.patch.set_facecolor('#0A0A1A')
        fig.suptitle('Unified Overflow Model — Rates from Stress, Modes from Geometry',
                      fontsize=14, fontweight='bold', color='white', y=0.98)

        def _style(ax, title):
            ax.set_facecolor('#0A0A1A')
            ax.set_title(title, fontsize=10, fontweight='bold', color='white')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.grid(True, alpha=0.15)

        # 1: Mode map by core fullness and |ε|
        ax = axes[0, 0]
        mode_colors = {'B-': '#3366CC', 'B+': '#CC3333', 'alpha': '#DDAA00',
                        'SF': '#FF00FF', 'n': '#00FFFF', 'p': '#FF6600', 'stable': '#333333'}
        for mode, color in mode_colors.items():
            if mode == 'stable':
                m_mask = gs_mask & data['is_stable']
            else:
                m_mask = gs_mask & (data['mode'] == mode) & ~data['is_stable']
            m_idx = np.where(m_mask)[0]
            if len(m_idx) > 0:
                ax.scatter(data['core_full'][m_idx], data['abs_eps'][m_idx],
                           s=2, alpha=0.3, c=color, edgecolors='none', label=mode)
        ax.set_xlabel('Core fullness N/N_max(Z)')
        ax.set_ylabel('|ε| (valley stress)')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 20)
        ax.legend(fontsize=7, facecolor='#1A1A2A', edgecolor='#444444',
                  labelcolor='white', markerscale=5)
        _style(ax, 'Decay Mode by Core Fullness vs Stress\n'
                    'Mode = geometry, Rate = stress')

        # 2: Mode map by peanut factor and ε (signed)
        ax = axes[0, 1]
        for mode, color in mode_colors.items():
            if mode == 'stable':
                m_mask = gs_mask & data['is_stable']
            else:
                m_mask = gs_mask & (data['mode'] == mode) & ~data['is_stable']
            m_idx = np.where(m_mask)[0]
            if len(m_idx) > 0:
                ax.scatter(data['peanut_f'][m_idx], data['eps'][m_idx],
                           s=2, alpha=0.3, c=color, edgecolors='none', label=mode)
        ax.set_xlabel('Peanut factor (A - A_CRIT) / WIDTH')
        ax.set_ylabel('ε (signed valley stress)')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-15, 15)
        ax.axvline(0, color='white', linewidth=0.5, alpha=0.3, linestyle=':')
        ax.axvline(1.0, color='#FF6600', linewidth=0.8, alpha=0.5, linestyle='--')
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.3, linestyle=':')
        ax.legend(fontsize=7, facecolor='#1A1A2A', edgecolor='#444444',
                  labelcolor='white', markerscale=5)
        _style(ax, 'Decay Mode by Peanut Depth vs Stress\n'
                    'Orange dashed = full peanut zone')

        # 3: Half-life vs |ε| colored by mode
        ax = axes[0, 2]
        for mode_key, color, label in [('B-', '#3366CC', 'β⁻'),
                                         ('B+', '#CC3333', 'β⁺'),
                                         ('alpha', '#DDAA00', 'α')]:
            m_mask = gs_mask & (data['mode'] == mode_key) & data['has_hl']
            m_idx = np.where(m_mask)[0]
            if len(m_idx) > 0:
                ax.scatter(data['abs_eps'][m_idx], data['log_hl'][m_idx],
                           s=2, alpha=0.2, c=color, edgecolors='none', label=label)
        ax.set_xlabel('|ε| (how overfull)')
        ax.set_ylabel('log₁₀(t½/s)')
        ax.set_xlim(0, 15)
        ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444',
                  labelcolor='white', markerscale=5)
        _style(ax, 'The Meniscus — Half-life vs Overflow\n'
                    'Small |ε| = barely strained = long-lived')

        # 4: Core fullness vs half-life
        ax = axes[1, 0]
        for mode_key, color, label in [('B-', '#3366CC', 'β⁻'),
                                         ('B+', '#CC3333', 'β⁺'),
                                         ('alpha', '#DDAA00', 'α')]:
            m_mask = gs_mask & (data['mode'] == mode_key) & data['has_hl']
            m_idx = np.where(m_mask)[0]
            if len(m_idx) > 0:
                ax.scatter(data['core_full'][m_idx], data['log_hl'][m_idx],
                           s=2, alpha=0.2, c=color, edgecolors='none', label=label)
        ax.set_xlabel('Core fullness N/N_max(Z)')
        ax.set_ylabel('log₁₀(t½/s)')
        ax.axvline(0.95, color='white', linewidth=0.5, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444',
                  labelcolor='white', markerscale=5)
        _style(ax, 'Half-life vs Core Fullness\n'
                    'Does approaching ceiling change rate?')

        # 5: Mode fraction vs peanut factor
        ax = axes[1, 1]
        pf_centers = np.arange(-0.5, 5.0, 0.25)
        for mode, color, label in [('B-', '#3366CC', 'β⁻'), ('B+', '#CC3333', 'β⁺'),
                                    ('alpha', '#DDAA00', 'α'), ('SF', '#FF00FF', 'SF')]:
            fracs = []
            centers = []
            for c in pf_centers:
                bin_mask = gs_mask & ~data['is_stable'] & \
                           (data['peanut_f'] >= c - 0.25) & (data['peanut_f'] < c + 0.25)
                n_bin = np.sum(bin_mask)
                if n_bin < 5:
                    continue
                frac = np.sum(data['mode'][np.where(bin_mask)[0]] == mode) / n_bin
                fracs.append(frac)
                centers.append(c)
            if fracs:
                ax.plot(centers, fracs, color=color, linewidth=2, alpha=0.8, label=label)
        ax.set_xlabel('Peanut factor')
        ax.set_ylabel('Mode fraction')
        ax.axvline(0, color='white', linewidth=0.5, alpha=0.3, linestyle=':')
        ax.axvline(1.0, color='#FF6600', linewidth=0.8, alpha=0.4, linestyle='--')
        ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
        _style(ax, 'Mode Fraction vs Peanut Depth\n'
                    'Shows mode switching at geometric boundaries')

        # 6: α and SF as function of ε and A
        ax = axes[1, 2]
        # Alpha emitters
        a_mask = gs_mask & (data['mode'] == 'alpha') & ~data['is_stable']
        a_idx = np.where(a_mask)[0]
        if len(a_idx) > 0:
            sc = ax.scatter(data['A'][a_idx], data['eps'][a_idx],
                           c=data['log_hl'][a_idx] if data['has_hl'][a_idx].any() else 'yellow',
                           s=4, alpha=0.6, cmap='viridis', vmin=-3, vmax=25,
                           edgecolors='none', label='α')
        # SF emitters
        sf_mask = gs_mask & (data['mode'] == 'SF') & ~data['is_stable']
        sf_idx = np.where(sf_mask)[0]
        if len(sf_idx) > 0:
            ax.scatter(data['A'][sf_idx], data['eps'][sf_idx],
                       c='magenta', s=20, alpha=0.8, marker='x',
                       edgecolors='none', label='SF')
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
        ax.axvline(A_CRIT, color='#FF6600', linewidth=0.8, alpha=0.4, linestyle='--',
                   label=f'A_CRIT={A_CRIT:.0f}')
        ax.axvline(A_ALPHA_ONSET, color='cyan', linewidth=0.8, alpha=0.4, linestyle='--',
                   label=f'α onset={A_ALPHA_ONSET:.0f}')
        ax.set_xlabel('Mass number A')
        ax.set_ylabel('ε (valley stress)')
        ax.legend(fontsize=7, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
        _style(ax, 'Alpha vs SF in (A, ε) space\n'
                    'Color = log₁₀(t½) for α; magenta X = SF')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, 'overflow_unified_test.png')
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  Saved: {path}")

    except ImportError:
        print("  matplotlib not available — skipping plots")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print("=" * 80)
    print("""
  The unified overflow model:
    RATES come from ε (stress) = how overfull the meniscus is
    MODES come from geometry = where the surface tension fails

  These appear disconnected because:
    - ε is a scalar (distance from valley)
    - Mode selection is a topological decision (peanut? neck width? N at ceiling?)
    - But BOTH derive from the same soliton geometry

  The connection: ε measures the strain on the topological winding (surface
  tension). The geometry determines WHERE the winding fails first (which
  mode). The combination of strain + failure location gives the full
  prediction — not two separate mechanisms, but two aspects of one overflow.
    """)


if __name__ == '__main__':
    main()
