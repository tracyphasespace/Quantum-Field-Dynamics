#!/usr/bin/env python3
"""
Isomer-Aware Clock Analysis — Reparametrized Fits + Residual Structure Hunt
============================================================================

Replicates AI 1's reparametrization findings using OUR valley definition
(rational z_star from α), then generates comprehensive heatmaps and
hunts for hidden structure in the residuals.

Key differences from AI 1's LaGrangian version:
  - ε = Z - z_star(A) from rational compression law (not 7-path polynomial)
  - All constants from Golden Loop (α → β)
  - Includes isomer-aware analysis

Clock variants fitted:
  V0: √|ε| + log₁₀(Z) + Z + const                     (our original 4-param)
  V1: √|ε| + ln(A/Z) + ln(A) + const                   (reparametrized 4-param)
  V2: V1 + ee + oo                                       (+ parity, 6-param)
  V3: V2 + 6×G(N,magic) + 5×G(Z,magic)                  (+ magic, 17-param)
  V4: Like V3 but alpha uses |ε| instead of √|ε|         (mode-specific stress)

Provenance:
  ε values: QFD_DERIVED (from compression law, zero free params)
  Clock coefficients: EMPIRICAL_FIT (from NUBASE2020 half-lives)
  Magic Gaussians: EMPIRICAL_PROXY (standard magic numbers, width=4 fitted)
"""

from __future__ import annotations
import math
import os
import sys

import numpy as np
from numpy.linalg import lstsq
from collections import defaultdict

# Import our engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_nuclide_topology import (
    ALPHA, BETA, PI, E_NUM,
    z_star, survival_score, predict_decay, normalize_nubase,
    load_nubase, group_nuclide_states, _parse_spin_value,
    ELEMENTS, MODE_COLORS, A_CRIT, WIDTH, A_ALPHA_ONSET,
    _format_halflife,
)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def find_nubase():
    """Find NUBASE2020 data file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, '..', 'data', 'raw', 'nubase2020_raw.txt'),
        os.path.join(script_dir, 'data', 'nubase2020_raw.txt'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"NUBASE2020 not found. Searched: {candidates}")


def build_dataframe(entries: list) -> dict:
    """Build numpy arrays from NUBASE entries for fitting.

    Returns dict of arrays keyed by field name.
    """
    n = len(entries)
    data = {
        'A': np.array([e['A'] for e in entries]),
        'Z': np.array([e['Z'] for e in entries]),
        'N': np.array([e['A'] - e['Z'] for e in entries]),
        'mode': np.array([normalize_nubase(e['dominant_mode']) for e in entries]),
        'is_stable': np.array([e['is_stable'] for e in entries]),
        'half_life_s': np.array([e['half_life_s'] for e in entries]),
        'state': np.array([e['state'] for e in entries]),
    }

    # Derived quantities using OUR valley
    data['eps'] = np.array([e['Z'] - z_star(e['A']) for e in entries])
    data['abs_eps'] = np.abs(data['eps'])
    data['sqrt_abs_eps'] = np.sqrt(np.clip(data['abs_eps'], 0.01, None))
    data['log10_Z'] = np.log10(np.clip(data['Z'], 1, None).astype(float))
    data['Z_float'] = data['Z'].astype(float)

    # Reparametrized regressors
    A_f = data['A'].astype(float)
    Z_f = np.clip(data['Z'].astype(float), 1, None)
    data['ln_A_over_Z'] = np.log(A_f / Z_f)
    data['ln_A'] = np.log(A_f)

    # Parity
    Z_even = (data['Z'] % 2 == 0)
    N_even = (data['N'] % 2 == 0)
    data['is_ee'] = (Z_even & N_even).astype(float)
    data['is_oo'] = (~Z_even & ~N_even).astype(float)

    # Parity label
    parity = np.full(n, 'eo', dtype='U2')
    parity[Z_even & N_even] = 'ee'
    parity[~Z_even & ~N_even] = 'oo'
    parity[Z_even & ~N_even] = 'en'
    parity[~Z_even & N_even] = 'ne'
    data['parity'] = parity

    # Magic number Gaussians
    MAGIC_N = [8, 20, 28, 50, 82, 126]
    MAGIC_Z = [8, 20, 28, 50, 82]

    def gauss(x, center, width=4.0):
        return np.exp(-((x - center) / width) ** 2)

    for m in MAGIC_N:
        data[f'gN{m}'] = gauss(data['N'].astype(float), m)
    for m in MAGIC_Z:
        data[f'gZ{m}'] = gauss(data['Z'].astype(float), m)

    # ── Correction features ──

    # 1. Valley proximity penalty: exp(-ε²) — near-valley β decays are slow
    #    (forbidden transitions with small Q, large ΔJ)
    data['valley_prox'] = np.exp(-(data['eps'] ** 2))

    # 2. Daughter-magic Gaussians for alpha:
    #    Alpha: (Z,A) → (Z-2, A-4), so daughter N = N-2, daughter Z = Z-2
    N_daughter = data['N'] - 2
    Z_daughter = data['Z'] - 2
    data['gNd126'] = gauss(N_daughter.astype(float), 126)  # daughter near N=126
    data['gZd82'] = gauss(Z_daughter.astype(float), 82)    # daughter near Z=82
    data['gNd82'] = gauss(N_daughter.astype(float), 82)    # daughter near N=82

    # 3. ε² (quadratic stress — captures forbidden transition suppression)
    data['eps_sq'] = data['eps'] ** 2

    # Log half-life (target)
    hl = data['half_life_s'].copy()
    valid_hl = np.isfinite(hl) & (hl > 0) & ~data['is_stable']
    data['log_hl'] = np.full(n, np.nan)
    data['log_hl'][valid_hl] = np.log10(hl[valid_hl])
    data['has_hl'] = valid_hl

    return data


# ═══════════════════════════════════════════════════════════════════
# CLOCK FITTING
# ═══════════════════════════════════════════════════════════════════

CLOCK_VARIANTS = {
    'V0_original': {
        'features': ['sqrt_abs_eps', 'log10_Z', 'Z_float', 'const'],
        'stress_fn': 'sqrt',
        'description': '√|ε| + log₁₀(Z) + Z + const  (original 4-param)',
    },
    'V1_reparam': {
        'features': ['sqrt_abs_eps', 'ln_A_over_Z', 'ln_A', 'const'],
        'stress_fn': 'sqrt',
        'description': '√|ε| + ln(A/Z) + ln(A) + const  (reparametrized)',
    },
    'V2_parity': {
        'features': ['sqrt_abs_eps', 'ln_A_over_Z', 'ln_A', 'is_ee', 'is_oo', 'const'],
        'stress_fn': 'sqrt',
        'description': 'V1 + parity  (6-param)',
    },
    'V3_magic': {
        'features': ['sqrt_abs_eps', 'ln_A_over_Z', 'ln_A', 'is_ee', 'is_oo',
                      'gN8', 'gN20', 'gN28', 'gN50', 'gN82', 'gN126',
                      'gZ8', 'gZ20', 'gZ28', 'gZ50', 'gZ82', 'const'],
        'stress_fn': 'sqrt',
        'description': 'V2 + magic Gaussians  (17-param)',
    },
    'V4_alpha_linear': {
        'features': ['abs_eps', 'ln_A_over_Z', 'ln_A', 'is_ee', 'is_oo',
                      'gN8', 'gN20', 'gN28', 'gN50', 'gN82', 'gN126',
                      'gZ8', 'gZ20', 'gZ28', 'gZ50', 'gZ82', 'const'],
        'stress_fn': 'linear',
        'description': '|ε| for alpha, √|ε| for beta  (mode-specific stress)',
    },
    'V5_corrections': {
        # Beta: add valley_prox (near-valley forbidden slowdown)
        # Alpha: add daughter-magic Gaussians
        'features_beta': ['sqrt_abs_eps', 'ln_A_over_Z', 'ln_A', 'is_ee', 'is_oo',
                          'valley_prox',
                          'gN8', 'gN20', 'gN28', 'gN50', 'gN82', 'gN126',
                          'gZ8', 'gZ20', 'gZ28', 'gZ50', 'gZ82', 'const'],
        'features_alpha': ['abs_eps', 'ln_A_over_Z', 'ln_A', 'is_ee', 'is_oo',
                           'gN8', 'gN20', 'gN28', 'gN50', 'gN82', 'gN126',
                           'gZ8', 'gZ20', 'gZ28', 'gZ50', 'gZ82',
                           'gNd126', 'gZd82', 'gNd82', 'const'],
        'features': None,  # mode-specific, handled in fit_clock
        'stress_fn': 'mixed',
        'description': 'V4 + valley_prox(β) + daughter-magic(α)',
    },
    'V6_full': {
        # Everything: valley_prox + daughter-magic for all modes
        'features_beta': ['sqrt_abs_eps', 'ln_A_over_Z', 'ln_A', 'is_ee', 'is_oo',
                          'valley_prox', 'eps_sq',
                          'gN8', 'gN20', 'gN28', 'gN50', 'gN82', 'gN126',
                          'gZ8', 'gZ20', 'gZ28', 'gZ50', 'gZ82', 'const'],
        'features_alpha': ['abs_eps', 'eps_sq', 'ln_A_over_Z', 'ln_A', 'is_ee', 'is_oo',
                           'valley_prox',
                           'gN8', 'gN20', 'gN28', 'gN50', 'gN82', 'gN126',
                           'gZ8', 'gZ20', 'gZ28', 'gZ50', 'gZ82',
                           'gNd126', 'gZd82', 'gNd82', 'const'],
        'features': None,
        'stress_fn': 'full',
        'description': 'V5 + ε² + valley_prox for all  (kitchen sink)',
    },
}

# Mode mapping for our labels → fit labels
MODE_MAP = {'B-': 'B-', 'B+': 'B+', 'alpha': 'alpha'}


def fit_clock(data: dict, mode: str, variant_name: str) -> dict | None:
    """Fit a clock variant for a single decay mode.

    Returns dict with: coefs, feature_names, r2, rmse, n, residuals, predictions.
    """
    variant = CLOCK_VARIANTS[variant_name]

    # Resolve mode-specific feature lists
    if variant.get('features') is not None:
        features = variant['features']
    elif mode == 'alpha' and 'features_alpha' in variant:
        features = variant['features_alpha']
    elif 'features_beta' in variant:
        features = variant['features_beta']
    else:
        features = variant.get('features', [])

    # For V4, swap stress function for alpha only
    if variant_name == 'V4_alpha_linear' and mode != 'alpha':
        # Beta modes use V3 features (sqrt stress)
        features = CLOCK_VARIANTS['V3_magic']['features']

    mask = (data['mode'] == mode) & data['has_hl'] & (data['A'] >= 3)

    # Ground states only for fitting
    gs_mask = mask & (data['state'] == 'gs')
    idx = np.where(gs_mask)[0]

    if len(idx) < 20:
        return None

    y = data['log_hl'][idx]

    # Build feature matrix
    # Add constant column if not in features
    feat_cols = []
    feat_names = []
    for f in features:
        if f == 'const':
            feat_cols.append(np.ones(len(idx)))
            feat_names.append('const')
        else:
            feat_cols.append(data[f][idx])
            feat_names.append(f)

    X = np.column_stack(feat_cols)

    coefs, _, _, _ = lstsq(X, y, rcond=None)
    pred = X @ coefs
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = math.sqrt(np.mean((y - pred) ** 2))
    residuals = y - pred

    # Also predict for ALL nuclides of this mode (including isomers)
    all_mask = (data['mode'] == mode) & (data['A'] >= 3)
    all_idx = np.where(all_mask)[0]
    all_feat_cols = []
    for f in features:
        if f == 'const':
            all_feat_cols.append(np.ones(len(all_idx)))
        else:
            all_feat_cols.append(data[f][all_idx])
    X_all = np.column_stack(all_feat_cols)
    pred_all = X_all @ coefs

    # Within-10x and within-100x for ground states with measured hl
    within_1 = np.sum(np.abs(residuals) <= 1.0)
    within_2 = np.sum(np.abs(residuals) <= 2.0)

    return {
        'coefs': coefs,
        'feature_names': feat_names,
        'r2': r2,
        'rmse': rmse,
        'n': len(idx),
        'residuals': residuals,
        'predictions_gs': pred,
        'gs_indices': idx,
        'all_indices': all_idx,
        'predictions_all': pred_all,
        'within_1': within_1,
        'within_2': within_2,
    }


def fit_all_variants(data: dict) -> dict:
    """Fit all clock variants for all modes. Returns nested dict."""
    results = {}
    for vname in CLOCK_VARIANTS:
        results[vname] = {}
        for mode in ['B-', 'B+', 'alpha']:
            result = fit_clock(data, mode, vname)
            if result is not None:
                results[vname][mode] = result
    return results


def print_r2_ladder(all_results: dict):
    """Print the R² ladder across variants and modes."""
    print(f"\n{'='*80}")
    print("  R² LADDER — Clock Variants (fitted to ground-state half-lives)")
    print("=" * 80)

    print(f"\n  {'Variant':<25s} {'β⁻ R²':>7s} {'β⁺ R²':>7s} {'α R²':>7s}  "
          f"{'β⁻ RMSE':>8s} {'β⁺ RMSE':>8s} {'α RMSE':>8s}  {'Params':>6s}")
    print(f"  {'-'*82}")

    for vname, vdef in CLOCK_VARIANTS.items():
        res = all_results.get(vname, {})
        parts = []
        for mode in ['B-', 'B+', 'alpha']:
            r = res.get(mode)
            if r:
                parts.append(f"{r['r2']:>7.4f}")
            else:
                parts.append(f"{'—':>7s}")

        rmse_parts = []
        for mode in ['B-', 'B+', 'alpha']:
            r = res.get(mode)
            if r:
                rmse_parts.append(f"{r['rmse']:>8.3f}")
            else:
                rmse_parts.append(f"{'—':>8s}")

        if vdef.get('features') is not None:
            n_params = len(vdef['features'])
        else:
            # Mode-specific: show max
            n_beta = len(vdef.get('features_beta', []))
            n_alpha = len(vdef.get('features_alpha', []))
            n_params = max(n_beta, n_alpha)
        label = vname.replace('_', ' ')
        print(f"  {label:<25s} {parts[0]} {parts[1]} {parts[2]}  "
              f"{rmse_parts[0]} {rmse_parts[1]} {rmse_parts[2]}  {n_params:>6d}")


def print_coefficients(all_results: dict, variant: str):
    """Print coefficients with algebraic identifications."""
    print(f"\n{'='*80}")
    print(f"  COEFFICIENTS — {variant}")
    print("=" * 80)

    # Algebraic candidates
    candidates = {
        'sqrt_abs_eps': [
            (-PI * BETA / E_NUM, '-πβ/e'),
            (-PI, '-π'),
            (-E_NUM, '-e'),
            (-4.0 / 3.0, '-4/3'),
            (-E_NUM * PI / 3.0, '-eπ/3'),
        ],
        'abs_eps': [
            (-3.0 / 2.0, '-3/2'),
            (-PI / 2.0, '-π/2'),
            (-BETA / 2.0, '-β/2'),
            (-E_NUM / 2.0, '-e/2'),
        ],
        'ln_A_over_Z': [
            (-PI, '-π'),
            (-BETA, '-β'),
            (PI ** 3, 'π³'),
            (BETA * PI ** 2, 'βπ²'),
            (-2 * E_NUM, '-2e'),
        ],
        'ln_A': [
            (-2 * E_NUM, '-2e'),
            (PI ** 3, 'π³'),
            (BETA * PI ** 2, 'βπ²'),
            (-PI * math.log(BETA), '-π·ln(β)'),
            (-E_NUM * PI, '-eπ'),
        ],
        'is_ee': [
            (E_NUM / (2 * BETA), 'e/(2β)'),
            (1.0 / BETA, '1/β'),
            (1.0 / PI, '1/π'),
        ],
        'is_oo': [
            (-PI / (2 * BETA), '-π/(2β)'),
            (-1.0 / BETA, '-1/β'),
            (-1.0 / PI, '-1/π'),
        ],
    }

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r = all_results.get(variant, {}).get(mode_key)
        if not r:
            continue

        print(f"\n  {mode_label} (n={r['n']}, R²={r['r2']:.4f}, RMSE={r['rmse']:.3f})")
        print(f"  {'Feature':<16s} {'Coefficient':>12s}  {'Best match':>12s} {'Error':>7s}")
        print(f"  {'-'*52}")

        for fname, coef in zip(r['feature_names'], r['coefs']):
            best_match = ''
            best_err = ''

            if fname in candidates:
                best_val, best_name = min(
                    candidates[fname],
                    key=lambda x: abs(coef - x[0]) / max(abs(x[0]), 0.01)
                )
                pct_err = abs(coef - best_val) / max(abs(best_val), 0.01) * 100
                if pct_err < 10:
                    best_match = best_name
                    best_err = f'{pct_err:.1f}%'

            print(f"  {fname:<16s} {coef:>+12.6f}  {best_match:>12s} {best_err:>7s}")


# ═══════════════════════════════════════════════════════════════════
# HEATMAP GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_heatmaps(data: dict, all_results: dict, output_dir: str):
    """Generate comprehensive (N,Z) heatmaps of clock predictions and residuals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Valley line
    A_vals = np.arange(1, 301)
    Z_line = np.array([z_star(A) for A in A_vals])
    N_line = A_vals - Z_line

    def _style_dark(ax, title):
        ax.set_title(title, fontsize=11, fontweight='bold', color='white')
        ax.set_xlabel('Neutrons (N)', color='white')
        ax.set_ylabel('Protons (Z)', color='white')
        ax.set_xlim(-5, 185)
        ax.set_ylim(-5, 120)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.1)
        ax.set_facecolor('#0A0A1A')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444444')

    # Use best available variant for heatmaps
    best_variant = 'V5_corrections' if 'V5_corrections' in all_results else 'V4_alpha_linear'
    res = all_results[best_variant]

    # Collect all predictions and residuals for ground states with measured hl
    gs_with_hl = (data['state'] == 'gs') & data['has_hl'] & (data['A'] >= 3)

    # Build per-nuclide prediction arrays using the best variant
    n_total = len(data['A'])
    pred_log_hl = np.full(n_total, np.nan)
    resid_log_hl = np.full(n_total, np.nan)

    for mode_key in ['B-', 'B+', 'alpha']:
        r = res.get(mode_key)
        if not r:
            continue
        for i, idx in enumerate(r['all_indices']):
            pred_log_hl[idx] = r['predictions_all'][i]
            if np.isfinite(data['log_hl'][idx]):
                resid_log_hl[idx] = data['log_hl'][idx] - r['predictions_all'][i]

    # Also for stable nuclides — mark separately
    stable_mask = data['is_stable'] & (data['state'] == 'gs')

    # ── Figure 1: 6-panel overview ──
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.patch.set_facecolor('#0A0A1A')

    # 1a: Actual half-life
    ax = axes[0, 0]
    mask = gs_with_hl
    idx = np.where(mask)[0]
    sc = ax.scatter(data['N'][idx], data['Z'][idx],
                    c=data['log_hl'][idx], s=3, alpha=0.6,
                    cmap='viridis', vmin=-3, vmax=20, edgecolors='none')
    # Stable
    s_idx = np.where(stable_mask)[0]
    ax.scatter(data['N'][s_idx], data['Z'][s_idx],
               c='#222222', s=2, alpha=0.5, edgecolors='none')
    ax.plot(N_line, Z_line, 'w-', linewidth=0.6, alpha=0.4)
    _style_dark(ax, 'Actual log₁₀(t½/s)\n(NUBASE2020 ground states)')
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, label='log₁₀(t½)')
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='white')

    # 1b: Predicted half-life
    ax = axes[0, 1]
    has_pred = np.isfinite(pred_log_hl) & (data['state'] == 'gs')
    idx = np.where(has_pred)[0]
    sc = ax.scatter(data['N'][idx], data['Z'][idx],
                    c=pred_log_hl[idx], s=3, alpha=0.6,
                    cmap='viridis', vmin=-3, vmax=20, edgecolors='none')
    ax.scatter(data['N'][s_idx], data['Z'][s_idx],
               c='#222222', s=2, alpha=0.5, edgecolors='none')
    ax.plot(N_line, Z_line, 'w-', linewidth=0.6, alpha=0.4)
    _style_dark(ax, f'Predicted log₁₀(t½/s)\n(V4 clock: √|ε| for β, |ε| for α)')
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, label='log₁₀(t½)')
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='white')

    # 1c: Residual (actual - predicted)
    ax = axes[0, 2]
    has_resid = np.isfinite(resid_log_hl) & (data['state'] == 'gs')
    idx = np.where(has_resid)[0]
    vmax_r = 4.0
    sc = ax.scatter(data['N'][idx], data['Z'][idx],
                    c=resid_log_hl[idx], s=3, alpha=0.6,
                    cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r, edgecolors='none')
    ax.plot(N_line, Z_line, 'w-', linewidth=0.6, alpha=0.4)
    _style_dark(ax, 'Residual (actual − predicted)\nBlue = lives longer, Red = dies faster')
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, label='Δlog₁₀(t½)')
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='white')

    # 2a: Residual by mode
    ax = axes[1, 0]
    for mode_key, color in [('B-', '#3366CC'), ('B+', '#CC3333'), ('alpha', '#DDAA00')]:
        r = res.get(mode_key)
        if not r:
            continue
        gs_idx = r['gs_indices']
        ax.scatter(data['N'][gs_idx], data['Z'][gs_idx],
                   c=color, s=2, alpha=0.3, edgecolors='none')
    ax.plot(N_line, Z_line, 'w-', linewidth=0.6, alpha=0.4)
    _style_dark(ax, 'Nuclides with Clock Predictions\nBlue=β⁻, Red=β⁺, Gold=α')

    # 2b: Residual by parity
    ax = axes[1, 1]
    parity_colors = {'ee': '#33CC33', 'oo': '#CC3333', 'en': '#3366CC', 'ne': '#DDAA00'}
    for p_type, p_color in parity_colors.items():
        p_mask = has_resid & (data['parity'] == p_type)
        p_idx = np.where(p_mask)[0]
        if len(p_idx) > 0:
            ax.scatter(data['N'][p_idx], data['Z'][p_idx],
                       c=resid_log_hl[p_idx], s=3, alpha=0.6,
                       cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r,
                       edgecolors='none')
    ax.plot(N_line, Z_line, 'w-', linewidth=0.6, alpha=0.4)
    _style_dark(ax, 'Residuals (all parities)\nSame as panel 1c, for reference')
    # Overlay parity type as small markers
    for p_type in ['ee', 'oo']:
        p_mask = has_resid & (data['parity'] == p_type)
        p_idx = np.where(p_mask)[0]
        r_vals = resid_log_hl[p_idx]
        big_resid = np.abs(r_vals) > 3.0
        if np.any(big_resid):
            big_idx = p_idx[big_resid]
            marker = 's' if p_type == 'ee' else 'D'
            ax.scatter(data['N'][big_idx], data['Z'][big_idx],
                       c='white', s=12, alpha=0.8, marker=marker,
                       edgecolors='none')

    # 2c: |Residual| — where is the clock worst?
    ax = axes[1, 2]
    idx = np.where(has_resid)[0]
    abs_resid = np.abs(resid_log_hl[idx])
    sc = ax.scatter(data['N'][idx], data['Z'][idx],
                    c=abs_resid, s=3, alpha=0.6,
                    cmap='hot_r', vmin=0, vmax=5, edgecolors='none')
    ax.plot(N_line, Z_line, 'w-', linewidth=0.6, alpha=0.4)
    _style_dark(ax, '|Residual| — Clock Error Magnitude\nBright = worse prediction')
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, label='|Δlog₁₀(t½)|')
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='white')

    plt.tight_layout()
    path1 = os.path.join(output_dir, 'clock_heatmap_overview.png')
    fig.savefig(path1, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ── Figure 2: Per-mode residual detail ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))
    fig2.patch.set_facecolor('#0A0A1A')

    for i, (mode_label, mode_key) in enumerate([('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]):
        ax = axes2[i]
        r = res.get(mode_key)
        if not r:
            _style_dark(ax, f'{mode_label} — no data')
            continue

        gs_idx = r['gs_indices']
        resids = r['residuals']
        vmax_m = min(max(np.percentile(np.abs(resids), 95), 2), 6)

        sc = ax.scatter(data['N'][gs_idx], data['Z'][gs_idx],
                        c=resids, s=4, alpha=0.6,
                        cmap='RdBu_r', vmin=-vmax_m, vmax=vmax_m,
                        edgecolors='none')
        ax.plot(N_line, Z_line, 'w-', linewidth=0.6, alpha=0.4)
        _style_dark(ax, f'{mode_label} Residuals\n'
                        f'R²={r["r2"]:.3f}, RMSE={r["rmse"]:.2f}, '
                        f'10×={r["within_1"]}/{r["n"]} ({r["within_1"]/r["n"]*100:.0f}%)')
        cb = fig2.colorbar(sc, ax=ax, shrink=0.7, label='Δlog₁₀(t½)')
        cb.ax.yaxis.label.set_color('white')
        cb.ax.tick_params(colors='white')

    plt.tight_layout()
    path2 = os.path.join(output_dir, 'clock_heatmap_per_mode.png')
    fig2.savefig(path2, dpi=150, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Saved: {path2}")

    return pred_log_hl, resid_log_hl


# ═══════════════════════════════════════════════════════════════════
# RESIDUAL STRUCTURE HUNT
# ═══════════════════════════════════════════════════════════════════

def hunt_residual_structure(data: dict, all_results: dict, pred_log_hl, resid_log_hl,
                            output_dir: str):
    """Analyze residuals for hidden structure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    best_variant = 'V5_corrections' if 'V5_corrections' in all_results else 'V4_alpha_linear'
    res = all_results[best_variant]

    print(f"\n{'='*80}")
    print(f"  RESIDUAL STRUCTURE HUNT (using {best_variant})")
    print("=" * 80)

    # Valley line for plots
    A_vals = np.arange(1, 301)
    Z_line = np.array([z_star(A) for A in A_vals])
    N_line = A_vals - Z_line

    def _style_dark(ax, title):
        ax.set_facecolor('#0A0A1A')
        ax.set_title(title, fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444444')

    # ── Analysis 1: Residual vs A — looking for periodic structure ──
    fig3, axes3 = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig3.patch.set_facecolor('#0A0A1A')

    for i, (mode_label, mode_key) in enumerate([('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]):
        ax = axes3[i]
        r = res.get(mode_key)
        if not r:
            continue

        gs_idx = r['gs_indices']
        A_arr = data['A'][gs_idx]
        resids = r['residuals']

        # Scatter
        ax.scatter(A_arr, resids, s=2, alpha=0.3, c='cyan', edgecolors='none')

        # Running mean (window=10 in A)
        A_unique = np.arange(A_arr.min(), A_arr.max() + 1)
        running_mean = np.full(len(A_unique), np.nan)
        running_std = np.full(len(A_unique), np.nan)
        for j, a in enumerate(A_unique):
            window = (A_arr >= a - 5) & (A_arr <= a + 5)
            if np.sum(window) >= 3:
                running_mean[j] = np.mean(resids[window])
                running_std[j] = np.std(resids[window])

        valid = np.isfinite(running_mean)
        ax.plot(A_unique[valid], running_mean[valid], 'yellow', linewidth=1.5, label='running mean (±5)')
        ax.fill_between(A_unique[valid],
                        running_mean[valid] - running_std[valid],
                        running_mean[valid] + running_std[valid],
                        alpha=0.15, color='yellow')

        ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
        ax.set_ylabel('Residual (decades)')
        ax.set_ylim(-8, 8)
        ax.grid(True, alpha=0.1)
        _style_dark(ax, f'{mode_label} Residual vs A  (R²={r["r2"]:.3f})')
        ax.legend(loc='upper right', fontsize=8, facecolor='#1A1A2A',
                  edgecolor='#444444', labelcolor='white')

        # Print structure
        print(f"\n  {mode_label} RESIDUAL vs A:")
        # Find systematic deviations (running mean > 1σ from zero)
        big_dev = np.where(valid & (np.abs(running_mean) > 1.5))[0]
        if len(big_dev) > 0:
            # Group consecutive A values
            groups = []
            current = [A_unique[big_dev[0]]]
            for k in range(1, len(big_dev)):
                if A_unique[big_dev[k]] - A_unique[big_dev[k-1]] <= 2:
                    current.append(A_unique[big_dev[k]])
                else:
                    groups.append(current)
                    current = [A_unique[big_dev[k]]]
            groups.append(current)

            print(f"    Systematic deviations (|mean| > 1.5 decades):")
            for g in groups[:10]:
                a_lo, a_hi = g[0], g[-1]
                g_mask = (A_arr >= a_lo) & (A_arr <= a_hi)
                mean_r = np.mean(resids[g_mask]) if np.sum(g_mask) > 0 else 0
                print(f"      A=[{a_lo},{a_hi}]: mean residual = {mean_r:+.2f} decades ({np.sum(g_mask)} nuclides)")

    axes3[-1].set_xlabel('Mass Number A')
    plt.tight_layout()
    path3 = os.path.join(output_dir, 'clock_residual_vs_A.png')
    fig3.savefig(path3, dpi=150, facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f"\n  Saved: {path3}")

    # ── Analysis 2: Residual by parity class ──
    print(f"\n  RESIDUAL BY PARITY:")
    gs_has_resid = (data['state'] == 'gs') & np.isfinite(resid_log_hl)
    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r = res.get(mode_key)
        if not r:
            continue
        gs_idx = r['gs_indices']
        resids = r['residuals']
        print(f"\n    {mode_label}:")
        for p_type in ['ee', 'en', 'ne', 'oo']:
            p_mask = data['parity'][gs_idx] == p_type
            if np.sum(p_mask) < 5:
                continue
            p_resids = resids[p_mask]
            print(f"      {p_type}: n={np.sum(p_mask):>4d}, mean={np.mean(p_resids):>+6.2f}, "
                  f"std={np.std(p_resids):>5.2f}, |max|={np.max(np.abs(p_resids)):>5.1f}")

    # ── Analysis 3: Magic number proximity effects ──
    print(f"\n  MAGIC NUMBER PROXIMITY:")
    MAGIC_N = [8, 20, 28, 50, 82, 126]
    MAGIC_Z = [8, 20, 28, 50, 82]

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r = res.get(mode_key)
        if not r:
            continue
        gs_idx = r['gs_indices']
        resids = r['residuals']
        N_arr = data['N'][gs_idx]
        Z_arr = data['Z'][gs_idx]

        print(f"\n    {mode_label}:")
        # Distance to nearest magic N
        for m in MAGIC_N:
            near = np.abs(N_arr - m) <= 3
            if np.sum(near) >= 3:
                mean_r = np.mean(resids[near])
                std_r = np.std(resids[near])
                if abs(mean_r) > 0.5:
                    print(f"      N≈{m:>3d}: n={np.sum(near):>3d}, "
                          f"mean={mean_r:>+6.2f} ± {std_r:.2f}  ← SIGNAL")
                else:
                    print(f"      N≈{m:>3d}: n={np.sum(near):>3d}, "
                          f"mean={mean_r:>+6.2f} ± {std_r:.2f}")

        for m in MAGIC_Z:
            near = np.abs(Z_arr - m) <= 3
            if np.sum(near) >= 3:
                mean_r = np.mean(resids[near])
                std_r = np.std(resids[near])
                if abs(mean_r) > 0.5:
                    print(f"      Z≈{m:>3d}: n={np.sum(near):>3d}, "
                          f"mean={mean_r:>+6.2f} ± {std_r:.2f}  ← SIGNAL")
                else:
                    print(f"      Z≈{m:>3d}: n={np.sum(near):>3d}, "
                          f"mean={mean_r:>+6.2f} ± {std_r:.2f}")

    # ── Analysis 4: Deformation region (A ~ 150-190) ──
    print(f"\n  DEFORMATION REGION (A=150-190):")
    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r = res.get(mode_key)
        if not r:
            continue
        gs_idx = r['gs_indices']
        resids = r['residuals']
        A_arr = data['A'][gs_idx]

        deform = (A_arr >= 150) & (A_arr <= 190)
        non_deform = (A_arr >= 50) & (A_arr < 150)
        if np.sum(deform) < 5 or np.sum(non_deform) < 5:
            continue

        mean_def = np.mean(resids[deform])
        std_def = np.std(resids[deform])
        mean_nd = np.mean(resids[non_deform])
        std_nd = np.std(resids[non_deform])
        print(f"    {mode_label}: deformed mean={mean_def:+.2f}±{std_def:.2f} (n={np.sum(deform)}), "
              f"non-def mean={mean_nd:+.2f}±{std_nd:.2f} (n={np.sum(non_deform)})")

    # ── Analysis 5: Worst residuals — what are they? ──
    print(f"\n  WORST RESIDUALS (|Δ| > 4 decades):")
    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r = res.get(mode_key)
        if not r:
            continue
        gs_idx = r['gs_indices']
        resids = r['residuals']

        bad = np.abs(resids) > 4.0
        if np.sum(bad) == 0:
            print(f"    {mode_label}: none")
            continue

        bad_idx = gs_idx[bad]
        bad_resids = resids[bad]
        order = np.argsort(-np.abs(bad_resids))

        print(f"\n    {mode_label} ({np.sum(bad)} outliers):")
        print(f"    {'Nuclide':>10s} {'|ε|':>6s} {'Actual':>12s} {'Predicted':>12s} {'Δlog':>7s} {'Parity':>6s}")
        print(f"    {'-'*58}")
        for j in order[:12]:
            ii = bad_idx[j]
            sym = ELEMENTS.get(data['Z'][ii], f"Z{data['Z'][ii]}")
            name = f"{sym}-{data['A'][ii]}"
            actual_s = data['half_life_s'][ii]
            pred_s = 10 ** pred_log_hl[ii] if np.isfinite(pred_log_hl[ii]) else 0
            actual_str = _format_halflife(actual_s) if np.isfinite(actual_s) and actual_s > 0 else '—'
            pred_str = _format_halflife(pred_s) if pred_s > 0 else '—'
            print(f"    {name:>10s} {data['abs_eps'][ii]:>6.2f} {actual_str:>12s} {pred_str:>12s} "
                  f"{bad_resids[j]:>+7.1f} {data['parity'][ii]:>6s}")

    # ── Analysis 6: Residual autocorrelation in A ──
    print(f"\n  RESIDUAL AUTOCORRELATION (periodicity in A):")
    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+')]:
        r = res.get(mode_key)
        if not r:
            continue
        gs_idx = r['gs_indices']
        resids = r['residuals']
        A_arr = data['A'][gs_idx]

        # Bin residuals by A
        A_min, A_max = int(A_arr.min()), int(A_arr.max())
        binned = np.full(A_max - A_min + 1, np.nan)
        for a in range(A_min, A_max + 1):
            mask = A_arr == a
            if np.sum(mask) >= 1:
                binned[a - A_min] = np.mean(resids[mask])

        valid = np.isfinite(binned)
        if np.sum(valid) < 30:
            continue

        # FFT on valid bins (interpolate gaps)
        signal = binned.copy()
        # Simple interpolation for gaps
        valid_idx = np.where(valid)[0]
        for j in range(len(signal)):
            if not np.isfinite(signal[j]):
                # Find nearest valid
                dists = np.abs(valid_idx - j)
                nearest = valid_idx[np.argmin(dists)]
                signal[j] = binned[nearest]

        signal -= np.mean(signal)
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal))
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

        # Find top 5 peaks (excluding DC and very low freq)
        interesting = (periods > 2) & (periods < 100)
        if np.sum(interesting) > 0:
            peak_idx = np.argsort(-fft[interesting])[:5]
            interesting_idx = np.where(interesting)[0]
            print(f"\n    {mode_label} — top periods in residual vs A:")
            for k in peak_idx:
                idx_k = interesting_idx[k]
                print(f"      Period = {periods[idx_k]:.1f} AMU, amplitude = {fft[idx_k]:.3f}")

    # ── Figure 4: Structure hunt summary ──
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
    fig4.patch.set_facecolor('#0A0A1A')

    # 4a: Residual vs |ε| (stress)
    ax = axes4[0, 0]
    for mode_label, mode_key, color in [('β⁻', 'B-', '#3366CC'),
                                         ('β⁺', 'B+', '#CC3333'),
                                         ('α', 'alpha', '#DDAA00')]:
        r = res.get(mode_key)
        if not r:
            continue
        gs_idx = r['gs_indices']
        ax.scatter(data['abs_eps'][gs_idx], r['residuals'],
                   s=2, alpha=0.2, c=color, edgecolors='none', label=mode_label)
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('|ε| (valley stress)')
    ax.set_ylabel('Residual (decades)')
    ax.set_xlim(0, 15)
    ax.set_ylim(-8, 8)
    ax.grid(True, alpha=0.1)
    ax.legend(fontsize=9, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
    _style_dark(ax, 'Residual vs |ε| — Any stress-dependent bias?')

    # 4b: Residual vs ln(A/Z)
    ax = axes4[0, 1]
    for mode_label, mode_key, color in [('β⁻', 'B-', '#3366CC'),
                                         ('β⁺', 'B+', '#CC3333'),
                                         ('α', 'alpha', '#DDAA00')]:
        r = res.get(mode_key)
        if not r:
            continue
        gs_idx = r['gs_indices']
        ax.scatter(data['ln_A_over_Z'][gs_idx], r['residuals'],
                   s=2, alpha=0.2, c=color, edgecolors='none', label=mode_label)
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('ln(A/Z) = ln(1 + N/Z)')
    ax.set_ylabel('Residual (decades)')
    ax.set_ylim(-8, 8)
    ax.grid(True, alpha=0.1)
    ax.legend(fontsize=9, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
    _style_dark(ax, 'Residual vs ln(A/Z) — Isospin bias?')

    # 4c: Residual histogram by mode
    ax = axes4[1, 0]
    for mode_label, mode_key, color in [('β⁻', 'B-', '#3366CC'),
                                         ('β⁺', 'B+', '#CC3333'),
                                         ('α', 'alpha', '#DDAA00')]:
        r = res.get(mode_key)
        if not r:
            continue
        ax.hist(r['residuals'], bins=60, range=(-8, 8), alpha=0.5,
                color=color, label=f"{mode_label} (σ={r['rmse']:.2f})")
    ax.axvline(0, color='white', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Residual (decades)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.1)
    ax.legend(fontsize=9, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
    _style_dark(ax, 'Residual Distribution by Mode')

    # 4d: Residual vs A (binned)
    ax = axes4[1, 1]
    for mode_label, mode_key, color in [('β⁻', 'B-', '#3366CC'),
                                         ('β⁺', 'B+', '#CC3333'),
                                         ('α', 'alpha', '#DDAA00')]:
        r = res.get(mode_key)
        if not r:
            continue
        gs_idx = r['gs_indices']
        A_arr = data['A'][gs_idx]
        resids = r['residuals']

        # Bin by A in windows of 10
        A_bins = np.arange(0, 300, 10)
        means = []
        centers = []
        for a_lo in A_bins:
            mask = (A_arr >= a_lo) & (A_arr < a_lo + 10)
            if np.sum(mask) >= 3:
                means.append(np.mean(resids[mask]))
                centers.append(a_lo + 5)

        ax.plot(centers, means, '-o', color=color, markersize=3,
                linewidth=1, alpha=0.7, label=mode_label)

    ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Mass Number A')
    ax.set_ylabel('Mean Residual (decades)')
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.1)
    ax.legend(fontsize=9, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
    _style_dark(ax, 'Binned Residual vs A — Systematic trends?')

    plt.tight_layout()
    path4 = os.path.join(output_dir, 'clock_residual_structure.png')
    fig4.savefig(path4, dpi=150, facecolor=fig4.get_facecolor())
    plt.close(fig4)
    print(f"\n  Saved: {path4}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("  ISOMER-AWARE CLOCK ANALYSIS — Reparametrized Fits + Structure Hunt")
    print("=" * 80)
    print(f"\n  Valley: z_star(A) from rational compression law")
    print(f"  α = {ALPHA}, β = {BETA:.10f}")

    # Load all NUBASE entries (gs + isomers)
    nubase_path = find_nubase()
    print(f"\n  Loading NUBASE2020 from: {nubase_path}")
    all_entries = load_nubase(nubase_path, include_isomers=True)
    print(f"  Total entries: {len(all_entries)}")

    gs_entries = [e for e in all_entries if e['state'] == 'gs']
    iso_entries = [e for e in all_entries if e['state'] != 'gs']
    print(f"  Ground states: {len(gs_entries)}, Isomers: {len(iso_entries)}")

    # Build data arrays
    data = build_dataframe(all_entries)
    print(f"  With measured half-life: {np.sum(data['has_hl'])}")

    # Mode census
    for mode in ['B-', 'B+', 'alpha', 'stable']:
        gs_mode = (data['mode'] == mode) & (data['state'] == 'gs')
        gs_hl = gs_mode & data['has_hl']
        print(f"    {mode:>8s}: {np.sum(gs_mode):>5d} gs total, {np.sum(gs_hl):>5d} with t½")

    # Fit all clock variants
    print(f"\n  Fitting clock variants...")
    all_results = fit_all_variants(data)

    # Print R² ladder
    print_r2_ladder(all_results)

    # Print coefficients for key variants
    print_coefficients(all_results, 'V1_reparam')
    print_coefficients(all_results, 'V4_alpha_linear')
    print_coefficients(all_results, 'V5_corrections')

    # ── Diagnostic: What did the corrections buy? ──
    print(f"\n{'='*80}")
    print("  CORRECTION IMPACT — What did V5 fix?")
    print("=" * 80)

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r4 = all_results.get('V4_alpha_linear', {}).get(mode_key)
        r5 = all_results.get('V5_corrections', {}).get(mode_key)
        if not r4 or not r5:
            continue

        delta_r2 = r5['r2'] - r4['r2']
        delta_rmse = r5['rmse'] - r4['rmse']
        delta_10x = r5['within_1'] - r4['within_1']

        print(f"\n  {mode_label}:")
        print(f"    R²:   {r4['r2']:.4f} → {r5['r2']:.4f}  ({delta_r2:+.4f})")
        print(f"    RMSE: {r4['rmse']:.3f} → {r5['rmse']:.3f}  ({delta_rmse:+.3f})")
        print(f"    10×:  {r4['within_1']}/{r4['n']} → {r5['within_1']}/{r5['n']}  ({delta_10x:+d})")

        # Show the new correction coefficients
        for fname, coef in zip(r5['feature_names'], r5['coefs']):
            if fname in ('valley_prox', 'gNd126', 'gZd82', 'gNd82', 'eps_sq'):
                print(f"    {fname}: {coef:+.4f}")

    # Compare worst outliers V4 vs V5 for β⁻
    print(f"\n  β⁻ OUTLIER IMPROVEMENT (V4 → V5):")
    r4 = all_results.get('V4_alpha_linear', {}).get('B-')
    r5 = all_results.get('V5_corrections', {}).get('B-')
    if r4 and r5:
        # Find the worst V4 outliers and see what V5 does
        bad4 = np.abs(r4['residuals']) > 4.0
        if np.any(bad4):
            bad_idx_in_gs = np.where(bad4)[0]
            print(f"    {'Nuclide':>10s} {'V4 resid':>9s} {'V5 resid':>9s} {'Δ':>7s}")
            print(f"    {'-'*40}")
            order = np.argsort(-np.abs(r4['residuals'][bad4]))
            for j in order[:10]:
                k = bad_idx_in_gs[j]
                gs_idx_4 = r4['gs_indices'][k]
                # Find same nuclide in V5
                Z_val = data['Z'][gs_idx_4]
                A_val = data['A'][gs_idx_4]
                sym = ELEMENTS.get(Z_val, f"Z{Z_val}")
                name = f"{sym}-{A_val}"
                r4_val = r4['residuals'][k]

                # Find in V5
                match = None
                for m, idx5 in enumerate(r5['gs_indices']):
                    if data['Z'][idx5] == Z_val and data['A'][idx5] == A_val:
                        match = r5['residuals'][m]
                        break
                if match is not None:
                    delta = match - r4_val
                    print(f"    {name:>10s} {r4_val:>+9.1f} {match:>+9.1f} {delta:>+7.1f}")

    # Same for alpha
    print(f"\n  α OUTLIER IMPROVEMENT (V4 → V5):")
    r4 = all_results.get('V4_alpha_linear', {}).get('alpha')
    r5 = all_results.get('V5_corrections', {}).get('alpha')
    if r4 and r5:
        bad4 = np.abs(r4['residuals']) > 4.0
        if np.any(bad4):
            bad_idx_in_gs = np.where(bad4)[0]
            print(f"    {'Nuclide':>10s} {'V4 resid':>9s} {'V5 resid':>9s} {'Δ':>7s}")
            print(f"    {'-'*40}")
            order = np.argsort(-np.abs(r4['residuals'][bad4]))
            for j in order[:10]:
                k = bad_idx_in_gs[j]
                gs_idx_4 = r4['gs_indices'][k]
                Z_val = data['Z'][gs_idx_4]
                A_val = data['A'][gs_idx_4]
                sym = ELEMENTS.get(Z_val, f"Z{Z_val}")
                name = f"{sym}-{A_val}"
                r4_val = r4['residuals'][k]

                match = None
                for m, idx5 in enumerate(r5['gs_indices']):
                    if data['Z'][idx5] == Z_val and data['A'][idx5] == A_val:
                        match = r5['residuals'][m]
                        break
                if match is not None:
                    delta = match - r4_val
                    print(f"    {name:>10s} {r4_val:>+9.1f} {match:>+9.1f} {delta:>+7.1f}")

    # Generate heatmaps
    print(f"\n{'='*80}")
    print("  GENERATING HEATMAPS...")
    print("=" * 80)

    try:
        pred_log_hl, resid_log_hl = generate_heatmaps(data, all_results, script_dir)

        # Hunt for structure
        hunt_residual_structure(data, all_results, pred_log_hl, resid_log_hl, script_dir)

    except ImportError as e:
        print(f"\n  matplotlib not available: {e}")
        print("  Skipping visualization, printing text analysis only.")

        # Still do the text analysis
        pred_log_hl = np.full(len(data['A']), np.nan)
        resid_log_hl = np.full(len(data['A']), np.nan)
        best_variant = 'V4_alpha_linear'
        res = all_results[best_variant]
        for mode_key in ['B-', 'B+', 'alpha']:
            r = res.get(mode_key)
            if not r:
                continue
            for i, idx in enumerate(r['all_indices']):
                pred_log_hl[idx] = r['predictions_all'][i]
                if np.isfinite(data['log_hl'][idx]):
                    resid_log_hl[idx] = data['log_hl'][idx] - r['predictions_all'][i]

        hunt_residual_structure(data, all_results, pred_log_hl, resid_log_hl, script_dir)

    print(f"\n{'='*80}")
    print("  ANALYSIS COMPLETE")
    print("=" * 80)
