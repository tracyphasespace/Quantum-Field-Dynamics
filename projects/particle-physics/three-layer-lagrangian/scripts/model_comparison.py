#!/usr/bin/env python3
"""
Unified Model Comparison — AI2 Atomic Clock vs Three-Layer LaGrangian

Head-to-head on the SAME data, SAME population, SAME metrics.

AI2 Model (Atomic Clock v6, QFD_NUCLIDE_ENGINE §23):
  log₁₀(t½) = a·√|ε| + b·log₁₀(Z) + c·Z + d
  3 modes (β⁻, β⁺, α), 12 parameters, ground states

AI2 Zero-Param (QFD_NUCLIDE_ENGINE §24):
  Same formula, coefficients derived from (α, β, π, e)
  0 free parameters

Our Model (Three-Layer LaGrangian, this session):
  L = L_vacuum(β) + V_ext(Q) + T_vib(shape)
  9 species, ~40 parameters, tracked population (incl isomers)
"""

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.environ.get('TLG_DATA_DIR', os.path.join(_ROOT_DIR, 'data'))
RESULTS_DIR = os.environ.get('TLG_RESULTS_DIR', os.path.join(_ROOT_DIR, 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

import sys
sys.path.insert(0, os.path.join(_ROOT_DIR, '..', '..', '..'))
from qfd.shared_constants import ALPHA, BETA

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# Constants (from shared_constants)
# ══════════════════════════════════════════════════════════════════════
alpha_em = ALPHA
beta_val = BETA
PI = np.pi
E_val = np.e
LN10 = np.log(10)
N_MAX = 2 * PI * beta_val**3
A_CRIT = 2 * E_val**2 * beta_val**2
WIDTH = 2 * PI * beta_val**2

# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════
cs = pd.read_csv(os.path.join(DATA_DIR, 'clean_species_sorted.csv'))
cs['N'] = cs['A'] - cs['Z']
tracked = cs[cs['tracking_bin'] == 'tracked'].copy()

print("=" * 80)
print("UNIFIED MODEL COMPARISON")
print("AI2 Atomic Clock  vs  Three-Layer LaGrangian")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════
# AI2 MODEL — Atomic Clock v6 (Section 23 of QFD_NUCLIDE_ENGINE)
# ══════════════════════════════════════════════════════════════════════

# Fitted coefficients (12 parameters)
AI2_FITTED = {
    'beta-': {'a': -3.699, 'b': 5.880, 'c': -0.05362, 'd': 0.873},
    'beta+': {'a': -3.997, 'b': 7.477, 'c': -0.01343, 'd': -2.323},
    'alpha': {'a': -3.168, 'b': 26.778, 'c': -0.16302, 'd': -30.606},
}

# Zero-parameter coefficients (0 parameters, all from α/β/π/e)
AI2_ZEROPARAM = {
    'beta-': {'a': -PI * beta_val / E_val,  # -3.5171
              'b': 2.0,
              'c': 0.0,
              'd': 4 * PI / 3},              # 4.1888
    'beta+': {'a': -PI,                       # -3.1416
              'b': 2 * beta_val,              # 6.0865
              'c': 0.0,
              'd': -2 * beta_val / E_val},    # -2.2391
    'alpha': {'a': -E_val,                    # -2.7183
              'b': beta_val + 1,              # 4.0432
              'c': 0.0,
              'd': -(beta_val - 1)},          # -2.0432
}

def predict_ai2(df, mode, coefs):
    """AI2 prediction: a*sqrt|eps| + b*log10(Z) + c*Z + d"""
    eps = np.sqrt(np.abs(df['epsilon'].values))
    logZ = np.log10(df['Z'].values.astype(float))
    Z = df['Z'].values.astype(float)
    return coefs['a'] * eps + coefs['b'] * logZ + coefs['c'] * Z + coefs['d']


# ══════════════════════════════════════════════════════════════════════
# OUR MODEL — Three-Layer LaGrangian (refit on same data)
# ══════════════════════════════════════════════════════════════════════

# Feature builders
def f_sqrt_eps(df): return np.sqrt(np.abs(df['epsilon'].values))
def f_abs_eps(df): return np.abs(df['epsilon'].values)
def f_logZ(df): return np.log10(df['Z'].values.astype(float))
def f_Z(df): return df['Z'].values.astype(float)
def f_N_NMAX(df): return df['N'].values / N_MAX
def f_N_Z(df): return df['N'].values / df['Z'].values.astype(float)
def f_lnA(df): return np.log(df['A'].values.astype(float))
def f_ee(df): return (df['parity'] == 'ee').astype(float).values
def f_oo(df): return (df['parity'] == 'oo').astype(float).values
def f_logQ(df): return np.log10(np.maximum(df['Q_keV'].values.astype(float), 1.0))
def f_inv_sqrtQ(df):
    Q = df['Q_keV'].values.astype(float) / 1000
    return 1.0 / np.sqrt(np.maximum(Q, 0.01))
def f_log_pen(df):
    Q = df['Q_keV'].values.astype(float)
    V = df['V_coulomb_keV'].values.astype(float)
    return np.log10(np.maximum(Q, 1.0) / np.maximum(V, 1.0))
def f_deficit(df):
    return (df['V_coulomb_keV'].values - df['Q_keV'].values) / 1000
def f_log_trans_E(df):
    E = df['transition_energy_keV'].values.astype(float)
    return np.log10(np.maximum(E, 1.0))
def f_lambda(df):
    med = df['correct_lambda'].median()
    if pd.isna(med): med = 3.0
    return df['correct_lambda'].fillna(med).values
def f_lambda_sq(df):
    med = df['correct_lambda'].median()
    if pd.isna(med): med = 3.0
    return df['correct_lambda'].fillna(med).values ** 2

# Species configs
OUR_CONFIGS = {
    'beta-': {
        'features': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 1.0,
    },
    'beta+': {
        'features': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 1.0,
    },
    'alpha': {
        'features': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                     ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 5.0,
    },
    'IT': {
        'features': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                     ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                     ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'ridge': 2.0,
    },
    'IT_platypus': {
        'features': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                     ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                     ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'ridge': 2.0,
    },
    'SF': {
        'features': [('N_NMAX', f_N_NMAX), ('abs_eps', f_abs_eps),
                     ('logZ', f_logZ), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'ridge': 5.0,
    },
}

iso_map = {
    'beta-_iso': 'beta-', 'beta+_iso': 'beta+',
    'alpha_iso': 'alpha',
}

def fit_our_model(df, features, ridge=1.0):
    """Fit our three-layer model via ridge regression."""
    y = df['log_hl'].values
    X_parts = [func(df) for _, func in features]
    X_parts.append(np.ones(len(df)))
    X = np.column_stack(X_parts)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = valid.sum()
    if n < len(features) + 2:
        return None, None, None
    X_v, y_v = X[valid], y[valid]
    I_mat = np.eye(X_v.shape[1]); I_mat[-1, -1] = 0
    try:
        coef = np.linalg.solve(X_v.T @ X_v + ridge * I_mat, X_v.T @ y_v)
    except:
        coef, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
    return coef, X_v, y_v


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Head-to-head on shared modes (β⁻, β⁺, α)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 1: HEAD-TO-HEAD — Shared Modes (same data)")
print("=" * 80)
print("\nComparing on the 3 modes both models cover: beta-, beta+, alpha")
print("Population: tracked nuclides (ground + isomers)\n")

# Map our clean_species to AI2 modes
mode_map = {
    'beta-': 'beta-', 'beta-_iso': 'beta-',
    'beta+': 'beta+', 'beta+_iso': 'beta+',
    'alpha': 'alpha', 'alpha_iso': 'alpha',
}

def score(y, pred):
    valid = np.isfinite(y) & np.isfinite(pred)
    if valid.sum() == 0:
        return {}
    y_v, p_v = y[valid], pred[valid]
    resid = y_v - p_v
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(resid**2))
    solved = np.sum(np.abs(resid) < 1.0)
    p90 = np.percentile(np.abs(resid), 90)
    return {'n': valid.sum(), 'r2': r2, 'rmse': rmse, 'p90': p90,
            'solved': solved, 'pct': 100 * solved / valid.sum()}

print(f"{'Mode':10s}  {'Model':20s}  {'n':>5s}  {'R²':>7s}  {'RMSE':>6s}  {'P90':>6s}  {'<1dec':>5s}  {'%sol':>5s}  {'params':>6s}")
print("─" * 85)

all_results = []

for ai2_mode in ['beta-', 'beta+', 'alpha']:
    # Gather ALL tracked nuclides for this mode (ground + isomeric variants)
    sp_list = [k for k, v in mode_map.items() if v == ai2_mode]
    data = tracked[tracked['clean_species'].isin(sp_list)].copy()

    if len(data) < 10:
        continue

    y = data['log_hl'].values
    valid_y = np.isfinite(y)

    # --- AI2 Fitted (12-param) ---
    pred_ai2f = predict_ai2(data, ai2_mode, AI2_FITTED[ai2_mode])
    s = score(y, pred_ai2f)
    print(f"{ai2_mode:10s}  {'AI2 fitted (12p)':20s}  {s['n']:5d}  {s['r2']:7.3f}  {s['rmse']:6.2f}  {s['p90']:6.2f}  {s['solved']:5d}  {s['pct']:4.1f}%  {4:6d}")
    all_results.append({'mode': ai2_mode, 'model': 'AI2_fitted', **s, 'params': 4})

    # --- AI2 Zero-Param (0p) ---
    pred_ai2z = predict_ai2(data, ai2_mode, AI2_ZEROPARAM[ai2_mode])
    s = score(y, pred_ai2z)
    print(f"{'':10s}  {'AI2 zero-param (0p)':20s}  {s['n']:5d}  {s['r2']:7.3f}  {s['rmse']:6.2f}  {s['p90']:6.2f}  {s['solved']:5d}  {s['pct']:4.1f}%  {0:6d}")
    all_results.append({'mode': ai2_mode, 'model': 'AI2_zeroparam', **s, 'params': 0})

    # --- Our Model: Layer A only (geometric, no Q) ---
    # Use the parent species config but only Layer A features
    parent = ai2_mode
    config = OUR_CONFIGS[parent]
    # Layer A = all features except Q-value related
    q_feats = {'logQ', '1/sqrtQ', 'log_pen', 'deficit', 'log_transE', 'lambda', 'lambda_sq'}
    layer_a_feats = [(n, f) for n, f in config['features'] if n not in q_feats]

    coef_a, X_a, y_a = fit_our_model(data, layer_a_feats, config['ridge'])
    if coef_a is not None:
        pred_a = X_a @ coef_a
        s = score(y_a, pred_a)
        n_params_a = len(layer_a_feats) + 1
        print(f"{'':10s}  {f'Ours Layer A ({n_params_a}p)':20s}  {s['n']:5d}  {s['r2']:7.3f}  {s['rmse']:6.2f}  {s['p90']:6.2f}  {s['solved']:5d}  {s['pct']:4.1f}%  {n_params_a:6d}")
        all_results.append({'mode': ai2_mode, 'model': 'Ours_LayerA', **s, 'params': n_params_a})

    # --- Our Model: Layer A+B (full) ---
    coef_ab, X_ab, y_ab = fit_our_model(data, config['features'], config['ridge'])
    if coef_ab is not None:
        pred_ab = X_ab @ coef_ab
        s = score(y_ab, pred_ab)
        n_params_ab = len(config['features']) + 1
        print(f"{'':10s}  {f'Ours A+B ({n_params_ab}p)':20s}  {s['n']:5d}  {s['r2']:7.3f}  {s['rmse']:6.2f}  {s['p90']:6.2f}  {s['solved']:5d}  {s['pct']:4.1f}%  {n_params_ab:6d}")
        all_results.append({'mode': ai2_mode, 'model': 'Ours_LayerAB', **s, 'params': n_params_ab})

    print()


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: What AI2 cannot cover
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 2: SPECIES AI2 DOES NOT COVER")
print("=" * 80)
print("\nAI2 has no model for IT, SF, or proton emission.")
print("Our model covers all tracked species.\n")

print(f"{'Species':18s}  {'n':>5s}  {'R²':>7s}  {'RMSE':>6s}  {'P90':>6s}  {'<1dec':>5s}  {'%sol':>5s}")
print("─" * 60)

uncovered_n = 0
uncovered_solved = 0

for sp in ['IT', 'IT_platypus', 'SF']:
    data = tracked[tracked['clean_species'] == sp]
    if len(data) < 10:
        continue

    config = OUR_CONFIGS.get(sp, OUR_CONFIGS.get(sp.replace('_platypus', '')))
    if config is None:
        continue

    coef, X_v, y_v = fit_our_model(data, config['features'], config['ridge'])
    if coef is None:
        continue

    pred = X_v @ coef
    s = score(y_v, pred)
    print(f"{sp:18s}  {s['n']:5d}  {s['r2']:7.3f}  {s['rmse']:6.2f}  {s['p90']:6.2f}  {s['solved']:5d}  {s['pct']:4.1f}%")
    uncovered_n += s['n']
    uncovered_solved += s['solved']

print(f"\n  AI2 has NO predictions for {uncovered_n} nuclides in these channels.")
print(f"  Our model solves {uncovered_solved}/{uncovered_n} ({100*uncovered_solved/uncovered_n:.1f}%) within 1 decade.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: Global scorecard
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 3: GLOBAL SCORECARD")
print("=" * 80)

# AI2 predictions on ALL tracked (using its 3-mode model)
all_y_ai2 = []
all_pred_ai2f = []
all_pred_ai2z = []
all_y_ours = []
all_pred_ours = []

n_ai2_covers = 0
n_ai2_misses = 0

for sp in tracked['clean_species'].unique():
    data = tracked[tracked['clean_species'] == sp]
    if len(data) < 5:
        continue

    y = data['log_hl'].values
    valid = np.isfinite(y)

    # Does AI2 cover this species?
    ai2_mode = mode_map.get(sp, None)
    if ai2_mode and ai2_mode in AI2_FITTED:
        pred_f = predict_ai2(data, ai2_mode, AI2_FITTED[ai2_mode])
        pred_z = predict_ai2(data, ai2_mode, AI2_ZEROPARAM[ai2_mode])
        both_valid = valid & np.isfinite(pred_f)
        all_y_ai2.extend(y[both_valid])
        all_pred_ai2f.extend(pred_f[both_valid])
        all_pred_ai2z.extend(pred_z[both_valid])
        n_ai2_covers += both_valid.sum()
    else:
        n_ai2_misses += valid.sum()

    # Our model always covers
    parent = iso_map.get(sp, sp)
    config = OUR_CONFIGS.get(parent)
    if config is None:
        continue

    coef, X_v, y_v = fit_our_model(data, config['features'], config['ridge'])
    if coef is not None:
        pred = X_v @ coef
        all_y_ours.extend(y_v)
        all_pred_ours.extend(pred)

# Score AI2
all_y_ai2 = np.array(all_y_ai2)
all_pred_ai2f = np.array(all_pred_ai2f)
all_pred_ai2z = np.array(all_pred_ai2z)
all_y_ours = np.array(all_y_ours)
all_pred_ours = np.array(all_pred_ours)

def global_score(y, p, label):
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - np.sum((y - p)**2) / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y - p)**2))
    solved = np.sum(np.abs(y - p) < 1.0)
    pct = 100 * solved / len(y)
    print(f"  {label:35s}  n={len(y):5d}  R²={r2:.3f}  RMSE={rmse:.2f}  "
          f"<1dec={solved:5d} ({pct:.1f}%)")

print(f"\nAI2 covers: {n_ai2_covers} nuclides ({100*n_ai2_covers/(n_ai2_covers+n_ai2_misses):.1f}%)")
print(f"AI2 misses: {n_ai2_misses} nuclides ({100*n_ai2_misses/(n_ai2_covers+n_ai2_misses):.1f}%)")
print()

# AI2 on its coverage
global_score(all_y_ai2, all_pred_ai2f, "AI2 fitted (12p, 3 modes)")
global_score(all_y_ai2, all_pred_ai2z, "AI2 zero-param (0p, 3 modes)")

# Our model on SAME coverage as AI2 (fair comparison)
# Need to get our predictions for just the AI2-covered species
ours_on_ai2_y = []
ours_on_ai2_p = []
for sp in tracked['clean_species'].unique():
    ai2_mode = mode_map.get(sp, None)
    if not (ai2_mode and ai2_mode in AI2_FITTED):
        continue
    data = tracked[tracked['clean_species'] == sp]
    parent = iso_map.get(sp, sp)
    config = OUR_CONFIGS.get(parent)
    if config is None:
        continue
    coef, X_v, y_v = fit_our_model(data, config['features'], config['ridge'])
    if coef is not None:
        pred = X_v @ coef
        ours_on_ai2_y.extend(y_v)
        ours_on_ai2_p.extend(pred)

ours_on_ai2_y = np.array(ours_on_ai2_y)
ours_on_ai2_p = np.array(ours_on_ai2_p)
global_score(ours_on_ai2_y, ours_on_ai2_p, "Ours A+B (on AI2's 3 modes)")

print()
# Our model on ALL species
global_score(all_y_ours, all_pred_ours, "Ours A+B (all 9 species)")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: Improvement attribution
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 4: WHERE DOES THE IMPROVEMENT COME FROM?")
print("=" * 80)
print("\nDecompose the R² gap between AI2 and our model.\n")

for ai2_mode in ['beta-', 'beta+', 'alpha']:
    sp_list = [k for k, v in mode_map.items() if v == ai2_mode]
    data = tracked[tracked['clean_species'].isin(sp_list)].copy()
    y = data['log_hl'].values

    # AI2 fitted
    pred_ai2 = predict_ai2(data, ai2_mode, AI2_FITTED[ai2_mode])
    valid = np.isfinite(y) & np.isfinite(pred_ai2)
    y_v = y[valid]
    p_ai2 = pred_ai2[valid]
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2_ai2 = 1 - np.sum((y_v - p_ai2)**2) / ss_tot

    # Our Layer A only
    parent = ai2_mode
    config = OUR_CONFIGS[parent]
    q_feats = {'logQ', '1/sqrtQ', 'log_pen', 'deficit'}
    layer_a_feats = [(n, f) for n, f in config['features'] if n not in q_feats]
    coef_a, X_a, y_a = fit_our_model(data, layer_a_feats, config['ridge'])
    pred_a = X_a @ coef_a
    r2_a = 1 - np.sum((y_a - pred_a)**2) / np.sum((y_a - y_a.mean())**2)

    # Our A+B
    coef_ab, X_ab, y_ab = fit_our_model(data, config['features'], config['ridge'])
    pred_ab = X_ab @ coef_ab
    r2_ab = 1 - np.sum((y_ab - pred_ab)**2) / np.sum((y_ab - y_ab.mean())**2)

    print(f"\n{ai2_mode}:")
    print(f"  AI2 fitted (4p):          R² = {r2_ai2:.3f}")
    print(f"  +Parity, N/Z, N/NMAX:     R² = {r2_a:.3f}  (ΔR² = {r2_a - r2_ai2:+.3f} — more Layer A features)")
    print(f"  +Q-values (Layer B):      R² = {r2_ab:.3f}  (ΔR² = {r2_ab - r2_a:+.3f} — external energy)")
    print(f"  Total improvement:                         ΔR² = {r2_ab - r2_ai2:+.3f}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: Summary table
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 5: SUMMARY")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL COMPARISON SUMMARY                        │
├─────────────────────┬──────────────────┬────────────────────────────┤
│                     │  AI2 Atomic Clock│  Three-Layer LaGrangian    │
├─────────────────────┼──────────────────┼────────────────────────────┤
│ Free parameters     │       12         │          ~40               │
│ Zero-param version  │       Yes (0p)   │       Partial (9p)         │
│ Modes covered       │       3          │          9                 │
│ Species             │  β⁻, β⁺, α      │  + IT, SF, isomers         │
│ IT coverage         │       None       │  555 nuclides              │
│ SF coverage         │       None       │  49 nuclides               │
│ Isomer coverage     │       None       │  478 nuclides              │
├─────────────────────┼──────────────────┼────────────────────────────┤
│ Physics layers      │       1          │          3                 │
│ Layer A (vacuum)    │       Yes        │       Yes (expanded)       │
│ Layer B (energy)    │       No         │       Yes (Q, V_C)         │
│ Layer C (Lyapunov)  │       No         │    Yes (statistical width) │
├─────────────────────┼──────────────────┼────────────────────────────┤
│ Key insight: stress │  √|ε| for all    │  √|ε| beta, |ε| alpha     │
│ Key insight: parity │  Not included    │  ee/oo shifts              │
│ Key insight: Q      │  Not included    │  Species-specific coupling │
│ Key insight: r₀     │  Not derived     │  π²/(β·e) = 1.193 fm      │
└─────────────────────┴──────────────────┴────────────────────────────┘
""")

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_results.csv'), index=False)
print(f"Saved: {os.path.join(RESULTS_DIR, 'model_comparison_results.csv')}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
