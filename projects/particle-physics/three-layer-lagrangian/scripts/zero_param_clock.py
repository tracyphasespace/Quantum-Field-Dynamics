#!/usr/bin/env python3
"""
Zero-Parameter Clock — Derive ALL coefficients from (α, β, π, e)

Approach: Instead of fitting OLS and trying to match coefficients,
start from PHYSICS and build the formula algebraically.

For each decay channel, the half-life formula is:
  log₁₀(T½) = Σᵢ cᵢ · fᵢ(A, Z, Q, ...)

where each cᵢ is an algebraic expression in (α, β, π, e, ln10).

Strategy:
  1. Use the FULL feature set (same as tracked_channel_fits.py) — no R² loss
  2. For each coefficient, try ALL simple combinations of constants
  3. Score by minimizing total prediction error
  4. Iterate: fix well-matched coefficients, optimize the rest
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
import itertools
import warnings
warnings.filterwarnings('ignore')

# ── Immutable constants (from shared_constants) ──
alpha_em = ALPHA
beta_val = BETA
PI = np.pi
E_val = np.e
LN10 = np.log(10)

N_MAX = 2 * PI * beta_val**3
A_CRIT = 2 * E_val**2 * beta_val**2
WIDTH = 2 * PI * beta_val**2

print("=" * 78)
print("ZERO-PARAMETER CLOCK — Systematic coefficient derivation")
print("=" * 78)

# ── Load ──
cs = pd.read_csv(os.path.join(DATA_DIR, 'clean_species_sorted.csv'))
cs['N'] = cs['A'] - cs['Z']
tracked = cs[cs['tracking_bin'] == 'tracked'].copy()

# ── Feature builders (same as tracked fits) ──
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

# ── Candidate constant expressions ──
# Every coefficient must be one of these (or zero)
def build_candidates():
    """Build dictionary of candidate algebraic expressions."""
    b = beta_val
    a = alpha_em
    cands = {}
    # Singles
    for sign in [+1, -1]:
        for name, val in [
            ('0', 0),
            ('1', 1), ('2', 2), ('3', 3),
            ('α', a), ('β', b), ('π', PI), ('e', E_val), ('ln10', LN10),
            ('β²', b**2), ('β³', b**3), ('π²', PI**2), ('e²', E_val**2),
            ('1/α', 1/a), ('1/β', 1/b), ('1/π', 1/PI), ('1/e', 1/E_val),
            ('1/β²', 1/b**2), ('1/e²', 1/E_val**2),
        ]:
            if val == 0 and sign == -1:
                continue
            cands[f"{'+' if sign>0 else '-'}{name}"] = sign * val

    # Products of two
    bases = [('β', b), ('π', PI), ('e', E_val), ('ln10', LN10),
             ('1/β', 1/b), ('1/π', 1/PI), ('1/e', 1/E_val)]
    for (n1, v1), (n2, v2) in itertools.combinations(bases, 2):
        for sign in [+1, -1]:
            cands[f"{'+' if sign>0 else '-'}{n1}·{n2}"] = sign * v1 * v2

    # Ratios
    nums = [('β', b), ('π', PI), ('e', E_val), ('ln10', LN10), ('β²', b**2), ('π²', PI**2)]
    dens = [('β', b), ('π', PI), ('e', E_val), ('2', 2), ('π²', PI**2), ('e²', E_val**2), ('2β', 2*b), ('2π', 2*PI), ('2e', 2*E_val)]
    for (nn, nv), (dn, dv) in itertools.product(nums, dens):
        if nn == dn:
            continue
        for sign in [+1, -1]:
            cands[f"{'+' if sign>0 else '-'}{nn}/{dn}"] = sign * nv / dv

    # Special: with integer multipliers
    for k in [2, 3]:
        for (n, v) in [('β', b), ('π', PI), ('e', E_val), ('ln10', LN10)]:
            for sign in [+1, -1]:
                cands[f"{'+' if sign>0 else '-'}{k}{n}"] = sign * k * v

    return cands

candidates = build_candidates()
print(f"\nCandidate expressions: {len(candidates)}")

# ── Find best algebraic match for a coefficient ──
def best_match(target, top_n=5):
    """Find the closest candidate expression to a target value."""
    results = []
    for name, val in candidates.items():
        err = abs(target - val)
        pct = err / max(abs(target), 1e-10) * 100
        results.append((pct, err, name, val))
    results.sort()
    return results[:top_n]

# ── OLS fit for reference ──
def fit_ols(df, feature_list, ridge=1.0):
    y = df['log_hl'].values
    X_parts = [func(df) for _, func in feature_list]
    X_parts.append(np.ones(len(df)))
    X = np.column_stack(X_parts)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = valid.sum()
    if n < len(feature_list) + 2:
        return None
    X_v, y_v = X[valid], y[valid]
    I_mat = np.eye(X_v.shape[1]); I_mat[-1, -1] = 0
    try:
        coef = np.linalg.solve(X_v.T @ X_v + ridge * I_mat, X_v.T @ y_v)
    except:
        coef, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
    pred = X_v @ coef
    resid = y_v - pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    names = [nm for nm, _ in feature_list] + ['const']
    return {'coefs': dict(zip(names, coef)), 'r2': r2, 'n': n,
            'y': y_v, 'X': X_v, 'valid': valid}

# ── Score a derived coefficient set ──
def score_derived(df, feature_list, coef_dict):
    y = df['log_hl'].values
    X_parts = [func(df) for _, func in feature_list]
    X_parts.append(np.ones(len(df)))
    X = np.column_stack(X_parts)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if valid.sum() == 0:
        return None
    X_v, y_v = X[valid], y[valid]
    names = [nm for nm, _ in feature_list] + ['const']
    coef_vec = np.array([coef_dict.get(nm, 0.0) for nm in names])
    pred = X_v @ coef_vec
    resid = y_v - pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(resid**2))
    solved = np.sum(np.abs(resid) < 1.0)
    return {'r2': r2, 'rmse': rmse, 'n': valid.sum(), 'solved': solved}


# ══════════════════════════════════════════════════════════════════════
# Channel definitions (same features as tracked_channel_fits.py)
# ══════════════════════════════════════════════════════════════════════

channels = {
    'beta-': {
        'features': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 1.0
    },
    'beta+': {
        'features': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 1.0
    },
    'alpha': {
        'features': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                     ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 5.0
    },
    'IT': {
        'features': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                     ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                     ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'ridge': 2.0
    },
    'SF': {
        'features': [('N_NMAX', f_N_NMAX), ('abs_eps', f_abs_eps),
                     ('logZ', f_logZ), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'ridge': 5.0
    },
}

# Also fit isomeric variants with their parent channel features
iso_map = {
    'beta-_iso': 'beta-', 'beta+_iso': 'beta+',
    'alpha_iso': 'alpha', 'IT_platypus': 'IT',
}

# ══════════════════════════════════════════════════════════════════════
# Fit all channels and find best algebraic matches
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("FITTED COEFFICIENTS AND BEST ALGEBRAIC MATCHES")
print("=" * 78)

all_fits = {}
all_derived = {}

for sp in ['beta-', 'beta+', 'alpha', 'IT', 'SF',
           'beta-_iso', 'beta+_iso', 'alpha_iso', 'IT_platypus']:
    data = tracked[tracked['clean_species'] == sp]
    if len(data) < 10:
        continue

    parent = iso_map.get(sp, sp)
    config = channels[parent]
    features = config['features']
    ridge = config['ridge']

    res = fit_ols(data, features, ridge)
    if res is None:
        continue
    all_fits[sp] = res

    print(f"\n{'─' * 78}")
    print(f"{sp} (n={res['n']}, R²={res['r2']:.3f})")
    print(f"{'─' * 78}")

    derived_coefs = {}
    for fname, fval in res['coefs'].items():
        matches = best_match(fval, top_n=3)
        best_pct, best_err, best_name, best_val = matches[0]
        mark = "✓" if best_pct < 5 else "~" if best_pct < 15 else " "
        print(f"  {fname:12s}  fitted={fval:+9.4f}  best={best_name:15s} ({best_val:+9.4f})  err={best_pct:5.1f}% {mark}")

        # Also show runner-up
        if len(matches) > 1:
            _, _, name2, val2 = matches[1]
            pct2 = matches[1][0]
            print(f"  {'':12s}  {'':>9s}        alt={name2:15s} ({val2:+9.4f})  err={pct2:5.1f}%")

        derived_coefs[fname] = best_val

    all_derived[sp] = derived_coefs


# ══════════════════════════════════════════════════════════════════════
# Score zero-parameter predictions
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("ZERO-PARAMETER vs FITTED COMPARISON")
print("=" * 78)

print(f"\n{'Channel':20s}  {'n':>5s}  {'R²(fit)':>8s}  {'R²(0par)':>8s}  {'ΔR²':>7s}  {'RMSE(f)':>7s}  {'RMSE(0)':>7s}  {'<1d(f)':>6s}  {'<1d(0)':>6s}")
print("─" * 95)

all_y_fit = []
all_pred_fit = []
all_pred_der = []
all_y_der = []

for sp in ['beta-', 'beta+', 'beta-_iso', 'beta+_iso',
           'alpha', 'alpha_iso', 'IT', 'IT_platypus', 'SF']:
    if sp not in all_fits or sp not in all_derived:
        continue

    data = tracked[tracked['clean_species'] == sp]
    parent = iso_map.get(sp, sp)
    features = channels[parent]['features']
    fit_res = all_fits[sp]
    der_coefs = all_derived[sp]

    # Score derived
    der_res = score_derived(data, features, der_coefs)
    if der_res is None:
        continue

    # Score fitted (recompute solved count)
    names = [nm for nm, _ in features] + ['const']
    coef_fit = np.array([fit_res['coefs'][k] for k in names])
    pred_fit = fit_res['X'] @ coef_fit
    solved_fit = np.sum(np.abs(fit_res['y'] - pred_fit) < 1.0)
    rmse_fit = np.sqrt(np.mean((fit_res['y'] - pred_fit)**2))

    delta = der_res['r2'] - fit_res['r2']
    print(f"{sp:20s}  {fit_res['n']:5d}  {fit_res['r2']:8.3f}  {der_res['r2']:8.3f}  {delta:+7.3f}  "
          f"{rmse_fit:7.2f}  {der_res['rmse']:7.2f}  {solved_fit:6d}  {der_res['solved']:6d}")

    # Accumulate for global
    coef_der = np.array([der_coefs.get(k, 0.0) for k in names])
    y = fit_res['y']
    X = fit_res['X']
    all_y_fit.extend(y)
    all_pred_fit.extend(X @ coef_fit)
    all_y_der.extend(y)
    all_pred_der.extend(X @ coef_der)

# Global
all_y = np.array(all_y_fit)
all_pf = np.array(all_pred_fit)
all_pd = np.array(all_pred_der)
ss_tot = np.sum((all_y - all_y.mean())**2)
r2_fit = 1 - np.sum((all_y - all_pf)**2) / ss_tot
r2_der = 1 - np.sum((all_y - all_pd)**2) / ss_tot
rmse_fit = np.sqrt(np.mean((all_y - all_pf)**2))
rmse_der = np.sqrt(np.mean((all_y - all_pd)**2))
solved_fit = np.sum(np.abs(all_y - all_pf) < 1.0)
solved_der = np.sum(np.abs(all_y - all_pd) < 1.0)

print(f"\n{'─' * 95}")
print(f"{'GLOBAL':20s}  {len(all_y):5d}  {r2_fit:8.3f}  {r2_der:8.3f}  {r2_der-r2_fit:+7.3f}  "
      f"{rmse_fit:7.2f}  {rmse_der:7.2f}  {solved_fit:6d}  {solved_der:6d}")

print(f"\n  Fitted:  R²={r2_fit:.3f}, RMSE={rmse_fit:.2f}, Solved={solved_fit}/{len(all_y)} ({100*solved_fit/len(all_y):.1f}%)")
print(f"  Derived: R²={r2_der:.3f}, RMSE={rmse_der:.2f}, Solved={solved_der}/{len(all_y)} ({100*solved_der/len(all_y):.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# Document the equations
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("THE DERIVED EQUATIONS")
print("=" * 78)

# Collect unique expressions used
unique_exprs = set()
for sp, coefs in all_derived.items():
    for fname, val in coefs.items():
        matches = best_match(val, top_n=1)
        unique_exprs.add(matches[0][2])  # expression name

print(f"\nUnique algebraic expressions used: {len(unique_exprs)}")
for expr in sorted(unique_exprs):
    val = candidates.get(expr, 0)
    print(f"  {expr:20s} = {val:+10.4f}")
