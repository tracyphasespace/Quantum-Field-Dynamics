#!/usr/bin/env python3
"""
LaGrangian Decomposition — Three-Layer Half-Life Structure

Tracy McSheery directive: "find the LaGrangian in terms of Beta and then we
add external energy and modes of vibration that are not Beta related but
are mechanistic"

Three layers:
  Layer A (beta-LaGrangian): Universal vacuum stiffness — stress, parity,
    geometric ratios. All coefficients derivable from (alpha, beta, pi, e).
    This is the part that comes from the vacuum density field.

  Layer B (External Energy): Q-values from AME2020, Coulomb barriers,
    transition energies, multipolarities. Measured per-nuclide, NOT
    derivable from beta. This is the energy landscape the soliton sits in.

  Layer C (Vibrational Modes / Dzhanibekov): The residual after A+B.
    Per-nucleus shape physics — triaxiality, moments of inertia,
    Lyapunov instability of the intermediate axis. Mechanistic but
    NOT reducible to beta because it depends on the specific 3D geometry
    of each nucleus (three principal moments of inertia).
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
# Constants (all from shared_constants)
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
# Load and filter
# ══════════════════════════════════════════════════════════════════════
cs = pd.read_csv(os.path.join(DATA_DIR, 'clean_species_sorted.csv'))
cs['N'] = cs['A'] - cs['Z']
tracked = cs[cs['tracking_bin'] == 'tracked'].copy()

print("=" * 80)
print("LAGRANGIAN DECOMPOSITION — Three Layers of Half-Life Physics")
print("=" * 80)
print(f"\nTracked population: {len(tracked)} nuclides")

# ══════════════════════════════════════════════════════════════════════
# Feature builders
# ══════════════════════════════════════════════════════════════════════

# --- Layer A: geometric / beta-derivable ---
def f_sqrt_eps(df): return np.sqrt(np.abs(df['epsilon'].values))
def f_abs_eps(df): return np.abs(df['epsilon'].values)
def f_logZ(df): return np.log10(df['Z'].values.astype(float))
def f_Z(df): return df['Z'].values.astype(float)
def f_N_NMAX(df): return df['N'].values / N_MAX
def f_N_Z(df): return df['N'].values / df['Z'].values.astype(float)
def f_lnA(df): return np.log(df['A'].values.astype(float))
def f_ee(df): return (df['parity'] == 'ee').astype(float).values
def f_oo(df): return (df['parity'] == 'oo').astype(float).values

# --- Layer B: external energy ---
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

# ══════════════════════════════════════════════════════════════════════
# Species: Layer A vs Layer B feature separation
# ══════════════════════════════════════════════════════════════════════

species_layers = {
    'beta-': {
        'layer_a': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                    ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                    ('ee', f_ee), ('oo', f_oo)],
        'layer_b': [('logQ', f_logQ)],
        'ridge': 1.0,
    },
    'beta+': {
        'layer_a': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                    ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                    ('ee', f_ee), ('oo', f_oo)],
        'layer_b': [('logQ', f_logQ)],
        'ridge': 1.0,
    },
    'alpha': {
        'layer_a': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                    ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                    ('ee', f_ee), ('oo', f_oo)],
        'layer_b': [('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                    ('deficit', f_deficit)],
        'ridge': 5.0,
    },
    'IT': {
        'layer_a': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ),
                    ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                    ('ee', f_ee), ('oo', f_oo)],
        'layer_b': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                    ('lambda_sq', f_lambda_sq)],
        'ridge': 2.0,
    },
    'SF': {
        'layer_a': [('N_NMAX', f_N_NMAX), ('abs_eps', f_abs_eps),
                    ('logZ', f_logZ), ('N_Z', f_N_Z),
                    ('ee', f_ee), ('oo', f_oo)],
        'layer_b': [],  # no external energy features
        'ridge': 5.0,
    },
}

# Map isomeric variants to parent config
iso_map = {
    'beta-_iso': 'beta-', 'beta+_iso': 'beta+',
    'alpha_iso': 'alpha', 'IT_platypus': 'IT',
}

# ══════════════════════════════════════════════════════════════════════
# Algebraic coefficient library (from zero_param_clock.py results)
# Best matches within 5% for each feature across channels
# ══════════════════════════════════════════════════════════════════════

b = beta_val

ALGEBRAIC = {
    # Stress slopes (Layer A, universal)
    'sqrt_eps': -PI**2 / (2 * b),   # -1.621  (beta: tunneling stress)
    'abs_eps':  -b**2 / E_val,       # -3.407  (alpha: neck strain)

    # Geometry scales
    'logZ':     PI**2 / E_val,       # +3.627  (charge scale)
    'Z':       -PI**2 / (2 * b * E_val**2),  # approximate: -0.0668
    'N_NMAX':  -b * E_val,           # -8.273  (proximity to ceiling)
    'N_Z':     -LN10 / PI,           # -0.733  (neutron excess)
    'lnA':     -PI**2 / E_val,       # -3.627  (mass scale, = -logZ)

    # Parity shifts
    'ee':       LN10 / PI**2,        # +0.234  (even-even bonus)
    'oo':      -LN10 / PI**2,        # -0.234  (odd-odd penalty)

    # External energy (Layer B — these NEED fitting, not from beta)
    'logQ':    -1.0,                  # placeholder (energy scale)
    '1/sqrtQ':  b * 1.2,             # ~beta * r0  (Gamow penetration)
    'log_pen': -b,                    # penetration probability
    'deficit':  0.1,                  # barrier deficit

    # IT external (Layer B)
    'log_transE': -2.0,              # Weisskopf
    'lambda':     1.0,               # multipolarity
    'lambda_sq':  0.1,               # quadratic
}


# ══════════════════════════════════════════════════════════════════════
# Fitter
# ══════════════════════════════════════════════════════════════════════

def fit_layer(df, features, ridge=1.0):
    """Fit OLS on given features. Returns coefs, R², predictions, etc."""
    y = df['log_hl'].values
    X_parts = [func(df) for _, func in features]
    X_parts.append(np.ones(len(df)))
    X = np.column_stack(X_parts)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = valid.sum()
    if n < len(features) + 2:
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
    rmse = np.sqrt(np.mean(resid**2))
    solved = np.sum(np.abs(resid) < 1.0)
    names = [nm for nm, _ in features] + ['const']
    return {
        'coefs': dict(zip(names, coef)),
        'r2': r2, 'rmse': rmse, 'n': n,
        'solved': solved, 'pct': 100 * solved / n,
        'y': y_v, 'X': X_v, 'pred': pred, 'resid': resid,
        'valid': valid, 'ss_tot': ss_tot,
    }


def score_with_coefs(df, features, coef_dict):
    """Score predictions using fixed (algebraic) coefficients."""
    y = df['log_hl'].values
    X_parts = [func(df) for _, func in features]
    X_parts.append(np.ones(len(df)))
    X = np.column_stack(X_parts)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if valid.sum() == 0:
        return None
    X_v, y_v = X[valid], y[valid]
    names = [nm for nm, _ in features] + ['const']
    coef_vec = np.array([coef_dict.get(nm, 0.0) for nm in names])
    pred = X_v @ coef_vec
    resid = y_v - pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(resid**2))
    solved = np.sum(np.abs(resid) < 1.0)
    return {
        'r2': r2, 'rmse': rmse, 'n': valid.sum(),
        'solved': solved, 'pct': 100 * solved / valid.sum(),
        'y': y_v, 'pred': pred, 'resid': resid,
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Layer-by-layer decomposition per species
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 1: THREE-LAYER DECOMPOSITION")
print("=" * 80)

header = (f"{'Species':18s}  {'n':>5s}  "
          f"{'R²_A':>7s}  {'R²_AB':>7s}  {'R²_full':>7s}  "
          f"{'A%':>5s}  {'B%':>5s}  {'C%':>5s}  "
          f"{'RMSE_A':>7s}  {'RMSE_AB':>7s}")
print(f"\n{header}")
print("─" * 100)

all_decomp = []

for sp in ['beta-', 'beta+', 'alpha', 'IT', 'IT_platypus', 'SF',
           'beta-_iso', 'beta+_iso', 'alpha_iso']:
    data = tracked[tracked['clean_species'] == sp]
    if len(data) < 15:
        continue

    parent = iso_map.get(sp, sp)
    config = species_layers[parent]
    feats_a = config['layer_a']
    feats_b = config['layer_b']
    feats_ab = feats_a + feats_b
    ridge = config['ridge']

    # --- Fit Layer A only (geometric/beta features) ---
    res_a = fit_layer(data, feats_a, ridge)
    if res_a is None:
        continue

    # --- Fit Layer A+B (add external energy) ---
    if len(feats_b) > 0:
        res_ab = fit_layer(data, feats_ab, ridge)
    else:
        res_ab = res_a  # no Layer B features

    if res_ab is None:
        continue

    # --- Compute variance fractions ---
    # Use a COMMON ss_tot (full dataset variance) so fractions add up
    ss_tot = res_a['ss_tot']
    r2_a = res_a['r2']
    r2_ab = res_ab['r2']

    # Layer A explains r2_a of total variance
    # Layer B adds r2_ab - r2_a
    # Layer C (residual) = 1 - r2_ab
    frac_a = r2_a * 100
    frac_b = (r2_ab - r2_a) * 100
    frac_c = (1 - r2_ab) * 100

    print(f"{sp:18s}  {res_a['n']:5d}  "
          f"{r2_a:7.3f}  {r2_ab:7.3f}  {r2_ab:7.3f}  "
          f"{frac_a:5.1f}  {frac_b:5.1f}  {frac_c:5.1f}  "
          f"{res_a['rmse']:7.2f}  {res_ab['rmse']:7.2f}")

    all_decomp.append({
        'species': sp, 'n': res_a['n'],
        'r2_a': r2_a, 'r2_ab': r2_ab,
        'frac_a': frac_a, 'frac_b': frac_b, 'frac_c': frac_c,
        'rmse_a': res_a['rmse'], 'rmse_ab': res_ab['rmse'],
        'solved_a': res_a['solved'], 'solved_ab': res_ab['solved'],
        'coefs_a': res_a['coefs'], 'coefs_ab': res_ab['coefs'],
    })

# Global summary
print(f"\n{'─' * 100}")
total_n = sum(d['n'] for d in all_decomp)
total_ss_tot = 0
total_ss_a = 0
total_ss_ab = 0
for d in all_decomp:
    # Rebuild from r2 and n, assuming roughly equal per-nuclide variance contribution
    # Actually use weighted average properly
    n = d['n']
    total_ss_a += d['rmse_a']**2 * n
    total_ss_ab += d['rmse_ab']**2 * n

# Compute global half-life stats for proper R²
all_y_vals = []
for sp in ['beta-', 'beta+', 'alpha', 'IT', 'IT_platypus', 'SF',
           'beta-_iso', 'beta+_iso', 'alpha_iso']:
    data = tracked[tracked['clean_species'] == sp]
    y = data['log_hl'].dropna().values
    all_y_vals.extend(y)
all_y_vals = np.array(all_y_vals)
global_var = np.var(all_y_vals) * len(all_y_vals)

global_r2_a = 1 - total_ss_a / global_var if global_var > 0 else 0
global_r2_ab = 1 - total_ss_ab / global_var if global_var > 0 else 0

print(f"{'GLOBAL':18s}  {total_n:5d}  "
      f"{global_r2_a:7.3f}  {global_r2_ab:7.3f}  {global_r2_ab:7.3f}  "
      f"{global_r2_a*100:5.1f}  {(global_r2_ab-global_r2_a)*100:5.1f}  {(1-global_r2_ab)*100:5.1f}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: Layer A — Algebraic (zero-param) vs Fitted
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 2: LAYER A — Zero-Parameter (algebraic) vs Fitted")
print("=" * 80)
print("\nFor each species: fit Layer A, then replace with algebraic coefficients")
print("from (alpha, beta, pi, e). The gap = cost of zero-parameter constraint.\n")

# First, fit the algebraic constant (intercept) per channel
# This is the ONE constant we allow — the overall offset
print(f"{'Species':18s}  {'n':>5s}  {'R²_fit':>7s}  {'R²_alg':>7s}  {'R²_alg+c':>8s}  {'ΔR²':>7s}  {'RMSE_f':>7s}  {'RMSE_a':>7s}")
print("─" * 85)

for d in all_decomp:
    sp = d['species']
    data = tracked[tracked['clean_species'] == sp]
    parent = iso_map.get(sp, sp)
    feats_a = species_layers[parent]['layer_a']

    # Build algebraic coefficient vector for Layer A features
    alg_coefs = {}
    for fname, _ in feats_a:
        if fname in ALGEBRAIC:
            alg_coefs[fname] = ALGEBRAIC[fname]
        else:
            alg_coefs[fname] = 0.0
    alg_coefs['const'] = 0.0  # start with zero intercept

    # Score with pure algebraic (no constant)
    res_alg0 = score_with_coefs(data, feats_a, alg_coefs)

    # Now find best-fit constant (just the intercept) with algebraic slopes
    y = data['log_hl'].values
    X_parts = [func(data) for _, func in feats_a]
    X = np.column_stack(X_parts)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_v, y_v = X[valid], y[valid]

    names_a = [nm for nm, _ in feats_a]
    alg_vec = np.array([alg_coefs.get(nm, 0.0) for nm in names_a])
    pred_slopes = X_v @ alg_vec
    # Best-fit constant = mean of (y - algebraic prediction)
    best_const = np.mean(y_v - pred_slopes)
    pred_alg_c = pred_slopes + best_const
    resid_alg_c = y_v - pred_alg_c
    ss_res_alg_c = np.sum(resid_alg_c**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2_alg_c = 1 - ss_res_alg_c / ss_tot if ss_tot > 0 else 0
    rmse_alg_c = np.sqrt(np.mean(resid_alg_c**2))

    r2_fit = d['r2_a']
    r2_alg = res_alg0['r2'] if res_alg0 else -999
    delta = r2_alg_c - r2_fit

    print(f"{sp:18s}  {d['n']:5d}  {r2_fit:7.3f}  {r2_alg:7.3f}  {r2_alg_c:8.3f}  {delta:+7.3f}  {d['rmse_a']:7.2f}  {rmse_alg_c:7.2f}")

    d['r2_alg'] = r2_alg
    d['r2_alg_c'] = r2_alg_c
    d['best_const'] = best_const

    # Print algebraic equations
    print(f"  Algebraic: log_hl = ", end="")
    terms = []
    for fname in names_a:
        c = alg_coefs.get(fname, 0)
        if abs(c) > 1e-6:
            terms.append(f"{c:+.4f}*{fname}")
    terms.append(f"{best_const:+.3f}")
    print(" ".join(terms))


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: Layer B coefficients — What external energy teaches us
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 3: LAYER B COEFFICIENTS — External Energy Physics")
print("=" * 80)
print("\nThese coefficients multiply measured energies (Q, V_C, transition E).")
print("They are NOT derivable from beta — they are the coupling constants")
print("between the soliton's internal geometry and the external energy landscape.\n")

for d in all_decomp:
    sp = d['species']
    parent = iso_map.get(sp, sp)
    feats_b = species_layers[parent]['layer_b']
    if not feats_b:
        print(f"{sp:18s}  (no Layer B features — all geometric)")
        continue

    coefs_ab = d['coefs_ab']
    print(f"\n{sp}:")
    for fname, _ in feats_b:
        c = coefs_ab.get(fname, 0)
        print(f"  {fname:15s}  coef = {c:+9.4f}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: Layer C — Residual structure (Dzhanibekov / Lyapunov)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 4: LAYER C — Residual Structure (Dzhanibekov Physics)")
print("=" * 80)
print("\nThe residual after Layers A+B encodes per-nucleus shape physics.")
print("This is the Dzhanibekov/intermediate-axis instability contribution.")
print("NOT reducible to beta because it depends on individual 3D geometry.\n")

for d in all_decomp:
    sp = d['species']
    parent = iso_map.get(sp, sp)
    config = species_layers[parent]
    feats_ab = config['layer_a'] + config['layer_b']
    data = tracked[tracked['clean_species'] == sp]
    ridge = config['ridge']

    res_ab = fit_layer(data, feats_ab, ridge)
    if res_ab is None:
        continue

    resid = res_ab['resid']
    y = res_ab['y']

    # Characterize the residual
    print(f"\n{sp} (n={res_ab['n']}):")
    print(f"  Layer C variance: {(1 - res_ab['r2'])*100:.1f}% of total")
    print(f"  Residual RMSE: {res_ab['rmse']:.2f} decades")
    print(f"  Residual P10/P50/P90: {np.percentile(np.abs(resid), 10):.2f} / "
          f"{np.percentile(np.abs(resid), 50):.2f} / {np.percentile(np.abs(resid), 90):.2f}")
    print(f"  Solved (<1 decade): {res_ab['solved']}/{res_ab['n']} ({res_ab['pct']:.1f}%)")

    # Check if residual correlates with parity (proxy for triaxiality)
    valid_idx = np.where(res_ab['valid'])[0]
    data_valid = data.iloc[valid_idx]
    par = data_valid['parity'].values

    for p in ['ee', 'eo', 'oo']:
        mask = par == p
        if mask.sum() > 5:
            r_mean = np.mean(resid[mask])
            r_std = np.std(resid[mask])
            print(f"  Parity {p}: mean_resid={r_mean:+.3f}, std={r_std:.3f} (n={mask.sum()})")

    # Check residual vs A (mass dependence = deformation proxy)
    A_vals = data_valid['A'].values
    corr = np.corrcoef(A_vals, resid)[0, 1]
    print(f"  Residual vs A: r = {corr:+.3f}")

    # Check residual vs N/Z (asymmetry = triaxiality proxy)
    NZ = data_valid['N'].values / data_valid['Z'].values.astype(float)
    corr_nz = np.corrcoef(NZ, resid)[0, 1]
    print(f"  Residual vs N/Z: r = {corr_nz:+.3f}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: GRAND SCORECARD
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("GRAND SCORECARD — LaGrangian Decomposition")
print("=" * 80)

print(f"""
THE THREE LAYERS OF NUCLEAR HALF-LIFE:

Layer A — beta-LaGrangian (universal vacuum stiffness)
  Features: stress (epsilon), geometric ratios (logZ, Z, N/N_MAX, N/Z, lnA), parity
  Coefficients: derivable from (alpha, beta, pi, e)
  These encode HOW the vacuum density field responds to geometric stress.

Layer B — External Energy (measured, per-nuclide)
  Features: Q-values, Coulomb barriers, transition energies, multipolarities
  Source: AME2020 mass table, NUBASE2020 spin-parity
  These encode the energy landscape the soliton sits in.
  NOT beta — these are the specific energetic environment.

Layer C — Vibrational Modes / Dzhanibekov (mechanistic, per-nuclide)
  Source: residual after A+B
  Physics: triaxiality, moments of inertia, intermediate-axis instability
  NOT beta — depends on specific 3D geometry of each nucleus.
  Lyapunov exponent requires 3 conditions to converge simultaneously.
""")

print(f"{'Species':18s}  {'n':>5s}  {'A(beta)':>7s}  {'B(energy)':>9s}  {'C(shape)':>8s}  {'A+B':>7s}")
print("─" * 65)

total_wa = 0
total_wb = 0
total_wc = 0
total_n = 0
for d in all_decomp:
    sp = d['species']
    fa = d['frac_a']
    fb = d['frac_b']
    fc = d['frac_c']
    n = d['n']
    print(f"{sp:18s}  {n:5d}  {fa:6.1f}%  {fb:8.1f}%  {fc:7.1f}%  {fa+fb:6.1f}%")
    total_wa += fa * n
    total_wb += fb * n
    total_wc += fc * n
    total_n += n

avg_a = total_wa / total_n
avg_b = total_wb / total_n
avg_c = total_wc / total_n
print(f"{'─' * 65}")
print(f"{'WEIGHTED AVG':18s}  {total_n:5d}  {avg_a:6.1f}%  {avg_b:8.1f}%  {avg_c:7.1f}%  {avg_a+avg_b:6.1f}%")

print(f"""
INTERPRETATION:
  {avg_a:.1f}% of half-life variance comes from the beta-LaGrangian (universal)
  {avg_b:.1f}% comes from external energy (Q-values, barriers)
  {avg_c:.1f}% is the Dzhanibekov residual (per-nucleus shape, NOT beta)

  The beta-LaGrangian IS the vacuum stiffness contribution.
  The external energy IS the potential landscape.
  The Dzhanibekov residual IS the limit of what any universal theory can predict —
  it requires knowledge of each nucleus's specific 3D geometry.
""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: The actual LaGrangian — explicit formula
# ══════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 6: THE LAGRANGIAN IN TERMS OF BETA")
print("=" * 80)

print(f"""
For a nuclear soliton with mass number A, charge Z, neutron number N = A - Z,
parity (Z mod 2, N mod 2), sitting in an energy landscape Q:

  L = L_vacuum(beta) + V_ext(Q) + T_vib(shape)

where:

  L_vacuum (Layer A):
    For BETA decay (charge correction):
      log_hl = -(pi^2/2*beta) * sqrt|eps|
             + (pi^2/e) * log10(Z)
             - (pi^2/2*beta*e^2) * Z
             - (beta*e) * N/N_MAX
             - (ln10/pi) * N/Z
             - (pi^2/e) * ln(A)
             + (ln10/pi^2) * [ee]
             - (ln10/pi^2) * [oo]
             + const(species)

    For ALPHA decay (barrier penetration):
      log_hl = -(beta^2/e) * |eps|
             + (pi^2/e) * log10(Z)
             - (beta*e) * N/N_MAX
             - (ln10/pi) * N/Z
             + (ln10/pi^2) * [ee]
             - (ln10/pi^2) * [oo]
             + const(species)

    For IT (electromagnetic transition):
      log_hl = -(pi^2/2*beta) * sqrt|eps|
             + (pi^2/e) * log10(Z)
             - (beta*e) * N/N_MAX
             - (ln10/pi) * N/Z
             + (ln10/pi^2) * [ee]
             - (ln10/pi^2) * [oo]
             + const(species)

  V_ext (Layer B):
    For BETA:  + c_Q * log10(Q_keV)
    For ALPHA: + c_pen * log10(Q/V_C) + c_inv * 1/sqrt(Q) + c_def * (V_C - Q)
    For IT:    + c_E * log10(E_trans) + c_lam * lambda + c_lam2 * lambda^2

  T_vib (Layer C):
    The RESIDUAL — encodes Dzhanibekov instability, triaxiality,
    specific moments of inertia. This is mechanistic (has physical
    origin) but NOT derivable from universal constants because it
    depends on each nucleus's 3D mass distribution.

Key algebraic constants (all from alpha/beta/pi/e):
  pi^2/(2*beta) = {PI**2 / (2*b):.4f}  — tunneling stress scale
  beta^2/e      = {b**2 / E_val:.4f}  — neck strain scale
  pi^2/e        = {PI**2 / E_val:.4f}  — charge/mass scale
  beta*e        = {b * E_val:.4f}  — ceiling proximity scale
  ln10/pi       = {LN10 / PI:.4f}  — neutron excess scale
  ln10/pi^2     = {LN10 / PI**2:.4f}  — parity shift scale
""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: Save per-nuclide predictions with layer attribution
# ══════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 7: PER-NUCLIDE PREDICTIONS")
print("=" * 80)

# Build predictions for each tracked nuclide
pred_rows = []

for d in all_decomp:
    sp = d['species']
    parent = iso_map.get(sp, sp)
    config = species_layers[parent]
    feats_a = config['layer_a']
    feats_ab = feats_a + config['layer_b']
    ridge = config['ridge']
    data = tracked[tracked['clean_species'] == sp].copy()

    if len(data) < 10:
        continue

    # Full A+B fit
    res_ab = fit_layer(data, feats_ab, ridge)
    if res_ab is None:
        continue

    # Layer A only fit
    res_a = fit_layer(data, feats_a, ridge)
    if res_a is None:
        continue

    # Algebraic Layer A prediction (with best-fit constant)
    names_a = [nm for nm, _ in feats_a]
    alg_coefs_a = {nm: ALGEBRAIC.get(nm, 0.0) for nm in names_a}

    y = data['log_hl'].values
    X_parts_a = [func(data) for _, func in feats_a]
    X_a = np.column_stack(X_parts_a)
    valid = np.isfinite(X_a).all(axis=1) & np.isfinite(y)

    alg_vec = np.array([alg_coefs_a.get(nm, 0.0) for nm in names_a])
    pred_alg_slopes = X_a[valid] @ alg_vec
    best_c = d.get('best_const', np.mean(y[valid] - pred_alg_slopes))
    pred_alg = pred_alg_slopes + best_c

    # Layer A fitted prediction
    names_a_full = names_a + ['const']
    coef_a = np.array([res_a['coefs'][k] for k in names_a_full])
    pred_a_fit = res_a['X'] @ coef_a

    # Layer A+B fitted prediction
    names_ab = [nm for nm, _ in feats_ab] + ['const']
    coef_ab = np.array([res_ab['coefs'][k] for k in names_ab])
    pred_ab_fit = res_ab['X'] @ coef_ab

    # Build output rows (only valid rows)
    valid_idx = np.where(valid)[0]
    # Match to res_ab's valid mask
    valid_ab = np.where(res_ab['valid'])[0]

    for i, idx in enumerate(valid_idx):
        row = data.iloc[idx]
        r = {
            'A': int(row['A']), 'Z': int(row['Z']),
            'element': row['element'], 'species': sp,
            'parity': row['parity'],
            'log_hl_obs': y[idx],
        }
        # Layer A algebraic
        if i < len(pred_alg):
            r['pred_A_alg'] = pred_alg[i]
            r['resid_A_alg'] = y[idx] - pred_alg[i]

        # Layer A fitted
        if i < len(pred_a_fit):
            r['pred_A_fit'] = pred_a_fit[i]
            r['resid_A_fit'] = y[idx] - pred_a_fit[i]

        # Layer A+B fitted
        if i < len(pred_ab_fit):
            r['pred_AB_fit'] = pred_ab_fit[i]
            r['resid_AB_fit'] = y[idx] - pred_ab_fit[i]
            r['layer_C_resid'] = y[idx] - pred_ab_fit[i]

        pred_rows.append(r)

pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(os.path.join(RESULTS_DIR, 'lagrangian_decomposition.csv'), index=False)
print(f"\nSaved: {os.path.join(RESULTS_DIR, 'lagrangian_decomposition.csv')} ({len(pred_df)} nuclides)")

# Quick stats
for col in ['resid_A_alg', 'resid_A_fit', 'resid_AB_fit']:
    vals = pred_df[col].dropna()
    solved = (np.abs(vals) < 1.0).sum()
    print(f"  {col:20s}: RMSE={vals.std():.2f}, <1dec={solved}/{len(vals)} ({100*solved/len(vals):.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# SECTION 8: Cross-species universality check
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 8: CROSS-SPECIES UNIVERSALITY OF LAYER A")
print("=" * 80)
print("\nIf the LaGrangian is universal, Layer A coefficients should be")
print("SIMILAR across species (same vacuum physics, different decay channel).\n")

# Collect Layer A coefficients across species
all_feat_names = set()
for d in all_decomp:
    sp = d['species']
    parent = iso_map.get(sp, sp)
    for fname, _ in species_layers[parent]['layer_a']:
        all_feat_names.add(fname)

# Build comparison table
common_feats = ['sqrt_eps', 'abs_eps', 'logZ', 'Z', 'N_NMAX', 'N_Z', 'lnA', 'ee', 'oo']

print(f"{'Feature':12s}", end="")
for d in all_decomp:
    sp = d['species'][:10]
    print(f"  {sp:>10s}", end="")
print(f"  {'Algebraic':>10s}")
print("─" * (14 + 12 * (len(all_decomp) + 1)))

for feat in common_feats:
    print(f"{feat:12s}", end="")
    for d in all_decomp:
        sp = d['species']
        c = d['coefs_a'].get(feat, None)
        if c is not None:
            print(f"  {c:+10.3f}", end="")
        else:
            print(f"  {'---':>10s}", end="")
    # Algebraic
    alg = ALGEBRAIC.get(feat, None)
    if alg is not None:
        print(f"  {alg:+10.3f}", end="")
    else:
        print(f"  {'---':>10s}", end="")
    print()


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
