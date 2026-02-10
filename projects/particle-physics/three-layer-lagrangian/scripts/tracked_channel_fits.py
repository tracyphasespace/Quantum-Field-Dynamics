#!/usr/bin/env python3
"""
Tracked-Only Channel Fits — Per-channel models on clean, well-characterized
nuclides only. Excludes ephemeral (< 1 μs) and suspect (man-made exotics).

Runs three model tiers per channel:
  Model A (baseline): Layer A+B features (stress, size, energy, constant parity)
  Model C (geometry): + soliton geometry (mass-dep parity, Dzhanibekov, neck)
  Model D (structural): + phase transition steps + split alpha + regime mass scaling

Structural discoveries (from residual analysis):
  - Alpha has TWO mechanisms: surface tunneling (A<160) vs neck-mediated (A≥160)
  - Each species sees the soliton phase transition at a different mass:
      Beta-:  A=124  (density-2 core nucleation, +0.71 decades slower)
      IT:     A=144  (core approaching criticality, -1.45 decades faster)
      Beta+:  A=160  (peanut bifurcation, -0.65 decades faster)
      Alpha:  A=160  (peanut creates neck tunneling channel)
  - Alpha mass scaling slope is 2× steeper in peanut regime (emergent time)

Tracy McSheery directive: "many man-made are possibly man-made up, with dubious
properties. Matching them is a Snark Hunt."

Bins:
  STABLE:    289  no decay
  TRACKED:  3674  well-characterized, worth modeling
  SUSPECT:   390  man-made exotics, |ε|>8 or higher-iso short-lived
  EPHEMERAL: 591  half-life < 1 μs
  RARE:        4  no dominant mode
"""

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.environ.get('TLG_DATA_DIR', os.path.join(_ROOT_DIR, 'data'))
RESULTS_DIR = os.environ.get('TLG_RESULTS_DIR', os.path.join(_ROOT_DIR, 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# QFD Constants — derived from α = 1/137.036 via β = 3.043233053
# ══════════════════════════════════════════════════════════════════════════════
beta_val = 3.043233053
PI = np.pi
E_CONST = np.e

r0 = PI**2 / (beta_val * E_CONST)       # ≈ 1.193 fm
N_MAX = 2 * PI * beta_val**3            # ≈ 177.09
A_CRIT = 2 * E_CONST**2 * beta_val**2   # ≈ 136.9
WIDTH = 2 * PI * beta_val**2            # ≈ 58.19
A_DRIP = 296

# Planetary core model thresholds
A_NUCLEATION = A_CRIT / E_CONST          # ≈ 50.4
A_PEANUT = 160
A_FROZEN = 225

# Species-specific phase transition masses (from boundary scan)
A_STEP_BETA_MINUS = 124    # density-2 core nucleation
A_STEP_IT = 144            # core approaching criticality
A_STEP_BETA_PLUS = 160     # peanut bifurcation
A_STEP_ALPHA = 160         # neck tunneling channel opens

# ══════════════════════════════════════════════════════════════════════════════
# Soliton Geometry Engine — planetary core model with neck-local moments
# ══════════════════════════════════════════════════════════════════════════════

def compute_geometry(A, Z, N, parity):
    """
    Compute soliton shape from the QFD planetary core model.

    Single-core regime (A ≤ 160): Prolate ellipsoid with core fraction.
    Peanut regime (A > 160): Two lobes + density-1 neck bridge.
    """
    R = r0 * A**(1.0/3)
    M = float(A)

    if A < A_NUCLEATION: f_core = 0.0
    elif A <= A_PEANUT: f_core = (A - A_NUCLEATION) / (A_PEANUT - A_NUCLEATION)
    else: f_core = 1.0

    n_unpaired = {'ee': 0, 'eo': 1, 'oe': 1, 'oo': 2}.get(parity, 1)
    NZ = N / Z if Z > 0 else 1.0
    nz_excess = max(0.0, NZ - 1.0)

    if A <= A_PEANUT:
        ecc = f_core * 0.15
        parity_triax = n_unpaired * 0.04 * f_core
        nz_triax = 0.015 * nz_excess * f_core
        total_triax = parity_triax + nz_triax
        a = R * (1 + ecc)
        perp = 1.0 / np.sqrt(1 + ecc)
        b = R * perp * (1 + total_triax / 2)
        c = R * perp * (1 - total_triax / 2)
        Ia = M / 5 * (b**2 + c**2)
        Ib = M / 5 * (a**2 + c**2)
        Ic = M / 5 * (a**2 + b**2)
        I1, I2, I3 = sorted([Ia, Ib, Ic])
        gamma = (I2 - I1) / (I3 - I1) if (I3 - I1) > 1e-12 else 0.0
        neck_ellipticity = 0.0
    else:
        eta = min(1.0, (A - A_PEANUT) / (A_DRIP - A_PEANUT))
        r_lobe = r0 * (M / 2)**(1.0/3)
        d = 2 * r_lobe * (1 + eta * 1.5)
        neck_length = d - 2 * r_lobe
        neck_radius = r_lobe * 0.4 * (1 - eta * 0.4)
        neck_mass_frac = 0.08 * (1 - eta * 0.3)
        base_ellipticity = n_unpaired * 0.20
        nz_ellipticity = 0.06 * nz_excess
        frozen_reduction = 0.0
        if A > A_FROZEN:
            frozen_reduction = 0.3 * min(1.0, (A - A_FROZEN) / (A_DRIP - A_FROZEN))
        neck_ellipticity = (base_ellipticity + nz_ellipticity) * (1 - frozen_reduction)
        b_neck = neck_radius * (1 + neck_ellipticity / 2)
        c_neck = neck_radius * (1 - neck_ellipticity / 2)
        M_neck = M * neck_mass_frac
        L = max(neck_length, 0.1)
        I_neck_x = M_neck / 4 * (b_neck**2 + c_neck**2)
        I_neck_y = M_neck / 12 * (3 * c_neck**2 + L**2)
        I_neck_z = M_neck / 12 * (3 * b_neck**2 + L**2)
        I1, I2, I3 = sorted([I_neck_x, I_neck_y, I_neck_z])
        gamma = (I2 - I1) / (I3 - I1) if (I3 - I1) > 1e-12 else 0.0

    dzhan = gamma * (1 - gamma) * 4
    lyapunov = dzhan * max(1.0, 1 + (A - A_PEANUT) / 80) if A > A_PEANUT else dzhan

    return {
        'f_core': f_core, 'gamma': gamma, 'dzhanibekov': dzhan,
        'log_lyapunov': np.log10(lyapunov + 1e-8),
        'neck_ellipticity': neck_ellipticity, 'n_unpaired': n_unpaired,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Load data and precompute all features
# ══════════════════════════════════════════════════════════════════════════════
cs = pd.read_csv(os.path.join(DATA_DIR, 'clean_species_sorted.csv'))
cs['N'] = cs['A'] - cs['Z']

print("=" * 85)
print("TRACKED-ONLY CHANNEL FITS — Baseline → Geometry → Structural")
print("=" * 85)

# Precompute soliton geometry
print("\nComputing soliton geometry (planetary core model)...")
geo_cols = ['f_core', 'gamma', 'dzhanibekov', 'log_lyapunov',
            'neck_ellipticity', 'n_unpaired']
for col in geo_cols:
    cs[col] = 0.0

for idx in cs.index:
    row = cs.loc[idx]
    geo = compute_geometry(int(row['A']), int(row['Z']),
                           int(row['N']), row.get('parity', 'eo'))
    for col in geo_cols:
        cs.at[idx, col] = geo[col]

# Mass-dependent parity (topology × core structure)
cs['ee_core'] = ((cs['parity'] == 'ee').astype(float) * cs['f_core']).values
cs['oo_core'] = ((cs['parity'] == 'oo').astype(float) * cs['f_core']).values

# Phase transition step indicators
cs['step_bm'] = (cs['A'] >= A_STEP_BETA_MINUS).astype(float)
cs['step_it'] = (cs['A'] >= A_STEP_IT).astype(float)
cs['step_bp'] = (cs['A'] >= A_STEP_BETA_PLUS).astype(float)
cs['step_alpha'] = (cs['A'] >= A_STEP_ALPHA).astype(float)

# Regime-dependent mass scaling (split lnA at peanut boundary)
cs['is_peanut'] = (cs['A'] >= A_PEANUT).astype(float)
cs['lnA_single'] = (1 - cs['is_peanut']) * np.log(cs['A'].values.astype(float))
cs['lnA_peanut'] = cs['is_peanut'] * np.log(cs['A'].values.astype(float))

print(f"  Geometry computed for {len(cs)} nuclides")
print(f"  Peanut regime (A>160): {(cs['A'] > A_PEANUT).sum()} nuclides")
print(f"  Phase transitions: β⁻ at A={A_STEP_BETA_MINUS}, IT at A={A_STEP_IT}, β⁺/α at A={A_STEP_BETA_PLUS}")

# ── Apply tracking filters ──
has_hl = cs['log_hl'].notna()
is_stable = cs['clean_species'] == 'stable'
is_rare = cs['clean_species'] == 'rare'
radioactive = has_hl & ~is_stable & ~is_rare

ephemeral = radioactive & (cs['log_hl'] < -6)
far_exotic = radioactive & (np.abs(cs['epsilon']) > 8)
higher_iso_short = radioactive & (cs['az_order'] >= 2) & (cs['log_hl'] < -3)
neutron = radioactive & (cs['clean_species'] == 'neutron')
suspect = (far_exotic | higher_iso_short | neutron) & ~ephemeral
tracked = radioactive & ~ephemeral & ~suspect

cs['tracking_bin'] = 'other'
cs.loc[is_stable, 'tracking_bin'] = 'stable'
cs.loc[tracked, 'tracking_bin'] = 'tracked'
cs.loc[suspect, 'tracking_bin'] = 'suspect'
cs.loc[ephemeral, 'tracking_bin'] = 'ephemeral'
cs.loc[is_rare, 'tracking_bin'] = 'rare'

print(f"\nPopulation:")
for b in ['stable', 'tracked', 'suspect', 'ephemeral', 'rare']:
    n = (cs['tracking_bin'] == b).sum()
    print(f"  {b:12s}  {n:5d}  ({100*n/len(cs):5.1f}%)")

decay = cs[cs['tracking_bin'] == 'tracked'].copy()
print(f"\nModeling {len(decay)} tracked nuclides")

# ══════════════════════════════════════════════════════════════════════════════
# Feature builders
# ══════════════════════════════════════════════════════════════════════════════

# --- Layer A (vacuum stiffness) ---
def f_sqrt_eps(df): return np.sqrt(np.abs(df['epsilon'].values))
def f_abs_eps(df): return np.abs(df['epsilon'].values)
def f_logZ(df): return np.log10(df['Z'].values.astype(float))
def f_Z(df): return df['Z'].values.astype(float)
def f_N_NMAX(df): return df['N'].values / N_MAX
def f_N_Z(df): return df['N'].values / df['Z'].values.astype(float)
def f_lnA(df): return np.log(df['A'].values.astype(float))
def f_ee(df): return (df['parity'] == 'ee').astype(float).values
def f_oo(df): return (df['parity'] == 'oo').astype(float).values

# --- Layer B (external energy) ---
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
    return np.log10(np.maximum(df['transition_energy_keV'].values.astype(float), 1.0))
def f_lambda(df):
    med = df['correct_lambda'].median()
    if pd.isna(med): med = 3.0
    return df['correct_lambda'].fillna(med).values
def f_lambda_sq(df):
    med = df['correct_lambda'].median()
    if pd.isna(med): med = 3.0
    return df['correct_lambda'].fillna(med).values ** 2

# --- Layer C (soliton geometry / Dzhanibekov) ---
def f_ee_core(df): return df['ee_core'].values
def f_oo_core(df): return df['oo_core'].values
def f_dzhanibekov(df): return df['dzhanibekov'].values
def f_log_lyapunov(df): return df['log_lyapunov'].values
def f_neck_ellip(df): return df['neck_ellipticity'].values

# --- Layer D (structural: phase transitions + regime mass scaling) ---
def f_step_bm(df): return df['step_bm'].values
def f_step_it(df): return df['step_it'].values
def f_step_bp(df): return df['step_bp'].values
def f_step_alpha(df): return df['step_alpha'].values
def f_lnA_single(df): return df['lnA_single'].values
def f_lnA_peanut(df): return df['lnA_peanut'].values
def f_neck_x_eps(df): return df['neck_ellipticity'].values * np.abs(df['epsilon'].values)

# ══════════════════════════════════════════════════════════════════════════════
# Species configs — three tiers: baseline (A), geometry (C), structural (D)
# ══════════════════════════════════════════════════════════════════════════════

species_configs = {
    'beta-': {
        'baseline': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'geometry': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo),
                     ('ee_core', f_ee_core), ('oo_core', f_oo_core)],
        'structural': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                       ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                       ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo),
                       ('ee_core', f_ee_core), ('oo_core', f_oo_core),
                       ('step_A124', f_step_bm)],
        'ridge': 1.0
    },
    'beta+': {
        'baseline': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'geometry': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo),
                     ('ee_core', f_ee_core), ('oo_core', f_oo_core),
                     ('dzhanibekov', f_dzhanibekov)],
        'structural': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                       ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                       ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo),
                       ('ee_core', f_ee_core), ('oo_core', f_oo_core),
                       ('dzhanibekov', f_dzhanibekov),
                       ('step_A160', f_step_bp)],
        'ridge': 1.0
    },
    'IT': {
        'baseline': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                     ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                     ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'geometry': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                     ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                     ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo),
                     ('ee_core', f_ee_core), ('oo_core', f_oo_core),
                     ('dzhanibekov', f_dzhanibekov), ('neck_ellip', f_neck_ellip)],
        'structural': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                       ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                       ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                       ('ee', f_ee), ('oo', f_oo),
                       ('ee_core', f_ee_core), ('oo_core', f_oo_core),
                       ('dzhanibekov', f_dzhanibekov), ('neck_ellip', f_neck_ellip),
                       ('step_A144', f_step_it)],
        'ridge': 2.0
    },
    'IT_platypus': {
        'baseline': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                     ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                     ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'geometry': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                     ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                     ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo),
                     ('ee_core', f_ee_core), ('oo_core', f_oo_core),
                     ('dzhanibekov', f_dzhanibekov), ('neck_ellip', f_neck_ellip)],
        'structural': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                       ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                       ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                       ('ee', f_ee), ('oo', f_oo),
                       ('ee_core', f_ee_core), ('oo_core', f_oo_core),
                       ('dzhanibekov', f_dzhanibekov), ('neck_ellip', f_neck_ellip),
                       ('step_A144', f_step_it)],
        'ridge': 2.0
    },
    'SF': {
        'baseline': [('N_NMAX', f_N_NMAX), ('abs_eps', f_abs_eps),
                     ('logZ', f_logZ), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'geometry': [('N_NMAX', f_N_NMAX), ('abs_eps', f_abs_eps),
                     ('logZ', f_logZ), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo),
                     ('dzhanibekov', f_dzhanibekov), ('neck_ellip', f_neck_ellip)],
        'structural': [('N_NMAX', f_N_NMAX), ('abs_eps', f_abs_eps),
                       ('logZ', f_logZ), ('N_Z', f_N_Z),
                       ('ee', f_ee), ('oo', f_oo),
                       ('dzhanibekov', f_dzhanibekov), ('neck_ellip', f_neck_ellip),
                       ('log_lyap', f_log_lyapunov)],
        'ridge': 5.0
    },
}

# Alpha uses SPLIT models (light surface vs heavy neck), not lumped
# Light alpha: surface tunneling from single-core soliton
alpha_light_feats = {
    'baseline': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                 ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                 ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                 ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo)],
    'structural': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                   ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                   ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                   ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo),
                   ('lnA', f_lnA)],
    'ridge': 5.0
}

# Heavy alpha: neck-mediated tunneling from peanut soliton
alpha_heavy_feats = {
    'baseline': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                 ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                 ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                 ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo)],
    'structural': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                   ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                   ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                   ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo),
                   ('ee_core', f_ee_core), ('oo_core', f_oo_core),
                   ('dzhanibekov', f_dzhanibekov), ('log_lyap', f_log_lyapunov),
                   ('neck_ellip', f_neck_ellip), ('neck_x_eps', f_neck_x_eps)],
    'ridge': 5.0
}

# Alpha lumped configs (for the non-split tiers)
species_configs['alpha'] = {
    'baseline': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                 ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                 ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                 ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo)],
    'geometry': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                 ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                 ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                 ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo),
                 ('ee_core', f_ee_core), ('oo_core', f_oo_core),
                 ('dzhanibekov', f_dzhanibekov), ('log_lyap', f_log_lyapunov),
                 ('neck_ellip', f_neck_ellip)],
    'structural': None,   # alpha uses split, not lumped structural
    'ridge': 5.0
}

# ── Regime boundaries ──
regimes = [
    ('light', 0, 60),
    ('medium', 60, A_CRIT),
    ('transition', A_CRIT, A_CRIT + WIDTH),
    ('heavy', A_CRIT + WIDTH, 300),
]

# ══════════════════════════════════════════════════════════════════════════════
# Fitter
# ══════════════════════════════════════════════════════════════════════════════

def fit_and_report(df, features, name, alpha_ridge=1.0):
    y = df['log_hl'].values
    X_parts = [func(df) for fn, func in features]
    X_parts.append(np.ones(len(df)))
    X = np.column_stack(X_parts)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = valid.sum()
    if n < max(len(features) + 2, 8):
        return None
    X_v, y_v = X[valid], y[valid]
    I_mat = np.eye(X_v.shape[1]); I_mat[-1, -1] = 0
    try:
        coef = np.linalg.solve(X_v.T @ X_v + alpha_ridge * I_mat, X_v.T @ y_v)
    except Exception:
        coef, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
    pred = X_v @ coef
    resid = y_v - pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(resid**2))
    p90 = np.percentile(np.abs(resid), 90)
    solved_1dec = np.sum(np.abs(resid) < 1.0)
    return {
        'n': n, 'r2': r2, 'rmse': rmse, 'p90': p90,
        'hl_std': y_v.std(), 'hl_range': y_v.max() - y_v.min(),
        'solved_1dec': solved_1dec, 'pct_solved': 100 * solved_1dec / n
    }

# ══════════════════════════════════════════════════════════════════════════════
# Run fits for all three tiers
# ══════════════════════════════════════════════════════════════════════════════
all_results = []

non_alpha_species = ['beta-', 'beta+', 'IT', 'IT_platypus', 'SF',
                     'beta-_iso', 'beta+_iso', 'alpha_iso', 'proton']

for tier_name, tier_key in [('BASELINE (A+B)', 'baseline'),
                             ('GEOMETRY (+ soliton)', 'geometry'),
                             ('STRUCTURAL (+ transitions)', 'structural')]:
    print(f"\n{'='*85}")
    print(f"MODEL: {tier_name}")
    print(f"{'='*85}")
    print(f"\n{'Channel':35s}  {'n':>5s}  {'R²':>7s}  {'RMSE':>6s}  {'P90':>6s}  {'<1dec':>5s}  {'%sol':>5s}")
    print("─" * 78)

    # Non-alpha species (lumped fitting)
    for sp in non_alpha_species:
        sp_data = decay[decay['clean_species'] == sp]
        if len(sp_data) < 8:
            if len(sp_data) > 0:
                print(f"{sp:35s}  {len(sp_data):5d}  {'(few)':>7s}")
            continue

        base_sp = sp.replace('_iso', '')
        config = species_configs.get(base_sp, species_configs.get('beta-'))
        features = config.get(tier_key)
        if features is None:
            features = config.get('geometry', config.get('baseline'))
        ridge = config['ridge']

        # Lumped
        res = fit_and_report(sp_data, features, f"{sp}_lumped", ridge)
        if res:
            print(f"{sp + ' (lumped)':35s}  {res['n']:5d}  {res['r2']:7.3f}  {res['rmse']:6.2f}  {res['p90']:6.2f}  {res['solved_1dec']:5d}  {res['pct_solved']:4.1f}%")
            all_results.append({'species': sp, 'parity': 'all', 'a_regime': 'lumped',
                               'model': tier_key, **res})

        # By regime
        for rname, a_lo, a_hi in regimes:
            rdata = sp_data[(sp_data['A'] >= a_lo) & (sp_data['A'] < a_hi)]
            if len(rdata) < 8: continue
            feats = features if len(rdata) > 20 else features[:5]
            res = fit_and_report(rdata, feats, f"{sp}_{rname}", ridge)
            if res:
                tag = f"  {sp}/{rname}"
                print(f"{tag:35s}  {res['n']:5d}  {res['r2']:7.3f}  {res['rmse']:6.2f}  {res['p90']:6.2f}  {res['solved_1dec']:5d}  {res['pct_solved']:4.1f}%")
                all_results.append({'species': sp, 'parity': 'all', 'a_regime': rname,
                                   'model': tier_key, **res})

        # By parity
        for par in ['ee', 'eo', 'oe', 'oo']:
            pdata = sp_data[sp_data['parity'] == par]
            if len(pdata) < 10: continue
            feats = features if len(pdata) > 20 else features[:5]
            res = fit_and_report(pdata, feats, f"{sp}_{par}", ridge)
            if res:
                tag = f"  {sp}/{par}"
                print(f"{tag:35s}  {res['n']:5d}  {res['r2']:7.3f}  {res['rmse']:6.2f}  {res['p90']:6.2f}  {res['solved_1dec']:5d}  {res['pct_solved']:4.1f}%")
                all_results.append({'species': sp, 'parity': par, 'a_regime': 'lumped',
                                   'model': tier_key, **res})

    # Alpha: lumped for baseline/geometry, SPLIT for structural
    alpha_data = decay[decay['clean_species'] == 'alpha']
    alpha_iso_data = decay[decay['clean_species'] == 'alpha_iso']

    if tier_key == 'structural':
        # Split alpha at peanut boundary
        alpha_light = alpha_data[alpha_data['A'] < A_PEANUT]
        alpha_heavy = alpha_data[alpha_data['A'] >= A_PEANUT]

        res_light = fit_and_report(alpha_light, alpha_light_feats['structural'],
                                   'alpha_light', alpha_light_feats['ridge'])
        res_heavy = fit_and_report(alpha_heavy, alpha_heavy_feats['structural'],
                                   'alpha_heavy', alpha_heavy_feats['ridge'])

        if res_light:
            print(f"{'alpha/light (surface)':35s}  {res_light['n']:5d}  {res_light['r2']:7.3f}  {res_light['rmse']:6.2f}  {res_light['p90']:6.2f}  {res_light['solved_1dec']:5d}  {res_light['pct_solved']:4.1f}%")
            all_results.append({'species': 'alpha', 'parity': 'all', 'a_regime': 'light_surface',
                               'model': tier_key, **res_light})

        if res_heavy:
            print(f"{'alpha/heavy (neck)':35s}  {res_heavy['n']:5d}  {res_heavy['r2']:7.3f}  {res_heavy['rmse']:6.2f}  {res_heavy['p90']:6.2f}  {res_heavy['solved_1dec']:5d}  {res_heavy['pct_solved']:4.1f}%")
            all_results.append({'species': 'alpha', 'parity': 'all', 'a_regime': 'heavy_neck',
                               'model': tier_key, **res_heavy})

        # Compute weighted split R² using GLOBAL mean
        if res_light and res_heavy:
            n_total = res_light['n'] + res_heavy['n']
            ss_res = res_light['rmse']**2 * res_light['n'] + res_heavy['rmse']**2 * res_heavy['n']
            y_all = alpha_data['log_hl'].dropna().values
            gm = y_all.mean()
            y_l = alpha_light['log_hl'].dropna().values
            y_h = alpha_heavy['log_hl'].dropna().values
            ss_tot = np.sum((y_l - gm)**2) + np.sum((y_h - gm)**2)
            split_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            split_solved = res_light['solved_1dec'] + res_heavy['solved_1dec']
            split_pct = 100 * split_solved / n_total
            split_rmse = np.sqrt(ss_res / n_total)
            print(f"{'alpha (split weighted)':35s}  {n_total:5d}  {split_r2:7.3f}  {split_rmse:6.2f}  {'':>6s}  {split_solved:5d}  {split_pct:4.1f}%")
            all_results.append({'species': 'alpha', 'parity': 'all', 'a_regime': 'lumped',
                               'model': tier_key, 'n': n_total, 'r2': split_r2,
                               'rmse': split_rmse, 'p90': 0, 'hl_std': y_all.std(),
                               'hl_range': y_all.max() - y_all.min(),
                               'solved_1dec': split_solved, 'pct_solved': split_pct})

        # Alpha_iso: use heavy feats (most are heavy)
        if len(alpha_iso_data) >= 8:
            config_iso = species_configs.get('alpha')
            res_iso = fit_and_report(alpha_iso_data, alpha_heavy_feats['structural'],
                                     'alpha_iso_struct', alpha_heavy_feats['ridge'])
            if res_iso:
                print(f"{'alpha_iso (lumped)':35s}  {res_iso['n']:5d}  {res_iso['r2']:7.3f}  {res_iso['rmse']:6.2f}  {res_iso['p90']:6.2f}  {res_iso['solved_1dec']:5d}  {res_iso['pct_solved']:4.1f}%")
                all_results.append({'species': 'alpha_iso', 'parity': 'all', 'a_regime': 'lumped',
                                   'model': tier_key, **res_iso})
    else:
        # Baseline/geometry: lumped alpha
        config = species_configs['alpha']
        features = config.get(tier_key, config['baseline'])
        ridge = config['ridge']

        res = fit_and_report(alpha_data, features, 'alpha_lumped', ridge)
        if res:
            print(f"{'alpha (lumped)':35s}  {res['n']:5d}  {res['r2']:7.3f}  {res['rmse']:6.2f}  {res['p90']:6.2f}  {res['solved_1dec']:5d}  {res['pct_solved']:4.1f}%")
            all_results.append({'species': 'alpha', 'parity': 'all', 'a_regime': 'lumped',
                               'model': tier_key, **res})

        # By regime
        for rname, a_lo, a_hi in regimes:
            rdata = alpha_data[(alpha_data['A'] >= a_lo) & (alpha_data['A'] < a_hi)]
            if len(rdata) < 8: continue
            feats = features if len(rdata) > 20 else features[:5]
            res = fit_and_report(rdata, feats, f"alpha_{rname}", ridge)
            if res:
                tag = f"  alpha/{rname}"
                print(f"{tag:35s}  {res['n']:5d}  {res['r2']:7.3f}  {res['rmse']:6.2f}  {res['p90']:6.2f}  {res['solved_1dec']:5d}  {res['pct_solved']:4.1f}%")
                all_results.append({'species': 'alpha', 'parity': 'all', 'a_regime': rname,
                                   'model': tier_key, **res})

        # Alpha_iso
        if len(alpha_iso_data) >= 8:
            res = fit_and_report(alpha_iso_data, features, 'alpha_iso', ridge)
            if res:
                print(f"{'alpha_iso (lumped)':35s}  {res['n']:5d}  {res['r2']:7.3f}  {res['rmse']:6.2f}  {res['p90']:6.2f}  {res['solved_1dec']:5d}  {res['pct_solved']:4.1f}%")
                all_results.append({'species': 'alpha_iso', 'parity': 'all', 'a_regime': 'lumped',
                                   'model': tier_key, **res})

# ══════════════════════════════════════════════════════════════════════════════
# THREE-TIER COMPARISON (lumped channels only)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*85}")
print("THREE-TIER COMPARISON — Baseline → Geometry → Structural")
print(f"{'='*85}")

all_species = ['beta-', 'beta+', 'alpha', 'IT', 'IT_platypus', 'SF',
               'beta-_iso', 'beta+_iso', 'alpha_iso']

print(f"\n{'Species':20s}  {'n':>5s}  {'R²(A)':>7s}  {'R²(C)':>7s}  {'R²(D)':>7s}  {'ΔR²(A→D)':>9s}  {'%sol(A)':>7s}  {'%sol(D)':>7s}  {'Δ%sol':>6s}")
print("─" * 95)

comparison = []
for sp in all_species:
    tiers = {}
    for tier_key in ['baseline', 'geometry', 'structural']:
        matches = [r for r in all_results if r['species'] == sp and r['model'] == tier_key
                   and r['a_regime'] == 'lumped' and r['parity'] == 'all']
        if matches:
            tiers[tier_key] = matches[-1]   # take latest if duplicates

    if 'baseline' not in tiers:
        continue

    b = tiers['baseline']
    c = tiers.get('geometry', b)
    d = tiers.get('structural', c)

    dr2_total = d['r2'] - b['r2']
    dsol_total = d['pct_solved'] - b['pct_solved']
    marker = '***' if dr2_total > 0.02 else ' * ' if dr2_total > 0.005 else '   '

    print(f"{sp:20s}  {b['n']:5d}  {b['r2']:7.3f}  {c['r2']:7.3f}  {d['r2']:7.3f}  {dr2_total:+9.3f}{marker}  {b['pct_solved']:6.1f}%  {d['pct_solved']:6.1f}%  {dsol_total:+5.1f}%")

    comparison.append({
        'species': sp, 'n': b['n'],
        'r2_baseline': b['r2'], 'r2_geometry': c['r2'], 'r2_structural': d['r2'],
        'delta_r2_total': dr2_total,
        'pct_solved_baseline': b['pct_solved'], 'pct_solved_structural': d['pct_solved'],
        'delta_pct_solved': dsol_total,
    })

# Global weighted metrics
print(f"\n--- Global metrics ---")
for tier_key, label in [('baseline', 'Baseline'), ('geometry', 'Geometry'), ('structural', 'Structural')]:
    lumped = [r for r in all_results if r['model'] == tier_key and r['a_regime'] == 'lumped'
              and r['parity'] == 'all' and r['species'] in all_species]
    # Deduplicate by species (take last)
    by_sp = {}
    for r in lumped:
        by_sp[r['species']] = r
    lumped = list(by_sp.values())
    if not lumped: continue
    ss_res = sum(r['rmse']**2 * r['n'] for r in lumped)
    ss_tot = sum(r['hl_std']**2 * r['n'] for r in lumped)
    g_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    t_n = sum(r['n'] for r in lumped)
    t_sol = sum(r['solved_1dec'] for r in lumped)
    print(f"  {label:12s}  weighted R² = {g_r2:.4f}  solve rate = {t_sol}/{t_n} = {100*t_sol/t_n:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# SCORECARD — Best model per species
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*85}")
print("SCORECARD — Best channel per species (structural model)")
print(f"{'='*85}")
print(f"\n{'Channel':35s}  {'n':>5s}  {'R²':>7s}  {'RMSE':>6s}  {'P90':>6s}  {'%sol':>5s}")
print("─" * 70)

struct_results = [r for r in all_results if r['model'] == 'structural']
seen_sp = set()
for r in sorted(struct_results, key=lambda x: -x['r2']):
    sp = r['species']
    if sp in seen_sp: continue
    seen_sp.add(sp)
    label = f"{sp}/{r['parity']}/{r['a_regime']}"
    print(f"{label:35s}  {r['n']:5d}  {r['r2']:7.3f}  {r['rmse']:6.2f}  {r['p90']:6.2f}  {r['pct_solved']:4.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, 'tracked_channel_scores.csv'), index=False)
print(f"\nSaved: {os.path.join(RESULTS_DIR, 'tracked_channel_scores.csv')} ({len(results_df)} channels)")

if comparison:
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(os.path.join(RESULTS_DIR, 'geometry_comparison.csv'), index=False)
    print(f"Saved: {os.path.join(RESULTS_DIR, 'geometry_comparison.csv')} ({len(comp_df)} species)")
