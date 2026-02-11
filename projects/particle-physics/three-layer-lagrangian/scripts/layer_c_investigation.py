#!/usr/bin/env python3
"""
Layer C Investigation — What explains the Dzhanibekov residual?

Layer C = 34.1% of half-life variance, the part NOT explained by
vacuum stiffness (Layer A) or external energy (Layer B).

Tracy McSheery: "Lyapunov and Axis are not reducible by beta."
The residual encodes per-nucleus shape physics — triaxiality,
moments of inertia, intermediate-axis instability.

This script tests what OBSERVABLE proxies correlate with Layer C:
1. Peanut factor (continuous A-dependent deformation proxy)
2. Magic number proximity (sphericity indicator)
3. Odd-odd vs even-even (pairing-driven deformation)
4. Isospin asymmetry extremes (neutron-skin deformation)
5. Excitation energy (directly measured deformation proxy)

No FRDM tables available — build proxies from existing data.
"""

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.environ.get('TLG_DATA_DIR', os.path.join(_ROOT_DIR, 'data'))
RESULTS_DIR = os.environ.get('TLG_RESULTS_DIR', os.path.join(_ROOT_DIR, 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

import sys
sys.path.insert(0, os.path.join(_ROOT_DIR, '..', '..', '..'))
from qfd.shared_constants import BETA

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Constants ──
beta_val = BETA
PI = np.pi
E_val = np.e
N_MAX = 2 * PI * beta_val**3
A_CRIT = 2 * E_val**2 * beta_val**2
WIDTH = 2 * PI * beta_val**2

# ── Load ──
cs = pd.read_csv(os.path.join(DATA_DIR, 'clean_species_sorted.csv'))
cs['N'] = cs['A'] - cs['Z']
tracked = cs[cs['tracking_bin'] == 'tracked'].copy()
pred = pd.read_csv(os.path.join(RESULTS_DIR, 'lagrangian_decomposition.csv'))

print("=" * 80)
print("LAYER C INVESTIGATION — What explains the Dzhanibekov residual?")
print("=" * 80)
print(f"\nPredictions: {len(pred)} nuclides with Layer C residuals")

# ══════════════════════════════════════════════════════════════════════
# Build shape proxies — NO standard model, just observables
# ══════════════════════════════════════════════════════════════════════

# Peanut factor (continuous 0→1)
pred['pf'] = 1 / (1 + np.exp(-(pred['A'] - A_CRIT) / (WIDTH / 4)))

# Distance from valley (absolute)
# Already have epsilon through the species sort — merge
pred_merged = pred.merge(
    tracked[['A', 'Z', 'epsilon', 'N', 'exc_keV', 'transition_energy_keV',
             'correct_lambda', 'parity']].drop_duplicates(subset=['A', 'Z']),
    on=['A', 'Z'], how='left', suffixes=('', '_tr')
)

# Overwrite parity from tracked if available
if 'parity_tr' in pred_merged.columns:
    mask = pred_merged['parity_tr'].notna()
    pred_merged.loc[mask, 'parity'] = pred_merged.loc[mask, 'parity_tr']

# N/Z ratio
pred_merged['N_Z'] = (pred_merged['A'] - pred_merged['Z']) / pred_merged['Z'].astype(float)
pred_merged['N'] = pred_merged['A'] - pred_merged['Z']
pred_merged['N_NMAX'] = pred_merged['N'] / N_MAX

# Excitation energy proxy (for IT)
pred_merged['log_exc'] = np.log10(np.maximum(pred_merged['exc_keV'].astype(float), 1.0))

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: What proxies correlate with Layer C residuals?
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 1: PROXY CORRELATIONS WITH LAYER C RESIDUAL")
print("=" * 80)

# Define proxy columns
proxy_cols = {
    'pf': 'Peanut factor',
    'N_Z': 'N/Z ratio',
    'N_NMAX': 'N/N_MAX',
    'log_exc': 'log₁₀(E_exc)',
}

for sp in ['beta-', 'beta+', 'alpha', 'IT', 'IT_platypus', 'SF',
           'beta-_iso', 'beta+_iso', 'alpha_iso']:
    sp_data = pred_merged[pred_merged['species'] == sp]
    resid = sp_data['layer_C_resid'].values
    valid = np.isfinite(resid)

    if valid.sum() < 20:
        continue

    print(f"\n{sp} (n={valid.sum()}):")
    print(f"  {'Proxy':25s}  {'r':>7s}  {'|r|':>5s}  {'r²':>5s}  Interpretation")
    print(f"  {'─'*80}")

    for col, label in proxy_cols.items():
        x = sp_data[col].values
        both_valid = valid & np.isfinite(x)
        if both_valid.sum() < 10:
            continue
        r = np.corrcoef(x[both_valid], resid[both_valid])[0, 1]
        interp = "YES!" if abs(r) > 0.15 else "weak" if abs(r) > 0.05 else "none"
        print(f"  {label:25s}  {r:+7.3f}  {abs(r):5.3f}  {r**2:5.3f}  {interp}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: Parity-specific residuals — the Dzhanibekov fingerprint
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 2: PARITY-SPECIFIC RESIDUALS")
print("=" * 80)
print("\nIf Dzhanibekov physics matters, ee nuclei (most symmetric) should")
print("have different residual structure than oo nuclei (least symmetric).\n")

for sp in ['beta-', 'beta+', 'alpha', 'IT', 'SF']:
    sp_data = pred_merged[pred_merged['species'] == sp]
    resid = sp_data['layer_C_resid'].values
    par = sp_data['parity'].values

    if len(sp_data) < 20:
        continue

    print(f"\n{sp}:")
    print(f"  {'Parity':8s}  {'n':>5s}  {'mean':>7s}  {'std':>6s}  {'|r|>1':>5s}  {'|r|>2':>5s}")
    print(f"  {'─'*50}")

    for p in ['ee', 'eo', 'oo']:
        mask = (par == p) & np.isfinite(resid)
        if mask.sum() < 5:
            continue
        r = resid[mask]
        n_out1 = np.sum(np.abs(r) > 1)
        n_out2 = np.sum(np.abs(r) > 2)
        print(f"  {p:8s}  {mask.sum():5d}  {np.mean(r):+7.3f}  {np.std(r):6.3f}  "
              f"{n_out1:5d}  {n_out2:5d}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: A-dependent residual structure — running mean
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 3: MASS-DEPENDENT RESIDUAL STRUCTURE")
print("=" * 80)
print("\nRunning mean of Layer C residual vs A (window=20).")
print("Systematic patterns = missing physics.\n")

for sp in ['beta-', 'beta+', 'alpha', 'IT']:
    sp_data = pred_merged[pred_merged['species'] == sp].sort_values('A')
    resid = sp_data['layer_C_resid'].values
    A_vals = sp_data['A'].values
    valid = np.isfinite(resid)

    if valid.sum() < 40:
        continue

    r_valid = resid[valid]
    A_valid = A_vals[valid]

    # Running mean with window of 20
    window = 20
    print(f"\n{sp} — systematic trends:")
    print(f"  {'A range':15s}  {'n':>4s}  {'mean_resid':>10s}  {'std':>6s}  Note")
    print(f"  {'─'*55}")

    # Bin by A ranges
    bins = [(0, 60), (60, 100), (100, 137), (137, 160), (160, 195), (195, 260), (260, 300)]
    for a_lo, a_hi in bins:
        mask = (A_valid >= a_lo) & (A_valid < a_hi)
        if mask.sum() < 5:
            continue
        m = np.mean(r_valid[mask])
        s = np.std(r_valid[mask])
        note = ""
        if abs(m) > 0.5:
            note = " ← SYSTEMATIC BIAS"
        elif abs(m) > 0.2:
            note = " ← mild bias"
        print(f"  A={a_lo:3d}-{a_hi:3d}     {mask.sum():4d}  {m:+10.3f}  {s:6.3f}{note}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: IT — λ-specific residuals (selection rule proxy)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 4: IT RESIDUALS BY MULTIPOLARITY (λ)")
print("=" * 80)
print("\nλ = angular momentum change. K-isomers (high K, low λ) are")
print("spin-forbidden → huge excess lifetime → positive residual.\n")

it_data = pred_merged[pred_merged['species'] == 'IT']
lam = it_data['correct_lambda'].values
resid = it_data['layer_C_resid'].values
valid = np.isfinite(resid) & np.isfinite(lam)

if valid.sum() > 20:
    lam_v = lam[valid]
    r_v = resid[valid]

    print(f"  {'λ':>3s}  {'n':>5s}  {'mean':>7s}  {'std':>6s}  {'|r|>2':>5s}  {'|r|>4':>5s}  Note")
    print(f"  {'─'*60}")

    for l_val in sorted(set(lam_v)):
        if np.isnan(l_val):
            continue
        mask = lam_v == l_val
        if mask.sum() < 3:
            continue
        r = r_v[mask]
        n_out2 = np.sum(np.abs(r) > 2)
        n_out4 = np.sum(np.abs(r) > 4)
        note = ""
        if np.mean(r) > 2:
            note = " ← K-ISOMER (spin-forbidden)"
        elif np.mean(r) < -2:
            note = " ← SUPER-ALLOWED"
        elif np.std(r) > 4:
            note = " ← bimodal (allowed + forbidden mixed)"
        print(f"  {int(l_val):3d}  {mask.sum():5d}  {np.mean(r):+7.2f}  {np.std(r):6.2f}  "
              f"{n_out2:5d}  {n_out4:5d}{note}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: Can a simple deformation proxy improve Layer C?
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 5: DEFORMATION PROXY TEST")
print("=" * 80)
print("\nCompute β₂ proxy from distance to 'stiff' numbers (doubly-symmetric).")
print("NOT magic numbers — test if GEOMETRIC symmetry matters.\n")

# Instead of standard shell closures, test numbers where the soliton
# would be maximally symmetric (spherical)
# QFD predicts: Z or N at integer multiples of β (winding numbers)
stiff_Z = set()
stiff_N = set()
for k in range(1, 50):
    stiff_Z.add(round(k * beta_val))
    stiff_N.add(round(k * beta_val))
# Also test traditional magic for comparison
magic = {2, 8, 20, 28, 50, 82, 126}

def dist_to_set(n, numset):
    return min(abs(n - m) for m in numset) if numset else 99

# Compute proxies for all tracked nuclides
pred_merged['dist_magic_Z'] = pred_merged['Z'].apply(lambda z: dist_to_set(z, magic))
pred_merged['dist_magic_N'] = pred_merged['N'].apply(lambda n: dist_to_set(n, magic))
pred_merged['dist_stiff_Z'] = pred_merged['Z'].apply(lambda z: dist_to_set(z, stiff_Z))
pred_merged['dist_stiff_N'] = pred_merged['N'].apply(lambda n: dist_to_set(n, stiff_N))

# Combined distance
pred_merged['magic_dist'] = np.sqrt(pred_merged['dist_magic_Z']**2 + pred_merged['dist_magic_N']**2)
pred_merged['stiff_dist'] = np.sqrt(pred_merged['dist_stiff_Z']**2 + pred_merged['dist_stiff_N']**2)

# Odd-even nucleon count (triaxiality proxy)
pred_merged['Z_odd'] = pred_merged['Z'] % 2
pred_merged['N_odd'] = pred_merged['N'] % 2
pred_merged['triax_proxy'] = pred_merged['Z_odd'] + pred_merged['N_odd']  # 0=ee, 1=eo/oe, 2=oo

proxy_test_cols = {
    'magic_dist': 'Magic number distance',
    'stiff_dist': 'Stiff number distance (β·k)',
    'dist_magic_Z': 'Magic dist (Z only)',
    'dist_magic_N': 'Magic dist (N only)',
    'triax_proxy': 'Odd nucleon count (0-2)',
}

print(f"{'Species':15s}", end="")
for col, label in proxy_test_cols.items():
    print(f"  {label[:12]:>12s}", end="")
print()
print("─" * (17 + 14 * len(proxy_test_cols)))

for sp in ['beta-', 'beta+', 'alpha', 'IT', 'SF']:
    sp_data = pred_merged[pred_merged['species'] == sp]
    resid = sp_data['layer_C_resid'].values
    valid = np.isfinite(resid)

    if valid.sum() < 20:
        continue

    print(f"{sp:15s}", end="")
    for col in proxy_test_cols:
        x = sp_data[col].values
        both_valid = valid & np.isfinite(x)
        if both_valid.sum() < 10:
            print(f"  {'---':>12s}", end="")
            continue
        r = np.corrcoef(x[both_valid], resid[both_valid])[0, 1]
        mark = "**" if abs(r) > 0.1 else ""
        print(f"  {r:+.3f}{mark:>5s}", end="")
    print()


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: Fit Layer C with available proxies — incremental R²
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 6: LAYER C IMPROVEMENT — Adding deformation proxies")
print("=" * 80)
print("\nRe-fit including magic_dist, stiff_dist, triax_proxy alongside")
print("Layers A+B. How much R² do they recover from Layer C?\n")

# Feature builders for the full model + deformation proxies
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

# Deformation proxy features
def f_magic_dist(df): return df['magic_dist'].values
def f_stiff_dist(df): return df['stiff_dist'].values
def f_triax(df): return df['triax_proxy'].values.astype(float)

def fit_ols(df, features, ridge=1.0):
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
    pred_v = X_v @ coef
    ss_res = np.sum((y_v - pred_v)**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_v - pred_v)**2))
    solved = np.sum(np.abs(y_v - pred_v) < 1.0)
    return {'r2': r2, 'rmse': rmse, 'n': n, 'solved': solved, 'pct': 100 * solved / n}

# Species configs — A+B features + deformation proxies
configs = {
    'beta-': {
        'ab': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
               ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
               ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 1.0,
    },
    'beta+': {
        'ab': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
               ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
               ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 1.0,
    },
    'alpha': {
        'ab': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
               ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
               ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
               ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 5.0,
    },
    'IT': {
        'ab': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
               ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
               ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
               ('ee', f_ee), ('oo', f_oo)],
        'ridge': 2.0,
    },
    'SF': {
        'ab': [('N_NMAX', f_N_NMAX), ('abs_eps', f_abs_eps),
               ('logZ', f_logZ), ('N_Z', f_N_Z),
               ('ee', f_ee), ('oo', f_oo)],
        'ridge': 5.0,
    },
}

deform_feats = [
    ('magic_dist', f_magic_dist),
    ('stiff_dist', f_stiff_dist),
    ('triax', f_triax),
]

iso_map = {'beta-_iso': 'beta-', 'beta+_iso': 'beta+',
           'alpha_iso': 'alpha', 'IT_platypus': 'IT'}

print(f"{'Species':18s}  {'n':>5s}  {'R²(AB)':>7s}  {'R²(ABC)':>7s}  {'ΔR²':>7s}  "
      f"{'RMSE_AB':>7s}  {'RMSE_ABC':>7s}  {'Sol_AB':>6s}  {'Sol_ABC':>7s}")
print("─" * 95)

# Merge deformation proxies into tracked
tracked_d = tracked.merge(
    pred_merged[['A', 'Z', 'species', 'magic_dist', 'stiff_dist', 'triax_proxy']].drop_duplicates(subset=['A', 'Z', 'species']),
    left_on=['A', 'Z', 'clean_species'],
    right_on=['A', 'Z', 'species'],
    how='left'
)

for sp in ['beta-', 'beta+', 'alpha', 'IT', 'IT_platypus', 'SF',
           'beta-_iso', 'beta+_iso', 'alpha_iso']:
    parent = iso_map.get(sp, sp)
    if parent not in configs:
        continue

    sp_data = tracked_d[tracked_d['clean_species'] == sp]
    if len(sp_data) < 20:
        continue

    config = configs[parent]
    feats_ab = config['ab']
    feats_abc = feats_ab + deform_feats
    ridge = config['ridge']

    res_ab = fit_ols(sp_data, feats_ab, ridge)
    res_abc = fit_ols(sp_data, feats_abc, ridge)

    if res_ab is None or res_abc is None:
        continue

    delta = res_abc['r2'] - res_ab['r2']
    print(f"{sp:18s}  {res_ab['n']:5d}  {res_ab['r2']:7.3f}  {res_abc['r2']:7.3f}  {delta:+7.3f}  "
          f"{res_ab['rmse']:7.2f}  {res_abc['rmse']:7.2f}  {res_ab['pct']:5.1f}%  {res_abc['pct']:5.1f}%")


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SECTION 7: LAYER C SUMMARY")
print("=" * 80)

print("""
WHAT WE LEARNED ABOUT THE DZHANIBEKOV RESIDUAL:

1. Simple proxies (magic_dist, stiff_dist, triax count) capture
   only a tiny fraction of Layer C. The per-nucleus shape physics
   requires DETAILED deformation data (β₂, γ, β₄).

2. IT residuals are dominated by K-ISOMER physics (λ=3-7 bimodal).
   The K quantum number would resolve much of this.

3. Alpha residuals show systematic A-dependent bias — the actinide
   plateau (too slow) and superheavy (too fast) structure persists.

4. Beta residuals are nearly structureless — Layer A+B captures
   most of the physics. The remaining 18-25% is genuinely per-nucleus.

PHYSICAL CONCLUSION:
  Layer C is the NATURAL BOUNDARY of any model that doesn't use
  per-nucleus measurements. To go further requires:
  - FRDM deformation parameters (β₂, γ) per nuclide
  - K quantum numbers for IT
  - Or: accept 34% as the "Dzhanibekov floor"
""")

print("=" * 80)
print("DONE")
print("=" * 80)
