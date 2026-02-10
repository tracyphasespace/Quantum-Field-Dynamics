#!/usr/bin/env python3
"""
Clean Species Sort — Separate NUBASE into proper decay channels.

Tracy McSheery directive: Isomers that decay to other isomers are platypuses.
They have wrong ΔJ, wrong transition energy, wrong physics. Separate them.

Classification:
  az_order=0: Ground state. Species = whatever it decays by.
  az_order=1: First isomer → ground state. Clean IT if dominant_mode=IT.
  az_order≥2: Higher isomer → lower isomer (PLATYPUS for IT).
              For non-IT species (alpha, beta), these are isomeric states
              that happen to decay by alpha/beta rather than IT.

After separation, fit each clean population with Q-values and measure per-channel.
"""

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.environ.get('TLG_DATA_DIR', os.path.join(_ROOT_DIR, 'data'))
RESULTS_DIR = os.environ.get('TLG_RESULTS_DIR', os.path.join(_ROOT_DIR, 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

# ── Constants ──
beta_val = 3.043233053
N_MAX = 2 * np.pi * beta_val**3
A_CRIT = 2 * np.e**2 * beta_val**2
WIDTH = 2 * np.pi * beta_val**2
r0_fm = 1.20
e2_MeV_fm = 1.4400
ME_He4 = 2424.916
ME_electron = 510.999

# ── Load data ──

print("=" * 72)
print("CLEAN SPECIES SORT — Platypus Separation")
print("=" * 72)

ip = pd.read_csv(os.path.join(DATA_DIR, 'iterative_peeling_predictions.csv'))
spin = pd.read_csv(os.path.join(DATA_DIR, 'nubase_excitation_spin.csv'))
ame = pd.read_csv(os.path.join(DATA_DIR, 'ame2020.csv'))

# Row-aligned
ip['az_order'] = spin['az_order'].values
ip['isomer_char'] = spin['isomer_char'].values
ip['spin_parity'] = spin['spin_parity'].values
ip['N'] = ip['A'] - ip['Z']

# Mass excess lookup
me_lookup = {}
for _, r in ame.iterrows():
    me_lookup[(int(r['A']), int(r['Z']))] = r['mass_excess_keV']

# ── Parse spin-parity ──

def parse_spin(sp_str):
    if pd.isna(sp_str):
        return None, None
    s = str(sp_str).strip()
    s = re.sub(r'\s+T=\S+', '', s)
    s = s.replace('(', '').replace(')', '').replace('*', '').replace('#', '').strip()
    if ',' in s:
        s = s.split(',')[0].strip()
    parity = None
    if '+' in s:
        parity = +1
        s = s.replace('+', '').strip()
    elif '-' in s:
        parity = -1
        s = s.replace('-', '').strip()
    s = s.strip()
    if not s:
        return None, parity
    try:
        if '/' in s:
            num, den = s.split('/')
            return float(num) / float(den), parity
        return float(s), parity
    except:
        return None, parity

# Build spin lookup: (A, Z, az_order) → (J, pi)
spin_lookup = {}
for _, row in spin.iterrows():
    J, pi = parse_spin(row['spin_parity'])
    spin_lookup[(int(row['A']), int(row['Z']), int(row['az_order']))] = (J, pi)

# ── Classify each nuclide ──

print("\n--- Classifying nuclides ---")

ip['clean_species'] = ip['species']
ip['is_platypus'] = False
ip['destination_az'] = np.nan
ip['correct_deltaJ'] = np.nan
ip['correct_lambda'] = np.nan
ip['transition_energy_keV'] = np.nan

for idx, row in ip.iterrows():
    A, Z, az = int(row['A']), int(row['Z']), int(row['az_order'])
    species = row['species']

    if az == 0:
        # Ground state — clean, no isomer issues
        ip.at[idx, 'destination_az'] = -1  # decays to daughter nucleus
        ip.at[idx, 'transition_energy_keV'] = 0  # not an isomeric transition
        continue

    if species == 'IT':
        if az == 1:
            # First isomer → ground state: clean IT
            ip.at[idx, 'destination_az'] = 0

            # Correct ΔJ
            J_src, pi_src = spin_lookup.get((A, Z, 1), (None, None))
            J_dst, pi_dst = spin_lookup.get((A, Z, 0), (None, None))

            if J_src is not None and J_dst is not None:
                dJ = abs(J_src - J_dst)
                ip.at[idx, 'correct_deltaJ'] = dJ
                lam = max(1, int(round(dJ))) if dJ > 0 else None
                if lam is not None and pi_src is not None and pi_dst is not None:
                    # Determine minimum multipole
                    pc = pi_src * pi_dst
                    for l in range(lam, lam + 5):
                        if pc == (-1)**l or pc == (-1)**(l+1):
                            ip.at[idx, 'correct_lambda'] = l
                            break

            # Transition energy = excitation energy of isomer
            ip.at[idx, 'transition_energy_keV'] = row.get('exc_keV', np.nan)

        else:
            # Higher isomer → PLATYPUS
            ip.at[idx, 'is_platypus'] = True
            ip.at[idx, 'clean_species'] = 'IT_platypus'
            ip.at[idx, 'destination_az'] = az - 1  # most likely destination

            # Compute correct ΔJ vs destination
            J_src, pi_src = spin_lookup.get((A, Z, az), (None, None))
            J_dst, pi_dst = spin_lookup.get((A, Z, az - 1), (None, None))

            if J_src is not None and J_dst is not None:
                dJ = abs(J_src - J_dst)
                ip.at[idx, 'correct_deltaJ'] = dJ
                lam = max(1, int(round(dJ))) if dJ > 0 else None
                if lam is not None and pi_src is not None and pi_dst is not None:
                    pc = pi_src * pi_dst
                    for l in range(lam, lam + 5):
                        if pc == (-1)**l or pc == (-1)**(l+1):
                            ip.at[idx, 'correct_lambda'] = l
                            break

            # Transition energy = E_x(this) - E_x(destination)
            exc_src = row.get('exc_keV', np.nan)
            # Get destination excitation energy
            dest_rows = spin[(spin['A'] == A) & (spin['Z'] == Z) & (spin['az_order'] == az - 1)]
            if len(dest_rows) > 0 and pd.notna(exc_src):
                exc_dst = dest_rows.iloc[0].get('exc_keV', 0)
                if pd.isna(exc_dst):
                    exc_dst = 0
                ip.at[idx, 'transition_energy_keV'] = exc_src - exc_dst

    elif species in ['alpha', 'beta-', 'beta+', 'SF', 'proton']:
        if az >= 2:
            # Isomeric state decaying by this species — not a platypus per se,
            # but flag it as an isomeric variant
            ip.at[idx, 'clean_species'] = f'{species}_iso'
        elif az == 1:
            # First isomer decaying by this species (not IT)
            ip.at[idx, 'clean_species'] = f'{species}_iso'

# ── Report ──

print("\n" + "=" * 72)
print("CLEAN POPULATION CENSUS")
print("=" * 72)

census = ip['clean_species'].value_counts()
print(f"\n{'Species':20s}  {'Count':>6s}  {'%':>6s}")
print("-" * 36)
for sp, n in census.items():
    print(f"{sp:20s}  {n:6d}  {100*n/len(ip):5.1f}%")

# ── Compute Q-values (only for ground states, properly) ──

print("\n" + "=" * 72)
print("Q-VALUES FOR CLEAN POPULATIONS")
print("=" * 72)

# Q-values per species
for idx, row in ip.iterrows():
    A, Z = int(row['A']), int(row['Z'])
    cs = row['clean_species']
    exc = row.get('exc_keV', 0)
    if pd.isna(exc):
        exc = 0

    me_par = me_lookup.get((A, Z))
    if me_par is not None:
        me_par += exc

    if cs in ['alpha', 'alpha_iso'] and me_par is not None:
        me_d = me_lookup.get((A-4, Z-2))
        if me_d is not None:
            ip.at[idx, 'Q_keV'] = me_par - me_d - ME_He4
            Z_d = Z - 2
            R_d = r0_fm * (A-4)**(1/3)
            R_a = r0_fm * 4**(1/3)
            ip.at[idx, 'V_coulomb_keV'] = e2_MeV_fm * Z_d * 2 / (R_d + R_a) * 1000
    elif cs in ['beta-', 'beta-_iso'] and me_par is not None:
        me_d = me_lookup.get((A, Z+1))
        if me_d is not None:
            ip.at[idx, 'Q_keV'] = me_par - me_d
    elif cs in ['beta+', 'beta+_iso'] and me_par is not None:
        me_d = me_lookup.get((A, Z-1))
        if me_d is not None:
            ip.at[idx, 'Q_keV'] = me_par - me_d - 2 * ME_electron
    elif cs in ['IT', 'IT_platypus']:
        ip.at[idx, 'Q_keV'] = row.get('transition_energy_keV', np.nan)

# ── Per-channel fitting ──

print("\n" + "=" * 72)
print("PER-CHANNEL FITS — Clean Populations with Q-values")
print("=" * 72)

def fit_and_report(df, features, name, alpha_ridge=1.0):
    y = df['log_hl'].values
    X_parts = []
    for fn, func in features:
        X_parts.append(func(df))
    X_parts.append(np.ones(len(df)))

    X = np.column_stack(X_parts)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = valid.sum()

    if n < max(len(features) + 2, 8):
        return None

    X_v, y_v = X[valid], y[valid]
    I = np.eye(X_v.shape[1]); I[-1,-1] = 0
    try:
        coef = np.linalg.solve(X_v.T @ X_v + alpha_ridge * I, X_v.T @ y_v)
    except:
        coef, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)

    pred = X_v @ coef
    resid = np.abs(y_v - pred)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2 = 1 - np.sum(resid**2) / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(resid**2))
    p90 = np.percentile(resid, 90)

    return {'n': n, 'r2': r2, 'rmse': rmse, 'p90': p90, 'hl_std': y_v.std()}

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
    return df['correct_lambda'].fillna(df['correct_lambda'].median()).values
def f_lambda_sq(df):
    lam = df['correct_lambda'].fillna(df['correct_lambda'].median()).values
    return lam**2

# Regime boundaries
regimes = [
    ('light', 0, 60),
    ('medium', 60, A_CRIT),
    ('transition', A_CRIT, A_CRIT + WIDTH),
    ('heavy', A_CRIT + WIDTH, 300),
]

# Species definitions
species_configs = {
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
    'IT_platypus': {
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

all_results = []
decay = ip[~ip['species'].isin(['stable', 'rare'])].copy()

print(f"\n{'Channel':35s}  {'n':>5s}  {'R²':>7s}  {'RMSE':>6s}  {'P90':>6s}  {'σ_hl':>5s}")
print("-" * 72)

for sp in ['beta-', 'beta+', 'alpha', 'IT', 'IT_platypus', 'SF',
           'beta-_iso', 'beta+_iso', 'alpha_iso', 'proton', 'neutron']:

    sp_data = decay[decay['clean_species'] == sp]
    if len(sp_data) < 8:
        if len(sp_data) > 0:
            print(f"{sp:35s}  {len(sp_data):5d}  {'(too few)':>7s}")
        continue

    # Get config (fall back to closest match)
    base_sp = sp.replace('_iso', '')
    config = species_configs.get(base_sp, species_configs.get('beta-'))
    features = config['features']
    ridge = config['ridge']

    # A. Lumped
    res = fit_and_report(sp_data, features, f"{sp}_lumped", ridge)
    if res:
        print(f"{sp + ' (lumped)':35s}  {res['n']:5d}  {res['r2']:7.4f}  {res['rmse']:6.2f}  {res['p90']:6.2f}  {res['hl_std']:5.2f}")
        all_results.append({'species': sp, 'parity': 'all', 'a_regime': 'all', **res})

    # B. By peanut regime
    for rname, a_lo, a_hi in regimes:
        rdata = sp_data[(sp_data['A'] >= a_lo) & (sp_data['A'] < a_hi)]
        if len(rdata) < 8:
            continue

        feats = features if len(rdata) > 20 else features[:5]
        res = fit_and_report(rdata, feats, f"{sp}_{rname}", ridge)
        if res:
            tag = f"  {sp}/{rname}"
            print(f"{tag:35s}  {res['n']:5d}  {res['r2']:7.4f}  {res['rmse']:6.2f}  {res['p90']:6.2f}  {res['hl_std']:5.2f}")
            all_results.append({'species': sp, 'parity': 'all', 'a_regime': rname, **res})

    # C. By parity (lumped across regimes)
    for par in ['ee', 'eo', 'oo']:
        pdata = sp_data[sp_data['parity'] == par]
        if len(pdata) < 10:
            continue
        feats = features if len(pdata) > 20 else features[:5]
        res = fit_and_report(pdata, feats, f"{sp}_{par}", ridge)
        if res:
            tag = f"  {sp}/{par}"
            print(f"{tag:35s}  {res['n']:5d}  {res['r2']:7.4f}  {res['rmse']:6.2f}  {res['p90']:6.2f}  {res['hl_std']:5.2f}")
            all_results.append({'species': sp, 'parity': par, 'a_regime': 'all', **res})

# ── IT Platypus: correct ΔJ analysis ──

print("\n" + "=" * 72)
print("IT vs IT_PLATYPUS — Corrected ΔJ Comparison")
print("=" * 72)

it_clean = decay[decay['clean_species'] == 'IT']
it_plat = decay[decay['clean_species'] == 'IT_platypus']

for label, data in [('IT (iso1→ground)', it_clean), ('IT_platypus (iso2+→iso)', it_plat)]:
    has_lam = data['correct_lambda'].notna()
    print(f"\n{label}:")
    print(f"  Total: {len(data)}, with correct λ: {has_lam.sum()}")
    if has_lam.sum() > 0:
        lam = data[has_lam]['correct_lambda']
        dj = data[has_lam]['correct_deltaJ']
        print(f"  ΔJ: mean={dj.mean():.1f}, median={dj.median():.1f}")
        print(f"  λ:  mean={lam.mean():.1f}, median={lam.median():.1f}")

        # λ distribution
        for l in sorted(lam.unique())[:10]:
            n_l = (lam == l).sum()
            sub_hl = data[has_lam & (data['correct_lambda'] == l)]['log_hl']
            if len(sub_hl) > 0:
                print(f"    λ={int(l):2d}: n={n_l:4d}, mean_hl={sub_hl.mean():+.2f}")

# ── Save clean dataset ──

print("\n" + "=" * 72)
print("SAVING CLEAN DATASET")
print("=" * 72)

out_cols = ['A', 'Z', 'N', 'element', 'species', 'clean_species', 'is_platypus',
            'az_order', 'parity', 'epsilon', 'log_hl', 'exc_keV',
            'transition_energy_keV', 'correct_deltaJ', 'correct_lambda',
            'Q_keV', 'V_coulomb_keV']
out = ip[[c for c in out_cols if c in ip.columns]].copy()
out.to_csv(os.path.join(RESULTS_DIR, 'clean_species_sorted.csv'), index=False)
print(f"Saved: {os.path.join(RESULTS_DIR, 'clean_species_sorted.csv')} ({len(out)} rows)")

pd.DataFrame(all_results).to_csv(os.path.join(RESULTS_DIR, 'clean_channel_scores.csv'), index=False)
print(f"Saved: {os.path.join(RESULTS_DIR, 'clean_channel_scores.csv')} ({len(all_results)} channels)")
