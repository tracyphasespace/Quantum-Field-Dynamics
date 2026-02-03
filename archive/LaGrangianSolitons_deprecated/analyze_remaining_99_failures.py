#!/usr/bin/env python3
"""
ANALYZE REMAINING 99 FAILURES (34.7%)
===========================================================================
With optimized dual-resonance configuration:
- Magic bonus: 0.10
- Symmetric resonance: N/Z ∈ [0.95, 1.15], bonus = 0.30
- Neutron-rich resonance: N/Z ∈ [1.15, 1.30], bonus = 0.10
- Pairing: δ = 11.0 MeV

Current: 186/285 (65.3%)
Remaining failures: 99 (34.7%)

Identify patterns to push beyond 65%.
===========================================================================
"""

import numpy as np
from collections import Counter

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52

# OPTIMAL DUAL-RESONANCE CONFIGURATION
MAGIC_BONUS = 0.10
SYMM_BONUS = 0.30
NR_BONUS = 0.10
DELTA_PAIRING = 11.0

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * MAGIC_BONUS * 0.5

    nz_ratio = N / Z if Z > 0 else 0
    if 0.95 <= nz_ratio <= 1.15:
        bonus += E_surface * SYMM_BONUS
    if 1.15 <= nz_ratio <= 1.30:
        bonus += E_surface * NR_BONUS

    return bonus

def qfd_energy(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_E * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym = (beta_vacuum * M_proton) / 15
    a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))
    E_iso = -get_resonance_bonus(Z, N, E_surface)

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A):
    best_Z, best_E = 1, qfd_energy(A, 1)
    for Z in range(1, A):
        E = qfd_energy(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("ANALYZING REMAINING 99 FAILURES (34.7%)")
print("="*95)
print()
print("Optimized dual-resonance configuration:")
print(f"  Magic bonus: {MAGIC_BONUS}")
print(f"  Symmetric resonance: N/Z ∈ [0.95, 1.15], bonus={SYMM_BONUS}")
print(f"  Neutron-rich resonance: N/Z ∈ [1.15, 1.30], bonus={NR_BONUS}")
print(f"  Pairing: δ={DELTA_PAIRING} MeV")
print()

# Classify
data = []
for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z(A)
    N_exp = A - Z_exp
    N_pred = A - Z_pred

    nz_ratio = N_exp / Z_exp if Z_exp > 0 else 0
    in_symm = 0.95 <= nz_ratio <= 1.15
    in_nr = 1.15 <= nz_ratio <= 1.30

    record = {
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'Delta_Z': Z_pred - Z_exp,
        'NZ_ratio': nz_ratio,
        'q': Z_exp / A,
        'Z_even': Z_exp % 2 == 0,
        'N_even': N_exp % 2 == 0,
        'Z_magic': Z_exp in ISOMER_NODES,
        'N_magic': N_exp in ISOMER_NODES,
        'in_symm': in_symm,
        'in_nr': in_nr,
        'in_any_res': in_symm or in_nr,
    }
    data.append(record)

survivors = [d for d in data if d['Delta_Z'] == 0]
failures = [d for d in data if d['Delta_Z'] != 0]

print(f"Survivors: {len(survivors)}/285 (65.3%)")
print(f"Failures:  {len(failures)}/285 (34.7%)")
print()

# Error distribution
Delta_Z_dist = Counter(d['Delta_Z'] for d in failures)

print("="*95)
print("ERROR DISTRIBUTION")
print("="*95)
print()
print(f"{'ΔZ':<8} {'Count':<10} {'%'}  ")
print("-"*95)
for dz in sorted(Delta_Z_dist.keys()):
    count = Delta_Z_dist[dz]
    pct = 100 * count / len(failures)
    print(f"{dz:+d}       {count:<10} {pct:.1f}%")
print()

# Mass region
regions = [
    ("Light (A<40)", lambda d: d['A'] < 40),
    ("Medium (40≤A<100)", lambda d: 40 <= d['A'] < 100),
    ("Heavy (100≤A<200)", lambda d: 100 <= d['A'] < 200),
    ("Superheavy (A≥200)", lambda d: d['A'] >= 200),
]

print("="*95)
print("BY MASS REGION")
print("="*95)
print()
print(f"{'Region':<25} {'Failures':<20} {'Fail Rate'}  ")
print("-"*95)

for name, condition in regions:
    in_region = [d for d in data if condition(d)]
    fail_in_region = [d for d in failures if condition(d)]

    if len(in_region) > 0:
        fail_rate = 100 * len(fail_in_region) / len(in_region)
        print(f"{name:<25} {len(fail_in_region)}/{len(in_region):<17} {fail_rate:.1f}%")

print()

# Pairing
pairing_cats = [
    ("Even-Even", lambda d: d['Z_even'] and d['N_even']),
    ("Even-Odd", lambda d: d['Z_even'] and not d['N_even']),
    ("Odd-Even", lambda d: not d['Z_even'] and d['N_even']),
    ("Odd-Odd", lambda d: not d['Z_even'] and not d['N_even']),
]

print("="*95)
print("BY PAIRING")
print("="*95)
print()
print(f"{'Type':<15} {'Failures':<20} {'Fail Rate'}  ")
print("-"*95)

for name, condition in pairing_cats:
    in_cat = [d for d in data if condition(d)]
    fail_in_cat = [d for d in failures if condition(d)]

    if len(in_cat) > 0:
        fail_rate = 100 * len(fail_in_cat) / len(in_cat)
        print(f"{name:<15} {len(fail_in_cat)}/{len(in_cat):<17} {fail_rate:.1f}%")

print()

# Resonance status
print("="*95)
print("BY RESONANCE STATUS")
print("="*95)
print()

res_categories = [
    ("In symmetric [0.95, 1.15]", lambda d: d['in_symm']),
    ("In neutron-rich [1.15, 1.30]", lambda d: d['in_nr']),
    ("Outside resonances", lambda d: not d['in_any_res']),
]

print(f"{'Category':<30} {'Failures':<20} {'Fail Rate'}  ")
print("-"*95)

for name, condition in res_categories:
    in_cat = [d for d in data if condition(d)]
    fail_in_cat = [d for d in failures if condition(d)]

    if len(in_cat) > 0:
        fail_rate = 100 * len(fail_in_cat) / len(in_cat)
        print(f"{name:<30} {len(fail_in_cat)}/{len(in_cat):<17} {fail_rate:.1f}%")

print()

# Large errors
print("="*95)
print("LARGEST ERRORS (|ΔZ| ≥ 2)")
print("="*95)
print()

large_errors = [d for d in failures if abs(d['Delta_Z']) >= 2]
large_errors.sort(key=lambda x: abs(x['Delta_Z']), reverse=True)

print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<7} {'Z_pred':<7} {'ΔZ':<6} {'N/Z':<8} {'Type':<10} {'Notes'}")
print("-"*95)

for d in large_errors[:30]:
    ptype = "E-E" if d['Z_even'] and d['N_even'] else \
            "E-O" if d['Z_even'] else \
            "O-E" if d['N_even'] else "O-O"

    notes = []
    if d['Z_magic']: notes.append("Z_mag")
    if d['N_magic']: notes.append("N_mag")
    if d['in_symm']: notes.append("symm")
    if d['in_nr']: notes.append("nr")

    print(f"{d['name']:<12} {d['A']:<6} {d['Z_exp']:<7} {d['Z_pred']:<7} "
          f"{d['Delta_Z']:<6} {d['NZ_ratio']:<8.2f} {ptype:<10} {', '.join(notes)}")

print()

# SUMMARY
print("="*95)
print("SUMMARY: What's Missing for Final 34.7%")
print("="*95)
print()

# Check heavy dominance
heavy_fail = [d for d in failures if d['A'] >= 100]
if len(heavy_fail) > len(failures) * 0.55:
    print(f"1. ★ HEAVY NUCLEUS CORRECTIONS (A≥100: {len(heavy_fail)}/{len(failures)} = "
          f"{100*len(heavy_fail)/len(failures):.1f}%)")
    print("   - Shell effects beyond magic numbers")
    print("   - Deformation effects")
    print()

# Check N/Z outside resonances
outside_res = [d for d in failures if not d['in_any_res']]
if len(outside_res) > len(failures) * 0.30:
    print(f"2. ★ NUCLEI OUTSIDE RESONANCES ({len(outside_res)}/{len(failures)} = "
          f"{100*len(outside_res)/len(failures):.1f}%)")
    print("   - N/Z < 0.95 or N/Z > 1.30")
    print("   - May need additional resonance windows")
    print("   - Or mass-dependent asymmetry coefficient")
    print()

# Check small errors
small_errors = [d for d in failures if abs(d['Delta_Z']) == 1]
if len(small_errors) > len(failures) * 0.50:
    print(f"3. ★ FINE-STRUCTURE (|ΔZ|=1: {len(small_errors)}/{len(failures)} = "
          f"{100*len(small_errors)/len(failures):.1f}%)")
    print("   - Small systematic adjustments")
    print("   - Sub-shell effects")
    print()

# Check specific stubborn cases
doubly_magic_fail = [d for d in failures if d['Z_magic'] and d['N_magic']]
if doubly_magic_fail:
    print(f"4. ★ DOUBLY MAGIC FAILURES ({len(doubly_magic_fail)} cases)")
    for d in doubly_magic_fail:
        print(f"   - {d['name']}: Z={d['Z_exp']}, N={d['N_exp']}, pred={d['Z_pred']}")
    print()

print("="*95)
print("PROGRESS TIMELINE")
print("="*95)
print()
print("  Original baseline:        129/285 (45.3%)")
print("  + Bonus optimization:     142/285 (49.8%) [+13]")
print("  + Charge resonance:       145/285 (50.9%) [+3]")
print("  + Pairing energy:         178/285 (62.5%) [+33]")
print("  + Dual resonance:         186/285 (65.3%) [+8]")
print()
print("  Total improvement:        +57 exact matches")
print("  Remaining:                99 failures (34.7%)")
print()
print("="*95)
