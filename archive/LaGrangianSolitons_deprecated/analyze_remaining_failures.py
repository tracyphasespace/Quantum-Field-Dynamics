#!/usr/bin/env python3
"""
ANALYZE THE 110 REMAINING FAILURES
===========================================================================
Both pure QFD and hybrid (full energy) achieve 175/285 (61.4%).

This means 110 nuclei are STILL wrong even with:
- Full energy functional (E_bulk + E_surf + E_asym + E_vac + E_pair)
- Correct pairing energy
- Vacuum displacement term
- Lambda_time Z-dependence

Question: What are these 110 failures? Is there a systematic pattern?
Are they clustered in specific mass regions, parities, or isotopic chains?
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_surface_coeff = beta_nuclear / 15
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def qfd_energy_full(A, Z):
    """Full QFD energy - complete Hamiltonian."""
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A
    lambda_time = KAPPA_E * Z

    E_volume_coeff = V_0 * (1 - lambda_time / (12 * np.pi))
    E_bulk = E_volume_coeff * A
    E_surf = E_surface_coeff * (A**(2/3))
    E_asym = a_sym_base * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    # Pairing energy
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def find_stable_Z_qfd(A):
    """Pure QFD: search all Z."""
    best_Z, best_E = 1, qfd_energy_full(A, 1)
    for Z in range(1, A):
        E = qfd_energy_full(A, Z)
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
print("ANALYZE THE 110 REMAINING FAILURES")
print("="*95)
print()

# Find all failures
failures = []
successes = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_qfd = find_stable_Z_qfd(A)
    error = Z_qfd - Z_exp

    data = {
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_qfd': Z_qfd,
        'error': error,
        'abs_error': abs(error),
        'mod_4': A % 4,
        'parity': 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else
                  'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A',
        'mass_region': 'light' if A < 60 else 'medium' if A < 140 else 'heavy',
    }

    if error == 0:
        successes.append(data)
    else:
        failures.append(data)

print(f"Total nuclei: {len(test_nuclides)}")
print(f"Correct: {len(successes)} ({100*len(successes)/len(test_nuclides):.1f}%)")
print(f"Failures: {len(failures)} ({100*len(failures)/len(test_nuclides):.1f}%)")
print()

# ============================================================================
# ERROR DIRECTION
# ============================================================================
print("="*95)
print("ERROR DIRECTION")
print("="*95)
print()

over = [f for f in failures if f['error'] > 0]
under = [f for f in failures if f['error'] < 0]

print(f"Overpredictions (Z too high):  {len(over)} ({100*len(over)/len(failures):.1f}%)")
print(f"Underpredictions (Z too low):  {len(under)} ({100*len(under)/len(failures):.1f}%)")
print()

if len(over) > len(under) * 1.5:
    print("★ Systematic OVERPREDICTION bias")
elif len(under) > len(over) * 1.5:
    print("★ Systematic UNDERPREDICTION bias")
else:
    print("→ No strong directional bias (roughly balanced)")

print()

# ============================================================================
# BY MASS REGION
# ============================================================================
print("="*95)
print("FAILURES BY MASS REGION")
print("="*95)
print()

print(f"{'Region':<15} {'Failures':<12} {'Total':<12} {'Failure %':<12} {'Success %'}")
print("-"*95)

for region in ['light', 'medium', 'heavy']:
    region_all = [n for n in test_nuclides if
                  (region == 'light' and n[2] < 60) or
                  (region == 'medium' and 60 <= n[2] < 140) or
                  (region == 'heavy' and n[2] >= 140)]
    region_fail = [f for f in failures if f['mass_region'] == region]

    total = len(region_all)
    n_fail = len(region_fail)
    fail_pct = 100 * n_fail / total if total > 0 else 0
    success_pct = 100 - fail_pct

    marker = "★★" if fail_pct > 50 else "★" if fail_pct > 40 else ""

    print(f"{region:<15} {n_fail:<12} {total:<12} {fail_pct:<12.1f} {success_pct:.1f}%  {marker}")

print()

# ============================================================================
# BY A MOD 4
# ============================================================================
print("="*95)
print("FAILURES BY A MOD 4")
print("="*95)
print()

print(f"{'A mod 4':<12} {'Failures':<12} {'Total':<12} {'Failure %':<12} {'Success %'}")
print("-"*95)

for mod in range(4):
    mod_all = [n for n in test_nuclides if n[2] % 4 == mod]
    mod_fail = [f for f in failures if f['mod_4'] == mod]

    total = len(mod_all)
    n_fail = len(mod_fail)
    fail_pct = 100 * n_fail / total if total > 0 else 0
    success_pct = 100 - fail_pct

    marker = "★★★" if success_pct > 70 else "★" if success_pct > 60 else ""

    print(f"{mod:<12} {n_fail:<12} {total:<12} {fail_pct:<12.1f} {success_pct:.1f}%  {marker}")

print()

# ============================================================================
# BY PARITY
# ============================================================================
print("="*95)
print("FAILURES BY PARITY")
print("="*95)
print()

print(f"{'Parity':<15} {'Failures':<12} {'Total':<12} {'Failure %':<12} {'Success %'}")
print("-"*95)

for parity in ['even-even', 'odd-odd', 'odd-A']:
    parity_all = []
    for name, Z_exp, A in test_nuclides:
        N_exp = A - Z_exp
        p = 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else \
            'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A'
        if p == parity:
            parity_all.append((name, Z_exp, A))

    parity_fail = [f for f in failures if f['parity'] == parity]

    total = len(parity_all)
    n_fail = len(parity_fail)
    fail_pct = 100 * n_fail / total if total > 0 else 0
    success_pct = 100 - fail_pct

    marker = "★" if success_pct > 65 else ""

    print(f"{parity:<15} {n_fail:<12} {total:<12} {fail_pct:<12.1f} {success_pct:.1f}%  {marker}")

print()

# ============================================================================
# BY ERROR MAGNITUDE
# ============================================================================
print("="*95)
print("ERROR MAGNITUDE DISTRIBUTION")
print("="*95)
print()

error_counts = Counter(f['abs_error'] for f in failures)

print(f"{'|Error|':<12} {'Count':<12} {'Percentage'}")
print("-"*95)

for abs_err in sorted(error_counts.keys()):
    count = error_counts[abs_err]
    pct = 100 * count / len(failures)
    print(f"{abs_err:<12} {count:<12} {pct:.1f}%")

print()

# Most common error
most_common_error = error_counts.most_common(1)[0]
print(f"Most common error magnitude: |error| = {most_common_error[0]} ({most_common_error[1]} cases)")
print()

# ============================================================================
# ISOTOPIC CHAINS
# ============================================================================
print("="*95)
print("FAILURES BY ISOTOPIC CHAIN (Z VALUE)")
print("="*95)
print()

# Group failures by Z_exp
failures_by_Z = defaultdict(list)
for f in failures:
    failures_by_Z[f['Z_exp']].append(f)

# Find Z values with most failures
Z_failure_counts = [(Z, len(nuclides)) for Z, nuclides in failures_by_Z.items()]
Z_failure_counts.sort(key=lambda x: x[1], reverse=True)

print("Top 15 elements with most failures:")
print(f"{'Z':<6} {'Element':<12} {'Failures':<12} {'Failed nuclei'}")
print("-"*95)

# Element names for context
element_names = {
    1: 'H', 2: 'He', 6: 'C', 8: 'O', 10: 'Ne', 12: 'Mg', 14: 'Si', 16: 'S',
    18: 'Ar', 20: 'Ca', 22: 'Ti', 24: 'Cr', 26: 'Fe', 28: 'Ni', 30: 'Zn',
    32: 'Ge', 34: 'Se', 36: 'Kr', 38: 'Sr', 40: 'Zr', 42: 'Mo', 44: 'Ru',
    46: 'Pd', 48: 'Cd', 50: 'Sn', 52: 'Te', 54: 'Xe', 56: 'Ba', 58: 'Ce',
    60: 'Nd', 62: 'Sm', 64: 'Gd', 66: 'Dy', 68: 'Er', 70: 'Yb', 72: 'Hf',
    74: 'W', 76: 'Os', 78: 'Pt', 80: 'Hg', 82: 'Pb', 83: 'Bi', 90: 'Th',
    92: 'U',
}

for Z, count in Z_failure_counts[:15]:
    elem = element_names.get(Z, '?')
    nuclei_list = [f['name'] for f in failures_by_Z[Z]]
    nuclei_str = ', '.join(nuclei_list[:5])
    if len(nuclei_list) > 5:
        nuclei_str += f", ... ({len(nuclei_list)} total)"

    print(f"{Z:<6} {elem:<12} {count:<12} {nuclei_str}")

print()

# ============================================================================
# SPECIFIC A RANGES
# ============================================================================
print("="*95)
print("FAILURE DENSITY BY MASS NUMBER")
print("="*95)
print()

# Bin by A ranges
A_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100),
          (100, 120), (120, 140), (140, 160), (160, 180), (180, 200), (200, 250)]

print(f"{'A range':<15} {'Failures':<12} {'Total':<12} {'Failure %':<12} {'Success %'}")
print("-"*95)

for A_min, A_max in A_bins:
    range_all = [n for n in test_nuclides if A_min < n[2] <= A_max]
    range_fail = [f for f in failures if A_min < f['A'] <= A_max]

    total = len(range_all)
    n_fail = len(range_fail)

    if total == 0:
        continue

    fail_pct = 100 * n_fail / total
    success_pct = 100 - fail_pct

    marker = "★★" if fail_pct > 50 else "★" if fail_pct > 40 else ""

    print(f"{A_min}-{A_max:<10} {n_fail:<12} {total:<12} {fail_pct:<12.1f} {success_pct:.1f}%  {marker}")

print()

# ============================================================================
# SAMPLE FAILURES (first 30)
# ============================================================================
print("="*95)
print("SAMPLE FAILURES (first 30)")
print("="*95)
print()

print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_QFD':<8} {'Error':<8} {'Mod 4':<8} {'Parity':<12} {'Region'}")
print("-"*95)

for f in failures[:30]:
    print(f"{f['name']:<12} {f['A']:<6} {f['Z_exp']:<8} {f['Z_qfd']:<8} {f['error']:+d}  "
          f"{f['mod_4']:<8} {f['parity']:<12} {f['mass_region']}")

if len(failures) > 30:
    print(f"... and {len(failures) - 30} more")

print()
print("="*95)
