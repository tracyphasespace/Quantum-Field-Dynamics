#!/usr/bin/env python3
"""
ANALYZE FINAL 107 FAILURES (37.5%)
===========================================================================
With full optimized configuration:
- Magic bonus: 0.10
- Charge resonance: N/Z ∈ [1.15, 1.30], bonus = 0.10
- Pairing: δ = 11.0 MeV

Current: 178/285 (62.5%)
Remaining failures: 107 (37.5%)

Find what's STILL missing after all optimizations.
===========================================================================
"""

import numpy as np
from collections import Counter

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52

# OPTIMAL CONFIGURATION
MAGIC_BONUS = 0.10
NZ_LOW, NZ_HIGH = 1.15, 1.30
NZ_BONUS = 0.10
DELTA_PAIRING = 11.0

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * MAGIC_BONUS * 0.5

    nz_ratio = N / Z if Z > 0 else 0
    if NZ_LOW <= nz_ratio <= NZ_HIGH:
        bonus += E_surface * NZ_BONUS

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

    # Pairing energy
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
print("ANALYZING FINAL 107 FAILURES (37.5%)")
print("="*95)
print()
print("Optimized configuration:")
print(f"  Magic bonus: {MAGIC_BONUS}")
print(f"  N/Z resonance: [{NZ_LOW}, {NZ_HIGH}], bonus={NZ_BONUS}")
print(f"  Pairing: δ={DELTA_PAIRING} MeV")
print()

# Classify
data = []
for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z(A)
    N_exp = A - Z_exp
    N_pred = A - Z_pred

    record = {
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'Delta_Z': Z_pred - Z_exp,
        'NZ_ratio': N_exp / Z_exp if Z_exp > 0 else 0,
        'q': Z_exp / A,
        'Z_even': Z_exp % 2 == 0,
        'N_even': N_exp % 2 == 0,
        'Z_magic': Z_exp in ISOMER_NODES,
        'N_magic': N_exp in ISOMER_NODES,
    }
    data.append(record)

survivors = [d for d in data if d['Delta_Z'] == 0]
failures = [d for d in data if d['Delta_Z'] != 0]

print(f"Survivors: {len(survivors)}/285 (62.5%)")
print(f"Failures:  {len(failures)}/285 (37.5%)")
print()

# ============================================================================
# DETAILED FAILURE BREAKDOWN
# ============================================================================
print("="*95)
print("FAILURE BREAKDOWN")
print("="*95)
print()

# By error magnitude
Delta_Z_dist = Counter(d['Delta_Z'] for d in failures)

print("Error distribution:")
print(f"{'ΔZ':<8} {'Count':<10} {'%'}")
print("-"*95)
for dz in sorted(Delta_Z_dist.keys()):
    count = Delta_Z_dist[dz]
    pct = 100 * count / len(failures)
    print(f"{dz:+d}       {count:<10} {pct:.1f}%")

print()

# By mass region
regions = [
    ("Light (A<40)", lambda d: d['A'] < 40),
    ("Medium (40≤A<100)", lambda d: 40 <= d['A'] < 100),
    ("Heavy (100≤A<200)", lambda d: 100 <= d['A'] < 200),
    ("Superheavy (A≥200)", lambda d: d['A'] >= 200),
]

print("By mass region:")
print(f"{'Region':<25} {'Failures':<20} {'Fail Rate'}")
print("-"*95)

for name, condition in regions:
    in_region = [d for d in data if condition(d)]
    fail_in_region = [d for d in failures if condition(d)]

    if len(in_region) > 0:
        fail_rate = 100 * len(fail_in_region) / len(in_region)
        print(f"{name:<25} {len(fail_in_region)}/{len(in_region):<17} {fail_rate:.1f}%")

print()

# By pairing
pairing_cats = [
    ("Even-Even", lambda d: d['Z_even'] and d['N_even']),
    ("Even-Odd", lambda d: d['Z_even'] and not d['N_even']),
    ("Odd-Even", lambda d: not d['Z_even'] and d['N_even']),
    ("Odd-Odd", lambda d: not d['Z_even'] and not d['N_even']),
]

print("By pairing (after pairing correction):")
print(f"{'Type':<15} {'Failures':<20} {'Fail Rate'}")
print("-"*95)

for name, condition in pairing_cats:
    in_cat = [d for d in data if condition(d)]
    fail_in_cat = [d for d in failures if condition(d)]

    if len(in_cat) > 0:
        fail_rate = 100 * len(fail_in_cat) / len(in_cat)
        print(f"{name:<15} {len(fail_in_cat)}/{len(in_cat):<17} {fail_rate:.1f}%")

print()

# ============================================================================
# SPECIFIC FAILURE CASES
# ============================================================================
print("="*95)
print("SPECIFIC FAILURE CASES")
print("="*95)
print()

print("All remaining failures:")
print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<7} {'Z_pred':<7} {'ΔZ':<6} {'N/Z':<8} {'Type':<12} {'Notes'}")
print("-"*95)

for d in sorted(failures, key=lambda x: abs(x['Delta_Z']), reverse=True):
    # Pairing type
    if d['Z_even'] and d['N_even']:
        ptype = "E-E"
    elif d['Z_even']:
        ptype = "E-O"
    elif d['N_even']:
        ptype = "O-E"
    else:
        ptype = "O-O"

    notes = []
    if d['Z_magic']: notes.append("Z_mag")
    if d['N_magic']: notes.append("N_mag")
    if NZ_LOW <= d['NZ_ratio'] <= NZ_HIGH: notes.append("res")

    print(f"{d['name']:<12} {d['A']:<6} {d['Z_exp']:<7} {d['Z_pred']:<7} "
          f"{d['Delta_Z']:<6} {d['NZ_ratio']:<8.2f} {ptype:<12} {', '.join(notes)}")

print()

# ============================================================================
# PATTERN SEARCH
# ============================================================================
print("="*95)
print("SEARCHING FOR REMAINING PATTERNS")
print("="*95)
print()

# Check if failures cluster in specific A ranges
A_values = [d['A'] for d in failures]
A_bins = {}
for A in A_values:
    # Bin by A mod 4
    key = A % 4
    A_bins[key] = A_bins.get(key, 0) + 1

print("A mod 4 distribution of failures:")
for key in sorted(A_bins.keys()):
    print(f"  A mod 4 = {key}: {A_bins[key]} failures")

print()

# Check Z mod patterns
Z_values = [d['Z_exp'] for d in failures]
Z_bins = {}
for Z in Z_values:
    key = Z % 4
    Z_bins[key] = Z_bins.get(key, 0) + 1

print("Z mod 4 distribution of failures:")
for key in sorted(Z_bins.keys()):
    print(f"  Z mod 4 = {key}: {Z_bins[key]} failures")

print()

# Check N/Z ratio distribution of failures
nz_ratios = [d['NZ_ratio'] for d in failures]

print("N/Z ratio distribution of failures:")
print(f"  Min: {min(nz_ratios):.2f}")
print(f"  Max: {max(nz_ratios):.2f}")
print(f"  Mean: {np.mean(nz_ratios):.2f}")
print(f"  In resonance [1.15, 1.30]: {sum(1 for r in nz_ratios if NZ_LOW <= r <= NZ_HIGH)} / {len(nz_ratios)}")

print()

# ============================================================================
# ENERGY LANDSCAPE ANALYSIS
# ============================================================================
print("="*95)
print("ENERGY LANDSCAPE ANALYSIS (Sample Failures)")
print("="*95)
print()

# For a few failures, show energy curve
sample_failures = [d for d in failures if abs(d['Delta_Z']) >= 2][:5]

for d in sample_failures:
    A = d['A']
    Z_exp = d['Z_exp']
    Z_pred = d['Z_pred']

    print(f"{d['name']} (A={A}): Z_exp={Z_exp}, Z_pred={Z_pred}")

    # Show energies around Z_exp
    Z_range = range(max(1, Z_exp-3), min(A, Z_exp+4))

    for Z in Z_range:
        E = qfd_energy(A, Z)
        marker = "←" if Z == Z_pred else ("*" if Z == Z_exp else "")
        print(f"  Z={Z}: E={E:.3f} MeV {marker}")

    # Energy gap
    E_exp = qfd_energy(A, Z_exp)
    E_pred = qfd_energy(A, Z_pred)
    gap = E_exp - E_pred

    print(f"  ΔE = {gap:.3f} MeV (energy to reach experimental)")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: What's Missing for Final 37.5%")
print("="*95)
print()

print("The 107 remaining failures likely require:")
print()

# Check if any mass region dominates
heavy_fail_count = sum(1 for d in failures if d['A'] >= 100)
if heavy_fail_count > len(failures) * 0.6:
    print("1. ★ HEAVY NUCLEUS CORRECTIONS (A≥100 failures dominate)")
    print("   - Shell effects beyond magic numbers")
    print("   - Deformation effects")
    print("   - Different energy scaling for heavy systems")
    print()

# Check remaining even-even
ee_fail = sum(1 for d in failures if d['Z_even'] and d['N_even'])
ee_total = sum(1 for d in data if d['Z_even'] and d['N_even'])
ee_fail_rate = 100 * ee_fail / ee_total

if ee_fail_rate > 45:
    print("2. ★ ADDITIONAL PAIRING STRUCTURE")
    print(f"   - Even-even still {ee_fail_rate:.1f}% fail rate")
    print("   - May need mass-dependent pairing or seniority coupling")
    print()

# Check if errors are small
small_errors = sum(1 for d in failures if abs(d['Delta_Z']) == 1)
if small_errors > len(failures) * 0.6:
    print("3. ★ FINE-STRUCTURE CORRECTIONS (most |ΔZ|=1)")
    print("   - Small systematic shifts needed")
    print("   - Possibly asymmetry coefficient adjustment")
    print("   - Or additional N/Z resonance windows")
    print()

print("="*95)
