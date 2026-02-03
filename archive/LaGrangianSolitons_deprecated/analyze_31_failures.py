#!/usr/bin/env python3
"""
ANALYZE THE 31 FAILURES - DEFORMATION PATTERNS
===========================================================================
Pure QFD: 175/285 correct
Empirical: 206/285 correct
Gap: 31 matches

Question: What do the 31 nuclei that empirical bonuses fix have in common?

Look for:
1. Deformation patterns (prolate/oblate regions)
2. Distance from magic numbers
3. Known deformed regions:
   - Rare earths (A~150-190, 60≤Z≤70)
   - Actinides (A~230-250, Z≥90)
   - Transitional nuclei (between shells)
4. Collective rotation/vibration signatures
5. Even-even vs odd-A vs odd-odd
6. A mod patterns
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict

# Constants (no lambda!)
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def qfd_energy_pure(A, Z):
    """Pure QFD - no bonuses, no lambda."""
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = KAPPA_E * Z

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

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def find_stable_Z_pure(A):
    """Find Z with pure QFD."""
    best_Z, best_E = 1, qfd_energy_pure(A, 1)
    for Z in range(1, A):
        E = qfd_energy_pure(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

def distance_to_magic(n):
    """Distance to nearest magic number."""
    if not ISOMER_NODES:
        return 999
    return min(abs(n - magic) for magic in ISOMER_NODES)

def is_deformed_region(Z, N, A):
    """Known deformed regions in nuclear chart."""
    # Rare earth region (Z~60-70, A~150-190)
    if 60 <= Z <= 70 and 150 <= A <= 190:
        return "Rare earth"

    # Actinide region (Z≥90, A~230-250)
    if Z >= 90 and 230 <= A <= 250:
        return "Actinide"

    # Transitional nuclei (midshell, far from closures)
    Z_dist = distance_to_magic(Z)
    N_dist = distance_to_magic(N)

    if Z_dist > 6 and N_dist > 6:
        return "Midshell"

    return "Spherical"

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("ANALYZE THE 31 FAILURES - WHAT EMPIRICAL BONUSES FIX")
print("="*95)
print()

# Classify all nuclei
nuclei_data = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred_pure = find_stable_Z_pure(A)
    correct_pure = (Z_pred_pure == Z_exp)

    # Determine deformation region
    deform_region = is_deformed_region(Z_exp, N_exp, A)

    nuclei_data.append({
        'name': name,
        'A': A,
        'Z': Z_exp,
        'N': N_exp,
        'Z_pred_pure': Z_pred_pure,
        'correct_pure': correct_pure,
        'deform_region': deform_region,
        'Z_dist_magic': distance_to_magic(Z_exp),
        'N_dist_magic': distance_to_magic(N_exp),
        'mod_4': A % 4,
        'mod_28': A % 28,
    })

# Identify the 31 that empirical fixes (correct with empirical, wrong with pure)
# We know empirical gets 206 correct, pure gets 175 correct
# So there are 31 that empirical gets right but pure gets wrong

pure_failures = [n for n in nuclei_data if not n['correct_pure']]
print(f"Pure QFD failures: {len(pure_failures)}/285")
print()

# We don't have empirical predictions directly, but we know there are 31 that
# empirical fixes. Let's analyze ALL pure failures to find patterns, then
# identify which are likely the "fixable" ones

print("="*95)
print("PURE QFD FAILURES BY DEFORMATION REGION")
print("="*95)
print()

failures_by_region = defaultdict(list)
for n in pure_failures:
    failures_by_region[n['deform_region']].append(n)

print(f"{'Region':<20} {'Failures':<12} {'Total in region':<18} {'Failure Rate %'}")
print("-"*95)

for region in ['Spherical', 'Midshell', 'Rare earth', 'Actinide']:
    region_failures = failures_by_region.get(region, [])
    region_total = sum(1 for n in nuclei_data if n['deform_region'] == region)

    if region_total > 0:
        fail_rate = 100 * len(region_failures) / region_total
        marker = "★" if fail_rate > 50 else ""

        print(f"{region:<20} {len(region_failures):<12} {region_total:<18} {fail_rate:.1f}  {marker}")

print()

# ============================================================================
# DISTANCE FROM MAGIC NUMBERS
# ============================================================================
print("="*95)
print("FAILURES BY DISTANCE FROM MAGIC NUMBERS")
print("="*95)
print()

print("Z distance from magic:")
Z_dist_failures = Counter(n['Z_dist_magic'] for n in pure_failures)
Z_dist_all = Counter(n['Z_dist_magic'] for n in nuclei_data)

print(f"{'Z dist':<12} {'Failures':<12} {'Total':<12} {'Failure Rate %'}")
print("-"*95)

for dist in sorted(set(Z_dist_all.keys())):
    failures = Z_dist_failures.get(dist, 0)
    total = Z_dist_all.get(dist, 0)
    rate = 100 * failures / total if total > 0 else 0
    marker = "★" if rate > 50 else ("•" if dist == 0 else "")

    print(f"{dist:<12} {failures:<12} {total:<12} {rate:.1f}  {marker}")

print()

print("N distance from magic:")
N_dist_failures = Counter(n['N_dist_magic'] for n in pure_failures)
N_dist_all = Counter(n['N_dist_magic'] for n in nuclei_data)

print(f"{'N dist':<12} {'Failures':<12} {'Total':<12} {'Failure Rate %'}")
print("-"*95)

for dist in sorted(set(N_dist_all.keys())):
    failures = N_dist_failures.get(dist, 0)
    total = N_dist_all.get(dist, 0)
    rate = 100 * failures / total if total > 0 else 0
    marker = "★" if rate > 50 else ("•" if dist == 0 else "")

    print(f"{dist:<12} {failures:<12} {total:<12} {rate:.1f}  {marker}")

print()

# ============================================================================
# SPECIFIC FAILURE REGIONS
# ============================================================================
print("="*95)
print("SPECIFIC FAILURE ANALYSIS")
print("="*95)
print()

print("Pure QFD failures (showing first 50):")
print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'Z_pred':<8} {'Region':<15} "
      f"{'Z_dist':<8} {'N_dist':<8} {'A mod 4'}")
print("-"*95)

for i, n in enumerate(pure_failures[:50]):
    region_short = n['deform_region'][:12]
    print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {n['Z_pred_pure']:<8} "
          f"{region_short:<15} {n['Z_dist_magic']:<8} {n['N_dist_magic']:<8} {n['mod_4']}")

if len(pure_failures) > 50:
    print(f"\n... and {len(pure_failures) - 50} more")

print()

# ============================================================================
# A-VALUE DISTRIBUTION
# ============================================================================
print("="*95)
print("FAILURES BY MASS NUMBER (A)")
print("="*95)
print()

# Group by A ranges
A_ranges = [
    (0, 40, "Light (A<40)"),
    (40, 100, "Medium (40≤A<100)"),
    (100, 150, "Heavy (100≤A<150)"),
    (150, 200, "Rare earth region (150≤A<200)"),
    (200, 300, "Very heavy (A≥200)"),
]

print(f"{'A Range':<30} {'Failures':<12} {'Total':<12} {'Failure Rate %'}")
print("-"*95)

for A_min, A_max, label in A_ranges:
    range_failures = [n for n in pure_failures if A_min <= n['A'] < A_max]
    range_total = [n for n in nuclei_data if A_min <= n['A'] < A_max]

    if range_total:
        fail_rate = 100 * len(range_failures) / len(range_total)
        marker = "★" if fail_rate > 50 else ""

        print(f"{label:<30} {len(range_failures):<12} {len(range_total):<12} {fail_rate:.1f}  {marker}")

print()

# ============================================================================
# EVEN-EVEN vs ODD-A vs ODD-ODD
# ============================================================================
print("="*95)
print("FAILURES BY NUCLEON PARITY")
print("="*95)
print()

parity_failures = defaultdict(list)
parity_all = defaultdict(list)

for n in nuclei_data:
    Z_par = "even" if n['Z'] % 2 == 0 else "odd"
    N_par = "even" if n['N'] % 2 == 0 else "odd"
    parity = f"Z-{Z_par}, N-{N_par}"

    parity_all[parity].append(n)
    if not n['correct_pure']:
        parity_failures[parity].append(n)

print(f"{'Parity':<20} {'Failures':<12} {'Total':<12} {'Failure Rate %'}")
print("-"*95)

for parity in sorted(parity_all.keys()):
    failures = len(parity_failures.get(parity, []))
    total = len(parity_all[parity])
    rate = 100 * failures / total if total > 0 else 0
    marker = "★" if rate > 50 else ""

    print(f"{parity:<20} {failures:<12} {total:<12} {rate:.1f}  {marker}")

print()

# ============================================================================
# GEOMETRIC PATTERNS IN FAILURES
# ============================================================================
print("="*95)
print("GEOMETRIC PATTERNS IN FAILURES")
print("="*95)
print()

print("A mod 4 distribution:")
mod4_failures = Counter(n['mod_4'] for n in pure_failures)
mod4_all = Counter(n['mod_4'] for n in nuclei_data)

print(f"{'A mod 4':<12} {'Failures':<12} {'Total':<12} {'Failure Rate %':<15} {'vs avg'}")
print("-"*95)

overall_fail_rate = 100 * len(pure_failures) / len(nuclei_data)

for mod4 in range(4):
    failures = mod4_failures.get(mod4, 0)
    total = mod4_all.get(mod4, 0)
    rate = 100 * failures / total if total > 0 else 0
    delta = rate - overall_fail_rate
    marker = "★" if abs(delta) > 10 else ""

    print(f"{mod4:<12} {failures:<12} {total:<12} {rate:<15.1f} {delta:+.1f}  {marker}")

print()

# ============================================================================
# KNOWN DEFORMED NUCLEI (examples)
# ============================================================================
print("="*95)
print("KNOWN DEFORMED REGIONS - SAMPLE FAILURES")
print("="*95)
print()

print("Rare earth region failures (60≤Z≤70, 150≤A≤190):")
rare_earth_failures = [n for n in pure_failures
                       if 60 <= n['Z'] <= 70 and 150 <= n['A'] <= 190]

if rare_earth_failures:
    print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'Z_pred':<8}")
    print("-"*60)
    for n in rare_earth_failures[:15]:
        print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {n['Z_pred_pure']:<8}")

    if len(rare_earth_failures) > 15:
        print(f"... and {len(rare_earth_failures) - 15} more")
else:
    print("No failures in rare earth region")

print()

print("Actinide region failures (Z≥90, A~230-250):")
actinide_failures = [n for n in pure_failures
                     if n['Z'] >= 90 and 230 <= n['A'] <= 250]

if actinide_failures:
    print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'Z_pred'}<8")
    print("-"*60)
    for n in actinide_failures:
        print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {n['Z_pred_pure']:<8}")
else:
    print("No failures in actinide region (likely not in dataset)")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: WHAT CAUSES THE 110 FAILURES?")
print("="*95)
print()

print(f"Total pure QFD failures: {len(pure_failures)}/285 ({100*len(pure_failures)/285:.1f}%)")
print()

print("Key patterns in failures:")
print()

# Find strongest patterns
max_fail_rate_region = max(
    [(region, 100*len(failures_by_region.get(region, []))/sum(1 for n in nuclei_data if n['deform_region'] == region))
     for region in ['Spherical', 'Midshell', 'Rare earth', 'Actinide']
     if sum(1 for n in nuclei_data if n['deform_region'] == region) > 0],
    key=lambda x: x[1]
)

print(f"1. DEFORMATION:")
print(f"   Highest failure rate: {max_fail_rate_region[0]} region ({max_fail_rate_region[1]:.1f}%)")

# Parity with highest failure rate
max_fail_parity = max(
    [(par, 100*len(parity_failures.get(par, []))/len(parity_all[par]))
     for par in parity_all.keys()],
    key=lambda x: x[1]
)

print(f"\n2. NUCLEON PARITY:")
print(f"   Highest failure rate: {max_fail_parity[0]} ({max_fail_parity[1]:.1f}%)")

# A mod 4 with highest failure rate
max_fail_mod4 = max(
    [(mod4, 100*mod4_failures.get(mod4, 0)/mod4_all.get(mod4, 1))
     for mod4 in range(4)],
    key=lambda x: x[1]
)

print(f"\n3. GEOMETRIC PATTERN:")
print(f"   Highest failure rate: A mod 4 = {max_fail_mod4[0]} ({max_fail_mod4[1]:.1f}%)")

print()
print("The 31-match gap (empirical bonuses fix) likely includes:")
print("  • Magic number bonuses (Z,N at 2,8,20,28,50,82,126)")
print("  • Symmetric bonuses (N/Z ∈ [0.95, 1.15])")
print("  • Neutron-rich bonuses (N/Z ∈ [1.15, 1.30])")
print()
print("Remaining failures after empirical bonuses (79 total) likely need:")
print("  • Deformation corrections (prolate/oblate shapes)")
print("  • Collective rotation/vibration")
print("  • Higher-order geometric terms")
print()

print("="*95)
