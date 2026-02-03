#!/usr/bin/env python3
"""
IDENTIFY THE EXACT 31 NUCLEI THAT EMPIRICAL BONUSES FIX
===========================================================================
We know:
  Pure QFD (no bonuses): 175/285 correct (110 failures)
  Empirical bonuses: 206/285 correct (79 failures)
  Gap: 31 nuclei that empirical fixes but pure fails

Question: WHICH specific 31 nuclei do empirical bonuses fix?

Strategy:
1. Run pure QFD (no bonuses) - get 175 correct
2. Run empirical QFD (magic, symm, nr) - get 206 correct
3. Find the 31 nuclei that are:
   - WRONG in pure QFD
   - CORRECT in empirical QFD

Then analyze THOSE 31 for patterns:
  - Are they magic nuclei?
  - Are they symmetric (N≈Z)?
  - Are they neutron-rich?
  - What's their A mod 4 distribution?
  - What's their deformation region?
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
SUBSHELL_NODES = {6, 14, 16, 32, 34, 38, 40, 56, 64, 70}

def qfd_energy_pure(A, Z):
    """Pure QFD - no bonuses."""
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

def qfd_energy_empirical(A, Z, magic, symm, nr, subshell):
    """Empirical QFD with bonuses."""
    N = A - Z
    E_base = qfd_energy_pure(A, Z)

    # Magic number bonuses
    E_magic = 0
    if Z in ISOMER_NODES:
        E_magic -= magic
    if N in ISOMER_NODES:
        E_magic -= magic

    # Symmetric bonus (N/Z ≈ 1)
    E_symm = 0
    nz_ratio = N / Z if Z > 0 else 0
    if 0.95 <= nz_ratio <= 1.15:
        E_symm -= symm

    # Neutron-rich bonus
    E_nr = 0
    if 1.15 <= nz_ratio <= 1.30:
        E_nr -= nr

    # Subshell bonus
    E_subshell = 0
    if Z in SUBSHELL_NODES:
        E_subshell -= subshell
    if N in SUBSHELL_NODES:
        E_subshell -= subshell

    return E_base + E_magic + E_symm + E_nr + E_subshell

def find_stable_Z_pure(A):
    """Find Z with pure QFD."""
    best_Z, best_E = 1, qfd_energy_pure(A, 1)
    for Z in range(1, A):
        E = qfd_energy_pure(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

def find_stable_Z_empirical(A, magic, symm, nr, subshell):
    """Find Z with empirical bonuses."""
    best_Z, best_E = 1, qfd_energy_empirical(A, 1, magic, symm, nr, subshell)
    for Z in range(1, A):
        E = qfd_energy_empirical(A, Z, magic, symm, nr, subshell)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

def distance_to_magic(n):
    """Distance to nearest magic number."""
    if not ISOMER_NODES:
        return 999
    return min(abs(n - magic) for magic in ISOMER_NODES)

def is_deformed_region(Z, N, A):
    """Known deformed regions."""
    if 60 <= Z <= 70 and 150 <= A <= 190:
        return "Rare earth"
    if Z >= 90 and 230 <= A <= 250:
        return "Actinide"
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

# Get empirical bonus values (from earlier optimization)
# These are the values that gave 206/285 correct
MAGIC_BONUS = 0.70
SYMM_BONUS = 0.40
NR_BONUS = 0.05
SUBSHELL_BONUS = 0.00

print("="*95)
print("IDENTIFY THE 31 NUCLEI THAT EMPIRICAL BONUSES FIX")
print("="*95)
print()

# Classify all nuclei
nuclei_data = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp

    Z_pred_pure = find_stable_Z_pure(A)
    Z_pred_empirical = find_stable_Z_empirical(A, MAGIC_BONUS, SYMM_BONUS,
                                                 NR_BONUS, SUBSHELL_BONUS)

    correct_pure = (Z_pred_pure == Z_exp)
    correct_empirical = (Z_pred_empirical == Z_exp)

    # Determine if empirical bonuses fix this nucleus
    empirical_fixes = (not correct_pure) and correct_empirical

    # Classify nucleus properties
    is_magic_Z = Z_exp in ISOMER_NODES
    is_magic_N = N_exp in ISOMER_NODES
    is_doubly_magic = is_magic_Z and is_magic_N

    nz_ratio = N_exp / Z_exp if Z_exp > 0 else 0
    is_symmetric = 0.95 <= nz_ratio <= 1.15
    is_neutron_rich = 1.15 <= nz_ratio <= 1.30

    is_subshell_Z = Z_exp in SUBSHELL_NODES
    is_subshell_N = N_exp in SUBSHELL_NODES

    nuclei_data.append({
        'name': name,
        'A': A,
        'Z': Z_exp,
        'N': N_exp,
        'correct_pure': correct_pure,
        'correct_empirical': correct_empirical,
        'empirical_fixes': empirical_fixes,
        'is_magic_Z': is_magic_Z,
        'is_magic_N': is_magic_N,
        'is_doubly_magic': is_doubly_magic,
        'is_symmetric': is_symmetric,
        'is_neutron_rich': is_neutron_rich,
        'is_subshell_Z': is_subshell_Z,
        'is_subshell_N': is_subshell_N,
        'deform_region': is_deformed_region(Z_exp, N_exp, A),
        'mod_4': A % 4,
        'mod_28': A % 28,
        'spin_type': 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0)
                     else 'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1)
                     else 'odd-A',
    })

# Count results
pure_correct = sum(1 for n in nuclei_data if n['correct_pure'])
empirical_correct = sum(1 for n in nuclei_data if n['correct_empirical'])
fixed_by_empirical = [n for n in nuclei_data if n['empirical_fixes']]

print(f"Pure QFD:        {pure_correct}/285 correct ({100*pure_correct/285:.1f}%)")
print(f"Empirical QFD:   {empirical_correct}/285 correct ({100*empirical_correct/285:.1f}%)")
print(f"Fixed by empirical: {len(fixed_by_empirical)} nuclei")
print()

# ============================================================================
# ANALYZE THE 31 FIXED NUCLEI
# ============================================================================
print("="*95)
print(f"THE {len(fixed_by_empirical)} NUCLEI THAT EMPIRICAL BONUSES FIX")
print("="*95)
print()

print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N':<6} {'Magic':<12} {'Symm':<8} {'NR':<8} "
      f"{'Subshell':<12} {'A mod 4'}  {'Region'}")
print("-"*95)

for n in sorted(fixed_by_empirical, key=lambda x: x['A']):
    magic_str = ""
    if n['is_doubly_magic']:
        magic_str = "Z+N ★★"
    elif n['is_magic_Z']:
        magic_str = "Z ★"
    elif n['is_magic_N']:
        magic_str = "N ★"

    symm_str = "Yes" if n['is_symmetric'] else ""
    nr_str = "Yes" if n['is_neutron_rich'] else ""

    subshell_str = ""
    if n['is_subshell_Z'] and n['is_subshell_N']:
        subshell_str = "Z+N"
    elif n['is_subshell_Z']:
        subshell_str = "Z"
    elif n['is_subshell_N']:
        subshell_str = "N"

    region_short = n['deform_region'][:12]

    print(f"{n['name']:<12} {n['A']:<6} {n['Z']:<6} {n['N']:<6} {magic_str:<12} "
          f"{symm_str:<8} {nr_str:<8} {subshell_str:<12} {n['mod_4']:<8} {region_short}")

print()

# ============================================================================
# STATISTICS ON THE 31 FIXED NUCLEI
# ============================================================================
print("="*95)
print("WHAT DO THE FIXED NUCLEI HAVE IN COMMON?")
print("="*95)
print()

# Count by bonus type
magic_Z_count = sum(1 for n in fixed_by_empirical if n['is_magic_Z'])
magic_N_count = sum(1 for n in fixed_by_empirical if n['is_magic_N'])
doubly_magic_count = sum(1 for n in fixed_by_empirical if n['is_doubly_magic'])
symmetric_count = sum(1 for n in fixed_by_empirical if n['is_symmetric'])
nr_count = sum(1 for n in fixed_by_empirical if n['is_neutron_rich'])
subshell_Z_count = sum(1 for n in fixed_by_empirical if n['is_subshell_Z'])
subshell_N_count = sum(1 for n in fixed_by_empirical if n['is_subshell_N'])

print(f"Bonus type breakdown (nuclei may have multiple bonuses):")
print(f"  Magic Z:          {magic_Z_count}/{len(fixed_by_empirical)} ({100*magic_Z_count/len(fixed_by_empirical):.1f}%)")
print(f"  Magic N:          {magic_N_count}/{len(fixed_by_empirical)} ({100*magic_N_count/len(fixed_by_empirical):.1f}%)")
print(f"  Doubly magic:     {doubly_magic_count}/{len(fixed_by_empirical)} ({100*doubly_magic_count/len(fixed_by_empirical):.1f}%)")
print(f"  Symmetric (N≈Z):  {symmetric_count}/{len(fixed_by_empirical)} ({100*symmetric_count/len(fixed_by_empirical):.1f}%)")
print(f"  Neutron-rich:     {nr_count}/{len(fixed_by_empirical)} ({100*nr_count/len(fixed_by_empirical):.1f}%)")
print(f"  Subshell Z:       {subshell_Z_count}/{len(fixed_by_empirical)} ({100*subshell_Z_count/len(fixed_by_empirical):.1f}%)")
print(f"  Subshell N:       {subshell_N_count}/{len(fixed_by_empirical)} ({100*subshell_N_count/len(fixed_by_empirical):.1f}%)")
print()

# Count nuclei with ONLY magic bonus (no symm, no nr, no subshell)
only_magic = [n for n in fixed_by_empirical
              if (n['is_magic_Z'] or n['is_magic_N'])
              and not n['is_symmetric']
              and not n['is_neutron_rich']
              and not n['is_subshell_Z']
              and not n['is_subshell_N']]

print(f"Nuclei fixed by ONLY magic bonus: {len(only_magic)}/{len(fixed_by_empirical)}")
if only_magic:
    print("  Examples:", ', '.join(n['name'] for n in only_magic[:5]))
print()

# Count by A mod 4
mod4_fixed = Counter(n['mod_4'] for n in fixed_by_empirical)
mod4_all_failures = Counter(n['mod_4'] for n in nuclei_data if not n['correct_pure'])

print("A mod 4 distribution of fixed nuclei:")
print(f"{'A mod 4':<12} {'Fixed':<12} {'Total failures':<18} {'Fix rate %'}")
print("-"*95)

for mod4 in range(4):
    fixed = mod4_fixed.get(mod4, 0)
    total_fail = mod4_all_failures.get(mod4, 0)
    rate = 100 * fixed / total_fail if total_fail > 0 else 0

    print(f"{mod4:<12} {fixed:<12} {total_fail:<18} {rate:.1f}")

print()

# Count by deformation region
region_fixed = Counter(n['deform_region'] for n in fixed_by_empirical)
region_all_failures = Counter(n['deform_region'] for n in nuclei_data if not n['correct_pure'])

print("Deformation region distribution of fixed nuclei:")
print(f"{'Region':<20} {'Fixed':<12} {'Total failures':<18} {'Fix rate %'}")
print("-"*95)

for region in ['Spherical', 'Midshell', 'Rare earth', 'Actinide']:
    fixed = region_fixed.get(region, 0)
    total_fail = region_all_failures.get(region, 0)
    rate = 100 * fixed / total_fail if total_fail > 0 else 0

    print(f"{region:<20} {fixed:<12} {total_fail:<18} {rate:.1f}")

print()

# Count by spin type
spin_fixed = Counter(n['spin_type'] for n in fixed_by_empirical)
spin_all_failures = Counter(n['spin_type'] for n in nuclei_data if not n['correct_pure'])

print("Spin type distribution of fixed nuclei:")
print(f"{'Spin type':<20} {'Fixed':<12} {'Total failures':<18} {'Fix rate %'}")
print("-"*95)

for spin_type in ['even-even', 'odd-A', 'odd-odd']:
    fixed = spin_fixed.get(spin_type, 0)
    total_fail = spin_all_failures.get(spin_type, 0)
    rate = 100 * fixed / total_fail if total_fail > 0 else 0

    print(f"{spin_type:<20} {fixed:<12} {total_fail:<18} {rate:.1f}")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: WHAT EMPIRICAL BONUSES ACTUALLY FIX")
print("="*95)
print()

print(f"Total nuclei fixed: {len(fixed_by_empirical)}")
print()

# Find dominant pattern
dominant_bonus = max([
    ("Magic (Z or N)", magic_Z_count + magic_N_count),
    ("Symmetric", symmetric_count),
    ("Neutron-rich", nr_count),
    ("Subshell", subshell_Z_count + subshell_N_count),
], key=lambda x: x[1])

print(f"Dominant pattern: {dominant_bonus[0]} ({dominant_bonus[1]}/{len(fixed_by_empirical)} nuclei, {100*dominant_bonus[1]/len(fixed_by_empirical):.1f}%)")
print()

# Check if they're fixing specific geometric classes
mod4_bias = max(mod4_fixed.values()) / len(fixed_by_empirical) if fixed_by_empirical else 0
if mod4_bias > 0.4:
    biased_mod4 = max(mod4_fixed.keys(), key=lambda k: mod4_fixed[k])
    print(f"Geometric bias: {100*mod4_bias:.1f}% of fixes are A mod 4 = {biased_mod4}")
else:
    print("No strong geometric bias (evenly distributed across A mod 4)")

print()

print("Interpretation:")
if dominant_bonus[1] / len(fixed_by_empirical) > 0.7:
    print(f"  ★ Empirical bonuses are primarily fixing {dominant_bonus[0]} nuclei")
    print("  → These bonuses capture REAL physics beyond pure QFD geometry")
else:
    print("  ★ Empirical bonuses fix a MIXED BAG of different nuclear types")
    print("  → No single physical mechanism dominates")

print()
print("="*95)
