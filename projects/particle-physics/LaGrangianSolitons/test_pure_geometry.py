#!/usr/bin/env python3
"""
PURE QFD GEOMETRY - NO EMPIRICAL CORRECTIONS
===========================================================================
User insight: "there are no shells or binding energies"

Strip away ALL empirical bonuses:
- magic = 0 (no shell closure bonuses)
- symm = 0 (no symmetric bonuses)
- nr = 0 (no neutron-rich bonuses)
- subshell = 0 (no subshell bonuses)

Keep ONLY pure QFD geometry:
- E_bulk (volume energy from vacuum stiffness β)
- E_surf (surface energy from field topology)
- E_asym (asymmetry energy from isospin)
- E_vac (Coulomb/vacuum polarization)
- E_pair (pairing energy from fermion statistics)

HYPOTHESIS: Geometric patterns (multiples of 7, mod 12 = 9, etc.)
will organize the FAILURES, revealing the true topological structure
that empirical "bonuses" were hiding!
===========================================================================
"""

import numpy as np
from collections import defaultdict, Counter

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

# Crossover reclassifications (keep for comparison)
CROSSOVER_RECLASSIFICATIONS = {
    ('Kr-84', 36, 84): 'Type_I',
    ('Rb-87', 37, 87): 'Type_II',
    ('Mo-94', 42, 94): 'Type_V',
    ('Ru-104', 44, 104): 'Type_I',
    ('Cd-114', 48, 114): 'Type_I',
    ('In-115', 49, 115): 'Type_I',
    ('Sn-122', 50, 122): 'Type_II',
    ('Ba-138', 56, 138): 'Type_II',
    ('La-139', 57, 139): 'Type_II',
}

def qfd_energy_pure(A, Z):
    """Pure QFD energy - NO bonuses, only fundamental geometry."""
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

    # Pairing energy (fundamental fermion statistics)
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def find_stable_Z_pure(A):
    """Find Z that minimizes energy - pure geometry."""
    best_Z, best_E = 1, qfd_energy_pure(A, 1)
    for Z in range(1, A):
        E = qfd_energy_pure(A, Z)
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
print("PURE QFD GEOMETRY - NO EMPIRICAL CORRECTIONS")
print("="*95)
print()
print("Testing with ONLY fundamental QFD:")
print("  ✓ E_bulk (vacuum volume energy, β = 3.058)")
print("  ✓ E_surf (topological surface energy)")
print("  ✓ E_asym (isospin asymmetry)")
print("  ✓ E_vac (Coulomb/vacuum polarization)")
print("  ✓ E_pair (fermion pairing)")
print()
print("Removed ALL bonuses:")
print("  ✗ magic = 0 (no shell closures)")
print("  ✗ symm = 0 (no symmetric bonus)")
print("  ✗ nr = 0 (no neutron-rich bonus)")
print("  ✗ subshell = 0 (no subshell bonus)")
print()

# Test pure geometry
nuclei_data = []
correct_count = 0

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred = find_stable_Z_pure(A)
    correct = (Z_pred == Z_exp)

    if correct:
        correct_count += 1

    nuclei_data.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'correct': correct,
        'is_magic_Z': Z_exp in ISOMER_NODES,
        'is_magic_N': N_exp in ISOMER_NODES,
        'is_crossover': (name, Z_exp, A) in CROSSOVER_RECLASSIFICATIONS,
    })

success_rate = 100 * correct_count / len(test_nuclides)

print(f"PURE GEOMETRY RESULT: {correct_count}/285 ({success_rate:.1f}%)")
print()
print(f"Comparison:")
print(f"  With bonuses (72.3%): 206/285")
print(f"  Pure geometry:        {correct_count}/285 ({success_rate:.1f}%)")
print(f"  Difference:           {correct_count - 206:+d} matches")
print()

if correct_count > 206:
    print("★★★ PURE GEOMETRY IS BETTER! Bonuses were hiding true structure! ★★★")
elif correct_count > 180:
    print("★ Pure geometry is competitive! Bonuses may not be needed.")
else:
    print("Pure geometry alone is insufficient. BUT check if failures show geometric patterns...")

print()

# ============================================================================
# ANALYZE FAILURES FOR GEOMETRIC PATTERNS
# ============================================================================
print("="*95)
print("GEOMETRIC ANALYSIS OF FAILURES")
print("="*95)
print()

failures = [n for n in nuclei_data if not n['correct']]
successes = [n for n in nuclei_data if n['correct']]

print(f"Failures: {len(failures)}")
print(f"Successes: {len(successes)}")
print()

# Check A mod 12 pattern (was strongest in earlier search)
print("PATTERN 1: A mod 12 distribution")
print(f"{'A mod 12':<12} {'Total':<10} {'Success':<10} {'Fail':<10} {'Success %':<12} {'Δ from avg'}")
print("-"*95)

A_mod12_all = Counter(n['A'] % 12 for n in nuclei_data)
A_mod12_success = Counter(n['A'] % 12 for n in successes)
A_mod12_fail = Counter(n['A'] % 12 for n in failures)

for mod_val in range(12):
    total = A_mod12_all.get(mod_val, 0)
    succ = A_mod12_success.get(mod_val, 0)
    fail = A_mod12_fail.get(mod_val, 0)
    rate = 100 * succ / total if total > 0 else 0
    delta = rate - success_rate
    marker = "★" if abs(delta) > 10.0 else ""

    print(f"{mod_val:<12} {total:<10} {succ:<10} {fail:<10} {rate:<12.1f} {delta:+.1f}  {marker}")

print()

# Check A mod 7 pattern
print("PATTERN 2: A mod 7 distribution")
print(f"{'A mod 7':<12} {'Total':<10} {'Success':<10} {'Fail':<10} {'Success %':<12} {'Δ from avg'}")
print("-"*95)

A_mod7_all = Counter(n['A'] % 7 for n in nuclei_data)
A_mod7_success = Counter(n['A'] % 7 for n in successes)
A_mod7_fail = Counter(n['A'] % 7 for n in failures)

for mod_val in range(7):
    total = A_mod7_all.get(mod_val, 0)
    succ = A_mod7_success.get(mod_val, 0)
    fail = A_mod7_fail.get(mod_val, 0)
    rate = 100 * succ / total if total > 0 else 0
    delta = rate - success_rate
    marker = "★" if abs(delta) > 10.0 else ""

    print(f"{mod_val:<12} {total:<10} {succ:<10} {fail:<10} {rate:<12.1f} {delta:+.1f}  {marker}")

print()

# Check magic nuclei
print("PATTERN 3: Magic nuclei in pure geometry")
print()

magic_nuclei = [n for n in nuclei_data if n['is_magic_Z'] or n['is_magic_N']]
magic_success = sum(1 for n in magic_nuclei if n['correct'])
magic_rate = 100 * magic_success / len(magic_nuclei) if magic_nuclei else 0

doubly_magic = [n for n in nuclei_data if n['is_magic_Z'] and n['is_magic_N']]
doubly_success = sum(1 for n in doubly_magic if n['correct'])
doubly_rate = 100 * doubly_success / len(doubly_magic) if doubly_magic else 0

print(f"Magic nuclei (Z or N magic): {magic_success}/{len(magic_nuclei)} ({magic_rate:.1f}%)")
print(f"Doubly magic (Z and N):      {doubly_success}/{len(doubly_magic)} ({doubly_rate:.1f}%)")
print(f"Non-magic nuclei:            {correct_count - magic_success}/{len(nuclei_data) - len(magic_nuclei)}")
print()

if magic_rate > success_rate + 10:
    print("★ Magic nuclei STILL succeed better - geometric resonances are real!")
elif magic_rate < success_rate - 10:
    print("★ Magic nuclei FAIL more - 'shells' are NOT fundamental!")
else:
    print("Magic nuclei show no special behavior - shells might be emergent")

print()

# Check crossovers
print("PATTERN 4: Crossover nuclei (heavy with light cores)")
print()

crossover_nuclei = [n for n in nuclei_data if n['is_crossover']]
crossover_success = sum(1 for n in crossover_nuclei if n['correct'])

print(f"Crossover nuclei: {crossover_success}/{len(crossover_nuclei)}")
if crossover_success == len(crossover_nuclei):
    print("★ ALL crossovers correct! Pure geometry understands multi-progenitor families!")
elif crossover_success == 0:
    print("★ ALL crossovers wrong! Pure geometry doesn't capture family structure.")
else:
    print(f"  Mixed results ({100*crossover_success/len(crossover_nuclei):.1f}%)")

print()

# ============================================================================
# SPECIFIC FAILURES - LOOK FOR PATTERNS
# ============================================================================
print("="*95)
print("SPECIFIC FAILURES - GEOMETRIC STRUCTURE")
print("="*95)
print()

print("First 30 failures (looking for patterns):")
print(f"{'Nuclide':<12} {'A':<8} {'Z_exp':<10} {'Z_pred':<10} {'A mod 7':<12} {'A mod 12':<12} {'Magic?'}")
print("-"*95)

for i, n in enumerate(failures[:30]):
    magic_str = ""
    if n['is_magic_Z'] and n['is_magic_N']:
        magic_str = "Doubly magic"
    elif n['is_magic_Z']:
        magic_str = f"Z={n['Z_exp']} magic"
    elif n['is_magic_N']:
        magic_str = f"N={n['N_exp']} magic"

    print(f"{n['name']:<12} {n['A']:<8} {n['Z_exp']:<10} {n['Z_pred']:<10} "
          f"{n['A'] % 7:<12} {n['A'] % 12:<12} {magic_str}")

print()

# ============================================================================
# SUCCESSES - GEOMETRIC STRUCTURE
# ============================================================================
print("="*95)
print("SAMPLE SUCCESSES - WHAT PURE GEOMETRY GETS RIGHT")
print("="*95)
print()

print("First 30 successes (looking for patterns):")
print(f"{'Nuclide':<12} {'A':<8} {'Z':<10} {'A mod 7':<12} {'A mod 12':<12} {'Magic?'}")
print("-"*95)

for i, n in enumerate(successes[:30]):
    magic_str = ""
    if n['is_magic_Z'] and n['is_magic_N']:
        magic_str = "Doubly magic"
    elif n['is_magic_Z']:
        magic_str = f"Z={n['Z_exp']} magic"
    elif n['is_magic_N']:
        magic_str = f"N={n['N_exp']} magic"

    print(f"{n['name']:<12} {n['A']:<8} {n['Z_exp']:<10} "
          f"{n['A'] % 7:<12} {n['A'] % 12:<12} {magic_str}")

print()

# ============================================================================
# INTEGER RATIO ANALYSIS
# ============================================================================
print("="*95)
print("FAILURE PATTERNS: Z/A RATIOS")
print("="*95)
print()

# Check if failures cluster at specific Z/A ratios
failure_ratios = [n['Z_exp'] / n['A'] for n in failures]
success_ratios = [n['Z_exp'] / n['A'] for n in successes]

print(f"Failure Z/A ratios:")
print(f"  Mean: {np.mean(failure_ratios):.4f}")
print(f"  Std:  {np.std(failure_ratios):.4f}")
print()

print(f"Success Z/A ratios:")
print(f"  Mean: {np.mean(success_ratios):.4f}")
print(f"  Std:  {np.std(success_ratios):.4f}")
print()

# Check if failures are at specific fractions
print("Checking if failures cluster near simple fractions Z/A:")
fractions = [(1, 2), (2, 5), (1, 3), (3, 7), (2, 7), (5, 12)]

for num, den in fractions:
    target = num / den
    near_frac_fail = sum(1 for r in failure_ratios if abs(r - target) < 0.02)
    near_frac_succ = sum(1 for r in success_ratios if abs(r - target) < 0.02)
    total_near = near_frac_fail + near_frac_succ

    if total_near > 0:
        fail_rate = 100 * near_frac_fail / total_near
        print(f"  Z/A ≈ {num}/{den} = {target:.3f}: {near_frac_fail}/{total_near} failures ({fail_rate:.1f}%)")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: PURE QFD GEOMETRY")
print("="*95)
print()

print(f"Pure geometry (no bonuses): {correct_count}/285 ({success_rate:.1f}%)")
print(f"With bonuses (previous):    206/285 (72.3%)")
print(f"Difference:                 {correct_count - 206:+d} matches")
print()

if correct_count >= 200:
    print("✓ Pure geometry is highly successful!")
    print("  → Bonuses may be hiding true geometric structure")
    print("  → Focus on geometric patterns in remaining failures")
elif correct_count >= 150:
    print("◐ Pure geometry captures major physics")
    print("  → Bonuses provide refinements")
    print("  → Look for geometric corrections, not empirical ones")
else:
    print("✗ Pure geometry needs corrections")
    print("  → But check if failures organize by geometric patterns")
    print("  → A mod 7, A mod 12, magic resonances, etc.")

print()
print("Key observations:")

# Find strongest pattern
max_deviation = 0
best_pattern = ""

for mod_val in range(12):
    total = A_mod12_all.get(mod_val, 0)
    if total > 0:
        succ = A_mod12_success.get(mod_val, 0)
        rate = 100 * succ / total
        deviation = abs(rate - success_rate)
        if deviation > max_deviation:
            max_deviation = deviation
            best_pattern = f"A mod 12 = {mod_val} ({rate:.1f}%)"

print(f"  • Strongest pattern: {best_pattern} (Δ = {max_deviation:.1f}%)")
print(f"  • Magic nuclei: {magic_rate:.1f}% success")
print(f"  • Crossovers: {crossover_success}/{len(crossover_nuclei)} correct")
print()

print("="*95)
