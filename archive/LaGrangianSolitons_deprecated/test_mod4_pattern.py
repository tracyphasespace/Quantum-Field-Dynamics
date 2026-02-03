#!/usr/bin/env python3
"""
TEST A MOD 4 = 1 PATTERN
===========================================================================
User insight: A mod 12 = {1,5,9} all satisfy A ≡ 1 (mod 4)

Maybe the fundamental pattern is simpler: A mod 4 = 1

Test:
1. Success rates by A mod 4
2. Compare to A mod 12 pattern strength
3. Check if A mod 4 = 1 is the fundamental resonance
4. Test geometric bonus based on A mod 4

If A mod 4 = 1 is fundamental, it might relate to:
- 4D spacetime structure
- Quaternions (4-dimensional)
- SU(2) × SU(2) ≈ SO(4) symmetry
- Pairing structure (but that's Z,N parity, not A)
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
DELTA_PAIRING = 11.0

def qfd_energy_pure(A, Z):
    """Pure QFD energy - NO bonuses."""
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

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def qfd_energy_mod4(A, Z, bonus_mod4):
    """QFD energy with A mod 4 = 1 bonus."""
    E = qfd_energy_pure(A, Z)

    # Get surface energy for bonus scale
    beta_nuclear = M_proton * beta_vacuum / 2
    E_surface = beta_nuclear / 15

    # Bonus if A mod 4 = 1
    if A % 4 == 1:
        E -= E_surface * bonus_mod4  # Stabilize A ≡ 1 (mod 4)

    return E

def find_stable_Z(A, bonus_mod4=0.0):
    """Find Z that minimizes energy."""
    best_Z, best_E = 1, qfd_energy_mod4(A, 1, bonus_mod4)
    for Z in range(1, A):
        E = qfd_energy_mod4(A, Z, bonus_mod4)
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
print("TEST A MOD 4 = 1 PATTERN")
print("="*95)
print()
print("Hypothesis: A mod 12 = {1,5,9} are all A ≡ 1 (mod 4)")
print("            Maybe fundamental pattern is simpler: A mod 4 = 1")
print()

# Verify the mathematical relationship
print("Checking A mod 12 values that succeed:")
print("  1 mod 4 =", 1 % 4, "✓")
print("  5 mod 4 =", 5 % 4, "✓")
print("  9 mod 4 =", 9 % 4, "✓")
print()
print("All three satisfy A ≡ 1 (mod 4)!")
print()

# Test pure geometry success rates by A mod 4
print("="*95)
print("PART 1: PURE GEOMETRY SUCCESS RATES BY A MOD 4")
print("="*95)
print()

nuclei_data = []
for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z(A, bonus_mod4=0.0)
    correct = (Z_pred == Z_exp)
    nuclei_data.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_pred,
        'correct': correct,
        'mod_4': A % 4,
        'mod_12': A % 12,
    })

successes = [n for n in nuclei_data if n['correct']]
failures = [n for n in nuclei_data if not n['correct']]

overall_rate = 100 * len(successes) / len(nuclei_data)

print(f"Overall success rate: {len(successes)}/285 ({overall_rate:.1f}%)")
print()

# A mod 4 distribution
mod4_all = Counter(n['mod_4'] for n in nuclei_data)
mod4_succ = Counter(n['mod_4'] for n in successes)

print(f"{'A mod 4':<12} {'Total':<10} {'Success':<10} {'Fail':<10} {'Success %':<12} {'Δ from avg'}")
print("-"*95)

for mod_val in range(4):
    total = mod4_all.get(mod_val, 0)
    succ = mod4_succ.get(mod_val, 0)
    fail = total - succ
    rate = 100 * succ / total if total > 0 else 0
    delta = rate - overall_rate
    marker = "★★★" if abs(delta) > 10.0 else ("★" if abs(delta) > 5.0 else "")

    print(f"{mod_val:<12} {total:<10} {succ:<10} {fail:<10} {rate:<12.1f} {delta:+.1f}  {marker}")

print()

# Compare to A mod 12
print("="*95)
print("COMPARISON: A MOD 4 vs A MOD 12")
print("="*95)
print()

print("A mod 4 = 1:")
mod4_1_nuclei = [n for n in nuclei_data if n['mod_4'] == 1]
mod4_1_succ = sum(1 for n in mod4_1_nuclei if n['correct'])
mod4_1_rate = 100 * mod4_1_succ / len(mod4_1_nuclei)
print(f"  {mod4_1_succ}/{len(mod4_1_nuclei)} ({mod4_1_rate:.1f}%)")
print()

print("A mod 12 ∈ {1,5,9}:")
mod12_favorable = [n for n in nuclei_data if n['mod_12'] in [1, 5, 9]]
mod12_fav_succ = sum(1 for n in mod12_favorable if n['correct'])
mod12_fav_rate = 100 * mod12_fav_succ / len(mod12_favorable)
print(f"  {mod12_fav_succ}/{len(mod12_favorable)} ({mod12_fav_rate:.1f}%)")
print()

print("A mod 12 = 0:")
mod12_0_nuclei = [n for n in nuclei_data if n['mod_12'] == 0]
mod12_0_succ = sum(1 for n in mod12_0_nuclei if n['correct'])
mod12_0_rate = 100 * mod12_0_succ / len(mod12_0_nuclei)
print(f"  {mod12_0_succ}/{len(mod12_0_nuclei)} ({mod12_0_rate:.1f}%)")
print()

# Check if A mod 4 = 0 overlaps with A mod 12 = 0
mod4_0_nuclei = [n for n in nuclei_data if n['mod_4'] == 0]
mod4_0_succ = sum(1 for n in mod4_0_nuclei if n['correct'])
mod4_0_rate = 100 * mod4_0_succ / len(mod4_0_nuclei)
print("A mod 4 = 0:")
print(f"  {mod4_0_succ}/{len(mod4_0_nuclei)} ({mod4_0_rate:.1f}%)")
print()

if abs(mod4_1_rate - mod12_fav_rate) < 2.0:
    print("★★★ A MOD 4 = 1 MATCHES A MOD 12 = {1,5,9} PATTERN!")
    print("    → A mod 4 = 1 might be the FUNDAMENTAL pattern!")
else:
    print("A mod 4 = 1 differs from A mod 12 = {1,5,9}")
    print(f"  Difference: {abs(mod4_1_rate - mod12_fav_rate):.1f} percentage points")

print()

# Detailed breakdown
print("="*95)
print("DETAILED BREAKDOWN: A MOD 4 = 1 SUBGROUPS")
print("="*95)
print()

print("A mod 4 = 1 includes A mod 12 values: 1, 5, 9 (favorable) and 13≡1 (mod 12)")
print()

# Split A mod 4 = 1 by A mod 12
mod4_1_by_mod12 = {}
for n in mod4_1_nuclei:
    mod12 = n['mod_12']
    if mod12 not in mod4_1_by_mod12:
        mod4_1_by_mod12[mod12] = []
    mod4_1_by_mod12[mod12].append(n)

print(f"{'A mod 12':<12} {'Count':<10} {'Success':<10} {'Rate %':<12} {'Note'}")
print("-"*95)

for mod12 in sorted(mod4_1_by_mod12.keys()):
    subgroup = mod4_1_by_mod12[mod12]
    succ = sum(1 for n in subgroup if n['correct'])
    rate = 100 * succ / len(subgroup)

    note = ""
    if mod12 in [1, 5, 9]:
        note = "Favorable (mod 12)"

    print(f"{mod12:<12} {len(subgroup):<10} {succ:<10} {rate:<12.1f} {note}")

print()

# Test A mod 4 = 0 subgroups
print("A mod 4 = 0 includes A mod 12 values: 0, 4, 8")
print()

mod4_0_by_mod12 = {}
for n in mod4_0_nuclei:
    mod12 = n['mod_12']
    if mod12 not in mod4_0_by_mod12:
        mod4_0_by_mod12[mod12] = []
    mod4_0_by_mod12[mod12].append(n)

print(f"{'A mod 12':<12} {'Count':<10} {'Success':<10} {'Rate %':<12} {'Note'}")
print("-"*95)

for mod12 in sorted(mod4_0_by_mod12.keys()):
    subgroup = mod4_0_by_mod12[mod12]
    succ = sum(1 for n in subgroup if n['correct'])
    rate = 100 * succ / len(subgroup)

    note = ""
    if mod12 == 0:
        note = "Unfavorable (mod 12)"

    print(f"{mod12:<12} {len(subgroup):<10} {succ:<10} {rate:<12.1f} {note}")

print()

# Test geometric bonus based on A mod 4
print("="*95)
print("PART 2: OPTIMIZE A MOD 4 = 1 BONUS")
print("="*95)
print()

print("Testing if adding bonus for A mod 4 = 1 improves predictions...")
print()

bonus_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

best_correct = len(successes)
best_bonus = 0.0

print(f"{'Bonus':<10} {'Correct':<10} {'Success %':<12} {'Improvement'}")
print("-"*95)

for bonus in bonus_values:
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z(A, bonus_mod4=bonus)
        if Z_pred == Z_exp:
            correct += 1

    rate = 100 * correct / len(test_nuclides)
    improvement = correct - len(successes)
    marker = "★" if correct > best_correct else ""

    print(f"{bonus:<10.2f} {correct:<10} {rate:<12.1f} {improvement:+d}  {marker}")

    if correct > best_correct:
        best_correct = correct
        best_bonus = bonus

print()

if best_correct > len(successes):
    print(f"★★ IMPROVEMENT FOUND! Best bonus = {best_bonus:.2f}")
    print(f"   Pure geometry:      {len(successes)}/285 ({overall_rate:.1f}%)")
    print(f"   With A mod 4 bonus: {best_correct}/285 ({100*best_correct/285:.1f}%)")
    print(f"   Improvement:        {best_correct - len(successes):+d} matches")
else:
    print("No improvement from A mod 4 = 1 bonus")
    print("Pattern is STATISTICAL but not a simple energy shift")

print()

# Physical interpretation
print("="*95)
print("PHYSICAL INTERPRETATION")
print("="*95)
print()

print("A mod 4 = 1 might relate to:")
print()
print("1. QUATERNIONS (4D algebra)")
print("   - SU(2) structure in nuclear physics")
print("   - Isospin symmetry (proton/neutron)")
print("   - A ≡ 1 (mod 4) = special quaternion structure?")
print()

print("2. SPACETIME STRUCTURE")
print("   - 4D spacetime (3 space + 1 time)")
print("   - Cl(3,3) → Cl(1,3) reduction (Dirac algebra)")
print("   - A ≡ 1 (mod 4) = geometric resonance?")
print()

print("3. PAIRING AND PARITY")
print("   - A mod 4 relates to (Z,N) parity combinations")
print("   - A=1: (odd Z, even N) or (even Z, odd N)")
print("   - A=0: both even or both odd (pairing energy already accounts for this)")
print()

print("4. SO(4) ≈ SU(2) × SU(2) SYMMETRY")
print("   - 4D rotations have special structure")
print("   - Related to nuclear angular momentum?")
print("   - A ≡ 1 (mod 4) = favorable J coupling?")
print()

# Check connection to Z,N parity
print("="*95)
print("CONNECTION TO Z,N PARITY")
print("="*95)
print()

print("Checking if A mod 4 correlates with (Z even/odd, N even/odd) patterns...")
print()

parity_by_mod4 = {0: [], 1: [], 2: [], 3: []}
for n in nuclei_data:
    Z = n['Z_exp']
    N = n['A'] - Z
    Z_parity = "even" if Z % 2 == 0 else "odd"
    N_parity = "even" if N % 2 == 0 else "odd"
    parity_by_mod4[n['mod_4']].append((Z_parity, N_parity))

print(f"{'A mod 4':<12} {'Z-even,N-even':<18} {'Z-even,N-odd':<18} {'Z-odd,N-even':<18} {'Z-odd,N-odd'}")
print("-"*95)

for mod_val in range(4):
    parities = parity_by_mod4[mod_val]
    ee = sum(1 for z, n in parities if z == "even" and n == "even")
    eo = sum(1 for z, n in parities if z == "even" and n == "odd")
    oe = sum(1 for z, n in parities if z == "odd" and n == "even")
    oo = sum(1 for z, n in parities if z == "odd" and n == "odd")

    print(f"{mod_val:<12} {ee:<18} {eo:<18} {oe:<18} {oo}")

print()
print("A mod 4 = 0 → Both even or both odd (pairing energy important)")
print("A mod 4 = 1 → One even, one odd (no pairing)")
print("A mod 4 = 2 → Both even or both odd (pairing energy important)")
print("A mod 4 = 3 → One even, one odd (no pairing)")
print()

# Summary
print("="*95)
print("SUMMARY")
print("="*95)
print()

print(f"A mod 4 = 1:  {mod4_1_rate:.1f}% success")
print(f"A mod 4 = 0:  {mod4_0_rate:.1f}% success")
print(f"Difference:   {mod4_1_rate - mod4_0_rate:+.1f} percentage points")
print()

print(f"A mod 12 = 1,5,9:  {mod12_fav_rate:.1f}% success")
print(f"A mod 12 = 0:      {mod12_0_rate:.1f}% success")
print(f"Difference:        {mod12_fav_rate - mod12_0_rate:+.1f} percentage points")
print()

if abs(mod4_1_rate - mod12_fav_rate) < 5.0:
    print("★★★ A MOD 4 = 1 IS THE FUNDAMENTAL PATTERN!")
    print()
    print("Evidence:")
    print("  • A mod 12 = {1,5,9} all satisfy A ≡ 1 (mod 4)")
    print(f"  • Success rates nearly identical ({abs(mod4_1_rate - mod12_fav_rate):.1f}% difference)")
    print("  • Simpler pattern (mod 4 vs mod 12)")
    print()
    print("Physical interpretation:")
    print("  → 4-fold structure fundamental in QFD")
    print("  → Related to quaternions, SU(2), or spacetime?")
    print("  → A ≡ 1 (mod 4) = optimal topological configuration")
else:
    print("A mod 12 pattern is richer than A mod 4")
    print("Both patterns exist but mod 12 provides additional structure")

print()
print("="*95)
