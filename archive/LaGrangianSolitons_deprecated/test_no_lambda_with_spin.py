#!/usr/bin/env python3
"""
REMOVE LAMBDA_TIME PERMANENTLY AND ADD SPIN
===========================================================================
User insight: "remove lambda time permanently and retest, also remember spin"

Lambda_time_0 = 0.42 had ZERO effect when removed, so remove it permanently
and see if previously ineffective bonuses now work!

Also add SPIN consideration:
- Even-even nuclei: J=0 (ground state spin zero)
- Odd-A nuclei: J = half-integer (unpaired nucleon)
- Odd-odd nuclei: J varies (both unpaired)

Test with lambda=0:
1. Baseline accuracy
2. A mod 4, mod 7, mod 28 bonuses (retest!)
3. (Z,N) mod 4 bonuses (retest!)
4. Spin-dependent bonuses (NEW!)
5. Combined patterns

Things that didn't work before might work now that lambda is gone!
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict
import itertools

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
# REMOVED: LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

def qfd_energy_no_lambda(A, Z, bonus_dict=None):
    """
    QFD energy WITHOUT lambda_time (removed permanently).

    Optional bonus_dict can contain:
        'mod4_1', 'mod7_6', 'mod28_13', 'spin_0', etc.
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    # NO LAMBDA TIME!
    lambda_time = KAPPA_E * Z  # Only Z-dependent part, no lambda_time_0

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

    # Apply bonuses if provided
    E_bonus = 0
    if bonus_dict:
        # A mod patterns
        if 'mod4_1' in bonus_dict and A % 4 == 1:
            E_bonus -= E_surf * bonus_dict['mod4_1']

        if 'mod7_6' in bonus_dict and A % 7 == 6:
            E_bonus -= E_surf * bonus_dict['mod7_6']

        if 'mod28_13' in bonus_dict and A % 28 == 13:
            E_bonus -= E_surf * bonus_dict['mod28_13']

        # (Z,N) mod 4 patterns
        Z_mod4 = Z % 4
        N_mod4 = N % 4

        if 'ZN_23' in bonus_dict and (Z_mod4, N_mod4) == (2, 3):
            E_bonus -= E_surf * bonus_dict['ZN_23']

        if 'ZN_32' in bonus_dict and (Z_mod4, N_mod4) == (3, 2):
            E_bonus -= E_surf * bonus_dict['ZN_32']

        # Spin-dependent bonuses
        # Even-even: J=0 (no spin)
        if 'spin_0' in bonus_dict and Z % 2 == 0 and N % 2 == 0:
            E_bonus -= E_surf * bonus_dict['spin_0']

        # Odd-A: J = half-integer (unpaired nucleon)
        if 'spin_half' in bonus_dict and A % 2 == 1:
            E_bonus -= E_surf * bonus_dict['spin_half']

        # Odd-odd: J varies
        if 'spin_odd_odd' in bonus_dict and Z % 2 == 1 and N % 2 == 1:
            E_bonus -= E_surf * bonus_dict['spin_odd_odd']

    return E_bulk + E_surf + E_asym + E_vac + E_pair + E_bonus

def find_stable_Z(A, bonus_dict=None):
    """Find Z that minimizes energy."""
    best_Z, best_E = 1, qfd_energy_no_lambda(A, 1, bonus_dict)
    for Z in range(1, A):
        E = qfd_energy_no_lambda(A, Z, bonus_dict)
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
print("NO LAMBDA + SPIN - COMPLETE RETEST")
print("="*95)
print()
print("Removed PERMANENTLY: LAMBDA_TIME_0 = 0.42 (had zero effect)")
print("Added: Spin considerations (J=0, J=half-integer, etc.)")
print()

# ============================================================================
# BASELINE: NO LAMBDA, NO BONUSES
# ============================================================================
print("="*95)
print("BASELINE: QFD WITHOUT LAMBDA_TIME_0")
print("="*95)
print()

nuclei_data = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred = find_stable_Z(A, bonus_dict=None)
    correct = (Z_pred == Z_exp)

    # Determine spin type
    if Z_exp % 2 == 0 and N_exp % 2 == 0:
        spin_type = "J=0 (even-even)"
    elif A % 2 == 1:
        spin_type = "J=1/2,3/2,... (odd-A)"
    else:
        spin_type = "J varies (odd-odd)"

    nuclei_data.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'correct': correct,
        'mod_4': A % 4,
        'mod_7': A % 7,
        'mod_28': A % 28,
        'spin_type': spin_type,
    })

successes = [n for n in nuclei_data if n['correct']]
failures = [n for n in nuclei_data if not n['correct']]

print(f"Total: {len(successes)}/285 ({100*len(successes)/285:.1f}%)")
print(f"vs. WITH lambda: 175/285 (61.4%)")
print(f"Change: {len(successes) - 175:+d} matches")
print()

if len(successes) != 175:
    print("★ LAMBDA REMOVAL CHANGED RESULTS!")
    print("  → Some structure was hidden by lambda_time_0")
else:
    print("Confirmed: lambda_time_0 has exactly zero effect")

print()

# Check patterns
mod4_1 = [n for n in nuclei_data if n['mod_4'] == 1]
mod4_1_succ = sum(1 for n in mod4_1 if n['correct'])
rate_mod4_1 = 100 * mod4_1_succ / len(mod4_1)

mod28_13 = [n for n in nuclei_data if n['mod_28'] == 13]
mod28_13_succ = sum(1 for n in mod28_13 if n['correct'])
rate_mod28_13 = 100 * mod28_13_succ / len(mod28_13) if mod28_13 else 0

print("Geometric patterns (without lambda):")
print(f"  A mod 4 = 1:  {mod4_1_succ}/{len(mod4_1)} ({rate_mod4_1:.1f}%)")
print(f"  A mod 28 = 13: {mod28_13_succ}/{len(mod28_13)} ({rate_mod28_13:.1f}%)")
print()

# ============================================================================
# SPIN ANALYSIS
# ============================================================================
print("="*95)
print("SPIN ANALYSIS")
print("="*95)
print()

spin_groups = defaultdict(list)
for n in nuclei_data:
    spin_groups[n['spin_type']].append(n)

print(f"{'Spin Type':<30} {'Total':<10} {'Success':<10} {'Rate %':<12} {'Δ from avg'}")
print("-"*95)

overall_rate = 100 * len(successes) / len(nuclei_data)

for spin_type in sorted(spin_groups.keys()):
    group = spin_groups[spin_type]
    succ = sum(1 for n in group if n['correct'])
    rate = 100 * succ / len(group)
    delta = rate - overall_rate
    marker = "★" if abs(delta) > 10 else ""

    print(f"{spin_type:<30} {len(group):<10} {succ:<10} {rate:<12.1f} {delta:+.1f}  {marker}")

print()

# Check connection to A mod 4
print("Spin vs A mod 4 correlation:")
print()
print(f"{'A mod 4':<12} {'Even-even %':<15} {'Odd-A %':<15} {'Odd-odd %'}")
print("-"*95)

for mod4 in range(4):
    mod4_nuclei = [n for n in nuclei_data if n['mod_4'] == mod4]
    if mod4_nuclei:
        ee = sum(1 for n in mod4_nuclei if 'J=0' in n['spin_type'])
        oa = sum(1 for n in mod4_nuclei if 'odd-A' in n['spin_type'])
        oo = sum(1 for n in mod4_nuclei if 'varies' in n['spin_type'])
        total = len(mod4_nuclei)

        print(f"{mod4:<12} {100*ee/total:<15.1f} {100*oa/total:<15.1f} {100*oo/total:.1f}")

print()

# ============================================================================
# RETEST: A MOD BONUSES (might work now without lambda!)
# ============================================================================
print("="*95)
print("RETEST 1: A MOD BONUSES (WITHOUT LAMBDA)")
print("="*95)
print()

print("Previously failed with lambda. Retesting...")
print()

bonus_values = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

print("Testing A mod 4 = 1 bonus:")
best_mod4 = 0
best_correct_mod4 = len(successes)

for bonus in bonus_values:
    correct = sum(1 for name, Z_exp, A in test_nuclides
                  if find_stable_Z(A, {'mod4_1': bonus}) == Z_exp)

    if correct > best_correct_mod4:
        best_correct_mod4 = correct
        best_mod4 = bonus

print(f"  Best: bonus={best_mod4:.2f}, result={best_correct_mod4}/285 "
      f"({best_correct_mod4 - len(successes):+d} vs baseline)")

print()
print("Testing A mod 28 = 13 bonus:")
best_mod28 = 0
best_correct_mod28 = len(successes)

for bonus in bonus_values:
    correct = sum(1 for name, Z_exp, A in test_nuclides
                  if find_stable_Z(A, {'mod28_13': bonus}) == Z_exp)

    if correct > best_correct_mod28:
        best_correct_mod28 = correct
        best_mod28 = bonus

print(f"  Best: bonus={best_mod28:.2f}, result={best_correct_mod28}/285 "
      f"({best_correct_mod28 - len(successes):+d} vs baseline)")

print()

# ============================================================================
# RETEST: (Z,N) MOD 4 BONUSES
# ============================================================================
print("="*95)
print("RETEST 2: (Z,N) MOD 4 BONUSES (WITHOUT LAMBDA)")
print("="*95)
print()

print("Testing (2,3) and (3,2) bonuses:")
best_ZN = (0, 0)
best_correct_ZN = len(successes)

for b23, b32 in itertools.product(bonus_values[:5], repeat=2):
    correct = sum(1 for name, Z_exp, A in test_nuclides
                  if find_stable_Z(A, {'ZN_23': b23, 'ZN_32': b32}) == Z_exp)

    if correct > best_correct_ZN:
        best_correct_ZN = correct
        best_ZN = (b23, b32)

print(f"  Best: (2,3)={best_ZN[0]:.2f}, (3,2)={best_ZN[1]:.2f}")
print(f"  Result: {best_correct_ZN}/285 ({best_correct_ZN - len(successes):+d} vs baseline)")

print()

# ============================================================================
# NEW: SPIN BONUSES
# ============================================================================
print("="*95)
print("NEW: SPIN-DEPENDENT BONUSES")
print("="*95)
print()

print("Testing spin bonuses:")
best_spin = (0, 0, 0)
best_correct_spin = len(successes)

for s0, sh, soo in itertools.product([0.0, 0.10, 0.20, 0.30], repeat=3):
    correct = sum(1 for name, Z_exp, A in test_nuclides
                  if find_stable_Z(A, {'spin_0': s0, 'spin_half': sh,
                                       'spin_odd_odd': soo}) == Z_exp)

    if correct > best_correct_spin:
        best_correct_spin = correct
        best_spin = (s0, sh, soo)

print(f"  Best: J=0 (even-even)={best_spin[0]:.2f}")
print(f"        J=half (odd-A)={best_spin[1]:.2f}")
print(f"        J varies (odd-odd)={best_spin[2]:.2f}")
print(f"  Result: {best_correct_spin}/285 ({best_correct_spin - len(successes):+d} vs baseline)")

print()

# ============================================================================
# COMBINED: BEST OF ALL
# ============================================================================
print("="*95)
print("COMBINED: BEST BONUSES WITHOUT LAMBDA")
print("="*95)
print()

print("Testing combined bonuses (A mod + spin)...")
print()

# Grid search with best individual values
test_combos = [
    {'mod4_1': best_mod4} if best_correct_mod4 > len(successes) else {},
    {'mod28_13': best_mod28} if best_correct_mod28 > len(successes) else {},
    {'ZN_23': best_ZN[0], 'ZN_32': best_ZN[1]} if best_correct_ZN > len(successes) else {},
    {'spin_0': best_spin[0], 'spin_half': best_spin[1],
     'spin_odd_odd': best_spin[2]} if best_correct_spin > len(successes) else {},
]

# Test combinations
best_overall = len(successes)
best_combo_dict = {}

# Try all combinations of effective bonuses
for i in range(len(test_combos)):
    for combo in itertools.combinations(range(len(test_combos)), i+1):
        bonus_dict = {}
        for idx in combo:
            bonus_dict.update(test_combos[idx])

        if bonus_dict:  # Only test non-empty combinations
            correct = sum(1 for name, Z_exp, A in test_nuclides
                         if find_stable_Z(A, bonus_dict) == Z_exp)

            if correct > best_overall:
                best_overall = correct
                best_combo_dict = bonus_dict.copy()

print(f"Best combined result: {best_overall}/285 ({100*best_overall/285:.1f}%)")
print(f"Improvement over baseline (no lambda, no bonuses): {best_overall - len(successes):+d}")
print(f"Improvement over original (WITH lambda): {best_overall - 175:+d}")
print()

if best_combo_dict:
    print("Optimal bonuses:")
    for key, val in best_combo_dict.items():
        print(f"  {key}: {val:.2f}")
else:
    print("No bonuses improve performance")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: NO LAMBDA RESULTS")
print("="*95)
print()

print(f"{'Configuration':<45} {'Result':<15} {'vs Original'}")
print("-"*95)
print(f"{'Original (WITH lambda_time_0=0.42)':<45} {'175/285 (61.4%)':<15} {'baseline'}")
print(f"{'No lambda, no bonuses':<45} {f'{len(successes)}/285 ({100*len(successes)/285:.1f}%)':<15} "
      f"{len(successes) - 175:+d} matches")

if best_correct_mod4 > len(successes):
    print(f"{'No lambda + A mod 4=1 bonus':<45} {f'{best_correct_mod4}/285':<15} "
          f"{best_correct_mod4 - 175:+d} matches  ★")

if best_correct_mod28 > len(successes):
    print(f"{'No lambda + A mod 28=13 bonus':<45} {f'{best_correct_mod28}/285':<15} "
          f"{best_correct_mod28 - 175:+d} matches  ★")

if best_correct_ZN > len(successes):
    print(f"{'No lambda + (Z,N) mod 4 bonuses':<45} {f'{best_correct_ZN}/285':<15} "
          f"{best_correct_ZN - 175:+d} matches  ★")

if best_correct_spin > len(successes):
    print(f"{'No lambda + spin bonuses':<45} {f'{best_correct_spin}/285':<15} "
          f"{best_correct_spin - 175:+d} matches  ★")

if best_overall > len(successes):
    print(f"{'No lambda + combined best bonuses':<45} {f'{best_overall}/285 ({100*best_overall/285:.1f}%)':<15} "
          f"{best_overall - 175:+d} matches  ★★")

print()
print(f"{'Empirical bonuses (magic, symm, nr)':<45} {'206/285 (72.3%)':<15} {'+31 matches'}")
print()

if best_overall > 175:
    print("★★ IMPROVEMENT FOUND!")
    print(f"   Removing lambda_time_0 and adding geometric bonuses: {best_overall - 175:+d} matches!")
    print(f"   New gap to empirical: {206 - best_overall} matches ({100*(206-best_overall)/285:.1f}%)")
elif len(successes) != 175:
    print(f"★ LAMBDA HAD HIDDEN EFFECT: {len(successes) - 175:+d} matches changed")
else:
    print("Confirmed: lambda_time_0 = 0.42 is completely redundant")
    print("Geometric patterns exist but aren't simple energy corrections")

print()
print("="*95)
