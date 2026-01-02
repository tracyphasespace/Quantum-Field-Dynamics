#!/usr/bin/env python3
"""
TEST COMBINED MOD 4 AND MOD 7 PATTERNS
===========================================================================
We found:
  • A mod 4 = 1: 77.4% success
  • A mod 7 = 6: 75.0% success

Questions:
1. Do nuclei satisfying BOTH conditions succeed even better?
2. Is there interaction between 4-fold and 7-fold structure?
3. Can we find unified pattern (A mod 28, since lcm(4,7) = 28)?
4. Do bonuses based on combined patterns work?

Test:
- Success rates for all (A mod 4, A mod 7) combinations
- Nuclei satisfying both favorable conditions
- A mod 28 pattern (full period)
- Combined geometric bonuses
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict
import itertools

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
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

def qfd_energy_combined(A, Z, bonus_mod4_1, bonus_mod7_6, bonus_both):
    """QFD energy with combined mod 4 and mod 7 bonuses."""
    E = qfd_energy_pure(A, Z)

    beta_nuclear = M_proton * beta_vacuum / 2
    E_surface = beta_nuclear / 15

    # Separate bonuses
    if A % 4 == 1:
        E -= E_surface * bonus_mod4_1

    if A % 7 == 6:
        E -= E_surface * bonus_mod7_6

    # Extra bonus if BOTH conditions satisfied
    if A % 4 == 1 and A % 7 == 6:
        E -= E_surface * bonus_both  # Synergy bonus

    return E

def find_stable_Z_pure(A):
    """Find Z that minimizes energy (pure geometry)."""
    best_Z, best_E = 1, qfd_energy_pure(A, 1)
    for Z in range(1, A):
        E = qfd_energy_pure(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

def find_stable_Z_combined(A, bonus_mod4_1, bonus_mod7_6, bonus_both):
    """Find Z that minimizes energy (with combined bonuses)."""
    best_Z, best_E = 1, qfd_energy_combined(A, 1, bonus_mod4_1, bonus_mod7_6, bonus_both)
    for Z in range(1, A):
        E = qfd_energy_combined(A, Z, bonus_mod4_1, bonus_mod7_6, bonus_both)
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
print("COMBINED MOD 4 AND MOD 7 PATTERN ANALYSIS")
print("="*95)
print()

# Classify all nuclei
nuclei_data = []
for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z_pure(A)
    correct = (Z_pred == Z_exp)

    nuclei_data.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'correct': correct,
        'mod_4': A % 4,
        'mod_7': A % 7,
        'mod_28': A % 28,
    })

successes = [n for n in nuclei_data if n['correct']]
failures = [n for n in nuclei_data if not n['correct']]

overall_rate = 100 * len(successes) / len(nuclei_data)

# ============================================================================
# PART 1: (A MOD 4, A MOD 7) COMBINATION MATRIX
# ============================================================================
print("="*95)
print("PART 1: SUCCESS RATES BY (A MOD 4, A MOD 7) COMBINATIONS")
print("="*95)
print()

# Create matrix
mod4_mod7_all = Counter((n['mod_4'], n['mod_7']) for n in nuclei_data)
mod4_mod7_succ = Counter((n['mod_4'], n['mod_7']) for n in successes)

print("Success rate matrix (rows = A mod 4, columns = A mod 7):")
print()
print(f"{'mod 4 \\ mod 7':<15}", end="")
for mod7 in range(7):
    print(f"{mod7:<12}", end="")
print("  Row Total")
print("-"*110)

for mod4 in range(4):
    print(f"{mod4:<15}", end="")
    row_total = 0
    row_succ = 0

    for mod7 in range(7):
        total = mod4_mod7_all.get((mod4, mod7), 0)
        succ = mod4_mod7_succ.get((mod4, mod7), 0)
        row_total += total
        row_succ += succ

        if total > 0:
            rate = 100 * succ / total
            print(f"{succ}/{total}({rate:.0f}%)<3", end="")
        else:
            print(f"{'—':<12}", end="")

    row_rate = 100 * row_succ / row_total if row_total > 0 else 0
    print(f"{row_succ}/{row_total}({row_rate:.1f}%)")

print()

# Column totals
print(f"{'Col Total':<15}", end="")
for mod7 in range(7):
    col_total = 0
    col_succ = 0
    for mod4 in range(4):
        total = mod4_mod7_all.get((mod4, mod7), 0)
        succ = mod4_mod7_succ.get((mod4, mod7), 0)
        col_total += total
        col_succ += succ

    col_rate = 100 * col_succ / col_total if col_total > 0 else 0
    print(f"{col_succ}/{col_total}({col_rate:.0f}%)<3", end="")

print(f"{len(successes)}/{len(nuclei_data)}({overall_rate:.1f}%)")
print()

# ============================================================================
# PART 2: SPECIAL COMBINATIONS
# ============================================================================
print("="*95)
print("PART 2: FAVORABLE CONDITION COMBINATIONS")
print("="*95)
print()

# A mod 4 = 1 only
mod4_1_only = [n for n in nuclei_data if n['mod_4'] == 1 and n['mod_7'] != 6]
mod4_1_only_succ = sum(1 for n in mod4_1_only if n['correct'])
rate_mod4_1_only = 100 * mod4_1_only_succ / len(mod4_1_only) if mod4_1_only else 0

# A mod 7 = 6 only
mod7_6_only = [n for n in nuclei_data if n['mod_7'] == 6 and n['mod_4'] != 1]
mod7_6_only_succ = sum(1 for n in mod7_6_only if n['correct'])
rate_mod7_6_only = 100 * mod7_6_only_succ / len(mod7_6_only) if mod7_6_only else 0

# BOTH conditions
both_conditions = [n for n in nuclei_data if n['mod_4'] == 1 and n['mod_7'] == 6]
both_succ = sum(1 for n in both_conditions if n['correct'])
rate_both = 100 * both_succ / len(both_conditions) if both_conditions else 0

# Neither condition
neither = [n for n in nuclei_data if n['mod_4'] != 1 and n['mod_7'] != 6]
neither_succ = sum(1 for n in neither if n['correct'])
rate_neither = 100 * neither_succ / len(neither) if neither else 0

print(f"{'Condition':<35} {'Count':<10} {'Success':<10} {'Rate %':<12} {'Δ from avg'}")
print("-"*95)
print(f"{'A mod 4 = 1 only (not mod 7 = 6)':<35} {len(mod4_1_only):<10} "
      f"{mod4_1_only_succ:<10} {rate_mod4_1_only:<12.1f} {rate_mod4_1_only - overall_rate:+.1f}")
print(f"{'A mod 7 = 6 only (not mod 4 = 1)':<35} {len(mod7_6_only):<10} "
      f"{mod7_6_only_succ:<10} {rate_mod7_6_only:<12.1f} {rate_mod7_6_only - overall_rate:+.1f}")
print(f"{'BOTH (mod 4 = 1 AND mod 7 = 6)':<35} {len(both_conditions):<10} "
      f"{both_succ:<10} {rate_both:<12.1f} {rate_both - overall_rate:+.1f}  ★")
print(f"{'Neither condition':<35} {len(neither):<10} "
      f"{neither_succ:<10} {rate_neither:<12.1f} {rate_neither - overall_rate:+.1f}")

print()

if rate_both > rate_mod4_1_only and rate_both > rate_mod7_6_only:
    print("★★★ SYNERGY! Both conditions together perform better than either alone!")
    print(f"    Expected (if independent): ~{(rate_mod4_1_only + rate_mod7_6_only)/2:.1f}%")
    print(f"    Observed: {rate_both:.1f}%")
elif rate_both < min(rate_mod4_1_only, rate_mod7_6_only):
    print("✗ ANTI-SYNERGY! Both conditions together perform worse than either alone!")
else:
    print("Patterns appear independent (no strong synergy or anti-synergy)")

print()

# Show the nuclei satisfying both conditions
if both_conditions:
    print(f"Nuclei with BOTH A mod 4 = 1 AND A mod 7 = 6:")
    print(f"{'Nuclide':<12} {'A':<8} {'A mod 28':<12} {'Correct?'}")
    print("-"*95)
    for n in both_conditions:
        status = "✓" if n['correct'] else "✗"
        print(f"{n['name']:<12} {n['A']:<8} {n['mod_28']:<12} {status}")
    print()

# ============================================================================
# PART 3: A MOD 28 ANALYSIS (FULL PERIOD)
# ============================================================================
print("="*95)
print("PART 3: A MOD 28 PATTERN (LCM OF 4 AND 7)")
print("="*95)
print()

print("Since 4 and 7 are coprime, lcm(4,7) = 28")
print("A mod 28 captures the full combined structure")
print()

mod28_all = Counter(n['mod_28'] for n in nuclei_data)
mod28_succ = Counter(n['mod_28'] for n in successes)

# Find best and worst mod 28 values
mod28_rates = {}
for mod_val in range(28):
    total = mod28_all.get(mod_val, 0)
    succ = mod28_succ.get(mod_val, 0)
    if total >= 3:  # Only show if at least 3 nuclei
        rate = 100 * succ / total
        mod28_rates[mod_val] = (total, succ, rate)

# Show top 10 and bottom 10
if mod28_rates:
    sorted_rates = sorted(mod28_rates.items(), key=lambda x: x[1][2], reverse=True)

    print("Top 10 A mod 28 values (highest success rates):")
    print(f"{'A mod 28':<12} {'Total':<10} {'Success':<10} {'Rate %':<12} {'mod 4, mod 7'}")
    print("-"*95)

    for mod_val, (total, succ, rate) in sorted_rates[:10]:
        mod4 = mod_val % 4
        mod7 = mod_val % 7
        marker = "★" if (mod4 == 1 and mod7 == 6) else ""
        print(f"{mod_val:<12} {total:<10} {succ:<10} {rate:<12.1f} ({mod4}, {mod7})  {marker}")

    print()
    print("Bottom 10 A mod 28 values (lowest success rates):")
    print(f"{'A mod 28':<12} {'Total':<10} {'Success':<10} {'Rate %':<12} {'mod 4, mod 7'}")
    print("-"*95)

    for mod_val, (total, succ, rate) in sorted_rates[-10:]:
        mod4 = mod_val % 4
        mod7 = mod_val % 7
        print(f"{mod_val:<12} {total:<10} {succ:<10} {rate:<12.1f} ({mod4}, {mod7})")

print()

# Check if A mod 28 = 13 is special (since 13 ≡ 1 (mod 4) and 13 ≡ 6 (mod 7))
print("Special value: A mod 28 = 13 satisfies BOTH favorable conditions")
print("  13 mod 4 =", 13 % 4, "(favorable!)")
print("  13 mod 7 =", 13 % 7, "(favorable!)")
print()

if 13 in mod28_all:
    total_13 = mod28_all[13]
    succ_13 = mod28_succ.get(13, 0)
    rate_13 = 100 * succ_13 / total_13
    print(f"A mod 28 = 13: {succ_13}/{total_13} ({rate_13:.1f}%)")
else:
    print("No nuclei with A mod 28 = 13 in dataset")

print()

# ============================================================================
# PART 4: TEST COMBINED BONUSES
# ============================================================================
print("="*95)
print("PART 4: OPTIMIZE COMBINED BONUSES")
print("="*95)
print()

print("Testing three bonus parameters:")
print("  • bonus_mod4_1: for A mod 4 = 1")
print("  • bonus_mod7_6: for A mod 7 = 6")
print("  • bonus_both: extra synergy if BOTH conditions met")
print()

# Grid search
bonus_values = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

best_correct = 175
best_params = (0.0, 0.0, 0.0)

print(f"Testing {len(bonus_values)**3} combinations...")
print()

for b4, b7, bb in itertools.product(bonus_values, repeat=3):
    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z_combined(A, b4, b7, bb)
        if Z_pred == Z_exp:
            correct += 1

    if correct > best_correct:
        best_correct = correct
        best_params = (b4, b7, bb)

print(f"Best result:")
print(f"  bonus_mod4_1 = {best_params[0]:.2f}")
print(f"  bonus_mod7_6 = {best_params[1]:.2f}")
print(f"  bonus_both (synergy) = {best_params[2]:.2f}")
print(f"  Result: {best_correct}/285 ({100*best_correct/285:.1f}%)")
print(f"  Improvement: {best_correct - 175:+d} matches")
print()

if best_correct > 175:
    print("★ IMPROVEMENT! Combined bonuses help!")
elif best_correct < 175:
    print("Bonuses made things worse")
else:
    print("No improvement from combined bonuses")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: MOD 4 AND MOD 7 INTERACTION")
print("="*95)
print()

print(f"{'Pattern':<40} {'Success Rate':<15} {'Δ from avg'}")
print("-"*95)
print(f"{'Overall (pure geometry)':<40} {overall_rate:<15.1f} {'baseline'}")
print(f"{'A mod 4 = 1 only':<40} {rate_mod4_1_only:<15.1f} {rate_mod4_1_only - overall_rate:+.1f}%")
print(f"{'A mod 7 = 6 only':<40} {rate_mod7_6_only:<15.1f} {rate_mod7_6_only - overall_rate:+.1f}%")
print(f"{'BOTH mod 4 = 1 AND mod 7 = 6':<40} {rate_both:<15.1f} {rate_both - overall_rate:+.1f}%  ★")
print(f"{'Neither condition':<40} {rate_neither:<15.1f} {rate_neither - overall_rate:+.1f}%")
print()

print("Interpretation:")
print()

if rate_both > max(rate_mod4_1_only, rate_mod7_6_only) + 5:
    print("★★★ STRONG SYNERGY!")
    print("  • 4-fold and 7-fold structures REINFORCE each other")
    print("  • Combined pattern A mod 28 = 13 might be fundamental")
    print("  • Suggests unified geometric origin (4×7 structure in Cl(3,3)?)")
elif abs(rate_both - (rate_mod4_1_only + rate_mod7_6_only)/2) < 5:
    print("Patterns appear INDEPENDENT:")
    print("  • 4-fold and 7-fold are separate mechanisms")
    print("  • Both relate to different aspects of topology")
    print("  • 4-fold: quaternion/SU(2) structure")
    print("  • 7-fold: related to β ≈ π ≈ 22/7")
else:
    print("Complex interaction:")
    print("  • Neither pure synergy nor pure independence")
    print("  • May need higher-order analysis")
    print("  • Could be interference pattern")

print()

print(f"With combined bonuses: {best_correct}/285 ({100*best_correct/285:.1f}%)")
print(f"Improvement over pure: {best_correct - 175:+d} matches")
print()

print("="*95)
print("CONCLUSIONS")
print("="*95)
print()

print("1. MOD 4 AND MOD 7 PATTERNS:")
if len(both_conditions) > 0:
    print(f"   • {len(both_conditions)} nuclei satisfy BOTH conditions")
    print(f"   • Success rate: {rate_both:.1f}%")
    print(f"   • A mod 28 = 13 is the key combined value")
else:
    print("   • No nuclei in dataset satisfy both conditions")
    print("   • Patterns don't overlap significantly")

print()

print("2. GEOMETRIC BONUSES:")
if best_correct > 175:
    print(f"   • Combined bonuses improve by {best_correct - 175} matches")
    print("   • 4-fold and 7-fold should be included together")
else:
    print("   • Patterns are STATISTICAL, not simple energy shifts")
    print("   • Describe which nuclei QFD predicts well, not how to fix it")

print()

print("3. NEXT STEPS:")
print("   • Derive mod 4 and mod 7 from Cl(3,3) algebra")
print("   • Check if β = 3.058 ≈ π relates to 7-fold (22/7 = 3.142...)")
print("   • Investigate why quaternion structure gives 4-fold")
print("   • Test if patterns extend to unstable isotopes")

print()
print("="*95)
