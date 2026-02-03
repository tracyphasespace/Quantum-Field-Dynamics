#!/usr/bin/env python3
"""
FAILURE MODE ANALYSIS - UNDERSTANDING THE 110 ERRORS
===========================================================================
Current success: 175/285 (61.4%) with pure QFD

Question: What constitutes "failure"?
- Exact match: Z_pred == Z_exp → success
- Off by 1: Z_pred = Z_exp ± 1 → failure (but how bad?)
- Off by 2+: Z_pred = Z_exp ± 2+ → failure (worse)

Analysis:
1. Distribution of errors: |Z_pred - Z_exp|
2. Systematic bias: mean(Z_pred - Z_exp) by region
3. Near-misses: How many are within ±1 charge?
4. Catastrophic failures: |error| > 3
5. Error patterns by A mod 4, parity, mass region

Goal: Understand if we need:
- Small correction (most errors ±1)
- Global shift (systematic bias)
- New physics (large random errors)
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# QFD Constants (PURE - NO BONUSES)
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived Constants
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_surface_coeff = beta_nuclear / 15
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def qfd_energy_pure(A, Z):
    """Pure QFD energy - no bonuses."""
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

def find_stable_Z_pure(A):
    """Find Z that minimizes pure QFD energy."""
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
print("FAILURE MODE ANALYSIS - UNDERSTANDING THE 110 ERRORS")
print("="*95)
print()

# Get all predictions
all_predictions = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred = find_stable_Z_pure(A)
    error = Z_pred - Z_exp

    all_predictions.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_pred,
        'N_exp': N_exp,
        'error': error,
        'abs_error': abs(error),
        'correct': (error == 0),
        'mod_4': A % 4,
        'parity': 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else
                  'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A',
    })

successes = [p for p in all_predictions if p['correct']]
failures = [p for p in all_predictions if not p['correct']]

print(f"Total predictions: {len(all_predictions)}")
print(f"Successes: {len(successes)} ({100*len(successes)/len(all_predictions):.1f}%)")
print(f"Failures: {len(failures)} ({100*len(failures)/len(all_predictions):.1f}%)")
print()

# ============================================================================
# ERROR MAGNITUDE DISTRIBUTION
# ============================================================================
print("="*95)
print("ERROR MAGNITUDE DISTRIBUTION")
print("="*95)
print()

error_counts = Counter(p['abs_error'] for p in all_predictions)

print(f"{'|Error|':<12} {'Count':<12} {'Percentage':<12} {'Cumulative'}\"")
print("-"*95)

cumulative = 0
for abs_err in sorted(error_counts.keys()):
    count = error_counts[abs_err]
    pct = 100 * count / len(all_predictions)
    cumulative += count
    cum_pct = 100 * cumulative / len(all_predictions)

    marker = ""
    if abs_err == 0:
        marker = "★★★ EXACT"
    elif abs_err == 1:
        marker = "★★ NEAR-MISS"
    elif abs_err == 2:
        marker = "★ CLOSE"
    elif abs_err >= 5:
        marker = "✗ CATASTROPHIC"

    print(f"{abs_err:<12} {count:<12} {pct:<12.1f} {cum_pct:<12.1f}  {marker}")

print()

# Near-miss analysis
near_misses = [p for p in failures if p['abs_error'] == 1]
close_calls = [p for p in failures if p['abs_error'] == 2]
catastrophic = [p for p in failures if p['abs_error'] >= 5]

print(f"CATEGORIZATION:")
print(f"  Exact matches:     {len(successes)}/285 ({100*len(successes)/285:.1f}%)")
print(f"  Near-misses (±1):  {len(near_misses)}/285 ({100*len(near_misses)/285:.1f}%)")
print(f"  Close (±2):        {len(close_calls)}/285 ({100*len(close_calls)/285:.1f}%)")
print(f"  Catastrophic (≥5): {len(catastrophic)}/285 ({100*len(catastrophic)/285:.1f}%)")
print()

if len(successes) + len(near_misses) > len(successes):
    relaxed_success = len(successes) + len(near_misses)
    print(f"★ RELAXED CRITERION (within ±1): {relaxed_success}/285 ({100*relaxed_success/285:.1f}%)")
    print()

# ============================================================================
# SYSTEMATIC BIAS ANALYSIS
# ============================================================================
print("="*95)
print("SYSTEMATIC BIAS (SIGNED ERRORS)")
print("="*95)
print()

errors_all = [p['error'] for p in all_predictions]
mean_error = np.mean(errors_all)
std_error = np.std(errors_all)

print(f"Overall error statistics:")
print(f"  Mean error (Z_pred - Z_exp): {mean_error:.3f} charges")
print(f"  Std deviation:               {std_error:.3f} charges")
print()

if mean_error < -0.1:
    print(f"★★ SYSTEMATIC UNDERPREDICTION: Z predicted is {abs(mean_error):.3f} too LOW")
elif mean_error > 0.1:
    print(f"★★ SYSTEMATIC OVERPREDICTION: Z predicted is {mean_error:.3f} too HIGH")
else:
    print(f"✓ No systematic bias (mean ≈ 0)")

print()

# Signed error distribution
print(f"Signed error distribution:")
error_signed_counts = Counter(p['error'] for p in all_predictions)

for err in sorted(error_signed_counts.keys()):
    if err != 0:  # Skip exact matches
        count = error_signed_counts[err]
        pct = 100 * count / len(all_predictions)

        sign = "too high" if err > 0 else "too low"
        print(f"  Error {err:+d} ({sign}): {count} nuclei ({pct:.1f}%)")

print()

# ============================================================================
# FAILURE PATTERNS BY A MOD 4
# ============================================================================
print("="*95)
print("FAILURE PATTERNS BY A MOD 4")
print("="*95)
print()

print(f"{'A mod 4':<12} {'Exact':<12} {'±1 error':<12} {'±2 error':<12} {'≥5 error':<12} {'Success %'}\"")
print("-"*95)

for mod in range(4):
    mod_preds = [p for p in all_predictions if p['mod_4'] == mod]

    exact = len([p for p in mod_preds if p['abs_error'] == 0])
    near = len([p for p in mod_preds if p['abs_error'] == 1])
    close = len([p for p in mod_preds if p['abs_error'] == 2])
    catast = len([p for p in mod_preds if p['abs_error'] >= 5])
    total = len(mod_preds)

    if total > 0:
        success_pct = 100 * exact / total
        marker = "★★★" if success_pct > 70 else "★" if success_pct > 60 else ""

        print(f"{mod:<12} {exact:<12} {near:<12} {close:<12} {catast:<12} {success_pct:.1f}%  {marker}")

print()

# ============================================================================
# FAILURE PATTERNS BY MASS REGION
# ============================================================================
print("="*95)
print("FAILURE PATTERNS BY MASS REGION")
print("="*95)
print()

mass_regions = [
    ("Light (A<60)", lambda A: A < 60),
    ("Medium (60≤A<140)", lambda A: 60 <= A < 140),
    ("Heavy (A≥140)", lambda A: A >= 140),
]

print(f"{'Region':<20} {'Exact':<12} {'±1 error':<12} {'Mean error':<15} {'Success %'}\"")
print("-"*95)

for region_name, region_filter in mass_regions:
    region_preds = [p for p in all_predictions if region_filter(p['A'])]

    if region_preds:
        exact = len([p for p in region_preds if p['abs_error'] == 0])
        near = len([p for p in region_preds if p['abs_error'] == 1])
        mean_err = np.mean([p['error'] for p in region_preds])
        total = len(region_preds)
        success_pct = 100 * exact / total

        print(f"{region_name:<20} {exact:<12} {near:<12} {mean_err:<+15.3f} {success_pct:.1f}%")

print()

# ============================================================================
# FAILURE PATTERNS BY PARITY
# ============================================================================
print("="*95)
print("FAILURE PATTERNS BY PARITY")
print("="*95)
print()

print(f"{'Parity':<20} {'Exact':<12} {'±1 error':<12} {'Mean error':<15} {'Success %'}\"")
print("-"*95)

for parity in ['even-even', 'odd-odd', 'odd-A']:
    parity_preds = [p for p in all_predictions if p['parity'] == parity]

    if parity_preds:
        exact = len([p for p in parity_preds if p['abs_error'] == 0])
        near = len([p for p in parity_preds if p['abs_error'] == 1])
        mean_err = np.mean([p['error'] for p in parity_preds])
        total = len(parity_preds)
        success_pct = 100 * exact / total

        marker = "★" if success_pct > 65 else ""

        print(f"{parity:<20} {exact:<12} {near:<12} {mean_err:<+15.3f} {success_pct:.1f}%  {marker}")

print()

# ============================================================================
# WORST FAILURES (EXAMPLES)
# ============================================================================
print("="*95)
print("WORST FAILURES (|error| ≥ 3)")
print("="*95)
print()

worst_failures = sorted([p for p in failures if p['abs_error'] >= 3],
                       key=lambda x: x['abs_error'], reverse=True)

if worst_failures:
    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_pred':<8} {'Error':<8} {'Parity':<12} {'A mod 4'}\"")
    print("-"*95)

    for p in worst_failures[:30]:
        print(f"{p['name']:<12} {p['A']:<6} {p['Z_exp']:<8} {p['Z_pred']:<8} {p['error']:+d}  "
              f"{p['parity']:<12} {p['mod_4']}")

    if len(worst_failures) > 30:
        print(f"\n... and {len(worst_failures) - 30} more worst failures")

    print()

    # Pattern in worst failures
    worst_mod4 = Counter(p['mod_4'] for p in worst_failures)
    worst_parity = Counter(p['parity'] for p in worst_failures)

    print(f"Patterns in worst failures:")
    print(f"  By A mod 4: {dict(worst_mod4)}")
    print(f"  By parity: {dict(worst_parity)}")
    print()

# ============================================================================
# NEAR-MISS EXAMPLES
# ============================================================================
print("="*95)
print("NEAR-MISSES (|error| = 1)")
print("="*95)
print()

print(f"We have {len(near_misses)} near-misses (off by exactly ±1 charge)")
print()

if near_misses:
    # Direction of near-misses
    over_by_one = [p for p in near_misses if p['error'] == +1]
    under_by_one = [p for p in near_misses if p['error'] == -1]

    print(f"Direction:")
    print(f"  Z_pred = Z_exp + 1 (overpredicted): {len(over_by_one)} nuclei")
    print(f"  Z_pred = Z_exp - 1 (underpredicted): {len(under_by_one)} nuclei")
    print()

    # Sample near-misses
    print(f"Sample near-misses:")
    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_pred':<8} {'Direction':<15} {'Parity'}\"")
    print("-"*95)

    for p in near_misses[:20]:
        direction = "overpredicted" if p['error'] > 0 else "underpredicted"
        print(f"{p['name']:<12} {p['A']:<6} {p['Z_exp']:<8} {p['Z_pred']:<8} {direction:<15} {p['parity']}")

    if len(near_misses) > 20:
        print(f"\n... and {len(near_misses) - 20} more near-misses")

    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: WHAT CONSTITUTES 'FAILURE'?")
print("="*95)
print()

print(f"CURRENT CRITERION: Exact match (Z_pred == Z_exp)")
print(f"  Success: {len(successes)}/285 (61.4%)")
print()

print(f"ALTERNATIVE CRITERIA:")
print(f"  Within ±1: {len(successes) + len(near_misses)}/285 ({100*(len(successes) + len(near_misses))/285:.1f}%)")
print(f"  Within ±2: {len(successes) + len(near_misses) + len(close_calls)}/285 ({100*(len(successes) + len(near_misses) + len(close_calls))/285:.1f}%)")
print()

print(f"ERROR PROFILE:")
print(f"  Mean error: {mean_error:.3f} charges (slight {'under' if mean_error < 0 else 'over'}prediction)")
print(f"  Std error: {std_error:.3f} charges")
print(f"  Near-misses: {len(near_misses)} (±1)")
print(f"  Catastrophic: {len(catastrophic)} (≥5)")
print()

print(f"KEY INSIGHTS:")

# Check if near-misses dominate
if len(near_misses) > len(catastrophic):
    print(f"  ★ Most failures are NEAR-MISSES (±1 charge)")
    print(f"    → Small correction might push many across threshold")
    print(f"    → Energy landscape is 'almost right' but slightly off")

if abs(mean_error) > 0.1:
    print(f"  ★ Systematic bias detected ({mean_error:+.3f} charges)")
    print(f"    → Global energy shift could help")

# Check mod 4 pattern
mod1_success = 100 * len([p for p in all_predictions if p['mod_4'] == 1 and p['correct']]) / len([p for p in all_predictions if p['mod_4'] == 1])
if mod1_success > 70:
    print(f"  ★ A mod 4 = 1 has {mod1_success:.1f}% success")
    print(f"    → Topology/geometry works best for this class")

# Check parity pattern
oddA_success = 100 * len([p for p in all_predictions if p['parity'] == 'odd-A' and p['correct']]) / len([p for p in all_predictions if p['parity'] == 'odd-A'])
if oddA_success > 65:
    print(f"  ★ Odd-A nuclei have {oddA_success:.1f}% success")
    print(f"    → Single minimum landscape (no pairing ambiguity)")

print()
print("="*95)
