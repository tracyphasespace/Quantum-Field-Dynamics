#!/usr/bin/env python3
"""
EMPIRICAL Z FORMULA TEST
===========================================================================
User's Formula:
Z = 0.529 × A^(2/3) + (1/3.058) × A

Physical interpretation:
- First term: 0.529 × A^(2/3)  [Surface-like scaling]
- Second term: β_vacuum^(-1) × A = 0.327 × A  [Linear bulk term]

This bypasses energy minimization and directly predicts Z from A.

Test: How well does this empirical fit predict the 285 stable nuclides?
===========================================================================
"""

import numpy as np
from collections import Counter

# Constants
BETA_VACUUM = 3.058231

def predict_Z_empirical(A):
    """User's empirical formula."""
    Z_raw = 0.529 * (A**(2.0/3.0)) + (1.0/BETA_VACUUM) * A
    return int(round(Z_raw))

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("EMPIRICAL Z FORMULA TEST")
print("="*95)
print()

print("Formula: Z = 0.529 × A^(2/3) + (1/3.058) × A")
print()
print(f"First term:  0.529 × A^(2/3)  (surface scaling)")
print(f"Second term: {1/BETA_VACUUM:.4f} × A  (bulk linear term, β_vacuum^(-1))")
print()

# Test predictions
correct = 0
errors = []
predictions = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred = predict_Z_empirical(A)
    error = Z_pred - Z_exp

    if Z_pred == Z_exp:
        correct += 1

    errors.append(error)
    predictions.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_pred,
        'error': error,
        'abs_error': abs(error),
        'mod_4': A % 4,
    })

success_rate = 100 * correct / 285

print("="*95)
print("RESULTS")
print("="*95)
print()

print(f"Empirical formula: {correct}/285 ({success_rate:.1f}%)")
print(f"Pure QFD baseline: 175/285 (61.4%)")
print(f"Difference: {correct - 175:+d} matches ({success_rate - 61.4:+.1f}%)")
print()

# Error statistics
mean_error = np.mean(errors)
std_error = np.std(errors)

print(f"Error statistics:")
print(f"  Mean error: {mean_error:.3f} charges")
print(f"  Std error:  {std_error:.3f} charges")
print()

if mean_error < -0.1:
    print(f"★ Systematic underprediction: {abs(mean_error):.3f} charges too low")
elif mean_error > 0.1:
    print(f"★ Systematic overprediction: {mean_error:.3f} charges too high")
else:
    print(f"✓ No systematic bias")

print()

# ============================================================================
# ERROR DISTRIBUTION
# ============================================================================
print("="*95)
print("ERROR DISTRIBUTION")
print("="*95)
print()

error_counts = Counter(abs(e) for e in errors)

print(f"{'|Error|':<12} {'Count':<12} {'Percentage'}\"")
print("-"*95)

cumulative = 0
for abs_err in sorted(error_counts.keys())[:10]:
    count = error_counts[abs_err]
    pct = 100 * count / 285
    cumulative += count
    cum_pct = 100 * cumulative / 285

    marker = ""
    if abs_err == 0:
        marker = "★★★ EXACT"
    elif abs_err == 1:
        marker = "★★ NEAR"
    elif abs_err == 2:
        marker = "★ CLOSE"

    print(f"{abs_err:<12} {count:<12} {pct:<12.1f}  {marker}")

print()

near_misses = [p for p in predictions if p['abs_error'] == 1]
within_2 = [p for p in predictions if p['abs_error'] <= 2]

print(f"Within ±1: {len([p for p in predictions if p['abs_error'] <= 1])}/285 ({100*len([p for p in predictions if p['abs_error'] <= 1])/285:.1f}%)")
print(f"Within ±2: {len(within_2)}/285 ({100*len(within_2)/285:.1f}%)")
print()

# ============================================================================
# BY A MOD 4
# ============================================================================
print("="*95)
print("SUCCESS BY A MOD 4")
print("="*95)
print()

print(f"{'A mod 4':<12} {'Correct':<12} {'Total':<12} {'Success %'}\"")
print("-"*95)

for mod in range(4):
    mod_preds = [p for p in predictions if p['mod_4'] == mod]
    if mod_preds:
        mod_correct = len([p for p in mod_preds if p['error'] == 0])
        mod_total = len(mod_preds)
        mod_pct = 100 * mod_correct / mod_total

        marker = "★★★" if mod_pct > 70 else "★" if mod_pct > 60 else ""

        print(f"{mod:<12} {mod_correct:<12} {mod_total:<12} {mod_pct:.1f}%  {marker}")

print()

# ============================================================================
# SAMPLE PREDICTIONS
# ============================================================================
print("="*95)
print("SAMPLE PREDICTIONS")
print("="*95)
print()

print(f"Light nuclei:")
print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_pred':<8} {'Error'}\"")
print("-"*95)

for name, Z_exp, A in test_nuclides[:20]:
    Z_pred = predict_Z_empirical(A)
    error = Z_pred - Z_exp
    marker = "✓" if error == 0 else f"{error:+d}"
    print(f"{name:<12} {A:<6} {Z_exp:<8} {Z_pred:<8} {marker}")

print()

print(f"Heavy nuclei:")
print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_pred':<8} {'Error'}\"")
print("-"*95)

heavy = [(name, Z, A) for name, Z, A in test_nuclides if A >= 140]
for name, Z_exp, A in heavy[:20]:
    Z_pred = predict_Z_empirical(A)
    error = Z_pred - Z_exp
    marker = "✓" if error == 0 else f"{error:+d}"
    print(f"{name:<12} {A:<6} {Z_exp:<8} {Z_pred:<8} {marker}")

print()

# ============================================================================
# COMPARISON TO BASELINE
# ============================================================================
print("="*95)
print("COMPARISON TO BASELINE")
print("="*95)
print()

if correct > 175:
    print(f"★★★ EMPIRICAL FORMULA OUTPERFORMS BASELINE!")
    print(f"  Empirical: {correct}/285 ({success_rate:.1f}%)")
    print(f"  Baseline:  175/285 (61.4%)")
    print(f"  Gain: {correct - 175:+d} matches")
    print()
    print(f"This suggests the form Z ~ A^(2/3) + A has physical significance")
elif correct == 175:
    print(f"✓ Empirical formula matches baseline")
    print(f"  Both achieve 175/285 (61.4%)")
else:
    print(f"✗ Empirical formula underperforms baseline")
    print(f"  Empirical: {correct}/285 ({success_rate:.1f}%)")
    print(f"  Baseline:  175/285 (61.4%)")
    print(f"  Loss: {175 - correct:+d} matches")

print()

# ============================================================================
# PHYSICAL INTERPRETATION
# ============================================================================
print("="*95)
print("PHYSICAL INTERPRETATION")
print("="*95)
print()

print("The formula Z = 0.529 A^(2/3) + 0.327 A combines:")
print()
print("1. Surface term (A^(2/3)):")
print("   Coefficient 0.529 ≈ 0.52 (similar to SHIELD_FACTOR)")
print("   Represents surface/boundary contribution to charge")
print()
print("2. Bulk term (A):")
print("   Coefficient 1/3.058 = β_vacuum^(-1)")
print("   Represents bulk volume contribution")
print()
print("This is a DIRECT fit, not derived from energy minimization.")
print("Compare to QFD approach: minimize E_total(A,Z) to find Z.")
print()

if abs(success_rate - 61.4) < 5:
    print("Similar performance suggests both approaches capture same physics:")
    print("  • Surface vs bulk balance")
    print("  • A^(2/3) and A scaling")
    print("  • β_vacuum = 3.058 as fundamental parameter")
else:
    print("Different performance suggests one approach misses key physics.")

print()
print("="*95)
