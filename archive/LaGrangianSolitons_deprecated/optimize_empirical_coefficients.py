#!/usr/bin/env python3
"""
OPTIMIZE EMPIRICAL Z FORMULA COEFFICIENTS
===========================================================================
User's formula: Z = 0.529 × A^(2/3) + (1/3.058) × A

This gave 59/285 (20.7%) with mean error +1.547 (overpredicts by ~1.5)

Let's optimize the coefficients:
Z = c1 × A^(2/3) + c2 × A + c3

to maximize exact matches on the 285 nuclides.
===========================================================================
"""

import numpy as np
from scipy.optimize import minimize
from collections import Counter

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

def predict_Z(A, c1, c2, c3=0):
    """Empirical formula with tunable coefficients."""
    Z_raw = c1 * (A**(2.0/3.0)) + c2 * A + c3
    return int(round(Z_raw))

def count_exact_matches(coeffs):
    """Count how many exact Z matches with given coefficients."""
    c1, c2, c3 = coeffs

    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = predict_Z(A, c1, c2, c3)
        if Z_pred == Z_exp:
            correct += 1

    return correct

def objective(coeffs):
    """Minimize negative of exact matches (to maximize matches)."""
    return -count_exact_matches(coeffs)

print("="*95)
print("OPTIMIZE EMPIRICAL Z FORMULA COEFFICIENTS")
print("="*95)
print()

print("Formula: Z = c1 × A^(2/3) + c2 × A + c3")
print()

# Test user's original coefficients
print("="*95)
print("USER'S ORIGINAL COEFFICIENTS")
print("="*95)
print()

c1_user = 0.529
c2_user = 1.0 / 3.058231
c3_user = 0.0

matches_user = count_exact_matches([c1_user, c2_user, c3_user])

print(f"c1 = {c1_user:.6f}")
print(f"c2 = {c2_user:.6f}  (1/β_vacuum)")
print(f"c3 = {c3_user:.6f}")
print()
print(f"Exact matches: {matches_user}/285 ({100*matches_user/285:.1f}%)")
print()

# Grid search for better coefficients
print("="*95)
print("GRID SEARCH OPTIMIZATION")
print("="*95)
print()

print("Searching c1, c2, c3 space...")
print()

best_coeffs = [c1_user, c2_user, c3_user]
best_matches = matches_user

# Grid search
c1_range = np.linspace(0.0, 2.0, 21)
c2_range = np.linspace(0.0, 0.5, 21)
c3_range = np.linspace(-5.0, 5.0, 21)

for c1 in c1_range:
    for c2 in c2_range:
        for c3 in c3_range:
            matches = count_exact_matches([c1, c2, c3])
            if matches > best_matches:
                best_matches = matches
                best_coeffs = [c1, c2, c3]

print(f"Best coefficients found:")
print(f"  c1 = {best_coeffs[0]:.6f}")
print(f"  c2 = {best_coeffs[1]:.6f}")
print(f"  c3 = {best_coeffs[2]:.6f}")
print()
print(f"Exact matches: {best_matches}/285 ({100*best_matches/285:.1f}%)")
print(f"Improvement: {best_matches - matches_user:+d} matches")
print()

# Fine-tune with local optimization
print("="*95)
print("LOCAL OPTIMIZATION (FINE-TUNING)")
print("="*95)
print()

from scipy.optimize import differential_evolution

# Use differential evolution (global optimizer for discrete problems)
bounds = [(0.0, 2.0), (0.0, 0.5), (-5.0, 5.0)]

result = differential_evolution(
    objective,
    bounds,
    maxiter=100,
    seed=42,
    atol=0,
    tol=0.01,
    workers=1,
)

optimized_coeffs = result.x
optimized_matches = count_exact_matches(optimized_coeffs)

print(f"Optimized coefficients:")
print(f"  c1 = {optimized_coeffs[0]:.6f}")
print(f"  c2 = {optimized_coeffs[1]:.6f}")
print(f"  c3 = {optimized_coeffs[2]:.6f}")
print()
print(f"Exact matches: {optimized_matches}/285 ({100*optimized_matches/285:.1f}%)")
print(f"Improvement over user: {optimized_matches - matches_user:+d} matches")
print(f"Comparison to baseline: {optimized_matches - 175:+d} matches")
print()

# Use best result
if optimized_matches > best_matches:
    best_coeffs = optimized_coeffs
    best_matches = optimized_matches

# ============================================================================
# ANALYSIS WITH BEST COEFFICIENTS
# ============================================================================
print("="*95)
print(f"BEST FORMULA: Z = {best_coeffs[0]:.4f}×A^(2/3) + {best_coeffs[1]:.4f}×A + {best_coeffs[2]:.4f}")
print("="*95)
print()

c1_best, c2_best, c3_best = best_coeffs

errors = []
predictions = []

for name, Z_exp, A in test_nuclides:
    Z_pred = predict_Z(A, c1_best, c2_best, c3_best)
    error = Z_pred - Z_exp

    errors.append(error)
    predictions.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_pred,
        'error': error,
        'abs_error': abs(error),
    })

mean_error = np.mean(errors)
std_error = np.std(errors)

print(f"Results: {best_matches}/285 ({100*best_matches/285:.1f}%)")
print(f"Mean error: {mean_error:.3f} charges")
print(f"Std error: {std_error:.3f} charges")
print()

# Error distribution
error_counts = Counter(abs(e) for e in errors)

print(f"Error distribution:")
for abs_err in sorted(error_counts.keys())[:5]:
    count = error_counts[abs_err]
    pct = 100 * count / 285
    marker = "★★★" if abs_err == 0 else "★★" if abs_err == 1 else "★" if abs_err == 2 else ""
    print(f"  |error| = {abs_err}: {count} ({pct:.1f}%)  {marker}")

print()

within_1 = len([p for p in predictions if p['abs_error'] <= 1])
within_2 = len([p for p in predictions if p['abs_error'] <= 2])

print(f"Within ±1: {within_1}/285 ({100*within_1/285:.1f}%)")
print(f"Within ±2: {within_2}/285 ({100*within_2/285:.1f}%)")
print()

# ============================================================================
# COMPARISON
# ============================================================================
print("="*95)
print("COMPARISON")
print("="*95)
print()

print(f"{'Method':<40} {'Exact':<12} {'Success %'}\"")
print("-"*95)
print(f"{'User formula (0.529, 0.327, 0)':<40} {matches_user:<12} {100*matches_user/285:.1f}%")
print(f"{'Optimized empirical formula':<40} {best_matches:<12} {100*best_matches/285:.1f}%")
print(f"{'Pure QFD (energy minimization)':<40} {'175':<12} {'61.4%'}")
print()

if best_matches > 175:
    print(f"★★★ OPTIMIZED EMPIRICAL FORMULA BEATS BASELINE!")
    print(f"  Gain: {best_matches - 175:+d} matches ({100*(best_matches - 175)/285:+.1f}%)")
elif best_matches == 175:
    print(f"✓ Optimized formula matches baseline performance")
else:
    print(f"Pure QFD energy minimization still superior")
    print(f"  QFD advantage: {175 - best_matches} matches ({100*(175 - best_matches)/285:.1f}%)")

print()

# ============================================================================
# PHYSICAL INTERPRETATION
# ============================================================================
print("="*95)
print("PHYSICAL INTERPRETATION")
print("="*95)
print()

print(f"Optimized formula:")
print(f"  Z = {c1_best:.4f} × A^(2/3) + {c2_best:.4f} × A {c3_best:+.4f}")
print()

print(f"Comparison to user's formula:")
print(f"  c1: {c1_best:.4f} vs {c1_user:.4f}  (ratio: {c1_best/c1_user:.3f})")
print(f"  c2: {c2_best:.4f} vs {c2_user:.4f}  (ratio: {c2_best/c2_user:.3f})")
print(f"  c3: {c3_best:.4f} vs {c3_user:.4f}  (shift)")
print()

if abs(c3_best) > 0.5:
    print(f"★ Constant offset c3 = {c3_best:.2f} needed")
    print(f"  Suggests missing physics or normalization")

if abs(c2_best - c2_user) / c2_user > 0.1:
    print(f"★ Linear term c2 differs significantly from β_vacuum^(-1)")
    print(f"  β_vacuum^(-1) = {c2_user:.4f}")
    print(f"  Optimized c2 = {c2_best:.4f}")

print()
print("="*95)
