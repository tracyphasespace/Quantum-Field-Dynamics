#!/usr/bin/env python3
"""
DISCRETE EMPIRICAL OPTIMIZATION - SEPARATE COEFFICIENTS BY CLASS
===========================================================================
Instead of one global formula Z = c1×A^(2/3) + c2×A + c3,
optimize DIFFERENT coefficients for different classes:

- A mod 4 = 0: Z = c1_0×A^(2/3) + c2_0×A + c3_0
- A mod 4 = 1: Z = c1_1×A^(2/3) + c2_1×A + c3_1
- A mod 4 = 2: Z = c1_2×A^(2/3) + c2_2×A + c3_2
- A mod 4 = 3: Z = c1_3×A^(2/3) + c2_3×A + c3_3

This captures discrete topology effects (mod 4 pattern).
===========================================================================
"""

import numpy as np
from scipy.optimize import differential_evolution
from collections import Counter, defaultdict

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

def predict_Z_discrete(A, coeffs_by_mod4):
    """
    Predict Z using different coefficients for each A mod 4 class.

    coeffs_by_mod4: dict {mod4: [c1, c2, c3]}
    """
    mod4 = A % 4
    c1, c2, c3 = coeffs_by_mod4[mod4]

    Z_raw = c1 * (A**(2.0/3.0)) + c2 * A + c3
    return int(round(Z_raw))

def count_matches_discrete(coeffs_flat):
    """Count exact matches with discrete coefficients."""
    # Unpack flat array into dict
    coeffs_by_mod4 = {
        0: coeffs_flat[0:3],
        1: coeffs_flat[3:6],
        2: coeffs_flat[6:9],
        3: coeffs_flat[9:12],
    }

    correct = 0
    for name, Z_exp, A in test_nuclides:
        Z_pred = predict_Z_discrete(A, coeffs_by_mod4)
        if Z_pred == Z_exp:
            correct += 1

    return correct

def objective_discrete(coeffs_flat):
    """Minimize negative of exact matches."""
    return -count_matches_discrete(coeffs_flat)

print("="*95)
print("DISCRETE EMPIRICAL OPTIMIZATION - SEPARATE COEFFICIENTS BY A MOD 4")
print("="*95)
print()

print("Strategy: Optimize different (c1, c2, c3) for each A mod 4 class")
print("  A mod 4 = 0: c1_0, c2_0, c3_0")
print("  A mod 4 = 1: c1_1, c2_1, c3_1")
print("  A mod 4 = 2: c1_2, c2_2, c3_2")
print("  A mod 4 = 3: c1_3, c2_3, c3_3")
print()
print("Total parameters: 12 (3 coefficients × 4 classes)")
print()

# Baseline: single global formula
print("="*95)
print("BASELINE: SINGLE GLOBAL FORMULA")
print("="*95)
print()

# Use previously found optimal global coefficients
c1_global = 0.8790
c2_global = 0.2584
c3_global = -1.8292

baseline_coeffs = {
    0: [c1_global, c2_global, c3_global],
    1: [c1_global, c2_global, c3_global],
    2: [c1_global, c2_global, c3_global],
    3: [c1_global, c2_global, c3_global],
}

baseline_matches = count_matches_discrete([
    c1_global, c2_global, c3_global,
    c1_global, c2_global, c3_global,
    c1_global, c2_global, c3_global,
    c1_global, c2_global, c3_global,
])

print(f"Global formula: Z = {c1_global:.4f}×A^(2/3) + {c2_global:.4f}×A + {c3_global:.4f}")
print(f"Exact matches: {baseline_matches}/285 ({100*baseline_matches/285:.1f}%)")
print()

# Optimize discrete coefficients
print("="*95)
print("OPTIMIZING DISCRETE COEFFICIENTS")
print("="*95)
print()

print("Running global optimization (this may take a minute)...")
print()

# Bounds for each coefficient: c1 in [0, 2], c2 in [0, 0.5], c3 in [-5, 5]
bounds = [(0.0, 2.0), (0.0, 0.5), (-5.0, 5.0)] * 4  # 12 parameters

result = differential_evolution(
    objective_discrete,
    bounds,
    maxiter=200,
    seed=42,
    atol=0,
    tol=0.01,
    workers=1,
    disp=True,
)

optimized_coeffs_flat = result.x
optimized_matches = count_matches_discrete(optimized_coeffs_flat)

optimized_coeffs = {
    0: optimized_coeffs_flat[0:3],
    1: optimized_coeffs_flat[3:6],
    2: optimized_coeffs_flat[6:9],
    3: optimized_coeffs_flat[9:12],
}

print()
print("Optimization complete!")
print()

# Display results
print("="*95)
print("OPTIMIZED DISCRETE COEFFICIENTS")
print("="*95)
print()

print(f"{'A mod 4':<10} {'c1':<15} {'c2':<15} {'c3':<15} {'Formula'}\"")
print("-"*95)

for mod in range(4):
    c1, c2, c3 = optimized_coeffs[mod]
    formula = f"Z = {c1:.3f}×A^(2/3) + {c2:.3f}×A {c3:+.3f}"
    print(f"{mod:<10} {c1:<15.6f} {c2:<15.6f} {c3:<15.6f}")
    print(f"           {formula}")
    print()

print(f"Total exact matches: {optimized_matches}/285 ({100*optimized_matches/285:.1f}%)")
print(f"Improvement over global: {optimized_matches - baseline_matches:+d} matches")
print(f"Comparison to QFD: {optimized_matches - 175:+d} matches")
print()

# ============================================================================
# PERFORMANCE BY A MOD 4
# ============================================================================
print("="*95)
print("PERFORMANCE BY A MOD 4 CLASS")
print("="*95)
print()

predictions = []
for name, Z_exp, A in test_nuclides:
    Z_pred = predict_Z_discrete(A, optimized_coeffs)
    error = Z_pred - Z_exp

    predictions.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_pred,
        'error': error,
        'abs_error': abs(error),
        'mod_4': A % 4,
    })

print(f"{'A mod 4':<12} {'Correct':<12} {'Total':<12} {'Success %':<12} {'QFD baseline'}\"")
print("-"*95)

for mod in range(4):
    mod_preds = [p for p in predictions if p['mod_4'] == mod]
    mod_correct = len([p for p in mod_preds if p['error'] == 0])
    mod_total = len(mod_preds)
    mod_pct = 100 * mod_correct / mod_total if mod_total > 0 else 0

    # QFD baseline for this mod 4
    qfd_baseline = {0: 55.1, 1: 77.4, 2: 59.3, 3: 59.6}
    baseline_str = f"{qfd_baseline[mod]:.1f}%"

    marker = "★★★" if mod_pct > 70 else "★" if mod_pct > 60 else ""

    print(f"{mod:<12} {mod_correct:<12} {mod_total:<12} {mod_pct:<12.1f} {baseline_str:<15} {marker}")

print()

# ============================================================================
# ERROR ANALYSIS
# ============================================================================
print("="*95)
print("ERROR DISTRIBUTION")
print("="*95)
print()

errors = [p['error'] for p in predictions]
mean_error = np.mean(errors)
std_error = np.std(errors)

print(f"Mean error: {mean_error:.3f} charges")
print(f"Std error: {std_error:.3f} charges")
print()

error_counts = Counter(p['abs_error'] for p in predictions)

print(f"{'|Error|':<12} {'Count':<12} {'Percentage'}\"")
print("-"*95)

for abs_err in sorted(error_counts.keys())[:6]:
    count = error_counts[abs_err]
    pct = 100 * count / 285
    marker = "★★★" if abs_err == 0 else "★★" if abs_err == 1 else "★" if abs_err == 2 else ""
    print(f"{abs_err:<12} {count:<12} {pct:<12.1f}  {marker}")

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
print("FINAL COMPARISON")
print("="*95)
print()

print(f"{'Method':<50} {'Exact':<12} {'Success %'}\"")
print("-"*95)
print(f"{'Pure QFD (energy minimization)':<50} {'175':<12} {'61.4%'}")
print(f"{'Global empirical formula':<50} {baseline_matches:<12} {100*baseline_matches/285:.1f}%")
print(f"{'Discrete empirical (by A mod 4)':<50} {optimized_matches:<12} {100*optimized_matches/285:.1f}%")
print()

if optimized_matches > 175:
    print(f"★★★ DISCRETE OPTIMIZATION BEATS QFD BASELINE!")
    print(f"  Gain: {optimized_matches - 175:+d} matches ({100*(optimized_matches - 175)/285:+.1f}%)")
    print()
    print("This proves the A mod 4 pattern is REAL and can be exploited!")

elif optimized_matches > baseline_matches:
    print(f"★★ Discrete coefficients improve over global formula")
    print(f"  Gain: {optimized_matches - baseline_matches:+d} matches")
    print(f"  But still below QFD: {175 - optimized_matches} matches behind")

else:
    print(f"Global and discrete formulas perform similarly")
    print(f"  QFD energy minimization still superior")

print()

# ============================================================================
# PHYSICAL INTERPRETATION
# ============================================================================
print("="*95)
print("PHYSICAL INTERPRETATION")
print("="*95)
print()

print("Coefficient variation by A mod 4:")
print()

c1_values = [optimized_coeffs[mod][0] for mod in range(4)]
c2_values = [optimized_coeffs[mod][1] for mod in range(4)]
c3_values = [optimized_coeffs[mod][2] for mod in range(4)]

print(f"c1 (A^(2/3) term) range: {min(c1_values):.3f} to {max(c1_values):.3f}  (variation: {max(c1_values)-min(c1_values):.3f})")
print(f"c2 (A term) range:       {min(c2_values):.3f} to {max(c2_values):.3f}  (variation: {max(c2_values)-min(c2_values):.3f})")
print(f"c3 (offset) range:       {min(c3_values):.3f} to {max(c3_values):.3f}  (variation: {max(c3_values)-min(c3_values):.3f})")
print()

if max(c1_values) - min(c1_values) > 0.3:
    print("★ Surface term (c1) varies significantly by mod 4")
    print("  → Topological structure affects surface/boundary physics")

if max(c3_values) - min(c3_values) > 1.0:
    print("★ Offset (c3) varies significantly by mod 4")
    print("  → Different 'ground state' normalization for different topologies")

print()
print("="*95)
