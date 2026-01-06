#!/usr/bin/env python3
"""
15-PATH GEOMETRIC QUANTIZATION MODEL
================================================================================
Hypothesis: Fundamental quantum is ΔN = 0.5, giving 15 paths total

Integer paths (N = -3, -2, -1, 0, +1, +2, +3): Even-even nuclei
Half-integer paths (N = -3.5, -2.5, ..., +3.5): Odd-A nuclei

Path N ∈ {-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
          +0.5, +1.0, +1.5, +2.0, +2.5, +3.0, +3.5}

Model: Z(A,N) = c₁(N)×A^(2/3) + c₂(N)×A + c₃(N)

where:
  c₁(N) = c₁⁰ + Δc₁ × N
  c₂(N) = c₂⁰ + Δc₂ × N
  c₃(N) = c₃⁰ + Δc₃ × N

Optimize 6 parameters: c₁⁰, c₂⁰, c₃⁰, Δc₁, Δc₂, Δc₃
================================================================================
"""

import numpy as np
from scipy.optimize import differential_evolution
from collections import defaultdict

# Load stable nuclei data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*80)
print("15-PATH GEOMETRIC QUANTIZATION OPTIMIZATION")
print("="*80)
print(f"\nLoaded {len(test_nuclides)} stable nuclei")
print(f"Path quantum number N ∈ {{-3.5, -3.0, -2.5, ..., +3.0, +3.5}} (15 paths)")
print(f"Increment: ΔN = 0.5")
print()

# Define 15 path values
PATH_VALUES = np.arange(-3.5, 4.0, 0.5)  # -3.5 to +3.5 in steps of 0.5
print(f"Path values: {PATH_VALUES}")
print(f"Total paths: {len(PATH_VALUES)}")
print()

def get_path_coefficients(c1_0, c2_0, c3_0, dc1, dc2, dc3, N):
    """Get coefficients for path N."""
    c1_N = c1_0 + dc1 * N
    c2_N = c2_0 + dc2 * N
    c3_N = c3_0 + dc3 * N
    return c1_N, c2_N, c3_N

def predict_Z_path_N(A, N, c1_0, c2_0, c3_0, dc1, dc2, dc3):
    """Predict Z using path N."""
    c1, c2, c3 = get_path_coefficients(c1_0, c2_0, c3_0, dc1, dc2, dc3, N)
    Z_pred = c1 * (A**(2/3)) + c2 * A + c3
    return int(round(Z_pred))

def classify_nucleus(A, Z_exp, c1_0, c2_0, c3_0, dc1, dc2, dc3):
    """
    Classify nucleus into one of 15 paths.
    Returns path N if found, else None.
    """
    for N in PATH_VALUES:
        Z_pred = predict_Z_path_N(A, N, c1_0, c2_0, c3_0, dc1, dc2, dc3)
        if Z_pred == Z_exp:
            return N
    return None

def objective_function(params):
    """
    Objective: Maximize correct classifications.
    Returns negative count (for minimization).
    """
    c1_0, c2_0, c3_0, dc1, dc2, dc3 = params

    correct = 0
    for name, Z, A in test_nuclides:
        N_assigned = classify_nucleus(A, Z, c1_0, c2_0, c3_0, dc1, dc2, dc3)
        if N_assigned is not None:
            correct += 1

    # Return negative count (we're minimizing, but want to maximize correct)
    return -correct

# Initial guess from 7-path model, but with half increments
# Original: c1_0=0.9618, dc1=-0.0295, c2_0=0.2475, dc2=+0.0064, c3_0=-2.4107, dc3=-0.8653
# For 15 paths with ΔN=0.5, increments should be approximately half
c1_0_init = 0.9618
c2_0_init = 0.2475
c3_0_init = -2.4107
dc1_init = -0.0295 / 2  # Half the original increment
dc2_init = +0.0064 / 2  # Half the original increment
dc3_init = -0.8653 / 2  # Half the original increment

print("Initial guess (from 7-path model scaled):")
print(f"  c₁⁰ = {c1_0_init:.6f}, Δc₁ = {dc1_init:.6f}")
print(f"  c₂⁰ = {c2_0_init:.6f}, Δc₂ = {dc2_init:.6f}")
print(f"  c₃⁰ = {c3_0_init:.6f}, Δc₃ = {dc3_init:.6f}")
print()

# Test initial guess
initial_correct = -objective_function([c1_0_init, c2_0_init, c3_0_init,
                                       dc1_init, dc2_init, dc3_init])
print(f"Initial classification: {initial_correct}/285 ({100*initial_correct/285:.1f}%)")
print()

# Define parameter bounds (allow reasonable variation)
bounds = [
    (0.85, 1.05),     # c1_0
    (0.20, 0.30),     # c2_0
    (-3.5, -1.5),     # c3_0
    (-0.025, -0.005), # dc1 (negative, half of original range)
    (0.001, 0.008),   # dc2 (positive, half of original range)
    (-0.6, -0.3),     # dc3 (negative, half of original range)
]

print("Optimizing 15-path model...")
print("This may take 2-5 minutes...")
print()

result = differential_evolution(
    objective_function,
    bounds,
    maxiter=300,
    popsize=20,
    seed=42,
    atol=0,
    tol=0.001,
    workers=1,
    polish=True
)

c1_0_opt, c2_0_opt, c3_0_opt, dc1_opt, dc2_opt, dc3_opt = result.x
correct_opt = -result.fun

print("="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print()
print("Optimized parameters:")
print(f"  c₁⁰ = {c1_0_opt:.6f}, Δc₁ = {dc1_opt:.6f}")
print(f"  c₂⁰ = {c2_0_opt:.6f}, Δc₂ = {dc2_opt:.6f}")
print(f"  c₃⁰ = {c3_0_opt:.6f}, Δc₃ = {dc3_opt:.6f}")
print()
print(f"Classification accuracy: {correct_opt}/285 ({100*correct_opt/285:.1f}%)")
print()

# Classify all nuclei with optimized parameters
path_populations = defaultdict(list)
failures = []

for name, Z, A in test_nuclides:
    N_assigned = classify_nucleus(A, Z, c1_0_opt, c2_0_opt, c3_0_opt,
                                   dc1_opt, dc2_opt, dc3_opt)

    neutrons = A - Z
    parity_type = 'even-even' if (Z % 2 == 0 and neutrons % 2 == 0) else 'odd-A'

    if N_assigned is not None:
        path_populations[N_assigned].append({
            'name': name,
            'A': A,
            'Z': Z,
            'parity': parity_type
        })
    else:
        failures.append({'name': name, 'A': A, 'Z': Z, 'parity': parity_type})

# Analyze path populations
print("="*80)
print("PATH POPULATIONS")
print("="*80)
print()

print(f"{'Path N':<10} {'Integer?':<12} {'Count':<8} {'Even-Even':<12} {'Odd-A':<8} {'% Even-Even'}")
print("-"*80)

total_integer_ee = 0
total_integer_odd = 0
total_half_ee = 0
total_half_odd = 0

for N in PATH_VALUES:
    is_integer = (N == int(N))
    count = len(path_populations[N])

    if count > 0:
        ee_count = sum(1 for nuc in path_populations[N] if nuc['parity'] == 'even-even')
        odd_count = count - ee_count
        pct_ee = 100 * ee_count / count

        # Track totals
        if is_integer:
            total_integer_ee += ee_count
            total_integer_odd += odd_count
        else:
            total_half_ee += ee_count
            total_half_odd += odd_count

        int_str = "Yes" if is_integer else "No"
        print(f"{N:^10.1f} {int_str:<12} {count:^8d} {ee_count:^12d} {odd_count:^8d} {pct_ee:>6.1f}%")

print("-"*80)
print(f"{'INTEGER PATHS':<10} {'':<12} {total_integer_ee + total_integer_odd:^8d} {total_integer_ee:^12d} {total_integer_odd:^8d} {100*total_integer_ee/(total_integer_ee+total_integer_odd) if (total_integer_ee+total_integer_odd)>0 else 0:>6.1f}%")
print(f"{'HALF-INT PATHS':<10} {'':<12} {total_half_ee + total_half_odd:^8d} {total_half_ee:^12d} {total_half_odd:^8d} {100*total_half_ee/(total_half_ee+total_half_odd) if (total_half_ee+total_half_odd)>0 else 0:>6.1f}%")

if len(failures) > 0:
    print()
    print(f"Unclassified: {len(failures)} nuclei")
    for f in failures[:5]:
        print(f"  {f['name']} (A={f['A']}, Z={f['Z']}, {f['parity']})")

print()
print("="*80)
print("HYPOTHESIS TEST: Integer vs Half-Integer Paths")
print("="*80)
print()
print("Hypothesis: Integer N (e.g., -3, -2, -1, 0, +1, +2, +3) prefer even-even")
print("           Half-integer N (e.g., -3.5, -2.5, ..., +3.5) prefer odd-A")
print()

total_integer = total_integer_ee + total_integer_odd
total_half = total_half_ee + total_half_odd

if total_integer > 0 and total_half > 0:
    pct_int_ee = 100 * total_integer_ee / total_integer
    pct_half_ee = 100 * total_half_ee / total_half

    print(f"Integer paths (N ∈ {{-3, -2, -1, 0, +1, +2, +3}}):")
    print(f"  Total: {total_integer} nuclei")
    print(f"  Even-even: {total_integer_ee} ({pct_int_ee:.1f}%)")
    print(f"  Odd-A: {total_integer_odd} ({100-pct_int_ee:.1f}%)")
    print()
    print(f"Half-integer paths (N ∈ {{-3.5, -2.5, ..., +3.5}}):")
    print(f"  Total: {total_half} nuclei")
    print(f"  Even-even: {total_half_ee} ({pct_half_ee:.1f}%)")
    print(f"  Odd-A: {total_half_odd} ({100-pct_half_ee:.1f}%)")
    print()

    if pct_int_ee > pct_half_ee + 10:
        print("✓ HYPOTHESIS CONFIRMED:")
        print(f"  Integer paths show {pct_int_ee - pct_half_ee:.1f}% stronger even-even preference")
        print(f"  → Integer N represent even-even fundamental sector")
        print(f"  → Half-integer N represent odd-A intermediate states")
    elif pct_half_ee > pct_int_ee + 10:
        print("✗ HYPOTHESIS REVERSED:")
        print(f"  Half-integer paths show stronger even-even preference")
    else:
        print("⚠ HYPOTHESIS UNCLEAR:")
        print(f"  Both path types show similar parity distributions")
else:
    print("Insufficient data for hypothesis test")

print()
print("="*80)
print("COMPARISON TO 7-PATH MODEL")
print("="*80)
print()
print(f"7-path model:  285/285 = 100% (7 paths, ΔN = 1.0)")
print(f"15-path model: {correct_opt}/285 = {100*correct_opt/285:.1f}% (15 paths, ΔN = 0.5)")
print()

if correct_opt == 285:
    print("✓ 15-path model maintains 100% classification!")
    print("  → Finer geometric resolution with no loss of accuracy")
elif correct_opt >= 280:
    print("✓ 15-path model maintains near-perfect classification")
else:
    print("⚠ 15-path model shows reduced accuracy")
    print("  → May need different parameterization")

# Save results
print()
print("="*80)
print("Saving results to: optimized_15path_results.txt")

with open('optimized_15path_results.txt', 'w') as f:
    f.write("15-PATH GEOMETRIC QUANTIZATION MODEL - OPTIMIZED PARAMETERS\n")
    f.write("="*80 + "\n\n")
    f.write(f"c₁(N) = {c1_0_opt:.6f} + ({dc1_opt:+.6f})×N\n")
    f.write(f"c₂(N) = {c2_0_opt:.6f} + ({dc2_opt:+.6f})×N\n")
    f.write(f"c₃(N) = {c3_0_opt:.6f} + ({dc3_opt:+.6f})×N\n\n")
    f.write(f"Path N ∈ {{-3.5, -3.0, -2.5, ..., +3.0, +3.5}} (15 paths)\n")
    f.write(f"Increment: ΔN = 0.5\n\n")
    f.write(f"Classification: {correct_opt}/285 ({100*correct_opt/285:.1f}%)\n\n")
    f.write(f"Path populations:\n")
    for N in PATH_VALUES:
        count = len(path_populations[N])
        if count > 0:
            ee = sum(1 for n in path_populations[N] if n['parity'] == 'even-even')
            f.write(f"  N={N:+5.1f}: {count:3d} nuclei ({ee} even-even, {count-ee} odd-A)\n")

print("Complete!")
print("="*80)
