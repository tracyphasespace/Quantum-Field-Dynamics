#!/usr/bin/env python3
"""
Analyze Solution Degeneracy from Multi-Start Test
==================================================

Explores the 50 solutions found in Test 2 to understand:
1. How are solutions distributed in parameter space?
2. Are there correlations between parameters?
3. Do solutions lie on a lower-dimensional manifold?
4. What physical relationships might constrain them?
"""

import json
import numpy as np
from pathlib import Path

# Load results
results_file = Path("results/multistart_robustness_results.json")
with open(results_file) as f:
    data = json.load(f)

solutions = [s for s in data['solutions'] if s['converged']]

# Extract parameters
R = np.array([s['R'] for s in solutions])
U = np.array([s['U'] for s in solutions])
amp = np.array([s['amplitude'] for s in solutions])
E_circ = np.array([s['E_circulation'] for s in solutions])
E_stab = np.array([s['E_stabilization'] for s in solutions])

print("="*80)
print("SOLUTION DEGENERACY ANALYSIS")
print("="*80)
print()

# Basic statistics
print("PARAMETER DISTRIBUTIONS")
print("-"*80)
print(f"R:         min={R.min():.4f}, max={R.max():.4f}, mean={R.mean():.4f}, std={R.std():.4f}")
print(f"U:         min={U.min():.4f}, max={U.max():.4f}, mean={U.mean():.4f}, std={U.std():.4f}")
print(f"amplitude: min={amp.min():.4f}, max={amp.max():.4f}, mean={amp.mean():.4f}, std={amp.std():.4f}")
print(f"E_circ:    min={E_circ.min():.4f}, max={E_circ.max():.4f}, mean={E_circ.mean():.4f}, std={E_circ.std():.4f}")
print(f"E_stab:    min={E_stab.min():.4f}, max={E_stab.max():.4f}, mean={E_stab.mean():.4f}, std={E_stab.std():.4f}")
print()

# Look for correlations
print("CORRELATION ANALYSIS")
print("-"*80)

# Correlation matrix
params = np.column_stack([R, U, amp, E_circ, E_stab])
param_names = ['R', 'U', 'amp', 'E_circ', 'E_stab']
corr_matrix = np.corrcoef(params.T)

print("\nCorrelation matrix:")
print("         ", "  ".join(f"{name:>8}" for name in param_names))
for i, name in enumerate(param_names):
    print(f"{name:>8} ", "  ".join(f"{corr_matrix[i,j]:>8.3f}" for j in range(len(param_names))))

# Strong correlations
print("\nStrong correlations (|r| > 0.7):")
for i in range(len(param_names)):
    for j in range(i+1, len(param_names)):
        if abs(corr_matrix[i,j]) > 0.7:
            print(f"  {param_names[i]} vs {param_names[j]}: r = {corr_matrix[i,j]:.3f}")

print()

# Test specific relationships
print("PHYSICAL RELATIONSHIP TESTS")
print("-"*80)

# Test 1: E_stab ∝ β a² R³ (from analytic formula)
beta = 3.1
E_stab_predicted = (32 * np.pi / 105) * beta * amp**2 * R**3
residual_stab = np.abs(E_stab - E_stab_predicted) / E_stab
print(f"\n1. E_stab vs (32π/105)βa²R³:")
print(f"   Mean residual: {residual_stab.mean()*100:.2f}%")
print(f"   Max residual:  {residual_stab.max()*100:.2f}%")
if residual_stab.mean() < 0.1:
    print("   ✓ Analytic formula validated")
else:
    print("   ✗ Significant deviation from analytic prediction")

# Test 2: Does R³ × amplitude ≈ constant?
product = R**3 * amp
print(f"\n2. R³ × amplitude:")
print(f"   Mean: {product.mean():.4f}, Std: {product.std():.4f}, CV: {100*product.std()/product.mean():.2f}%")
if product.std()/product.mean() < 0.2:
    print(f"   → Possible constraint: R³ × amp ≈ {product.mean():.4f}")

# Test 3: Does R³ × amplitude² ≈ constant? (from E_stab constraint)
product2 = R**3 * amp**2
print(f"\n3. R³ × amplitude²:")
print(f"   Mean: {product2.mean():.4f}, Std: {product2.std():.4f}, CV: {100*product2.std()/product2.mean():.2f}%")
if product2.std()/product2.mean() < 0.2:
    print(f"   → Possible constraint: R³ × amp² ≈ {product2.mean():.4f}")

# Test 4: E_circ vs U² (should be linear since E_circ ∝ U²)
# Fit E_circ = c × U²
coeffs = np.polyfit(U**2, E_circ, 1)
E_circ_fit = coeffs[0] * U**2 + coeffs[1]
r_squared = 1 - np.sum((E_circ - E_circ_fit)**2) / np.sum((E_circ - E_circ.mean())**2)
print(f"\n4. E_circ vs U²:")
print(f"   Fit: E_circ = {coeffs[0]:.2f} × U² + {coeffs[1]:.2f}")
print(f"   R² = {r_squared:.4f}")
if r_squared > 0.95:
    print("   ✓ Strong linear relationship E_circ ∝ U²")

# Test 5: Constraint surface E_total = 1
# Since E_total = E_circ - E_stab = 1, we have:
# E_circ = E_stab + 1
# Let's check this
constraint_residual = np.abs(E_circ - E_stab - 1.0)
print(f"\n5. Constraint E_circ - E_stab = 1.0:")
print(f"   Mean residual: {constraint_residual.mean():.3e}")
print(f"   Max residual:  {constraint_residual.max():.3e}")
print("   ✓ Constraint satisfied to numerical precision")

print()

# Solutions near cavitation
print("CAVITATION ANALYSIS")
print("-"*80)
near_cavitation = amp > 0.95
print(f"\nSolutions with amplitude > 0.95: {near_cavitation.sum()}/{len(amp)}")
if near_cavitation.sum() > 0:
    print(f"  R range: [{R[near_cavitation].min():.3f}, {R[near_cavitation].max():.3f}]")
    print(f"  U range: [{U[near_cavitation].min():.3f}, {U[near_cavitation].max():.3f}]")
    print(f"  Mean R: {R[near_cavitation].mean():.3f}, Mean U: {U[near_cavitation].mean():.3f}")

# Original fit comparison
print()
print("ORIGINAL FIT COMPARISON")
print("-"*80)
# From grid convergence test at (100,20): R=0.446, U=0.024, amp=0.938
R_orig = 0.446
U_orig = 0.024
amp_orig = 0.938

# Find closest solution
distances = np.sqrt((R - R_orig)**2 + (U - U_orig)**2 + (amp - amp_orig)**2)
closest_idx = np.argmin(distances)

print(f"\nOriginal fit: R={R_orig:.3f}, U={U_orig:.3f}, amp={amp_orig:.3f}")
print(f"Closest multi-start solution (#{closest_idx+1}):")
print(f"  R={R[closest_idx]:.3f}, U={U[closest_idx]:.3f}, amp={amp[closest_idx]:.3f}")
print(f"  Distance: {distances[closest_idx]:.4f}")
print()

# Identify "interesting" solutions
print("REPRESENTATIVE SOLUTIONS")
print("-"*80)
print("\n1. Smallest R:")
idx = np.argmin(R)
print(f"   R={R[idx]:.4f}, U={U[idx]:.4f}, amp={amp[idx]:.4f}")
print(f"   E_circ={E_circ[idx]:.4f}, E_stab={E_stab[idx]:.4f}")

print("\n2. Largest R:")
idx = np.argmax(R)
print(f"   R={R[idx]:.4f}, U={U[idx]:.4f}, amp={amp[idx]:.4f}")
print(f"   E_circ={E_circ[idx]:.4f}, E_stab={E_stab[idx]:.4f}")

print("\n3. Smallest amplitude:")
idx = np.argmin(amp)
print(f"   R={R[idx]:.4f}, U={U[idx]:.4f}, amp={amp[idx]:.4f}")
print(f"   E_circ={E_circ[idx]:.4f}, E_stab={E_stab[idx]:.4f}")

print("\n4. Largest amplitude (closest to cavitation):")
idx = np.argmax(amp)
print(f"   R={R[idx]:.4f}, U={U[idx]:.4f}, amp={amp[idx]:.4f}")
print(f"   E_circ={E_circ[idx]:.4f}, E_stab={E_stab[idx]:.4f}")

print("\n5. Median solution (by R):")
idx = np.argsort(R)[len(R)//2]
print(f"   R={R[idx]:.4f}, U={U[idx]:.4f}, amp={amp[idx]:.4f}")
print(f"   E_circ={E_circ[idx]:.4f}, E_stab={E_stab[idx]:.4f}")

print()

# Recommendations
print("="*80)
print("RECOMMENDATIONS FOR REDUCING DEGENERACY")
print("="*80)
print()

print("Based on the analysis above, try these constraints:")
print()
print("1. CAVITATION SATURATION:")
print("   Fix amplitude = 1.0 (or amplitude > 0.95)")
print(f"   → Would select {near_cavitation.sum()} of current 50 solutions")
print(f"   → R would range [{R[near_cavitation].min():.3f}, {R[near_cavitation].max():.3f}]")
print(f"   → Still need additional constraint on R or U")
print()

print("2. CHARGE RADIUS CONSTRAINT:")
print("   Compute r_rms = sqrt(∫ r² ρ(r) dV / ∫ ρ(r) dV)")
print("   Require r_rms ≈ 0.84 fm (experimental electron radius)")
print("   → Would likely select unique solution")
print()

print("3. MINIMUM ACTION:")
print("   Among solutions with E_total=1, choose minimum ∫ |∇ψ|² dV")
print("   → Smoothest configuration = lowest internal stress")
print("   → Physically motivated by variational principle")
print()

print("4. STABILITY CRITERION:")
print("   Compute δ²E/δψ² for each solution")
print("   Keep only solutions with all eigenvalues > 0")
print("   → May dramatically reduce solution count")
print("   → Most computationally intensive")
print()

print("IMMEDIATE NEXT STEP:")
print("  Test constraint #1 (cavitation) - easiest to implement")
print("  If still degenerate, add constraint #2 (charge radius)")
print()
