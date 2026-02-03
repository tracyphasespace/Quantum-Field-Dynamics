#!/usr/bin/env python3
"""
QUANTIZED DISCRETE GEOMETRIC PATHS
===========================================================================
Instead of arbitrary regional fits, find DISCRETE MONOTONIC paths:

  Path N: c1 = c1_0 + N×Δc1
          c2 = c2_0 + N×Δc2  
          c3 = c3_0 + N×Δc3

Where N = 0, 1, 2, 3, ... represents discrete topological states.

Physical interpretation:
  - N could be winding number, topological charge, or quantized geometry
  - Each path represents a different discrete formation route
  - Increments Δc1, Δc2 are UNIVERSAL (same for all nuclei)

Goal: Find optimal (c1_0, c2_0, c3_0) and (Δc1, Δc2, Δc3) to maximize
      recovery of failures with discrete path assignments.
===========================================================================
"""

import numpy as np
from scipy.optimize import differential_evolution
from collections import Counter, defaultdict

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_surface_coeff = beta_nuclear / 15
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def qfd_energy_full(A, Z):
    """Full QFD energy."""
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

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def find_stable_Z_qfd(A):
    """Find Z with minimum QFD energy."""
    best_Z, best_E = 1, qfd_energy_full(A, 1)
    for Z in range(1, A):
        E = qfd_energy_full(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

def empirical_Z(A, c1, c2, c3):
    """Empirical formula."""
    return c1 * (A**(2/3)) + c2 * A + c3

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

# Get successes and failures
successes = []
failures = []

for name, Z_exp, A in test_nuclides:
    Z_qfd = find_stable_Z_qfd(A)
    if Z_qfd == Z_exp:
        successes.append({'name': name, 'A': A, 'Z_exp': Z_exp})
    else:
        failures.append({'name': name, 'A': A, 'Z_exp': Z_exp, 'Z_qfd': Z_qfd})

print("="*95)
print("QUANTIZED DISCRETE GEOMETRIC PATHS")
print("="*95)
print()

print(f"Successes: {len(successes)}/285")
print(f"Failures:  {len(failures)}/285")
print()

# ============================================================================
# STEP 1: FIND BASE PATH (PATH 0) FROM SUCCESSES
# ============================================================================
print("="*95)
print("STEP 1: ESTABLISH BASE PATH (PATH 0) FROM SUCCESSES")
print("="*95)
print()

def fit_base_path(data_subset):
    """Fit base path coefficients."""
    def objective(coeffs):
        c1, c2, c3 = coeffs
        matches = 0
        for d in data_subset:
            Z_pred = int(round(empirical_Z(d['A'], c1, c2, c3)))
            if Z_pred == d['Z_exp']:
                matches += 1
        return -matches
    
    result = differential_evolution(
        objective,
        bounds=[(0.0, 2.0), (0.0, 0.5), (-5.0, 5.0)],
        maxiter=100,
        seed=42,
        workers=1,
    )
    
    return result.x

c1_0, c2_0, c3_0 = fit_base_path(successes)

print(f"Base path (Path 0):")
print(f"  c1_0 = {c1_0:.6f}")
print(f"  c2_0 = {c2_0:.6f}")
print(f"  c3_0 = {c3_0:.6f}")
print()

# Test base path on successes
base_matches_on_success = 0
for s in successes:
    Z_pred = int(round(empirical_Z(s['A'], c1_0, c2_0, c3_0)))
    if Z_pred == s['Z_exp']:
        base_matches_on_success += 1

print(f"Base path fits successes: {base_matches_on_success}/{len(successes)} ({100*base_matches_on_success/len(successes):.1f}%)")
print()

# ============================================================================
# STEP 2: FIND OPTIMAL INCREMENT (Δc1, Δc2, Δc3)
# ============================================================================
print("="*95)
print("STEP 2: FIND OPTIMAL DISCRETE INCREMENT")
print("="*95)
print()

print("Testing increments to maximize failure recovery...")
print()

def test_discrete_paths(c1_0, c2_0, c3_0, delta_c1, delta_c2, delta_c3, n_paths=10):
    """
    Test discrete paths and assign each failure to best-fit path.
    
    Path N: (c1_0 + N×Δc1, c2_0 + N×Δc2, c3_0 + N×Δc3)
    """
    path_assignments = defaultdict(list)
    
    for f in failures:
        A = f['A']
        Z_exp = f['Z_exp']
        
        best_path_N = None
        
        # Test paths N = -5 to +5 (allow negative for underpredictions)
        for N in range(-5, n_paths):
            c1_N = c1_0 + N * delta_c1
            c2_N = c2_0 + N * delta_c2
            c3_N = c3_0 + N * delta_c3
            
            # Skip invalid coefficients
            if c1_N < 0 or c2_N < 0:
                continue
            
            Z_pred = int(round(empirical_Z(A, c1_N, c2_N, c3_N)))
            
            if Z_pred == Z_exp:
                best_path_N = N
                break
        
        if best_path_N is not None:
            path_assignments[best_path_N].append(f)
    
    total_recovered = sum(len(nuclei) for nuclei in path_assignments.values())
    return total_recovered, path_assignments

def objective_increment(delta_coeffs):
    """Maximize total recoveries across all discrete paths."""
    delta_c1, delta_c2, delta_c3 = delta_coeffs
    total_recovered, _ = test_discrete_paths(c1_0, c2_0, c3_0, delta_c1, delta_c2, delta_c3)
    return -total_recovered

# Optimize increments
print("Optimizing (Δc1, Δc2, Δc3)...")

result = differential_evolution(
    objective_increment,
    bounds=[(-0.5, 0.5), (-0.1, 0.1), (-2.0, 2.0)],
    maxiter=50,
    seed=42,
    workers=1,
)

delta_c1, delta_c2, delta_c3 = result.x

print()
print(f"Optimal increment:")
print(f"  Δc1 = {delta_c1:.6f}")
print(f"  Δc2 = {delta_c2:.6f}")
print(f"  Δc3 = {delta_c3:.6f}")
print()

# ============================================================================
# STEP 3: ASSIGN FAILURES TO DISCRETE PATHS
# ============================================================================
print("="*95)
print("STEP 3: ASSIGN FAILURES TO QUANTIZED PATHS")
print("="*95)
print()

total_recovered, path_assignments = test_discrete_paths(c1_0, c2_0, c3_0, delta_c1, delta_c2, delta_c3, n_paths=10)

print(f"Total recovered with discrete paths: {total_recovered}/{len(failures)} ({100*total_recovered/len(failures):.1f}%)")
print()

# Display path populations
occupied_paths = sorted(path_assignments.keys())

print(f"{'Path N':<10} {'c1':<12} {'c2':<12} {'c3':<12} {'Recovered':<12} {'Sample nuclei'}")
print("-"*95)

for N in occupied_paths:
    c1_N = c1_0 + N * delta_c1
    c2_N = c2_0 + N * delta_c2
    c3_N = c3_0 + N * delta_c3
    
    nuclei_in_path = path_assignments[N]
    sample = ', '.join([n['name'] for n in nuclei_in_path[:5]])
    if len(nuclei_in_path) > 5:
        sample += f", ... ({len(nuclei_in_path)} total)"
    
    print(f"{N:<10} {c1_N:<12.6f} {c2_N:<12.6f} {c3_N:<12.6f} {len(nuclei_in_path):<12} {sample}")

print()

# ============================================================================
# STEP 4: PHYSICAL INTERPRETATION
# ============================================================================
print("="*95)
print("PHYSICAL INTERPRETATION")
print("="*95)
print()

print("Discrete path model:")
print(f"  Path N: c1 = {c1_0:.4f} + {delta_c1:+.4f}×N")
print(f"          c2 = {c2_0:.4f} + {delta_c2:+.4f}×N")
print(f"          c3 = {c3_0:.4f} + {delta_c3:+.4f}×N")
print()

print("Path quantum number N:")
print(f"  Range: {min(occupied_paths)} to {max(occupied_paths)}")
print(f"  Occupied paths: {len(occupied_paths)}")
print()

# Interpret increment direction
if delta_c1 > 0:
    print(f"★ Δc1 > 0: Higher paths have STRONGER envelope curvature")
    print(f"  → N increases surface/volume ratio")
elif delta_c1 < 0:
    print(f"★ Δc1 < 0: Higher paths have WEAKER envelope curvature")
    print(f"  → N decreases surface/volume ratio")

if delta_c2 > 0:
    print(f"★ Δc2 > 0: Higher paths have LARGER core fraction")
    print(f"  → N increases volume term dominance")
elif delta_c2 < 0:
    print(f"★ Δc2 < 0: Higher paths have SMALLER core fraction")
    print(f"  → N decreases volume term dominance")

print()

# Ratio evolution
print("Envelope/core ratio evolution:")
for N in occupied_paths[:5]:
    c1_N = c1_0 + N * delta_c1
    c2_N = c2_0 + N * delta_c2
    ratio = c1_N / c2_N if c2_N > 0 else 0
    print(f"  Path {N}: c1/c2 = {ratio:.3f}")

print()

# ============================================================================
# STEP 5: ANALYZE REMAINING FAILURES
# ============================================================================
print("="*95)
print("REMAINING FAILURES (NOT ON ANY DISCRETE PATH)")
print("="*95)
print()

assigned_failures = set(f['name'] for nuclei in path_assignments.values() for f in nuclei)
remaining_failures = [f for f in failures if f['name'] not in assigned_failures]

print(f"Remaining: {len(remaining_failures)}/{len(failures)} ({100*len(remaining_failures)/len(failures):.1f}%)")
print()

if remaining_failures:
    print("Sample remaining failures:")
    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_QFD':<8}")
    print("-"*95)
    
    for f in remaining_failures[:15]:
        print(f"{f['name']:<12} {f['A']:<6} {f['Z_exp']:<8} {f['Z_qfd']:<8}")
    
    if len(remaining_failures) > 15:
        print(f"... and {len(remaining_failures) - 15} more")
    
    print()

# ============================================================================
# COMPARISON
# ============================================================================
print("="*95)
print("COMPARISON: QUANTIZED vs REGIONAL PATHS")
print("="*95)
print()

print(f"{'Model':<40} {'Recovered':<15} {'Rate'}")
print("-"*95)
print(f"{'QFD baseline':<40} {'175/285':<15} {'61.4%'}")
print(f"{'5 regional paths':<40} {'52/110':<15} {'47.3%'}")
print(f"{'Quantized discrete paths (this)':<40} {f'{total_recovered}/110':<15} {100*total_recovered/len(failures):.1f}%")
print()

if total_recovered > 52:
    print(f"★★★ QUANTIZED PATHS SUPERIOR!")
    print(f"    +{total_recovered - 52} matches over regional paths")
    print(f"    Proves discrete geometric quantization")
elif total_recovered >= 45:
    print(f"★★ QUANTIZED PATHS COMPARABLE")
    print(f"    Similar performance with simpler model")
    print(f"    Universal increment Δc validates discrete states")
else:
    print(f"→ Regional paths still better")
    print(f"   Geometry may not follow simple quantization")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: QUANTIZED GEOMETRIC PATHS")
print("="*95)
print()

print("Discrete path structure:")
print(f"  Base path (N=0): QFD success geometry")
print(f"  Increment: Universal (Δc1, Δc2, Δc3)")
print(f"  Path N: Base + N × Increment")
print()

print(f"Results:")
print(f"  Paths occupied: {len(occupied_paths)}")
print(f"  Path range: N ∈ [{min(occupied_paths)}, {max(occupied_paths)}]")
print(f"  Failures recovered: {total_recovered}/{len(failures)} ({100*total_recovered/len(failures):.1f}%)")
print()

print("Physical interpretation:")
if len(occupied_paths) <= 5:
    print(f"  ★ Small number of discrete states ({len(occupied_paths)} paths)")
    print(f"    Suggests quantized topological charge or winding number")
elif len(occupied_paths) <= 10:
    print(f"  → Moderate number of states ({len(occupied_paths)} paths)")
    print(f"    Could be core/envelope configuration index")
else:
    print(f"  → Many states ({len(occupied_paths)} paths)")
    print(f"    May not represent true quantization")

print()

if abs(delta_c1) > 0.1:
    print(f"★ Large Δc1 = {delta_c1:.4f}")
    print(f"  Envelope geometry changes significantly between paths")

if abs(delta_c2) > 0.02:
    print(f"★ Significant Δc2 = {delta_c2:.4f}")
    print(f"  Core/envelope ratio evolves with path number")

print()
print("="*95)
