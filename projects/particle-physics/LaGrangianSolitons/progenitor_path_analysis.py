#!/usr/bin/env python3
"""
PROGENITOR PATH ANALYSIS - SUCCESSES VS FAILURES
===========================================================================
Hypothesis: Successes and failures follow DIFFERENT geometric pathways.

Test:
1. Separate nuclei into successes (QFD correct) and failures (QFD wrong)
2. Fit empirical coefficients separately for each group
3. Plot both populations in (A, Z) space
4. Check if failures cluster along different geometry

This reveals if there are multiple "formation paths" with different
core/envelope structures.
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
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

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("PROGENITOR PATH ANALYSIS - SUCCESSES VS FAILURES")
print("="*95)
print()

# Separate into successes and failures
successes = []
failures = []

for name, Z_exp, A in test_nuclides:
    Z_qfd = find_stable_Z_qfd(A)
    error = Z_qfd - Z_exp
    
    data_point = {
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_qfd': Z_qfd,
        'error': error,
    }
    
    if error == 0:
        successes.append(data_point)
    else:
        failures.append(data_point)

print(f"Successes: {len(successes)}/285 ({100*len(successes)/285:.1f}%)")
print(f"Failures:  {len(failures)}/285 ({100*len(failures)/285:.1f}%)")
print()

# ============================================================================
# FIT COEFFICIENTS SEPARATELY FOR SUCCESSES AND FAILURES
# ============================================================================
print("="*95)
print("SEPARATE GEOMETRIC PATHWAYS")
print("="*95)
print()

def empirical_Z(A, c1, c2, c3):
    """Empirical formula."""
    return c1 * (A**(2/3)) + c2 * A + c3

def fit_coeffs(data_subset):
    """Fit c1, c2, c3 to maximize exact matches."""
    def objective(coeffs):
        c1, c2, c3 = coeffs
        matches = 0
        for d in data_subset:
            Z_pred = int(round(empirical_Z(d['A'], c1, c2, c3)))
            if Z_pred == d['Z_exp']:
                matches += 1
        return -matches  # Minimize negative
    
    result = differential_evolution(
        objective,
        bounds=[(0.0, 2.0), (0.0, 0.5), (-5.0, 5.0)],
        maxiter=150,
        seed=42,
        workers=1,
    )
    
    return result.x

# Fit separately
print("Fitting coefficients for SUCCESSES only...")
c1_success, c2_success, c3_success = fit_coeffs(successes)

print("Fitting coefficients for FAILURES only...")
c1_failure, c2_failure, c3_failure = fit_coeffs(failures)

print()
print(f"{'Group':<15} {'c1':<15} {'c2':<15} {'c3':<15} {'Ratio c1/c2'}")
print("-"*95)
print(f"{'Successes':<15} {c1_success:<15.6f} {c2_success:<15.6f} {c3_success:<15.6f} {c1_success/c2_success:.3f}")
print(f"{'Failures':<15} {c1_failure:<15.6f} {c2_failure:<15.6f} {c3_failure:<15.6f} {c1_failure/c2_failure:.3f}")
print()

# Compare
delta_c1 = abs(c1_success - c1_failure) / c1_success
delta_c2 = abs(c2_success - c2_failure) / c2_success

print("Difference between paths:")
print(f"  Δc1: {100*delta_c1:.1f}%")
print(f"  Δc2: {100*delta_c2:.1f}%")
print()

if delta_c1 > 0.1 or delta_c2 > 0.1:
    print("★★★ DIFFERENT PROGENITOR PATHS DETECTED!")
    print()
    if delta_c1 > delta_c2:
        print(f"  Primary difference: c1 (surface/envelope geometry)")
        print(f"  Successes have different envelope curvature than failures")
    else:
        print(f"  Primary difference: c2 (core/volume ratio)")
        print(f"  Successes have different core structure than failures")
else:
    print("→ No significant difference in geometric pathways")

print()

# ============================================================================
# TEST: CAN FAILURE COEFFICIENTS PREDICT FAILURES?
# ============================================================================
print("="*95)
print("TEST: DO FAILURE COEFFICIENTS FIT FAILURES BETTER?")
print("="*95)
print()

def test_coeffs_on_data(data_subset, c1, c2, c3):
    """Count how many exact matches with given coefficients."""
    matches = 0
    for d in data_subset:
        Z_pred = int(round(empirical_Z(d['A'], c1, c2, c3)))
        if Z_pred == d['Z_exp']:
            matches += 1
    return matches

# Test success coeffs on successes
success_on_success = test_coeffs_on_data(successes, c1_success, c2_success, c3_success)

# Test failure coeffs on failures
failure_on_failure = test_coeffs_on_data(failures, c1_failure, c2_failure, c3_failure)

# Cross-test
success_on_failure = test_coeffs_on_data(failures, c1_success, c2_success, c3_success)
failure_on_success = test_coeffs_on_data(successes, c1_failure, c2_failure, c3_failure)

print(f"{'Coefficients':<20} {'On Successes':<15} {'On Failures':<15}")
print("-"*95)
print(f"{'Success coeffs':<20} {success_on_success}/{len(successes):<15} {success_on_failure}/{len(failures):<15}")
print(f"{'Failure coeffs':<20} {failure_on_success}/{len(successes):<15} {failure_on_failure}/{len(failures):<15}")
print()

if failure_on_failure > success_on_failure:
    print(f"★★ Failure coefficients fit failures BETTER!")
    print(f"  Improvement: {failure_on_failure - success_on_failure} matches")
    print(f"  → Failures follow a distinct geometric pathway")
else:
    print(f"→ No improvement - failures don't follow consistent alternative path")

print()

# ============================================================================
# PLOT: VISUALIZE BOTH PATHWAYS
# ============================================================================
print("="*95)
print("CREATING VISUALIZATION...")
print("="*95)
print()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: A vs Z scatter (successes vs failures)
ax1 = axes[0, 0]
A_success = [d['A'] for d in successes]
Z_success = [d['Z_exp'] for d in successes]
A_failure = [d['A'] for d in failures]
Z_failure = [d['Z_exp'] for d in failures]

ax1.scatter(A_success, Z_success, c='green', alpha=0.6, s=30, label='QFD Successes')
ax1.scatter(A_failure, Z_failure, c='red', alpha=0.6, s=30, label='QFD Failures')

# Plot fitted paths
A_range = np.linspace(1, 240, 200)
Z_success_fit = empirical_Z(A_range, c1_success, c2_success, c3_success)
Z_failure_fit = empirical_Z(A_range, c1_failure, c2_failure, c3_failure)

ax1.plot(A_range, Z_success_fit, 'g--', linewidth=2, label='Success path')
ax1.plot(A_range, Z_failure_fit, 'r--', linewidth=2, label='Failure path')

ax1.set_xlabel('Mass Number A', fontsize=12)
ax1.set_ylabel('Proton Number Z', fontsize=12)
ax1.set_title('Progenitor Paths: Successes vs Failures', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals (Z_exp - Z_fit) for successes
ax2 = axes[0, 1]
residuals_success = []
for d in successes:
    Z_fit = empirical_Z(d['A'], c1_success, c2_success, c3_success)
    residuals_success.append(d['Z_exp'] - Z_fit)

A_success_res = [d['A'] for d in successes]
ax2.scatter(A_success_res, residuals_success, c='green', alpha=0.5, s=20)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax2.set_xlabel('Mass Number A', fontsize=12)
ax2.set_ylabel('Z_exp - Z_fit (charges)', fontsize=12)
ax2.set_title('Residuals: Successes from Success Path', fontsize=14)
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals for failures
ax3 = axes[1, 0]
residuals_failure = []
for d in failures:
    Z_fit = empirical_Z(d['A'], c1_failure, c2_failure, c3_failure)
    residuals_failure.append(d['Z_exp'] - Z_fit)

A_failure_res = [d['A'] for d in failures]
ax3.scatter(A_failure_res, residuals_failure, c='red', alpha=0.5, s=20)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax3.set_xlabel('Mass Number A', fontsize=12)
ax3.set_ylabel('Z_exp - Z_fit (charges)', fontsize=12)
ax3.set_title('Residuals: Failures from Failure Path', fontsize=14)
ax3.grid(True, alpha=0.3)

# Plot 4: Charge fraction q = Z/A evolution
ax4 = axes[1, 1]
q_success = [d['Z_exp']/d['A'] for d in successes]
q_failure = [d['Z_exp']/d['A'] for d in failures]

ax4.scatter(A_success, q_success, c='green', alpha=0.5, s=20, label='Successes')
ax4.scatter(A_failure, q_failure, c='red', alpha=0.5, s=20, label='Failures')

# Valley of stability
q_success_fit = Z_success_fit / A_range
q_failure_fit = Z_failure_fit / A_range

ax4.plot(A_range, q_success_fit, 'g--', linewidth=2, label='Success q(A)')
ax4.plot(A_range, q_failure_fit, 'r--', linewidth=2, label='Failure q(A)')

ax4.set_xlabel('Mass Number A', fontsize=12)
ax4.set_ylabel('Charge Fraction q = Z/A', fontsize=12)
ax4.set_title('Valley of Stability: q(A)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('progenitor_paths.png', dpi=150, bbox_inches='tight')
print("Saved: progenitor_paths.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: PROGENITOR PATH ANALYSIS")
print("="*95)
print()

print("Geometric pathways:")
print()
print(f"Success path: Z = {c1_success:.4f}×A^(2/3) + {c2_success:.4f}×A {c3_success:+.4f}")
print(f"Failure path: Z = {c1_failure:.4f}×A^(2/3) + {c2_failure:.4f}×A {c3_failure:+.4f}")
print()

print(f"Pathway differences:")
print(f"  Surface term (c1): {100*delta_c1:.1f}% different")
print(f"  Volume term (c2):  {100*delta_c2:.1f}% different")
print()

if delta_c1 > 0.1 or delta_c2 > 0.1:
    print("★★★ DUAL PATHWAY MODEL SUPPORTED!")
    print()
    print("Physical interpretation:")
    print("  • Successes: Follow one core/envelope assembly sequence")
    print("  • Failures:  Follow different geometric formation path")
    print()
    print("This suggests:")
    print("  → QFD energy functional captures ONE pathway correctly")
    print("  → Failures may represent different topological assembly")
    print("  → Need multi-pathway model or modified energy functional")
else:
    print("→ No clear dual pathway")
    print("  Failures scattered, not following coherent alternative path")

print()
print("="*95)
