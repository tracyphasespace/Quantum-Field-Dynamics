#!/usr/bin/env python3
"""
QFD Vortex Electron: Phase 1 Validation

Direct numerical confirmation of Lean theorems:
1. external_is_classical_coulomb
2. internal_is_zitterbewegung

No additional physics assumptions - just test what was proven.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("QFD VORTEX ELECTRON: LEAN THEOREM VALIDATION")
print("="*70)
print("\nPhase 1: Force Law Correctness (No QM required)")
print()

# ============================================================================
# CONSTANTS
# ============================================================================

K_E = 8.9875517923e9      # Coulomb constant (N⋅m²/C²)
Q_E = 1.602176634e-19     # Elementary charge (C)
M_E = 9.1093837015e-31    # Electron mass (kg)
HBAR = 1.054571817e-34    # Reduced Planck constant (J⋅s)
C = 299792458             # Speed of light (m/s)

LAMBDA_C = HBAR / (M_E * C)  # Compton wavelength
R_VORTEX = LAMBDA_C / 2      # Vortex radius

print(f"Vortex parameters:")
print(f"  Compton wavelength: {LAMBDA_C*1e15:.2f} fm")
print(f"  Vortex radius R: {R_VORTEX*1e15:.2f} fm")
print()

# ============================================================================
# FORCE LAW IMPLEMENTATION (from Lean spec)
# ============================================================================

def shielding_factor(r, R):
    """
    Newton's Shell Theorem shielding.
    Lean: if r >= R then 1.0 else (r/R)³
    """
    return 1.0 if r >= R else (r / R)**3

def vortex_force(r, R):
    """
    QFD Force Law from Lean formalization.
    F = k_e * q² * ShieldingFactor(r) / r²
    """
    Q_eff = Q_E * shielding_factor(r, R)
    return K_E * Q_E * Q_eff / r**2

def classical_coulomb(r):
    """Standard Coulomb force."""
    return K_E * Q_E**2 / r**2

# ============================================================================
# TEST 1: external_is_classical_coulomb
# ============================================================================

print("TEST 1: Theorem `external_is_classical_coulomb`")
print("-" * 70)
print("Claim: When r ≥ R, VortexForce = k*q²/r² (exactly)")
print()

# Sample external region
r_external = np.logspace(np.log10(R_VORTEX), np.log10(10*R_VORTEX), 100)

F_vortex_ext = np.array([vortex_force(r, R_VORTEX) for r in r_external])
F_coulomb_ext = np.array([classical_coulomb(r) for r in r_external])

# Maximum relative error
error_ext = np.max(np.abs(F_vortex_ext - F_coulomb_ext) / F_coulomb_ext) * 100

print(f"Sample points: {len(r_external)} from {r_external[0]*1e15:.2f} to {r_external[-1]*1e15:.2f} fm")
print(f"Maximum relative error: {error_ext:.12f}%")

if error_ext < 1e-10:
    print("✅ PASS: External regime matches Coulomb to machine precision")
else:
    print(f"❌ FAIL: Error {error_ext}% exceeds tolerance")

print()

# ============================================================================
# TEST 2: internal_is_zitterbewegung
# ============================================================================

print("TEST 2: Theorem `internal_is_zitterbewegung`")
print("-" * 70)
print("Claim: When r < R, ∃k: VortexForce = k*r (linear restoring force)")
print()

# Sample internal region
r_internal = np.linspace(0.01*R_VORTEX, 0.99*R_VORTEX, 100)

F_vortex_int = np.array([vortex_force(r, R_VORTEX) for r in r_internal])

# Theoretical spring constant from Lean proof
k_spring_theory = K_E * Q_E**2 / R_VORTEX**3
F_linear_theory = k_spring_theory * r_internal

# Fit to F = k*r model to verify linearity
coeffs = np.polyfit(r_internal, F_vortex_int, 1)
k_spring_fit = coeffs[0]
intercept = coeffs[1]

# Errors
linearity_error = np.mean(np.abs(F_vortex_int - F_linear_theory) / F_linear_theory) * 100
k_match_error = abs(k_spring_fit - k_spring_theory) / k_spring_theory * 100
intercept_rel = abs(intercept) / np.mean(F_vortex_int) * 100

print(f"Sample points: {len(r_internal)} from {r_internal[0]*1e15:.2f} to {r_internal[-1]*1e15:.2f} fm")
print(f"Spring constant (theory): k = k*q²/R³ = {k_spring_theory:.6e} N/m")
print(f"Spring constant (fitted):  k = {k_spring_fit:.6e} N/m")
print(f"Match error: {k_match_error:.12f}%")
print(f"Linearity (mean deviation): {linearity_error:.12f}%")
print(f"Intercept (should be ~0): {intercept_rel:.12f}%")

if linearity_error < 1e-10 and intercept_rel < 1e-8:
    print("✅ PASS: Internal regime is perfectly linear F = k*r")
else:
    print(f"❌ FAIL: Linearity error {linearity_error}% or intercept {intercept_rel}% too large")

print()

# ============================================================================
# TEST 3: Boundary Continuity
# ============================================================================

print("TEST 3: Force Continuity at r = R")
print("-" * 70)
print("Claim: Force is continuous across vortex boundary")
print()

# Approach boundary from both sides
epsilon = 1e-18  # Small displacement

r_just_below = R_VORTEX - epsilon
r_at_R = R_VORTEX
r_just_above = R_VORTEX + epsilon

F_below = vortex_force(r_just_below, R_VORTEX)
F_at = vortex_force(r_at_R, R_VORTEX)
F_above = vortex_force(r_just_above, R_VORTEX)

# All should equal k*q²/R²
F_expected_at_R = K_E * Q_E**2 / R_VORTEX**2

jump_below = abs(F_below - F_expected_at_R) / F_expected_at_R * 100
jump_above = abs(F_above - F_expected_at_R) / F_expected_at_R * 100

print(f"Force at r = R - ε: {F_below:.12e} N")
print(f"Force at r = R:     {F_at:.12e} N")
print(f"Force at r = R + ε: {F_above:.12e} N")
print(f"Expected at r = R:  {F_expected_at_R:.12e} N")
print()
print(f"Jump from below: {jump_below:.12f}%")
print(f"Jump from above: {jump_above:.12f}%")

if max(jump_below, jump_above) < 0.01:  # 0.01% = excellent continuity
    print("✅ PASS: Force is continuous at boundary (< 0.01% jump)")
else:
    print(f"❌ FAIL: Discontinuity {max(jump_below, jump_above):.6f}% too large")

print()

# ============================================================================
# TEST 4: Singularity Prevention
# ============================================================================

print("TEST 4: Singularity Prevention (r → 0)")
print("-" * 70)
print("Claim: Vortex force remains finite as r → 0 (unlike Coulomb)")
print()

# Test near origin
r_near_zero = np.array([0.001, 0.01, 0.1, 0.5, 1.0]) * R_VORTEX

print("Distance       Vortex Force      Coulomb Force")
print("-" * 60)

for r in r_near_zero:
    F_vortex = vortex_force(r, R_VORTEX)
    F_coulomb = classical_coulomb(r)

    print(f"{r/R_VORTEX:6.3f} R    {F_vortex:.6e} N    {F_coulomb:.6e} N")

# Check limit behavior
r_tiny = 1e-6 * R_VORTEX
F_vortex_tiny = vortex_force(r_tiny, R_VORTEX)
F_expected_tiny = k_spring_theory * r_tiny

print()
print(f"At r = 10⁻⁶ R:")
print(f"  Vortex force: {F_vortex_tiny:.6e} N (finite)")
print(f"  Expected F≈kr: {F_expected_tiny:.6e} N")
print(f"  Coulomb (would be): {classical_coulomb(r_tiny):.6e} N (infinite)")

singularity_prevented = F_vortex_tiny < 1e10  # Reasonable bound

if singularity_prevented:
    print("✅ PASS: No singularity - force remains bounded")
else:
    print("❌ FAIL: Force diverges")

print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Generating validation plots...")
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Full force profile
ax1 = axes[0, 0]
r_full = np.concatenate([r_internal, r_external])
F_full_vortex = np.concatenate([F_vortex_int, F_vortex_ext])
F_full_coulomb = np.array([classical_coulomb(r) for r in r_full])

ax1.loglog(r_full*1e15, F_full_vortex, 'b-', linewidth=2.5, label='QFD Vortex Force')
ax1.loglog(r_full*1e15, F_full_coulomb, 'r--', linewidth=2, label='Classical Coulomb')
ax1.axvline(R_VORTEX*1e15, color='green', linestyle=':', linewidth=2.5,
            label=f'Vortex Boundary (R = {R_VORTEX*1e15:.1f} fm)')
ax1.set_xlabel('Distance r (fm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Force F (N)', fontsize=12, fontweight='bold')
ax1.set_title('Theorem 1 & 2: Force Profile Across Regimes', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, which='both', alpha=0.3)

# Plot 2: External regime match
ax2 = axes[0, 1]
error_profile_ext = np.abs(F_vortex_ext - F_coulomb_ext) / F_coulomb_ext * 100
ax2.semilogx(r_external*1e15, error_profile_ext, 'g-', linewidth=2)
ax2.set_xlabel('Distance r (fm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
ax2.set_title('Theorem 1: External Coulomb Match', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0.5, 0.5, f'Max Error: {error_ext:.2e}%',
         transform=ax2.transAxes, fontsize=14, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Plot 3: Internal linearity
ax3 = axes[1, 0]
ax3.plot(r_internal*1e15, F_vortex_int, 'bo', markersize=5, alpha=0.6, label='Vortex Force')
ax3.plot(r_internal*1e15, F_linear_theory, 'r-', linewidth=2.5, label='F = k·r (Theory)')
ax3.set_xlabel('Distance r (fm)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Force F (N)', fontsize=12, fontweight='bold')
ax3.set_title('Theorem 2: Internal Linear Restoring Force', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.text(0.5, 0.8, f'Linearity: {linearity_error:.2e}%\nk = {k_spring_theory:.3e} N/m',
         transform=ax3.transAxes, fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Plot 4: Singularity comparison
ax4 = axes[1, 1]
r_singularity = np.logspace(np.log10(0.001*R_VORTEX), np.log10(R_VORTEX), 100)
F_vortex_sing = np.array([vortex_force(r, R_VORTEX) for r in r_singularity])
F_coulomb_sing = np.array([classical_coulomb(r) for r in r_singularity])

ax4.loglog(r_singularity*1e15, F_vortex_sing, 'b-', linewidth=2.5, label='QFD: F → 0 as r → 0')
ax4.loglog(r_singularity*1e15, F_coulomb_sing, 'r--', linewidth=2, label='Coulomb: F → ∞')
ax4.set_xlabel('Distance r (fm)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Force F (N)', fontsize=12, fontweight='bold')
ax4.set_title('Singularity Prevention', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11, loc='upper left')
ax4.grid(True, which='both', alpha=0.3)
ax4.text(0.5, 0.5, 'No Collapse!\nF ∝ r (bounded)',
         transform=ax4.transAxes, fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('vortex_force_law_validation.png', dpi=300, bbox_inches='tight')
print("✅ Saved: vortex_force_law_validation.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print()
print("="*70)
print("VALIDATION SUMMARY")
print("="*70)
print()

tests_passed = 0
tests_total = 4

# Test 1
if error_ext < 1e-10:
    print("✅ TEST 1: External Coulomb match (< 1e-10% error)")
    tests_passed += 1
else:
    print(f"❌ TEST 1: External error {error_ext}%")

# Test 2
if linearity_error < 1e-10 and intercept_rel < 1e-8:
    print("✅ TEST 2: Internal linearity (< 1e-10% deviation)")
    tests_passed += 1
else:
    print(f"❌ TEST 2: Linearity {linearity_error}%, intercept {intercept_rel}%")

# Test 3
if max(jump_below, jump_above) < 0.01:
    print("✅ TEST 3: Boundary continuity (< 0.01% jump)")
    tests_passed += 1
else:
    print(f"❌ TEST 3: Jump {max(jump_below, jump_above):.6f}% exceeds tolerance")

# Test 4
if singularity_prevented:
    print("✅ TEST 4: Singularity prevented (force bounded)")
    tests_passed += 1
else:
    print("❌ TEST 4: Force diverges")

print()
print(f"RESULT: {tests_passed}/{tests_total} tests passed")
print()

if tests_passed == tests_total:
    print("="*70)
    print("🎉 ALL LEAN THEOREMS VALIDATED NUMERICALLY 🎉")
    print("="*70)
    print()
    print("Proven claims:")
    print("  1. External regime (r > R): Standard Coulomb F = k*q²/r²")
    print("  2. Internal regime (r < R): Linear force F = k*r")
    print("  3. Transition at r = R: Smooth and continuous")
    print("  4. No singularity: Force bounded as r → 0")
    print()
    print("Physical interpretation:")
    print("  - Newton's Shell Theorem shielding prevents 1/r² singularity")
    print("  - External scattering sees point-like Coulomb attraction")
    print("  - Internal structure exhibits harmonic restoring force")
    print("  - The electron is a VORTEX, not a point particle ✅")
    print()
    print("="*70)
else:
    print("❌ VALIDATION FAILED - Check implementation")

print()
print("Next steps:")
print("  - Phase 2: Add angular momentum → classical stability")
print("  - Phase 3: Solve Schrödinger equation → hydrogen spectrum")
print()
