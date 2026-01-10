#!/usr/bin/env python3
"""
STAIRCASE KAPPA SWEEP - Find Optimal Phase Stiffness
===========================================================================
Test different values of κ (Berry phase barrier strength) to find
if there's an optimal topological tension parameter.

If κ → 0: Recovers static solver (no memory)
If κ → ∞: Complete phase-lock (Z never changes)
If optimal κ exists: Sweet spot between flexibility and memory
===========================================================================
"""

import numpy as np

# ============================================================================
# CONSTANTS (same as staircase solver)
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
lambda_time  = 0.42
M_proton     = 938.272

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15
a_sym     = (beta_vacuum * M_proton) / 15

SHIELD_FACTOR = 0.52
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

def get_resonance_bonus(Z, N):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_base_energy(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_resonance_bonus(Z, N)
    return E_bulk + E_surf + E_asym + E_vac + E_iso

def qfd_staircase_energy(A, Z, Z_prev, kappa):
    E_base = qfd_base_energy(A, Z)
    if Z != Z_prev:
        E_phase_slip = kappa * E_surface * abs(Z - Z_prev)
        return E_base + E_phase_slip
    return E_base

def build_stability_path_with_kappa(test_nuclides, kappa):
    """Build staircase path with given κ value."""
    A_max = max(A for _, _, A in test_nuclides)

    # Build path from A=1 to A_max
    Z_current = 1
    path_lookup = {1: 1}

    for A in range(2, A_max + 1):
        best_Z = Z_current
        best_E = qfd_staircase_energy(A, Z_current, Z_current, kappa)

        for Z_test in range(1, A):
            E_test = qfd_staircase_energy(A, Z_test, Z_current, kappa)
            if E_test < best_E:
                best_E = E_test
                best_Z = Z_test

        Z_current = best_Z
        path_lookup[A] = Z_current

    # Evaluate on test set
    errors = []
    for name, Z_exp, A in test_nuclides:
        Z_pred = path_lookup.get(A, 1)
        errors.append(abs(Z_pred - Z_exp))

    exact = sum(e == 0 for e in errors)
    mean_error = np.mean(errors)

    return exact, mean_error

# ============================================================================
# LOAD TEST DATA
# ============================================================================
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()

start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*80)
print("STAIRCASE KAPPA SWEEP - Finding Optimal Berry Phase Barrier")
print("="*80)
print()
print("Testing κ values from 0.0 (static) to 1.0 (strong phase-lock)")
print()

# ============================================================================
# PARAMETER SWEEP
# ============================================================================
kappa_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]

results = []

for kappa in kappa_values:
    exact, mean_err = build_stability_path_with_kappa(test_nuclides, kappa)
    exact_pct = 100 * exact / len(test_nuclides)

    results.append({
        'kappa': kappa,
        'exact': exact,
        'exact_pct': exact_pct,
        'mean_error': mean_err,
    })

    print(f"κ = {kappa:.3f}  →  Exact: {exact}/{len(test_nuclides)} ({exact_pct:.1f}%)  "
          f"Mean|ΔZ|: {mean_err:.3f}")

print()
print("="*80)
print("OPTIMAL CONFIGURATION")
print("="*80)
print()

# Find best by exact matches
best_by_exact = max(results, key=lambda r: (r['exact'], -r['mean_error']))

print(f"Best κ (by exact matches):  {best_by_exact['kappa']:.3f}")
print(f"  Exact matches:            {best_by_exact['exact']}/{len(test_nuclides)} ({best_by_exact['exact_pct']:.1f}%)")
print(f"  Mean |ΔZ|:                {best_by_exact['mean_error']:.3f}")
print()

# Compare to static (κ=0)
static_result = next(r for r in results if r['kappa'] == 0.0)

print(f"Static (κ=0) baseline:      {static_result['exact']}/{len(test_nuclides)} ({static_result['exact_pct']:.1f}%)")
print()

if best_by_exact['exact'] > static_result['exact']:
    improvement = best_by_exact['exact'] - static_result['exact']
    print(f"✓ IMPROVEMENT: +{improvement} exact matches with κ={best_by_exact['kappa']:.3f}")
    print("  Berry phase memory helps!")
elif best_by_exact['exact'] == static_result['exact'] and best_by_exact['kappa'] == 0.0:
    print("= NEUTRAL: κ=0 is optimal (static solver is best)")
    print("  Phase memory doesn't help - nuclei optimize independently")
else:
    print(f"✗ OPTIMAL κ={best_by_exact['kappa']:.3f} but still worse than static")
    print("  Path-dependent approach fundamentally wrong for this system")

print()
print("="*80)
print("INTERPRETATION")
print("="*80)
print()

if best_by_exact['kappa'] == 0.0:
    print("κ = 0 is optimal → NO TOPOLOGICAL MEMORY")
    print()
    print("Physical interpretation:")
    print("  - Each nucleus optimizes Z independently at its mass A")
    print("  - No hysteresis or phase-locking to previous configuration")
    print("  - Solitons are NOT path-dependent; they're in equilibrium")
    print()
    print("This suggests:")
    print("  - Nuclei in stability valley are ground states (not metastable)")
    print("  - Berry phase term doesn't apply (or is averaged out)")
    print("  - Static energy minimization is the correct model")
elif best_by_exact['kappa'] < 0.1:
    print(f"κ ≈ {best_by_exact['kappa']:.3f} (very small) is optimal")
    print()
    print("Physical interpretation:")
    print("  - Weak topological memory (small phase-slip barrier)")
    print("  - Nuclei mostly follow static optimization")
    print("  - Minor corrections from history dependence")
else:
    print(f"κ = {best_by_exact['kappa']:.3f} is optimal")
    print()
    print("Physical interpretation:")
    print("  - Significant topological memory")
    print("  - Path-dependent buildup matters")
    print("  - Nuclei retain 'memory' of previous winding configuration")

print()
print("="*80)
