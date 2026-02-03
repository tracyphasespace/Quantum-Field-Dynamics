#!/usr/bin/env python3
"""
PARAMETER SWEEP - FIND OPTIMAL CONFIGURATION
===========================================================================
Systematically test combinations of:
1. Shielding factor (0.45 to 0.70)
2. Isomer bonus strength (0.5× to 1.5× E_surface)

Find configuration that maximizes exact predictions.
===========================================================================
"""

import numpy as np

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
lambda_time  = 0.42
M_proton     = 938.272

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15
a_sym     = (beta_vacuum * M_proton) / 15

hbar_c = 197.327
r_0 = 1.2

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

# ============================================================================
# PARAMETERIZED FUNCTIONS
# ============================================================================
def get_isomer_bonus(Z, N, bonus_strength):
    """Isomer bonus with adjustable strength."""
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * bonus_strength
    if N in ISOMER_NODES: bonus += E_surface * bonus_strength
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_energy(A, Z, shield_factor, bonus_strength):
    """Energy functional with adjustable parameters."""
    N = A - Z
    q = Z / A if A > 0 else 0

    a_disp = (alpha_fine * hbar_c / r_0) * shield_factor

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_isomer_bonus(Z, N, bonus_strength)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z(A, shield_factor, bonus_strength):
    """Find optimal Z with given parameters."""
    if A <= 2:
        return 1

    best_Z = 1
    best_E = qfd_energy(A, 1, shield_factor, bonus_strength)

    for Z in range(1, A):
        E = qfd_energy(A, Z, shield_factor, bonus_strength)
        if E < best_E:
            best_E = E
            best_Z = Z

    return best_Z

# ============================================================================
# TEST SET (reduced for speed)
# ============================================================================
test_nuclides = [
    # Representative sample across all regions
    ("H-2", 1, 2), ("He-4", 2, 4),
    ("C-12", 6, 12), ("O-16", 8, 16),
    ("Ne-20", 10, 20), ("Mg-24", 12, 24),
    ("Si-28", 14, 28), ("S-32", 16, 32),
    ("Ca-40", 20, 40), ("Ca-48", 20, 48),
    ("Fe-56", 26, 56), ("Ni-58", 28, 58),
    ("Kr-84", 36, 84), ("Zr-90", 40, 90),
    ("Mo-98", 42, 98), ("Cd-112", 48, 112),
    ("Sn-120", 50, 120), ("Xe-132", 54, 132),
    ("Ba-138", 56, 138), ("Nd-144", 60, 144),
    ("Pb-208", 82, 208), ("U-238", 92, 238),
]

# ============================================================================
# PARAMETER SWEEP
# ============================================================================
print("="*80)
print("PARAMETER SWEEP - FINDING OPTIMAL CONFIGURATION")
print("="*80)
print()

# Test ranges
shield_factors = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]
bonus_strengths = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5]

best_exact = 0
best_config = None
results_table = []

print("Testing combinations...")
print(f"  Shield factors: {len(shield_factors)} values")
print(f"  Bonus strengths: {len(bonus_strengths)} values")
print(f"  Total combinations: {len(shield_factors) * len(bonus_strengths)}")
print(f"  Test nuclides: {len(test_nuclides)}")
print()

for shield in shield_factors:
    for bonus in bonus_strengths:
        # Test this configuration
        correct = 0
        errors = []

        for name, Z_exp, A in test_nuclides:
            Z_pred = find_stable_Z(A, shield, bonus)
            Delta_Z = Z_pred - Z_exp
            errors.append(abs(Delta_Z))
            if Delta_Z == 0:
                correct += 1

        exact_pct = 100 * correct / len(test_nuclides)
        mean_error = np.mean(errors)

        results_table.append({
            'shield': shield,
            'bonus': bonus,
            'exact': correct,
            'exact_pct': exact_pct,
            'mean_error': mean_error,
        })

        if correct > best_exact:
            best_exact = correct
            best_config = (shield, bonus)

# ============================================================================
# RESULTS
# ============================================================================
print("="*80)
print("TOP 10 CONFIGURATIONS (by exact matches)")
print("="*80)
print(f"{'Shield':>8} {'Bonus':>8} {'Exact':>8} {'Exact %':>10} {'Mean |ΔZ|':>12}")
print("-"*80)

# Sort by exact matches (descending), then by mean error (ascending)
sorted_results = sorted(results_table,
                       key=lambda r: (-r['exact'], r['mean_error']))

for i, r in enumerate(sorted_results[:10]):
    marker = "★" if i == 0 else " "
    print(f"{marker} {r['shield']:>6.2f}   {r['bonus']:>6.2f}   "
          f"{r['exact']:>2}/{len(test_nuclides):>2}   {r['exact_pct']:>8.1f}%   "
          f"{r['mean_error']:>10.3f}")

print()
print(f"OPTIMAL CONFIGURATION:")
print(f"  Shielding factor: {best_config[0]:.2f}")
print(f"  Isomer bonus: {best_config[1]:.2f} × E_surface")
print(f"  Performance: {best_exact}/{len(test_nuclides)} exact ({100*best_exact/len(test_nuclides):.1f}%)")
print()

# Test optimal on key nuclei
print("="*80)
print("OPTIMAL CONFIGURATION - KEY NUCLEI TEST")
print("="*80)

key_nuclei = [
    ("He-4", 2, 4, "Doubly magic"),
    ("O-16", 8, 16, "Doubly magic"),
    ("Ca-40", 20, 40, "Doubly magic"),
    ("Fe-56", 26, 56, "Most stable"),
    ("Ni-58", 28, 58, "Magic Z=28"),
    ("Sn-120", 50, 120, "Magic Z=50"),
    ("Pb-208", 82, 208, "Doubly magic"),
    ("U-238", 92, 238, "Heaviest natural"),
]

print(f"{'Nuclide':<10} {'A':>4} {'Z_exp':>6} {'Z_pred':>6} {'ΔZ':>6} {'Description'}")
print("-"*80)

for name, Z_exp, A, desc in key_nuclei:
    Z_pred = find_stable_Z(A, best_config[0], best_config[1])
    Delta_Z = Z_pred - Z_exp
    status = "✓" if Delta_Z == 0 else f"{Delta_Z:+d}"
    print(f"{name:<10} {A:>4} {Z_exp:>6} {Z_pred:>6} {status:>6} {desc}")

print("="*80)
