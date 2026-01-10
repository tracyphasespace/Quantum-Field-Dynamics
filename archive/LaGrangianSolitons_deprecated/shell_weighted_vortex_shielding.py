#!/usr/bin/env python3
"""
SHELL-WEIGHTED VORTEX SHIELDING
===========================================================================
User insight: "Outer shells less impact due to distance and shielding by
               inner electrons"

PHYSICS:
- Inner shell electrons (K, L, M) close to core → strong shielding
- Outer shell electrons (N, O, P, Q) far from core → weak shielding
- Outer electrons are themselves shielded by inner electrons

IMPLEMENTATION:
Instead of shield ∝ Z (all electrons equal), use:

    shield_eff = Σ_shells (n_electrons × weight(shell))

where weight(shell) decays with distance:
    - K shell (n=1): weight = 1.0
    - L shell (n=2): weight = 1/4
    - M shell (n=3): weight = 1/9
    - N shell (n=4): weight = 1/16
    - etc.

This gives 1/r² falloff for shielding effectiveness.
===========================================================================
"""

import numpy as np

# Constants
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
M_proton     = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

# Electron shell structure (cumulative Z for shell closures)
# Noble gas configurations + intermediate closures
SHELL_CLOSURES = {
    # K shell (n=1)
    2: ('K', 1),
    # L shell (n=2)
    10: ('L', 2),
    # M shell (n=3)
    18: ('M', 3),
    28: ('M', 3),  # 3d subshell
    # N shell (n=4)
    36: ('N', 4),
    48: ('N', 4),  # 4d subshell
    # O shell (n=5)
    54: ('O', 5),
    70: ('O', 5),  # 4f subshell
    # P shell (n=6)
    86: ('P', 6),
}

def get_shell_structure(Z):
    """
    Determine shell occupancy for atom with Z protons/electrons.

    Returns: list of (n_shell, electrons_in_shell) tuples
    """
    shells = []

    # Simplified: assign electrons to shells based on closures
    shell_edges = sorted(SHELL_CLOSURES.keys())

    Z_remaining = Z
    prev_Z = 0

    for Z_closure in shell_edges:
        if Z <= Z_closure:
            # Partially filled shell
            shell_name, n = SHELL_CLOSURES[Z_closure]
            electrons_in_shell = Z_remaining
            if electrons_in_shell > 0:
                shells.append((n, electrons_in_shell))
            break
        else:
            # Filled shell
            shell_name, n = SHELL_CLOSURES[Z_closure]
            electrons_in_shell = Z_closure - prev_Z
            shells.append((n, electrons_in_shell))
            Z_remaining -= electrons_in_shell
            prev_Z = Z_closure

    # Any remaining electrons in outermost shell
    if Z > shell_edges[-1]:
        # Beyond Z=86, assume high n
        n = 7  # Q shell
        shells.append((n, Z_remaining))

    return shells

def shell_weighted_shielding(Z):
    """
    Calculate effective shielding from electron vortex configuration.

    Weight each shell by 1/n² (distance falloff).

    Returns: effective electron count for shielding
    """
    shells = get_shell_structure(Z)

    Z_eff = 0
    for n_shell, electrons in shells:
        # Distance weighting: 1/n²
        weight = 1.0 / (n_shell**2)
        Z_eff += electrons * weight

    return Z_eff

# ============================================================================
# ENERGY FUNCTIONAL
# ============================================================================

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_energy_shell_weighted(A, Z, shield_base=0.40, kappa_shell=0.02):
    """
    QFD energy with shell-weighted vortex shielding.

    Parameters:
    - shield_base: Base shielding (bare nucleus)
    - kappa_shell: Strength of shell-weighted shielding effect
    """
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_E * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)

    # Shell-weighted shielding
    Z_eff = shell_weighted_shielding(Z)
    shield = shield_base * (1 + kappa_shell * Z_eff)

    a_disp_bare = (alpha_fine * 197.327 / 1.2)
    a_disp = a_disp_bare * shield
    E_vac = a_disp * (Z**2) / (A**(1/3))

    E_iso = -get_resonance_bonus(Z, N, E_surface)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_shell_weighted(A, **kwargs):
    best_Z = 1
    best_E = qfd_energy_shell_weighted(A, 1, **kwargs)
    for Z in range(1, A):
        E = qfd_energy_shell_weighted(A, Z, **kwargs)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

# ============================================================================
# DEMONSTRATE SHELL WEIGHTING
# ============================================================================

print("="*95)
print("SHELL-WEIGHTED VORTEX SHIELDING")
print("="*95)
print()
print("Physics: Inner electrons shield strongly (1/r² weight), outer weakly")
print()
print("Shell weighting examples:")
print(f"{'Z':<6} {'Shells':<35} {'Z_eff':<12} {'Ratio Z_eff/Z'}")
print("-"*95)

for Z in [1, 2, 6, 8, 10, 18, 20, 26, 28, 36, 50, 54, 82, 86, 92]:
    shells = get_shell_structure(Z)
    Z_eff = shell_weighted_shielding(Z)

    # Format shell structure
    shell_str = ", ".join([f"n={n}({e}e)" for n, e in shells])

    ratio = Z_eff / Z if Z > 0 else 0

    print(f"{Z:<6} {shell_str:<35} {Z_eff:<12.3f} {ratio:.3f}")

print()
print("Key observation: Z_eff/Z decreases with Z (outer shells contribute less)")
print()

# ============================================================================
# CALIBRATE
# ============================================================================

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("CALIBRATION: Shell-Weighted Shielding")
print("="*95)
print()

# Baseline
def qfd_baseline(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_E * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    a_disp = (alpha_fine * 197.327 / 1.2) * 0.52

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_resonance_bonus(Z, N, E_surface)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_baseline(A):
    best_Z = 1
    best_E = qfd_baseline(A, 1)
    for Z in range(1, A):
        E = qfd_baseline(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

baseline_exact = sum(1 for name, Z_exp, A in test_nuclides
                     if find_stable_Z_baseline(A) == Z_exp)

print(f"Baseline (fixed shield=0.52): {baseline_exact}/{len(test_nuclides)} ({100*baseline_exact/len(test_nuclides):.1f}%)")
print()

# Grid search shell-weighted parameters
print("Grid search for optimal (shield_base, κ_shell)...")
print()

shield_base_values = [0.30, 0.35, 0.40, 0.45, 0.52, 0.60]
kappa_shell_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]

best_config = {
    'exact': baseline_exact,
    'shield_base': 0.52,
    'kappa_shell': 0.0,
}

for shield_base in shield_base_values:
    for kappa_shell in kappa_shell_values:
        exact = 0
        for name, Z_exp, A in test_nuclides:
            Z_pred = find_stable_Z_shell_weighted(A, shield_base=shield_base,
                                                  kappa_shell=kappa_shell)
            if Z_pred == Z_exp:
                exact += 1

        if exact > best_config['exact']:
            best_config = {
                'exact': exact,
                'shield_base': shield_base,
                'kappa_shell': kappa_shell,
            }

            pct = 100 * exact / len(test_nuclides)
            improvement = exact - baseline_exact
            print(f"  New best: {exact}/{len(test_nuclides)} ({pct:.1f}%)  "
                  f"shield_base={shield_base:.2f}, κ_shell={kappa_shell:.3f}  "
                  f"(+{improvement})")

# ============================================================================
# RESULTS
# ============================================================================

print()
print("="*95)
print("RESULTS")
print("="*95)
print()

print(f"Baseline:          {baseline_exact}/{len(test_nuclides)} ({100*baseline_exact/len(test_nuclides):.1f}%)")
print(f"Shell-Weighted:    {best_config['exact']}/{len(test_nuclides)} ({100*best_config['exact']/len(test_nuclides):.1f}%)")
print()

if best_config['kappa_shell'] > 0:
    print("Optimal parameters:")
    print(f"  shield_base: {best_config['shield_base']:.3f}")
    print(f"  κ_shell:     {best_config['kappa_shell']:.4f}")
    print()

improvement = best_config['exact'] - baseline_exact

if improvement > 0:
    print(f"✓ IMPROVEMENT: +{improvement} exact matches (+{improvement/len(test_nuclides)*100:.1f} pp)")
    print()
    print("Shell-weighted vortex shielding captures electron distance effect!")
    print("Inner electrons (K, L shells) dominate shielding; outer shells contribute weakly.")
else:
    print("= No improvement from shell weighting")
    print()
    print("Conclusion: Vortex shielding (even with 1/r² weighting) doesn't resolve failures.")
    print()
    print("This suggests missing physics is NOT in displacement shielding.")
    print("Must be in angular momentum quantization or discrete spin coupling.")

print()
print("="*95)
