#!/usr/bin/env python3
"""
VORTEX LOCKING CONSTRAINT - Angular Momentum Quantization
===========================================================================
QFD Framework: "Survivors" are configurations where core spin and electron
vortex angular momentum are in PHASE HARMONY.

KEY INSIGHT: The failures happen NOT because the energy is wrong, but because
certain (A, Z) configurations have FORBIDDEN vortex states.

HYPOTHESIS:
- Each nucleus has ground state spin J_nucleus
- Electron vortices have total angular momentum L_electron
- Stable configuration requires: J_nucleus ⊗ L_electron = Integer × ℏ
- This creates DISCRETE allowed Z values for each A

IMPLEMENTATION:
Instead of continuous minimization over all Z ∈ [1, A-1], we:
1. Calculate allowed Z values based on vortex locking rules
2. Only search over ALLOWED configurations
3. This creates the "stair-step" in Z(A) curve

WITHOUT FULL SPIN DATA:
Use heuristic rules based on:
- Pairing (even/odd Z, N)
- Magic numbers (known to have specific J)
- Shell closures (L quantization)
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

# ============================================================================
# NUCLEAR SPIN HEURISTICS (Ground State)
# ============================================================================

def estimate_nuclear_spin(Z, N):
    """
    Heuristic estimate of ground state nuclear spin J.

    Rules (empirical, without full spectroscopic data):
    1. Even Z, Even N: J = 0 (paired nucleons)
    2. Odd A: J = 1/2, 3/2, 5/2, ... (odd nucleon)
    3. Magic numbers: J depends on shell closure

    Returns: J (total angular momentum quantum number)
    """
    A = Z + N

    # Even-even nuclei (most stable, J=0 ground state)
    if Z % 2 == 0 and N % 2 == 0:
        return 0

    # Odd-A nuclei: J from unpaired nucleon
    if A % 2 == 1:
        # Simplification: assume J = 1/2 for odd-A
        # (Real nuclei vary: 1/2, 3/2, 5/2, 7/2, 9/2)
        return 0.5

    # Odd-odd nuclei (rare in stable chart)
    # Higher J possible from unpaired proton + neutron
    return 1.0

# ============================================================================
# ELECTRON VORTEX ANGULAR MOMENTUM
# ============================================================================

def electron_vortex_L(Z):
    """
    Total orbital angular momentum of electron vortex configuration.

    In atomic physics: L = Σ l_i for each electron in shell
    - Filled shells: L = 0 (paired)
    - Partially filled: L from unpaired electrons

    Simplified: L_total depends on whether Z is at noble gas closure.
    """
    # Noble gas configurations (closed shells → L = 0)
    noble_gases = {2, 10, 18, 36, 54, 86}

    if Z in noble_gases:
        return 0

    # Partially filled shells: estimate L from valence electrons
    # Crude approximation: L ≈ (Z mod shell_size)

    # For simplicity, return Z mod 8 (octet rule approximation)
    L = Z % 8

    return L

# ============================================================================
# VORTEX LOCKING CONDITION
# ============================================================================

def is_vortex_locked(Z, N):
    """
    Check if (Z, N) configuration satisfies vortex locking condition.

    Hypothesis: J_nucleus and L_electron must satisfy coupling rule.

    Simple rule (to test):
    - If J = 0 (even-even), require L = 0, 1, 2 (low L)
    - If J = 1/2 (odd-A), allow wider L range
    - If at magic number, always locked (geometric resonance)

    Returns: True if configuration is ALLOWED, False if FORBIDDEN
    """
    A = Z + N

    # Magic numbers always locked (core resonances)
    if Z in ISOMER_NODES or N in ISOMER_NODES:
        return True

    J_nuc = estimate_nuclear_spin(Z, N)
    L_elec = electron_vortex_L(Z)

    # Even-even nuclei (J=0): prefer closed electron shells (L=0)
    if J_nuc == 0:
        # Allow L = 0, 1, 2 (small angular momentum)
        if L_elec <= 2:
            return True
        else:
            return False

    # Odd-A nuclei (J=1/2): more flexible
    if J_nuc == 0.5:
        # Allow L = 0 to 4
        if L_elec <= 4:
            return True
        else:
            return False

    # Odd-odd nuclei (J=1): allow wider range
    if L_elec <= 6:
        return True

    return False

# ============================================================================
# ENERGY FUNCTIONAL (Baseline for Comparison)
# ============================================================================

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_energy_baseline(A, Z):
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

# ============================================================================
# STABILITY SEARCH WITH VORTEX LOCKING
# ============================================================================

def find_stable_Z_baseline(A):
    """Find stable Z without vortex locking constraint."""
    best_Z = 1
    best_E = qfd_energy_baseline(A, 1)
    for Z in range(1, A):
        E = qfd_energy_baseline(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

def find_stable_Z_with_locking(A):
    """
    Find stable Z WITH vortex locking constraint.

    Only search over Z values where is_vortex_locked(Z, N) = True.
    """
    best_Z = 1
    best_E = qfd_energy_baseline(A, 1)

    for Z in range(1, A):
        N = A - Z

        # Check vortex locking constraint
        if not is_vortex_locked(Z, N):
            continue  # SKIP this Z (forbidden by vortex locking)

        # Allowed configuration: evaluate energy
        E = qfd_energy_baseline(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z

    return best_Z

# ============================================================================
# EVALUATE
# ============================================================================

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("VORTEX LOCKING CONSTRAINT - Angular Momentum Quantization")
print("="*95)
print()
print("HYPOTHESIS: Survivors are configurations where J_nucleus and L_electron")
print("            satisfy discrete locking condition.")
print()
print("IMPLEMENTATION:")
print("  - Even-even (J=0): Allow only small L_electron (≤2)")
print("  - Odd-A (J=1/2): Allow L_electron ≤ 4")
print("  - Magic numbers: Always allowed (core resonances)")
print()

# Check how many configurations are forbidden
total_configs = 0
forbidden_configs = 0

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    total_configs += 1

    if not is_vortex_locked(Z_exp, N_exp):
        forbidden_configs += 1

print(f"Experimental configurations:")
print(f"  Total: {total_configs}")
print(f"  Forbidden by locking: {forbidden_configs} ({100*forbidden_configs/total_configs:.1f}%)")
print(f"  Allowed: {total_configs - forbidden_configs} ({100*(total_configs - forbidden_configs)/total_configs:.1f}%)")
print()

if forbidden_configs > 0:
    print("⚠️  Some experimental stable nuclei are FORBIDDEN by current locking rules!")
    print("    This means the locking heuristic needs refinement.")
    print()

# ============================================================================
# COMPARE PREDICTIONS
# ============================================================================

print("="*95)
print("PREDICTIONS: Baseline vs Vortex Locking")
print("="*95)
print()

results_baseline = []
results_locking = []

for name, Z_exp, A in test_nuclides:
    Z_base = find_stable_Z_baseline(A)
    Z_lock = find_stable_Z_with_locking(A)

    Delta_base = Z_base - Z_exp
    Delta_lock = Z_lock - Z_exp

    results_baseline.append({'name': name, 'Z_pred': Z_base, 'Delta_Z': Delta_base})
    results_locking.append({'name': name, 'Z_pred': Z_lock, 'Delta_Z': Delta_lock})

# Statistics
errors_base = [abs(r['Delta_Z']) for r in results_baseline]
errors_lock = [abs(r['Delta_Z']) for r in results_locking]

exact_base = sum(e == 0 for e in errors_base)
exact_lock = sum(e == 0 for e in errors_lock)

print(f"{'Model':<35} {'Exact':<20} {'Mean |ΔZ|':<15} {'Median |ΔZ|'}")
print("-"*95)
print(f"{'Baseline (no locking)':<35} {exact_base}/{len(results_baseline)} ({100*exact_base/len(results_baseline):.1f}%)  "
      f"{np.mean(errors_base):<15.3f} {np.median(errors_base):.1f}")
print(f"{'With Vortex Locking':<35} {exact_lock}/{len(results_locking)} ({100*exact_lock/len(results_locking):.1f}%)  "
      f"{np.mean(errors_lock):<15.3f} {np.median(errors_lock):.1f}")
print()

improvement = exact_lock - exact_base

if improvement > 0:
    print(f"✓ IMPROVEMENT: +{improvement} exact matches")
    print()
    print("Vortex locking constraint successfully filters forbidden configurations!")
else:
    print(f"Current vortex locking rules: {improvement:+d} change")
    print()
    if improvement < 0:
        print("Locking rules TOO RESTRICTIVE - forbidding valid configurations")
    else:
        print("Locking rules have no effect with current heuristics")

print()

# Show examples where locking changes prediction
print("="*95)
print("EXAMPLES: Where Locking Changes Prediction")
print("="*95)
print()

changed = [(name, Z_exp, A, results_baseline[i]['Z_pred'], results_locking[i]['Z_pred'])
           for i, (name, Z_exp, A) in enumerate(test_nuclides)
           if results_baseline[i]['Z_pred'] != results_locking[i]['Z_pred']]

if changed:
    print(f"{'Nuclide':<12} {'Z_exp':<8} {'Baseline':<12} {'Locked':<12} {'Status'}")
    print("-"*95)

    for name, Z_exp, A, Z_base, Z_lock in changed[:20]:  # First 20
        base_str = "✓" if Z_base == Z_exp else f"{Z_base} (ΔZ={Z_base-Z_exp:+d})"
        lock_str = "✓" if Z_lock == Z_exp else f"{Z_lock} (ΔZ={Z_lock-Z_exp:+d})"

        if abs(Z_lock - Z_exp) < abs(Z_base - Z_exp):
            status = "✓ Better"
        elif abs(Z_lock - Z_exp) > abs(Z_base - Z_exp):
            status = "✗ Worse"
        else:
            status = "= Same error"

        print(f"{name:<12} {Z_exp:<8} {base_str:<12} {lock_str:<12} {status}")

    print()
    print(f"Total configurations changed by locking: {len(changed)}")
else:
    print("No predictions changed (locking rules ineffective)")

print()
print("="*95)
print("VERDICT")
print("="*95)
print()

if improvement > 20:
    print("✓✓✓ BREAKTHROUGH: Vortex locking resolves major failures!")
    print()
    print("This confirms the discrete quantization picture:")
    print("  - Angular momentum J_nucleus ⊗ L_electron must be integer-locked")
    print("  - 'Survivors' are resonant phase-locked configurations")
    print("  - 'Failures' were energetically favorable but topologically FORBIDDEN")
elif improvement > 5:
    print("✓✓ SIGNIFICANT: Vortex locking improves predictions")
    print()
    print("Next steps:")
    print("  - Refine locking rules with real nuclear spin data")
    print("  - Add L-S coupling (total J = L + S)")
    print("  - Test different locking thresholds")
elif improvement > 0:
    print("✓ MODEST: Vortex locking has measurable effect")
    print()
    print("Locking rules need calibration - current heuristics too simple")
else:
    print("Current vortex locking heuristic doesn't improve predictions")
    print()
    print("POSSIBLE ISSUES:")
    print("  1. Locking rules incorrect (need real spectroscopic data)")
    print("  2. L_electron estimation wrong (need proper orbital calculation)")
    print("  3. Coupling rule too simplistic (need J = |L - S| to L + S)")
    print("  4. Missing physics is NOT in vortex locking")
    print()
    print("CONCLUSION:")
    print("  Without full nuclear spin data and electron orbital details,")
    print("  we cannot properly test the vortex locking hypothesis.")
    print()
    print("  To proceed, need:")
    print("  - Nuclear ground state J values for all 285 nuclides")
    print("  - Proper electron vortex L calculation from QFD framework")
    print("  - Coupling rules from Cl(3,3) algebra")

print()
print("="*95)
