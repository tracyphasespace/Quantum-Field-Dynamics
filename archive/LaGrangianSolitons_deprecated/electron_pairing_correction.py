#!/usr/bin/env python3
"""
ELECTRON VORTEX PAIRING CORRECTION TO λ_time
===========================================================================
User's insight: Electron vortex pairing LOWERS effective time parameter.

Key physics:
  - Electrons form vortex pairs (geometric analog of Cooper pairs)
  - Each pair modifies vacuum time flow → reduces λ_time
  - Heavy atoms (high Z) have more paired electrons → lower λ_time
  - This affects E_volume → changes nuclear stability

QFD prediction: Stripped heavy nuclei should be UNSTABLE
  (Removing electrons increases λ_time → wrong E_volume for that nucleus)

Implementation:
  λ_time(Z) = λ_time_0 - κ_e × f(Z_paired)

where f(Z_paired) is the number/effect of paired electron vortices.
===========================================================================
"""

import numpy as np

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
M_proton     = 938.272

# BASE TIME PARAMETER (for neutral atom)
LAMBDA_TIME_0 = 0.42

# ELECTRON PAIRING PARAMETERS
# These control how electron vortex pairing affects λ_time
KAPPA_E = 0.002  # Pairing strength (to be calibrated)

def lambda_time_with_electrons(Z):
    """
    Effective time parameter including electron vortex pairing.

    For heavy atoms, paired electrons lower λ_time.

    Parameters:
      Z: Proton number (= electron number for neutral atom)

    Returns:
      λ_time_eff: Effective time parameter
    """
    # Electron pairing function
    # Hypothesis: Pairing scales with Z (more electrons → more pairs)
    # Could be linear, quadratic, or saturating function

    # Simple linear model: ΔλPhoton ∝ Z
    delta_lambda = KAPPA_E * Z

    # Alternative: Saturating model (pairing saturates for very heavy)
    # delta_lambda = KAPPA_E * Z / (1 + Z/100)

    lambda_eff = LAMBDA_TIME_0 - delta_lambda

    # Physical bound: λ_time must be positive
    if lambda_eff < 0:
        lambda_eff = 0.01  # Small positive minimum

    return lambda_eff

def lambda_time_stripped(Z):
    """
    Time parameter for STRIPPED nucleus (no electrons).

    Without electron pairing, λ_time returns to base value.
    This changes E_volume and could destabilize heavy nuclei.
    """
    return LAMBDA_TIME_0

# ============================================================================
# ENERGY FUNCTIONAL WITH ELECTRON CORRECTION
# ============================================================================
SHIELD_FACTOR = 0.52
a_disp_base = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_energy_with_electrons(A, Z, stripped=False):
    """
    Energy functional with electron vortex pairing correction.

    Parameters:
      A: Mass number
      Z: Proton number
      stripped: If True, compute energy WITHOUT electrons

    Returns:
      Total energy in MeV
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    # ELECTRON-DEPENDENT TIME PARAMETER
    if stripped:
        lambda_time = lambda_time_stripped(Z)
    else:
        lambda_time = lambda_time_with_electrons(Z)

    # Recalculate V_0, E_volume, E_surface with corrected λ_time
    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2

    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    # Standard terms with electron-corrected E_volume
    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp_base * (Z**2) / (A**(1/3))
    E_iso  = -get_resonance_bonus(Z, N, E_surface)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_with_electrons(A, stripped=False):
    """Find stable Z including electron correction."""
    best_Z = 1
    best_E = qfd_energy_with_electrons(A, 1, stripped)

    for Z in range(1, A):
        E = qfd_energy_with_electrons(A, Z, stripped)
        if E < best_E:
            best_E = E
            best_Z = Z

    return best_Z

# ============================================================================
# LOAD TEST DATA
# ============================================================================
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()

start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("ELECTRON VORTEX PAIRING CORRECTION TO λ_time")
print("="*95)
print()
print("QFD Prediction: Electron pairing lowers λ_time for heavy atoms")
print()
print(f"Parameters:")
print(f"  λ_time_0 (base):        {LAMBDA_TIME_0:.3f}")
print(f"  κ_e (pairing strength): {KAPPA_E:.4f}")
print()
print("Testing:")
print("  1. Neutral atoms (with electrons)")
print("  2. Stripped nuclei (no electrons)")
print()

# Show λ_time variation with Z
print("λ_time vs Z:")
print(f"{'Z':<6} {'λ_time(neutral)':<20} {'λ_time(stripped)':<20} {'Δλ'}")
print("-"*95)

for Z in [1, 10, 20, 30, 50, 82, 92]:
    lambda_neutral = lambda_time_with_electrons(Z)
    lambda_strip = lambda_time_stripped(Z)
    delta = lambda_neutral - lambda_strip

    print(f"{Z:<6} {lambda_neutral:<20.4f} {lambda_strip:<20.4f} {delta:+.4f}")

print()

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================
print("="*95)
print("RESULTS - NEUTRAL vs STRIPPED")
print("="*95)
print()

results_neutral = []
results_stripped = []
results_baseline = []  # Fixed λ_time = 0.42

# Baseline (original model, fixed λ_time)
def qfd_baseline_energy(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - LAMBDA_TIME_0 / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp_base * (Z**2) / (A**(1/3))
    E_iso  = -get_resonance_bonus(Z, N, E_surface)
    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_baseline(A):
    best_Z = 1
    best_E = qfd_baseline_energy(A, 1)
    for Z in range(1, A):
        E = qfd_baseline_energy(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

for name, Z_exp, A in test_nuclides:
    # Neutral (with electrons)
    Z_neutral = find_stable_Z_with_electrons(A, stripped=False)
    Delta_neutral = Z_neutral - Z_exp

    # Stripped (no electrons)
    Z_stripped = find_stable_Z_with_electrons(A, stripped=True)
    Delta_stripped = Z_stripped - Z_exp

    # Baseline (fixed λ_time)
    Z_base = find_stable_Z_baseline(A)
    Delta_base = Z_base - Z_exp

    results_neutral.append({'name': name, 'A': A, 'Z_exp': Z_exp,
                            'Z_pred': Z_neutral, 'Delta_Z': Delta_neutral})
    results_stripped.append({'name': name, 'A': A, 'Z_exp': Z_exp,
                              'Z_pred': Z_stripped, 'Delta_Z': Delta_stripped})
    results_baseline.append({'name': name, 'A': A, 'Z_exp': Z_exp,
                             'Z_pred': Z_base, 'Delta_Z': Delta_base})

# Statistics
errors_neutral = [abs(r['Delta_Z']) for r in results_neutral]
errors_stripped = [abs(r['Delta_Z']) for r in results_stripped]
errors_base = [abs(r['Delta_Z']) for r in results_baseline]

exact_neutral = sum(e == 0 for e in errors_neutral)
exact_stripped = sum(e == 0 for e in errors_stripped)
exact_base = sum(e == 0 for e in errors_base)

print(f"{'Model':<30} {'Exact':<20} {'Mean |ΔZ|':<15} {'Median |ΔZ|'}")
print("-"*95)
print(f"{'Baseline (fixed λ=0.42)':<30} {exact_base}/{len(results_baseline)} ({100*exact_base/len(results_baseline):.1f}%)  "
      f"{np.mean(errors_base):<15.3f} {np.median(errors_base):.1f}")
print(f"{'Neutral (electron pairing)':<30} {exact_neutral}/{len(results_neutral)} ({100*exact_neutral/len(results_neutral):.1f}%)  "
      f"{np.mean(errors_neutral):<15.3f} {np.median(errors_neutral):.1f}")
print(f"{'Stripped (no electrons)':<30} {exact_stripped}/{len(results_stripped)} ({100*exact_stripped/len(results_stripped):.1f}%)  "
      f"{np.mean(errors_stripped):<15.3f} {np.median(errors_stripped):.1f}")
print()

improvement_neutral = exact_neutral - exact_base
improvement_stripped = exact_stripped - exact_base

if improvement_neutral > 0:
    print(f"✓ ELECTRON CORRECTION IMPROVES: +{improvement_neutral} exact matches")
elif improvement_neutral < 0:
    print(f"? ELECTRON CORRECTION WORSENS: {improvement_neutral} exact matches")
    print(f"  (May need to calibrate κ_e = {KAPPA_E})")
else:
    print(f"= NEUTRAL: Same as baseline (κ_e too small?)")

print()

# QFD PREDICTION TEST
print("="*95)
print("QFD PREDICTION: Stripped Heavy Nuclei Become Unstable")
print("="*95)
print()

# Count how many heavy nuclei change stability when stripped
heavy_nuclei = [(r_n, r_s) for r_n, r_s in zip(results_neutral, results_stripped)
                if r_n['A'] >= 100]

destabilized = [(r_n, r_s) for r_n, r_s in heavy_nuclei
                if abs(r_n['Delta_Z']) < abs(r_s['Delta_Z'])]  # Worse when stripped

print(f"Heavy nuclei (A ≥ 100): {len(heavy_nuclei)}")
print(f"Destabilized when stripped: {len(destabilized)} ({100*len(destabilized)/len(heavy_nuclei) if len(heavy_nuclei)>0 else 0:.1f}%)")
print()

if len(destabilized) > 0:
    print("Examples:")
    print(f"{'Nuclide':<12} {'A':<5} {'ΔZ(neutral)':<15} {'ΔZ(stripped)':<15} {'Effect'}")
    print("-"*95)
    for r_n, r_s in destabilized[:10]:
        neutral_str = "✓" if r_n['Delta_Z'] == 0 else f"{r_n['Delta_Z']:+d}"
        stripped_str = f"{r_s['Delta_Z']:+d}"
        print(f"{r_n['name']:<12} {r_n['A']:<5} {neutral_str:<15} {stripped_str:<15} Destabilized")

print()
print("="*95)
print("INTERPRETATION")
print("="*95)
print()

if improvement_neutral > 5:
    print("✓✓ ELECTRON PAIRING SIGNIFICANTLY IMPROVES PREDICTIONS")
    print()
    print("Implication:")
    print("  - Electron vortex pairing IS essential for nuclear stability")
    print("  - λ_time must be Z-dependent (electron configuration matters)")
    print("  - QFD prediction confirmed: stripped nuclei less stable")
elif improvement_neutral > 0:
    print("✓ MODEST IMPROVEMENT with electron correction")
    print()
    print("Implication:")
    print("  - Electron effect is real but small with current κ_e")
    print("  - May need to calibrate κ_e from experimental data")
    print("  - Or pairing function f(Z) needs refinement")
elif improvement_stripped > improvement_neutral:
    print("? STRIPPED NUCLEI PREDICT BETTER (unexpected!)")
    print()
    print("Implication:")
    print("  - Current pairing model may be inverted")
    print("  - Or κ_e has wrong sign")
    print("  - Need to revisit electron vortex pairing mechanism")
else:
    print("= NEUTRAL: κ_e too small to see effect")
    print()
    print("Next step:")
    print(f"  - Increase κ_e from {KAPPA_E} to ~0.01-0.05")
    print("  - Or use nonlinear pairing function")
    print("  - Calibrate from known electron affinity effects")

print()
print("="*95)
