#!/usr/bin/env python3
"""
VORTEX SHIELDING COUPLING - QFD Book Implementation
===========================================================================
Based on QFD Book (Jan 1, 2026) and Lean 4 formalizations.

THEORY: Electrons are not point particles but VORTEX SOLITONS that create
a topological shield around the nuclear Q-ball. This shield "refracts"
the vacuum displacement field, reducing the stress on the core.

MISSING PHYSICS:
1. Aharonov-Bohm Shielding: Electron vortices smooth ∇ρ_vacuum gradient
2. Dynamic Shielding: Shield strength depends on Z and electron configuration
3. Saturation: Finite shielding capacity at high Z

IMPLEMENTATION:
Instead of fixed shield_factor = 0.52, use Z-dependent shielding:

    shield(Z, A) = shield_base × vortex_refraction(Z, A)

where vortex_refraction increases with electron density but saturates.

This explains why heavy nuclei survive: electrons don't just neutralize
charge; they REFRACT VACUUM PRESSURE.
===========================================================================
"""

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
M_proton     = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001  # Temporal metric modulation (linear baseline)

# Nuclear magic numbers (geometric resonance nodes)
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

# VORTEX SHIELDING PARAMETERS (to calibrate)
SHIELD_BASE = 0.40      # Bare core shielding (lower than previous 0.52)
KAPPA_VORTEX = 0.015    # Vortex refraction strength
ZETA_SATURATION = 0.01  # Saturation rate at high Z

def vortex_shielding_factor(Z, A):
    """
    Electron vortex shielding as function of (Z, A).

    Physics:
    - More electrons (higher Z) → stronger topological shield
    - Neutron-rich (lower Z/A) → weaker shielding per nucleon
    - High Z → saturation (finite refraction capacity)

    Functional forms to test:
    1. Linear: 1 + κ × Z
    2. Ratio-dependent: 1 + κ × (Z/A)
    3. Saturating: 1 + κ×Z / (1 + ζ×Z)
    4. Hybrid: Combines ratio and saturation
    """
    q = Z / A if A > 0 else 0

    # OPTION A: Linear in Z (simple scaling)
    # refraction = 1 + KAPPA_VORTEX * Z

    # OPTION B: Ratio-dependent (Z/A captures electron density)
    # refraction = 1 + KAPPA_VORTEX * q * A**(1/3)  # Scale with radius

    # OPTION C: Saturating (finite shielding capacity)
    # refraction = 1 + KAPPA_VORTEX * Z / (1 + ZETA_SATURATION * Z)

    # OPTION D: Hybrid (ratio + saturation)
    # Shielding increases with Z but relative to neutron background
    # Heavy neutron-rich nuclei get less benefit than proton-rich
    electron_density = Z  # Number of vortices
    saturation_term = 1 + ZETA_SATURATION * Z

    refraction = 1 + (KAPPA_VORTEX * electron_density) / saturation_term

    return refraction

def get_displacement_energy(Z, A):
    """
    Vacuum displacement energy with DYNAMIC vortex shielding.

    Previous: E_vac = a_disp_base × Z² / A^(1/3)  with fixed shield
    New:      E_vac = a_disp_bare × shield(Z,A) × Z² / A^(1/3)

    The electron vortex cloud REFRACTS the vacuum pressure gradient,
    reducing the displacement penalty for high-Z nuclei.
    """
    a_disp_bare = (alpha_fine * 197.327 / 1.2)  # Bare displacement coefficient

    # Dynamic shielding from electron vortices
    shield = SHIELD_BASE * vortex_shielding_factor(Z, A)

    a_disp_effective = a_disp_bare * shield

    E_vac = a_disp_effective * (Z**2) / (A**(1/3))

    return E_vac

# ============================================================================
# TEMPORAL METRIC MODULATION
# ============================================================================
def lambda_time_effective(Z, A):
    """
    Time metric modulation from combined soliton-vortex density.

    QFD Theory: High field density slows local clock (gravitational redshift).

    Previous: λ_time = λ₀ + κ_e × Z  (linear electron effect)
    New:      Include core density contribution

    The nucleus+electrons create a "time well" - the metric modulation
    depends on TOTAL density, not just electron count.
    """
    # Linear baseline (from calibration)
    lambda_linear = LAMBDA_TIME_0 + KAPPA_E * Z

    # Core density contribution (could add nonlinear terms)
    # For now, keep linear form but acknowledge it's a metric effect

    return lambda_linear

# ============================================================================
# ENERGY FUNCTIONAL WITH VORTEX SHIELDING
# ============================================================================
def get_resonance_bonus(Z, N, E_surface):
    """Magic number resonance (unchanged from baseline)."""
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_energy_vortex_shielding(A, Z):
    """
    QFD energy with electron vortex shielding coupling.

    Key change: Displacement term now has Z-dependent shielding
    reflecting electron vortex refraction of vacuum field.
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    # Temporal metric (electron contribution)
    lambda_time = lambda_time_effective(Z, A)

    # Standard QFD terms
    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    # Energy components
    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)

    # VORTEX-SHIELDED DISPLACEMENT (the key new physics)
    E_vac = get_displacement_energy(Z, A)

    # Geometric resonance
    E_iso = -get_resonance_bonus(Z, N, E_surface)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_vortex(A):
    """Find stable Z with vortex shielding."""
    best_Z = 1
    best_E = qfd_energy_vortex_shielding(A, 1)
    for Z in range(1, A):
        E = qfd_energy_vortex_shielding(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

# ============================================================================
# LOAD DATA
# ============================================================================
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()

start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("VORTEX SHIELDING COUPLING - QFD Book Implementation")
print("="*95)
print()
print("THEORY: Electron vortices create topological shield around nuclear Q-ball")
print("        Aharonov-Bohm coupling refracts vacuum displacement gradient")
print()
print("KEY PHYSICS:")
print("  1. Electrons are VORTEX SOLITONS, not point particles")
print("  2. They SHIELD the core by refracting ∇ρ_vacuum")
print("  3. Shielding increases with Z but saturates at high electron count")
print()
print(f"Parameters:")
print(f"  shield_base:    {SHIELD_BASE:.3f}  (bare core shielding)")
print(f"  κ_vortex:       {KAPPA_VORTEX:.4f}  (vortex refraction strength)")
print(f"  ζ_saturation:   {ZETA_SATURATION:.4f}  (high-Z saturation rate)")
print(f"  λ_time_0:       {LAMBDA_TIME_0:.3f}")
print(f"  κ_e:            {KAPPA_E:.4f}  (temporal metric modulation)")
print()

# Show shielding variation
print("Vortex shielding factor for representative nuclei:")
print(f"{'Nuclide':<12} {'Z':<5} {'A':<6} {'Z/A':<8} {'Shield(Z,A)':<15} {'vs Fixed (0.52)'}")
print("-"*95)

test_cases = [
    ("H-1", 1, 1),
    ("He-4", 2, 4),
    ("O-16", 8, 16),
    ("Ca-40", 20, 40),
    ("Fe-56", 26, 56),
    ("Ni-58", 28, 58),
    ("Sn-112", 50, 112),
    ("Xe-136", 54, 136),
    ("Pb-208", 82, 208),
    ("U-238", 92, 238),
]

for name, Z, A in test_cases:
    q = Z / A
    shield_dynamic = SHIELD_BASE * vortex_shielding_factor(Z, A)
    shield_fixed = 0.52
    ratio = shield_dynamic / shield_fixed

    marker = "↑" if shield_dynamic > shield_fixed else ("↓" if shield_dynamic < shield_fixed else "=")

    print(f"{name:<12} {Z:<5} {A:<6} {q:<8.3f} {shield_dynamic:<15.4f} {ratio:.3f}× {marker}")

print()

# ============================================================================
# EVALUATE ON 285 NUCLIDES
# ============================================================================
print("="*95)
print("EVALUATION ON 285 NUCLIDES")
print("="*95)
print()

results_vortex = []
results_baseline = []

# Baseline (fixed shielding)
def qfd_baseline(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_E * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    # Fixed shielding (previous optimum)
    a_disp_fixed = (alpha_fine * 197.327 / 1.2) * 0.52

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp_fixed * (Z**2) / (A**(1/3))
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

for name, Z_exp, A in test_nuclides:
    # With vortex shielding
    Z_vortex = find_stable_Z_vortex(A)
    Delta_vortex = Z_vortex - Z_exp

    # Baseline
    Z_base = find_stable_Z_baseline(A)
    Delta_base = Z_base - Z_exp

    results_vortex.append({'name': name, 'A': A, 'Z_exp': Z_exp,
                          'Z_pred': Z_vortex, 'Delta_Z': Delta_vortex})
    results_baseline.append({'name': name, 'A': A, 'Z_exp': Z_exp,
                            'Z_pred': Z_base, 'Delta_Z': Delta_base})

# ============================================================================
# STATISTICS
# ============================================================================
errors_vortex = [abs(r['Delta_Z']) for r in results_vortex]
errors_base = [abs(r['Delta_Z']) for r in results_baseline]

exact_vortex = sum(e == 0 for e in errors_vortex)
exact_base = sum(e == 0 for e in errors_base)

print(f"{'Model':<45} {'Exact':<20} {'Mean |ΔZ|':<15} {'Median |ΔZ|'}")
print("-"*95)
print(f"{'Baseline (fixed shield=0.52)':<45} {exact_base}/{len(results_baseline)} ({100*exact_base/len(results_baseline):.1f}%)  "
      f"{np.mean(errors_base):<15.3f} {np.median(errors_base):.1f}")
print(f"{'Vortex Shielding (dynamic)':<45} {exact_vortex}/{len(results_vortex)} ({100*exact_vortex/len(results_vortex):.1f}%)  "
      f"{np.mean(errors_vortex):<15.3f} {np.median(errors_vortex):.1f}")
print()

improvement = exact_vortex - exact_base
delta_pct = improvement / len(results_vortex) * 100

if improvement > 0:
    print(f"✓ IMPROVEMENT: +{improvement} exact matches (+{delta_pct:.1f} percentage points)")
    print()
    print("Vortex shielding coupling successfully models electron-nucleus interaction!")
    print("Electrons REFRACT vacuum pressure, allowing heavy nuclei to stabilize.")
elif improvement < 0:
    print(f"✗ REGRESSION: {improvement} exact matches")
    print()
    print("Current vortex shielding parameters may need calibration.")
    print("Try adjusting κ_vortex, ζ_saturation, or functional form.")
else:
    print("= NEUTRAL: No change from baseline")
    print()
    print("Vortex shielding effect may be too weak with current parameters.")

print()

# By mass region
print("="*95)
print("PERFORMANCE BY MASS REGION")
print("="*95)
print()

for model_name, results in [("BASELINE", results_baseline), ("VORTEX SHIELDING", results_vortex)]:
    light = [r for r in results if r['A'] < 40]
    medium = [r for r in results if 40 <= r['A'] < 100]
    heavy = [r for r in results if 100 <= r['A'] < 200]
    superheavy = [r for r in results if r['A'] >= 200]

    print(f"{model_name}:")
    print("-"*95)
    for region_name, group in [("Light (A<40)", light), ("Medium (40≤A<100)", medium),
                                ("Heavy (100≤A<200)", heavy), ("Superheavy (A≥200)", superheavy)]:
        if len(group) > 0:
            errs = [abs(r['Delta_Z']) for r in group]
            ex = sum(e == 0 for e in errs)
            print(f"  {region_name:<25} {ex}/{len(group)} ({100*ex/len(group):>5.1f}%)  "
                  f"Mean|ΔZ|={np.mean(errs):.2f}")
    print()

# ============================================================================
# KEY NUCLEI TEST
# ============================================================================
print("="*95)
print("KEY NUCLEI (Critical Test Cases)")
print("="*95)
print()

key_nuclei = [
    ("He-4", 2, 4),
    ("O-16", 8, 16),
    ("Ca-40", 20, 40),
    ("Fe-56", 26, 56),
    ("Ni-58", 28, 58),
    ("Sn-112", 50, 112),
    ("Pb-208", 82, 208),
    ("U-238", 92, 238),
]

print(f"{'Nuclide':<12} {'Z_exp':<8} {'Baseline':<12} {'Vortex':<12} {'Status'}")
print("-"*95)

for name, Z_exp, A in key_nuclei:
    r_vortex = next((r for r in results_vortex if r['name'] == name), None)
    r_base = next((r for r in results_baseline if r['name'] == name), None)

    if r_vortex and r_base:
        base_str = "✓" if r_base['Delta_Z'] == 0 else f"{r_base['Delta_Z']:+d}"
        vortex_str = "✓" if r_vortex['Delta_Z'] == 0 else f"{r_vortex['Delta_Z']:+d}"

        if abs(r_vortex['Delta_Z']) < abs(r_base['Delta_Z']):
            status = "✓ Improved"
        elif abs(r_vortex['Delta_Z']) > abs(r_base['Delta_Z']):
            status = "✗ Worse"
        else:
            status = "= Same"

        print(f"{name:<12} {Z_exp:<8} {base_str:<12} {vortex_str:<12} {status}")

print()
print("="*95)
print("VERDICT")
print("="*95)
print()

if improvement > 20:
    print("✓✓✓ BREAKTHROUGH: Vortex shielding resolves major failure modes!")
    print()
    print("QFD Interpretation:")
    print("  - Electrons are VORTEX SOLITONS in far-field of Q-ball")
    print("  - Aharonov-Bohm coupling creates topological shield")
    print("  - Shield refracts ∇ρ_vacuum, reducing displacement stress")
    print("  - This is why heavy nuclei can sustain high Z")
    print()
    print("This confirms the coupled soliton-vortex system picture!")
elif improvement > 10:
    print("✓✓ SIGNIFICANT IMPROVEMENT from vortex shielding")
    print()
    print("Next steps:")
    print("  1. Calibrate κ_vortex and ζ_saturation for optimal fit")
    print("  2. Add angular momentum quantization (vortex locking)")
    print("  3. Include temporal metric modulation (nonlinear λ_time)")
elif improvement > 0:
    print("✓ MODEST IMPROVEMENT - vortex shielding has measurable effect")
    print()
    print("Next steps:")
    print("  1. Test different functional forms for shield(Z,A)")
    print("  2. Add electron shell structure to shielding")
    print("  3. Implement saturation more carefully")
else:
    print("Current vortex shielding model doesn't improve predictions")
    print()
    print("Possible issues:")
    print("  1. Functional form wrong (try different shield(Z,A))")
    print("  2. Parameters need optimization (κ_vortex, ζ_saturation)")
    print("  3. May need to couple to OTHER terms (not just displacement)")
    print("  4. Angular momentum locking may be the primary missing physics")

print()
print("="*95)
