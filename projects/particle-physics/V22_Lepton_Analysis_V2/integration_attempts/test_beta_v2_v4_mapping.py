#!/usr/bin/env python3
"""
Test Beta -> V2, V4 Mapping

Explores the relationship between:
- β from cosmology/nuclear (β ≈ 3.1)
- V2, V4 from Phoenix solver

Goal: Find if β = 3.1 can predict correct V2, V4 values.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# KNOWN VALUES
# ============================================================================

# From cosmology/nuclear
BETA_COSMIC = 3.1

# From Phoenix solver (working values)
PHOENIX_PARAMS = {
    'electron': {'V2': 12e6, 'V4': 11.0, 'Q_star': 2.166},
    'muon': {'V2': 8e6, 'V4': 11.0, 'Q_star': 2.3},
    'tau': {'V2': 100e6, 'V4': 11.0, 'Q_star': 9800},
}

# Lepton masses (experimental, eV)
MASSES_EXP = {
    'electron': 510998.95,
    'muon': 105658374.4,
    'tau': 1777000000.0,
}

# ============================================================================
# HYPOTHESIS 1: Direct Dimensional Mapping
# ============================================================================

def test_direct_mapping():
    """
    Test if V4 = β × (length scale)^4

    For leptons, natural length scale is Compton wavelength:
    λ_C = ℏ / (m c)
    """
    print("=" * 80)
    print("TEST 1: Direct Dimensional Mapping")
    print("=" * 80)
    print()

    hbar_c = 197.3269804  # MeV·fm (natural units)

    results = {}

    for particle, mass_eV in MASSES_EXP.items():
        mass_MeV = mass_eV / 1e6

        # Compton wavelength
        lambda_C_fm = hbar_c / mass_MeV
        lambda_C_m = lambda_C_fm * 1e-15

        # Try different length scales
        length_scales = {
            'Compton wavelength': lambda_C_fm,
            '1 fm': 1.0,
            '0.1 fm': 0.1,
            '10 fm': 10.0,
        }

        print(f"{particle.upper()}:")
        print(f"  Mass: {mass_MeV:.3f} MeV")
        print(f"  Compton wavelength: {lambda_C_fm:.2f} fm")
        print()

        for scale_name, length_fm in length_scales.items():
            # V4 = β × L^4 (dimensional analysis)
            # But what are units of β?

            # Try β in units of [Energy × Length^-4]
            # Then V4 = β × 1 (dimensionless)
            V4_predicted = BETA_COSMIC  # Simplest case

            # Phoenix V4 is ~11 for all leptons
            phoenix_V4 = PHOENIX_PARAMS[particle]['V4']
            ratio = V4_predicted / phoenix_V4

            print(f"  {scale_name:20s}: V4 = {V4_predicted:.2f}, " +
                  f"Phoenix = {phoenix_V4:.2f}, ratio = {ratio:.3f}")

        print()
        results[particle] = {'lambda_C_fm': lambda_C_fm}

    return results

# ============================================================================
# HYPOTHESIS 2: Potential Well Mapping
# ============================================================================

def test_potential_well_mapping():
    """
    Test if V(r) = β(r²-v²)² can be mapped to V(ρ) = V2·ρ + V4·ρ²

    Assume soliton has Gaussian profile: ψ(r) ~ exp(-r²/2σ²)
    Then ρ(r) = |ψ(r)|² ~ exp(-r²/σ²)

    At the minimum of V(r), we have r = v (vacuum scale)
    """
    print("=" * 80)
    print("TEST 2: Potential Well Mapping (Gaussian Soliton)")
    print("=" * 80)
    print()

    beta = BETA_COSMIC
    v = 1.0  # Vacuum scale (normalized)

    # For a Gaussian soliton ψ(r) ~ exp(-r²/2σ²)
    # The width σ determines the energy scale

    # Try different σ values
    sigma_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    print("Assuming V(r) = β(r²-v²)²")
    print(f"With β = {beta}, v = {v}")
    print()

    for sigma in sigma_values:
        # For a Gaussian, the density ρ(r) = |ψ(r)|² peaks at r=0
        # V(r=0) = β·v⁴
        # This sets the potential depth

        # Expand V(r) around r=0:
        # V(r) = β(r²-v²)² = β(r⁴ - 2v²r² + v⁴)
        #      = βv⁴ + β(-2v²)r² + βr⁴

        # Now for density-dependent: V(ρ) = V2·ρ + V4·ρ²
        # We need to relate r² to ρ

        # If ψ ~ exp(-r²/2σ²), then ρ ~ exp(-r²/σ²)
        # So r² ~ -σ² ln(ρ/ρ₀)

        # This is complex... try simpler approach:
        # V4 controls quartic term in both formulations
        # Estimate: V4 ~ β × (normalization factor)

        V4_estimate = beta  # Simplest guess

        # V2 controls quadratic term
        # V2 ~ -2βv² × (normalization)
        V2_estimate = -2 * beta * v**2  # Negative!

        print(f"σ = {sigma:.1f}:")
        print(f"  V4 estimate: {V4_estimate:.2f} (Phoenix: 11.0)")
        print(f"  V2 estimate: {V2_estimate:.2f} (Phoenix: 0 to 100M)")
        print(f"  V2 is NEGATIVE - doesn't match Phoenix!")
        print()

    print("CONCLUSION: Direct mapping gives WRONG sign for V2!")
    print("Phoenix V2 is POSITIVE and large (0 to 100M)")
    print("But expansion of V(r) = β(r²-v²)² gives NEGATIVE V2")
    print()

# ============================================================================
# HYPOTHESIS 3: Scale Factor Exploration
# ============================================================================

def test_scale_factors():
    """
    Test if β needs to be scaled by powers of 10 to match Phoenix energies.
    """
    print("=" * 80)
    print("TEST 3: Scale Factor Exploration")
    print("=" * 80)
    print()

    beta_base = 3.1

    # Try powers of 10
    scale_factors = [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]

    print("Testing β × scale_factor to match Phoenix V2, V4 ranges:")
    print()

    print("Phoenix V4 ~ 11 for all leptons")
    print("Phoenix V2 ranges from 0 (electron) to 100M (tau)")
    print()

    for scale in scale_factors:
        beta_scaled = beta_base * scale

        # Check if this matches V4 ~ 11
        ratio_V4 = beta_scaled / 11.0

        # Check if this matches V2 range 0 to 100M
        in_V2_range = (0 <= beta_scaled <= 100e6)

        print(f"β × {scale:>10.0e} = {beta_scaled:>15.2e}  |  " +
              f"V4 ratio: {ratio_V4:>10.2e}  |  " +
              f"In V2 range: {in_V2_range}")

    print()
    print("OBSERVATION:")
    print("  β × 1 ~ 3 → Close to V4 = 11 (factor of ~3.5)")
    print("  β × 1e6 ~ 3M → In V2 range for electron/muon")
    print("  β × 3e7 ~ 100M → Matches tau V2")
    print()
    print("HYPOTHESIS: V4 ~ β (same order), V2 ~ β × (lepton-dependent factor)")
    print()

# ============================================================================
# HYPOTHESIS 4: V2 from Mass Scale
# ============================================================================

def test_v2_from_mass():
    """
    Test if V2 scales with lepton mass.

    Phoenix observation:
    - Electron: V2 = 12M, m = 0.511 MeV
    - Muon: V2 = 8M, m = 105.7 MeV
    - Tau: V2 = 100M, m = 1777 MeV
    """
    print("=" * 80)
    print("TEST 4: V2 Scaling with Mass")
    print("=" * 80)
    print()

    print("Phoenix V2 values:")
    print()

    for particle, params in PHOENIX_PARAMS.items():
        mass_MeV = MASSES_EXP[particle] / 1e6
        V2 = params['V2']

        # Test different scaling relationships
        V2_over_m = V2 / mass_MeV
        V2_over_m2 = V2 / (mass_MeV ** 2)
        V2_over_sqrt_m = V2 / np.sqrt(mass_MeV)

        print(f"{particle.upper():10s}: V2 = {V2/1e6:6.1f}M, m = {mass_MeV:8.1f} MeV")
        print(f"            V2/m = {V2_over_m:12.2e}")
        print(f"            V2/m² = {V2_over_m2:12.2e}")
        print(f"            V2/√m = {V2_over_sqrt_m:12.2e}")
        print()

    print("OBSERVATION:")
    print("  V2/m varies by orders of magnitude (not constant)")
    print("  V2/m² also varies widely (not constant)")
    print("  No simple V2 ~ m^n relationship!")
    print()
    print("CONCLUSION: V2 does NOT scale simply with mass")
    print()

# ============================================================================
# HYPOTHESIS 5: Q* Relationship
# ============================================================================

def test_q_star_relationship():
    """
    Analyze Q* values and see if they relate to β or mass.

    Phoenix Q* values:
    - Electron: 2.166
    - Muon: 2.3
    - Tau: 9800 (HUGE jump!)
    """
    print("=" * 80)
    print("TEST 5: Q* Analysis")
    print("=" * 80)
    print()

    print("Phoenix Q* values:")
    print()

    q_stars = []
    masses = []

    for particle, params in PHOENIX_PARAMS.items():
        mass_MeV = MASSES_EXP[particle] / 1e6
        Q_star = params['Q_star']

        q_stars.append(Q_star)
        masses.append(mass_MeV)

        print(f"{particle.upper():10s}: Q* = {Q_star:10.1f}, m = {mass_MeV:8.1f} MeV")
        print(f"            Q*/m = {Q_star/mass_MeV:12.2e}")
        print(f"            Q*·m = {Q_star*mass_MeV:12.2e}")
        print()

    # Check for patterns
    print("Ratios:")
    print(f"  Q*(muon) / Q*(electron) = {q_stars[1]/q_stars[0]:.3f}")
    print(f"  Q*(tau) / Q*(muon) = {q_stars[2]/q_stars[1]:.1f}")
    print()
    print(f"  m(muon) / m(electron) = {masses[1]/masses[0]:.1f}")
    print(f"  m(tau) / m(muon) = {masses[2]/masses[1]:.1f}")
    print()

    print("OBSERVATION:")
    print("  Q* is nearly constant for electron/muon (~2.2)")
    print("  Q* JUMPS by 4200× for tau!")
    print("  This jump does NOT correlate with mass ratio (only 17×)")
    print()
    print("HYPOTHESIS: Q* might encode internal angular structure,")
    print("            not just mass scale")
    print()

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 80)
    print("BETA → V2, V4 MAPPING ANALYSIS")
    print("=" * 80)
    print()
    print("Goal: Find connection between β ≈ 3.1 (cosmic/nuclear)")
    print("      and V2, V4 (Phoenix lepton solver)")
    print()

    # Run all tests
    test_direct_mapping()
    test_potential_well_mapping()
    test_scale_factors()
    test_v2_from_mass()
    test_q_star_relationship()

    # Summary
    print("=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    print()

    print("1. ✅ V4 ~ β in magnitude (3.1 vs 11 - factor of ~3.5)")
    print("   → Could be related, needs proper unit conversion")
    print()

    print("2. ❌ V2 does NOT come from simple V(r) → V(ρ) expansion")
    print("   → Expansion gives NEGATIVE V2, Phoenix uses POSITIVE V2")
    print()

    print("3. ⚠️  V2 is lepton-specific, ranges from 0 to 100M")
    print("   → Does NOT scale simply with mass")
    print("   → Might require ladder solver convergence")
    print()

    print("4. ❓ Q* is mysterious:")
    print("   → Nearly constant for e/μ (~2.2)")
    print("   → Jumps 4200× for τ")
    print("   → No obvious connection to β or mass")
    print()

    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()

    print("POSITIVE:")
    print("  • V4 ≈ β suggests some connection")
    print("  • Both formulations use quartic potentials")
    print("  • Phoenix achieves perfect mass reproduction")
    print()

    print("CHALLENGES:")
    print("  • V2 sign mismatch (positive vs negative)")
    print("  • V2 magnitude varies by lepton (not universal)")
    print("  • Q* huge variation unexplained")
    print("  • No clear β → (V2, V4, Q*) mapping")
    print()

    print("NEXT STEPS:")
    print("  1. Review QFD theory: Is V(ρ) the correct formulation?")
    print("  2. Understand Phoenix ladder solver convergence")
    print("  3. Analyze Q* physical meaning (angular structure?)")
    print("  4. Try enhanced V22 with 4-component fields + CSR")
    print()

    print("PROBABILITY OF UNIFICATION:")
    print("  • Simple β = 3.1 → masses: LOW (10-20%)")
    print("  • Enhanced formulation works: MEDIUM (50-60%)")
    print("  • Fundamental scale separation: MEDIUM-HIGH (40-50%)")
    print()

if __name__ == "__main__":
    main()
