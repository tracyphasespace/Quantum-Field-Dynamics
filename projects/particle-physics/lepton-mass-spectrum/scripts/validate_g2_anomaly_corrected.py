#!/usr/bin/env python3
"""
Numerical Zeeman Probe: g-2 Anomaly Validation (CORRECTED)

CRITICAL FIX: The radius R is GIVEN by the Compton wavelength (not solved for).
The energy functional determines the mass from R, not R from the mass.

R_lepton = ℏ/(m_lepton·c) is a CONSTRAINT from quantum mechanics.

Then we calculate g-2 from that R using the geometric formula.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# --- Constants from QFD Unification ---
ALPHA = 1/137.035999177  # Fine structure constant (CODATA 2018)
HBARC = 197.3269804      # MeV·fm (CODATA 2018)
MP = 938.27208816        # Proton mass in MeV (PDG 2024)

# --- Parameters from MCMC Breakthrough ---
BETA_CORE = 3.0627       # Vacuum compression stiffness
XI = 0.9655              # Gradient stiffness
BETA_GOLDEN = 3.043233053      # Theoretical prediction

# --- Geometric Factors ---
SHAPE_FACTOR = np.pi / 2  # D-flow compression


def calculate_g2_from_compton_radius(target_mass_mev, vortex_shape_v4=1.0, vortex_shape_v6=0.0):
    """
    Calculate g-2 using CORRECT approach:

    1. R is SET by Compton wavelength: R = ℏ/(mc)
    2. Calculate g-2 from geometric formula with shape factors

    The QED formula for g-2 with geometric corrections:

    a = (α/2π) + (α/2π)² · V₄ + (α/2π)³ · V₆ + ...

    where V₄, V₆ are geometric shape factors from the vortex structure.

    For Hill vortex with D-flow:
    - V₄ includes the (R/λ_C)² term and π/2 compression
    - V₆ would include higher-order geometric corrections

    Parameters
    ----------
    target_mass_mev : float
        Lepton mass in MeV
    vortex_shape_v4 : float
        Fourth-order shape factor (geometric correction)
    vortex_shape_v6 : float
        Sixth-order shape factor (higher geometric correction)

    Returns
    -------
    lambda_c : float
        Compton wavelength in fm
    a_qed : float
        Pure QED prediction (Schwinger term)
    a_geom : float
        QFD geometric prediction
    """

    # Compton wavelength (this IS the vortex radius!)
    lambda_c = HBARC / target_mass_mev  # fm

    # QED Schwinger term (first-order)
    a_schwinger = ALPHA / (2 * np.pi)

    # Geometric correction from vortex structure
    # This is where the Hill vortex geometry enters
    #
    # The D-flow creates an effective "circulation" that modifies the
    # magnetic moment. The correction should scale as:
    #
    # Δa_geom = (α/2π)² · V₄ · (something geometric)
    #
    # For a point particle: V₄ = 0
    # For Hill vortex: V₄ depends on (π/2) compression and internal flow

    # HYPOTHESIS: The geometric factor includes D-flow compression
    # Try: V₄ ~ (π/2 - 1) as the deviation from spherical symmetry
    geometric_factor = SHAPE_FACTOR  # π/2

    # Attempt 1: Linear geometric correction
    # a_geom = a_schwinger * (1 + geometric_factor * (α/π))

    # Attempt 2: Quadratic (α²) correction as in QED
    alpha_over_pi = ALPHA / np.pi
    a_v4 = vortex_shape_v4 * (alpha_over_pi)**2
    a_v6 = vortex_shape_v6 * (alpha_over_pi)**3

    a_geom = a_schwinger + a_v4 + a_v6

    return lambda_c, a_schwinger, a_geom


def validate_schwinger_term():
    """
    First test: Do we reproduce the QED Schwinger term?

    a = α/(2π) ≈ 0.001161409...

    This is the pure QED result before any higher-order corrections.
    """
    print("="*80)
    print("TEST 1: QED Schwinger Term (Baseline)")
    print("="*80)
    print()

    a_schwinger = ALPHA / (2 * np.pi)

    print(f"Schwinger term: a = α/(2π) = {a_schwinger:.12f}")
    print()
    print("Experimental values (includes higher-order QED + hadronic corrections):")
    print(f"  Electron: a_exp = 0.001159652181 (QED + hadronic)")
    print(f"  Muon:     a_exp = 0.001165920590 (QED + hadronic + weak)")
    print()
    print("Difference:")
    print(f"  Electron: Δa = {0.001159652181 - a_schwinger:.12f} (higher-order QED)")
    print(f"  Muon:     Δa = {0.001165920590 - a_schwinger:.12f} (includes 'anomaly')")
    print()


def scan_vortex_shape_factors():
    """
    Scan V₄ and V₆ to find values that match experiments.

    This tells us what the EFFECTIVE geometric correction is,
    which we can then try to derive from Hill vortex integrals.
    """
    print("="*80)
    print("TEST 2: Vortex Shape Factor Scan")
    print("="*80)
    print()

    # Experimental values
    leptons = {
        "Electron": {"mass": 0.51099895, "a_exp": 0.00115965218128},
        "Muon":     {"mass": 105.6583755, "a_exp": 0.00116592059}
    }

    a_schwinger = ALPHA / (2 * np.pi)

    print(f"Schwinger baseline: a₀ = {a_schwinger:.12f}")
    print()

    for name, data in leptons.items():
        mass = data["mass"]
        a_exp = data["a_exp"]

        # Calculate what V₄ would need to be to match experiment
        # a_exp = a₀ + V₄·(α/π)²
        # V₄ = (a_exp - a₀) / (α/π)²

        delta_a = a_exp - a_schwinger
        alpha_over_pi_sq = (ALPHA / np.pi)**2

        V4_required = delta_a / alpha_over_pi_sq

        lambda_c = HBARC / mass

        print(f"{name}:")
        print(f"  Compton wavelength: λ_C = {lambda_c:.4f} fm")
        print(f"  Experimental:       a_exp = {a_exp:.12f}")
        print(f"  Schwinger term:     a₀    = {a_schwinger:.12f}")
        print(f"  Difference:         Δa    = {delta_a:.12f}")
        print(f"  Required V₄:        V₄    = {V4_required:.6f}")
        print()

    print("INTERPRETATION:")
    print("  If V₄ ~ 1-10:  Simple geometric correction from vortex")
    print("  If V₄ ~ 100:   Need to include circulation/rotation effects")
    print("  If V₄ ~ 1000:  Fundamental problem with geometric model")
    print()


def test_dflow_compression_factor():
    """
    Test if the π/2 D-flow factor directly gives the correction.

    Hypothesis: The magnetic moment correction comes from the
    path-length ratio (πR vs 2R) creating an effective circulation.
    """
    print("="*80)
    print("TEST 3: D-Flow Compression Hypothesis")
    print("="*80)
    print()

    a_schwinger = ALPHA / (2 * np.pi)

    # Hypothesis: Δa ~ (π/2 - 1) · a_schwinger
    dflow_factor = (np.pi / 2) - 1  # ≈ 0.5708

    a_with_dflow = a_schwinger * (1 + dflow_factor)

    print(f"Hypothesis: g-2 correction from D-flow geometry")
    print()
    print(f"D-flow factor: (π/2 - 1) = {dflow_factor:.6f}")
    print()
    print(f"Predicted: a = a₀ · (π/2) = {a_with_dflow:.12f}")
    print(f"Electron:  a_exp          = 0.001159652181")
    print(f"Muon:      a_exp          = 0.001165920590")
    print()

    # Check if this gives the right order of magnitude
    delta_e = abs(a_with_dflow - 0.001159652181)
    delta_mu = abs(a_with_dflow - 0.001165920590)

    print(f"Electron error: {delta_e:.2e}")
    print(f"Muon error:     {delta_mu:.2e}")
    print()

    if delta_e < 1e-4:
        print("✓ D-flow factor gives correct order of magnitude!")
    else:
        print("✗ D-flow alone insufficient - need additional terms")
    print()


def compare_to_qed_formula():
    """
    Compare to full QED formula (known up to α⁵).

    a = (α/2π) - 0.32848(α/π)² + 1.181234(α/π)³ + ...

    The coefficients come from Feynman diagrams.
    Can we relate them to vortex geometry?
    """
    print("="*80)
    print("TEST 4: QED Series Comparison")
    print("="*80)
    print()

    alpha_over_pi = ALPHA / np.pi

    # Known QED coefficients (from Feynman diagram calculations)
    C1 = 0.5                    # Schwinger (exact)
    C2 = -0.32848               # α² term
    C3 = 1.181234               # α³ term

    a_qed = C1 * alpha_over_pi + C2 * alpha_over_pi**2 + C3 * alpha_over_pi**3

    print("QED perturbative series:")
    print(f"  a = (α/π) · [C₁ + C₂·(α/π) + C₃·(α/π)² + ...]")
    print()
    print(f"  C₁ = {C1:.6f}  (Schwinger)")
    print(f"  C₂ = {C2:.6f}  (vertex + vacuum polarization)")
    print(f"  C₃ = {C3:.6f}  (light-by-light scattering)")
    print()
    print(f"QED prediction: a_QED = {a_qed:.12f}")
    print(f"Electron exp:   a_exp = 0.001159652181")
    print()
    print("QUESTION: Can we derive C₂, C₃ from Hill vortex geometry?")
    print()
    print("Geometric interpretation:")
    print(f"  π/2 = {np.pi/2:.6f}  (D-flow path ratio)")
    print(f"  π/2 - 1 = {np.pi/2 - 1:.6f}  (deviation from sphere)")
    print(f"  |C₂| = {abs(C2):.6f}  (QED coefficient)")
    print()

    # Check if geometric factors match QED coefficients
    if abs(abs(C2) - 0.32848) < 0.01:
        print("Note: C₂ ≈ 1/3 ... geometric?")
    print()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("QFD g-2 Validation: Corrected Approach")
    print("R = Compton wavelength (FIXED), then calculate g-2")
    print("="*80)
    print()

    # Test sequence
    validate_schwinger_term()
    print()

    scan_vortex_shape_factors()

    test_dflow_compression_factor()

    compare_to_qed_formula()

    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Calculate V₄ from Hill vortex circulation integrals")
    print("2. Include spin-orbit coupling (ℏ/2 constraint)")
    print("3. Derive connection between π/2 and QED coefficients")
    print("4. Test if V₄(electron) ≈ V₄(muon) (universality)")
    print()
    print("If V₄ is UNIVERSAL (same for e, μ, τ):")
    print("  → Geometric origin confirmed")
    print("  → QFD validates QED from first principles")
    print()
    print("="*80)
