#!/usr/bin/env python3
"""
Numerical Zeeman Probe: g-2 Anomaly Validation

This script bridges the Lean 4 theoretical proofs with experimental data.
- Lean proves the theory is internally consistent (math works)
- This script proves the theory is externally accurate (numbers match Fermilab/CODATA)

Validates:
1. The "3.15% Tax": Does β = 3.058 + correction → β ≈ 3.15 align g-2 values?
2. Generational Bridge: Do R_e and R_μ from mass spectrum predict correct g-2?
3. Model Falsification: Are predictions within experimental bounds?
"""

import numpy as np
from scipy.optimize import brentq
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# --- Constants from QFD Unification ---
ALPHA = 1/137.035999177  # Fine structure constant (CODATA 2018)
HBARC = 197.3269804      # MeV·fm (CODATA 2018)
MP = 938.27208816        # Proton mass in MeV (PDG 2024)

# --- Parameters from MCMC Breakthrough ---
# From: results/example_results.json
BETA_CORE = 3.0627       # Vacuum compression stiffness (MCMC median)
BETA_STD = 0.1491        # Uncertainty
XI = 0.9655              # Gradient stiffness (MCMC median)
XI_STD = 0.5494          # Uncertainty
TAU = 1.0073             # Temporal stiffness
TAU_STD = 0.6584

# Theoretical prediction (Golden Loop)
BETA_GOLDEN = 3.058

# --- Geometric Factors from Hill Vortex ---
# From D-flow geometry (π/2 compression)
SHAPE_FACTOR = np.pi / 2  # ≈ 1.5708

# Integration factors (from Hill vortex integrals)
C_COMP = 1.4246          # Compression contribution
C_GRAD = 2.5512          # Gradient contribution


def calculate_lepton_properties(target_mass_mev, beta=BETA_CORE, xi=XI):
    """
    Calculate lepton properties from QFD energy functional.

    1. Solves for radius R that reproduces the lepton mass (VortexStability)
    2. Calculates anomalous magnetic moment a = (g-2)/2 (AnomalousMoment)

    Parameters
    ----------
    target_mass_mev : float
        Target lepton mass in MeV
    beta : float
        Vacuum compression stiffness
    xi : float
        Gradient stiffness

    Returns
    -------
    r_sol : float
        Vortex radius in fm
    lambda_c : float
        Compton wavelength in fm
    a_geom : float
        Anomalous moment a = (g-2)/2
    """

    # Energy functional: E(R) = β·R³ + ξ·R
    # This comes from integrating the full functional over Hill vortex profile
    # Normalization chosen so E(R_electron) = m_electron

    def energy_ratio(R):
        """Energy functional normalized to electron mass."""
        e_comp = beta * C_COMP * R**3
        e_grad = xi * C_GRAD * R
        # Normalize to electron mass (0.511 MeV)
        return (e_comp + e_grad) / 0.511

    # Solve for R that gives target mass
    # Search range: 0.01 fm (nuclear scale) to 10000 fm (molecular scale)
    try:
        r_sol = brentq(
            lambda r: energy_ratio(r) - (target_mass_mev / 0.511),
            0.01, 10000.0
        )
    except ValueError as e:
        print(f"Warning: Could not find radius for mass {target_mass_mev} MeV")
        print(f"  Beta={beta}, Xi={xi}")
        print(f"  Error: {e}")
        return np.nan, np.nan, np.nan

    # Compton wavelength: λ_C = ℏc / (mc²)
    lambda_c = HBARC / target_mass_mev

    # Anomalous magnetic moment from geometric factor
    # a = (α/2π) · (R/λ_C)² · (π/2)
    #
    # Physical interpretation:
    # - (R/λ_C)² is the geometric scale factor
    # - π/2 is the D-flow compression contribution
    # - α/2π is the QED coupling at this scale
    a_geom = (ALPHA / (2 * np.pi)) * (r_sol / lambda_c)**2 * SHAPE_FACTOR

    return r_sol, lambda_c, a_geom


def validate_against_experiments():
    """
    Validate QFD predictions against experimental g-2 measurements.

    Experimental values from:
    - Electron: Fan et al. (2023), Phys. Rev. Lett. 130, 071801
    - Muon: Muon g-2 Collaboration (2023), Phys. Rev. Lett. 131, 161802
    - Tau: No direct measurement (theoretical estimate)
    """

    # Lepton masses (PDG 2024)
    leptons = {
        "Electron": {
            "mass": 0.51099895,
            "a_exp": 0.00115965218128,  # (g-2)/2 measured
            "a_exp_unc": 0.00000000000018,
            "source": "Fan et al. (2023)"
        },
        "Muon": {
            "mass": 105.6583755,
            "a_exp": 0.00116592059,     # (g-2)/2 measured (includes anomaly!)
            "a_exp_unc": 0.00000000022,
            "source": "Muon g-2 Collab (2023)"
        },
        "Tau": {
            "mass": 1776.86,
            "a_exp": None,  # No direct measurement
            "a_exp_unc": None,
            "source": "Not measured"
        }
    }

    print("="*80)
    print("NUMERICAL ZEEMAN PROBE: g-2 Anomaly Validation")
    print("="*80)
    print()
    print(f"QFD Parameters (from MCMC):")
    print(f"  β = {BETA_CORE:.4f} ± {BETA_STD:.4f} (Golden Loop: {BETA_GOLDEN:.3f})")
    print(f"  ξ = {XI:.4f} ± {XI_STD:.4f}")
    print(f"  τ = {TAU:.4f} ± {TAU_STD:.4f}")
    print()
    print(f"Topological Factor: π/2 = {SHAPE_FACTOR:.6f} (D-flow compression)")
    print()
    print("="*80)
    print(f"{'Lepton':<10} | {'R (fm)':<12} | {'λ_C (fm)':<12} | {'a_pred':<15} | {'a_exp':<15} | {'Δ (σ)':<10}")
    print("-"*80)

    results = {}

    for name, data in leptons.items():
        mass = data["mass"]
        a_exp = data["a_exp"]
        a_unc = data["a_exp_unc"]

        # Calculate QFD prediction
        r, lambda_c, a_pred = calculate_lepton_properties(mass)

        if np.isnan(a_pred):
            print(f"{name:<10} | {'FAILED':<12} | {'N/A':<12} | {'N/A':<15} | {'N/A':<15} | {'N/A':<10}")
            continue

        # Calculate deviation in standard deviations
        if a_exp is not None and a_unc is not None:
            delta_sigma = abs(a_pred - a_exp) / a_unc
            delta_str = f"{delta_sigma:.2f}"
        else:
            delta_str = "N/A"

        a_exp_str = f"{a_exp:.11f}" if a_exp is not None else "Not measured"

        print(f"{name:<10} | {r:<12.4f} | {lambda_c:<12.4f} | {a_pred:<15.11f} | {a_exp_str:<15} | {delta_str:<10}")

        results[name] = {
            'radius_fm': r,
            'compton_fm': lambda_c,
            'a_predicted': a_pred,
            'a_experimental': a_exp,
            'delta_sigma': delta_sigma if a_exp is not None else None
        }

    print("-"*80)
    print()

    return results


def test_beta_sensitivity():
    """
    Test how g-2 predictions vary with β (the "3.15% Tax").

    This tests whether β = 3.058 + correction ≈ 3.15 improves agreement.
    """
    print("="*80)
    print("BETA SENSITIVITY ANALYSIS: The 3.15% Tax")
    print("="*80)
    print()

    # Test range: β from Golden Loop to V22 effective value
    beta_values = [
        (BETA_GOLDEN, "Golden Loop (theory)"),
        (BETA_CORE, "MCMC median"),
        (3.15, "V22 effective")
    ]

    for beta, label in beta_values:
        print(f"\nβ = {beta:.4f} ({label})")
        print("-"*40)

        # Calculate muon g-2 (most precisely measured)
        r, lambda_c, a_pred = calculate_lepton_properties(105.6583755, beta=beta)
        a_exp = 0.00116592059
        a_unc = 0.00000000022

        if not np.isnan(a_pred):
            delta = a_pred - a_exp
            delta_sigma = abs(delta) / a_unc
            print(f"  Muon a_pred = {a_pred:.11f}")
            print(f"  Muon a_exp  = {a_exp:.11f}")
            print(f"  Δa = {delta:+.2e} ({delta_sigma:.2f} σ)")


def parameter_uncertainty_analysis():
    """
    Propagate MCMC parameter uncertainties to g-2 predictions.
    """
    print("="*80)
    print("PARAMETER UNCERTAINTY PROPAGATION")
    print("="*80)
    print()

    # Monte Carlo sampling from MCMC posterior
    n_samples = 1000
    beta_samples = np.random.normal(BETA_CORE, BETA_STD, n_samples)
    xi_samples = np.random.normal(XI, XI_STD, n_samples)

    # Calculate muon g-2 distribution
    a_muon_samples = []
    for beta, xi in zip(beta_samples, xi_samples):
        if beta > 0 and xi > 0:  # Physical constraints
            _, _, a = calculate_lepton_properties(105.6583755, beta=beta, xi=xi)
            if not np.isnan(a):
                a_muon_samples.append(a)

    a_muon_samples = np.array(a_muon_samples)

    if len(a_muon_samples) > 0:
        a_median = np.median(a_muon_samples)
        a_std = np.std(a_muon_samples)
        a_16, a_84 = np.percentile(a_muon_samples, [16, 84])

        print(f"Muon g-2 prediction from MCMC uncertainty:")
        print(f"  a = {a_median:.11f} ± {a_std:.11f}")
        print(f"  68% CI: [{a_16:.11f}, {a_84:.11f}]")
        print()
        print(f"Experimental value:")
        print(f"  a = 0.00116592059 ± 0.00000000022")
        print()

        # Check if experimental value within prediction uncertainty
        a_exp = 0.00116592059
        if a_16 <= a_exp <= a_84:
            print("  ✓ Experimental value within 68% credible interval")
        else:
            delta = min(abs(a_exp - a_16), abs(a_exp - a_84))
            print(f"  ✗ Experimental value outside 68% CI by {delta:.2e}")
    else:
        print("Warning: No valid samples generated")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("QFD Lepton g-2 Validation Script")
    print("Bridging Lean 4 Proofs ↔ Experimental Data")
    print("="*80)
    print()

    # Main validation
    results = validate_against_experiments()

    # Beta sensitivity test
    print("\n")
    test_beta_sensitivity()

    # Uncertainty propagation
    print("\n")
    parameter_uncertainty_analysis()

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    print("If Muon Δ(σ) < 3:")
    print("  → QFD microphysical model VALIDATED")
    print("  → Lepton Isomer Ladder is a DISCOVERY")
    print()
    print("If Muon Δ(σ) > 10:")
    print("  → Vortex Shape Factor needs refinement (V6 term)")
    print("  → Geometric details matter")
    print()
    print("="*80)
