#!/usr/bin/env python3
"""
V22 QFD Supernova Analysis - Proper Implementation

Built on V21 foundation with:
1. Unified schema integration (15+ QFD parameters)
2. Static QFD physics (from V21's v17_qfd_model.py)
3. V21 Stage1 results (8,253 SNe from raw DES5yr)
4. Proper parameter fitting with optional Lean constraints

Author: QFD Research Team
Date: December 22, 2025
Version: V22 (Proper - Built on V21)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json

# Add schema directory to path
schema_path = Path("/home/tracy/development/QFD_SpectralGap/Background_and_Schema")
sys.path.insert(0, str(schema_path))

# Import unified schema
try:
    from qfd_unified_schema import QFDCouplings, CosmologyParams
    HAVE_SCHEMA = True
except ImportError:
    print("Warning: Could not import unified schema")
    HAVE_SCHEMA = False

# ============================================================================
# QFD PHYSICS MODEL (From V21 - Self-Contained)
# ============================================================================

def qfd_distance_modulus(z, k_J, eta_prime, xi=0.0, H0_baseline=70.0):
    """
    QFD distance modulus using V21's static universe physics.

    Model:
        D = (c/k_J_total) * z  (linear Hubble law, static universe)
        Dimming: η'*z + ξ*z/(1+z)  (plasma veil + thermal effects)

    Parameters:
        z: Redshift
        k_J: Total J·A interaction rate (km/s/Mpc)
        eta_prime: Plasma veil opacity (FDR photon self-interaction)
        xi: Thermal processing (optional, defaults to 0)
        H0_baseline: Only for initial scaling (default 70)

    Note: This matches V21's physics exactly.
    """
    # Speed of light
    c_km_s = 299792.458  # km/s

    # Distance in static QFD universe (linear Hubble law)
    D_static = (c_km_s / k_J) * z  # Mpc
    mu_geometric = 5.0 * np.log10(D_static) + 25.0

    # Dimming from QFD effects
    # eta_prime: plasma veil opacity
    # xi: thermal processing (FDR scattering, broadening)
    dimming = eta_prime * z + xi * z / (1 + z)

    return mu_geometric + dimming


def qfd_distance_modulus_v21_compat(z, eta, H0=70.0):
    """
    V21-compatible wrapper using simplified parameterization.

    V21 uses: mu = mu_static + eta * z^1.5
    This is a simplified version for comparison.
    """
    c_km_s = 299792.458
    D_static = (c_km_s / H0) * z
    mu_static = 5.0 * np.log10(D_static) + 25.0
    mu_qfd = mu_static + eta * (z ** 1.5)
    return mu_qfd


# ============================================================================
# LEAN CONSTRAINTS (Optional - To Be Properly Derived)
# ============================================================================

class LeanConstraints:
    """
    Parameter constraints from Lean 4 formal proofs.

    NOTE: Current Lean proofs (AdjointStability, SpacetimeEmergence,
    BivectorClasses) prove vacuum stability but don't directly constrain
    cosmology parameters. These bounds are physically motivated placeholders
    until proper Lean theorems are proven.

    TODO: Prove Lean theorems that formally derive these bounds.
    """

    # k_J: Universal J·A interaction (quantum field drag rate)
    # Physical reasoning: Must be positive, ~70 km/s/Mpc baseline
    K_J_MIN = 50.0   # km/s/Mpc (lower bound from observations)
    K_J_MAX = 100.0  # km/s/Mpc (upper bound from observations)

    # eta_prime: Plasma veil opacity (photon self-interaction)
    # Physical reasoning: Can be negative (brightening) or positive (dimming)
    ETA_PRIME_MIN = -10.0  # Reasonable physical range
    ETA_PRIME_MAX = 0.0    # Typically negative or zero

    # xi: Thermal processing (FDR scattering)
    # Physical reasoning: Similar to eta_prime
    XI_MIN = -10.0
    XI_MAX = 0.0

    @classmethod
    def validate(cls, k_J, eta_prime, xi):
        """Validate parameters against constraints."""
        if not (cls.K_J_MIN <= k_J <= cls.K_J_MAX):
            raise ValueError(f"k_J = {k_J} outside range [{cls.K_J_MIN}, {cls.K_J_MAX}]")
        if not (cls.ETA_PRIME_MIN <= eta_prime <= cls.ETA_PRIME_MAX):
            raise ValueError(f"eta_prime = {eta_prime} outside range [{cls.ETA_PRIME_MIN}, {cls.ETA_PRIME_MAX}]")
        if not (cls.XI_MIN <= xi <= cls.XI_MAX):
            raise ValueError(f"xi = {xi} outside range [{cls.XI_MIN}, {cls.XI_MAX}]")
        return True


# ============================================================================
# FITTING
# ============================================================================

def chi_squared(params, z_obs, mu_obs, sigma_mu):
    """
    Chi-squared for QFD model fit.

    params = [k_J, eta_prime, xi]
    """
    k_J, eta_prime, xi = params

    # Validate constraints
    try:
        LeanConstraints.validate(k_J, eta_prime, xi)
    except ValueError:
        return 1e10  # Penalty for invalid parameters

    # Predicted distance modulus
    mu_pred = qfd_distance_modulus(z_obs, k_J, eta_prime, xi)

    # Chi-squared
    chi2 = np.sum(((mu_obs - mu_pred) / sigma_mu) ** 2)

    return chi2


def fit_qfd_model(data, initial_guess=None, use_lean_constraints=True):
    """
    Fit QFD model to supernova data.

    Parameters:
        data: DataFrame with columns [redshift, distance_modulus, sigma_mu]
        initial_guess: [k_J, eta_prime, xi] (defaults to [70, -6, -6])
        use_lean_constraints: Apply Lean-derived bounds

    Returns:
        result: scipy.optimize.OptimizeResult
        best_fit: dict with best-fit parameters and statistics
    """
    if initial_guess is None:
        initial_guess = [70.0, -6.0, -6.0]  # k_J, eta_prime, xi

    # Extract data
    z_obs = data['redshift'].values
    mu_obs = data['distance_modulus'].values
    sigma_mu = data['sigma_mu'].values

    # Set bounds
    if use_lean_constraints:
        bounds = [
            (LeanConstraints.K_J_MIN, LeanConstraints.K_J_MAX),
            (LeanConstraints.ETA_PRIME_MIN, LeanConstraints.ETA_PRIME_MAX),
            (LeanConstraints.XI_MIN, LeanConstraints.XI_MAX)
        ]
    else:
        bounds = [
            (40.0, 120.0),   # k_J: wider range
            (-20.0, 10.0),   # eta_prime: very wide
            (-20.0, 10.0)    # xi: very wide
        ]

    # Minimize chi-squared
    result = minimize(
        chi_squared,
        x0=initial_guess,
        args=(z_obs, mu_obs, sigma_mu),
        method='L-BFGS-B',
        bounds=bounds
    )

    k_J_fit, eta_prime_fit, xi_fit = result.x
    chi2_fit = result.fun
    dof = len(z_obs) - 3  # 3 free parameters
    chi2_per_dof = chi2_fit / dof

    # Validate final parameters
    if use_lean_constraints:
        LeanConstraints.validate(k_J_fit, eta_prime_fit, xi_fit)

    best_fit = {
        'k_J': k_J_fit,
        'eta_prime': eta_prime_fit,
        'xi': xi_fit,
        'chi2': chi2_fit,
        'dof': dof,
        'chi2_per_dof': chi2_per_dof,
        'n_sne': len(z_obs),
        'converged': result.success,
        'rms': np.sqrt(chi2_fit / len(z_obs))
    }

    return result, best_fit


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_v22_analysis():
    """
    Run V22 QFD analysis using V21 foundation.
    """
    print("=" * 80)
    print("V22 QFD SUPERNOVA ANALYSIS - PROPER IMPLEMENTATION")
    print("Built on V21 Foundation")
    print("=" * 80)
    print()

    # Load V21 Stage1 results (converted to distance moduli)
    # Try multiple possible data sources in order of preference
    data_sources = [
        Path("/home/tracy/development/QFD_SpectralGap/data/raw/des5yr_v21_exact.csv"),  # V21 exact processing
        Path("/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/results/stage3_hubble/hubble_data.csv"),
    ]

    data = None
    data_path = None
    for path in data_sources:
        if path.exists():
            print(f"Loading data from:\n  {path}")
            try:
                df = pd.read_csv(path)

                # Check for required columns
                if 'z' in df.columns and 'mu_obs' in df.columns:
                    # V18 format
                    data = pd.DataFrame({
                        'redshift': df['z'],
                        'distance_modulus': df['mu_obs'],
                        'sigma_mu': df.get('sigma_mu', 0.5)  # Default if not present
                    })
                    data_path = path
                    print(f"  Format: V18 Stage3 output")
                    break
                elif 'redshift' in df.columns and 'distance_modulus' in df.columns:
                    # Already in correct format
                    data = df
                    data_path = path
                    print(f"  Format: Standard V22 format")
                    break
            except Exception as e:
                print(f"  Error loading {path}: {e}")
                continue

    if data is None:
        print("ERROR: No valid data source found!")
        print("Tried:")
        for path in data_sources:
            print(f"  - {path}")
        return None, None

    print(f"\nTotal SNe: {len(data)}")
    print(f"Redshift range: {data['redshift'].min():.3f} - {data['redshift'].max():.3f}")
    print()

    # Display schema info if available
    if HAVE_SCHEMA:
        print("UNIFIED SCHEMA PARAMETERS:")
        print("  QFDCouplings: V2, V4, V6, V8, lambda_R1-R4, k_J, k_c2, k_EM, k_csr, xi, g_c, eta_prime")
        print("  CosmologyParams: t0, ln_A, A_plasma, beta, eta_prime, A_lens, k_J_correction")
        print()

    print("V22 FITTING PARAMETERS (from unified schema):")
    print("  k_J: Universal J·A interaction (km/s/Mpc)")
    print("  eta_prime: Plasma veil opacity (FDR photon self-interaction)")
    print("  xi: Thermal processing (FDR scattering)")
    print()

    print("LEAN CONSTRAINTS (Physically Motivated - Formal Proofs Pending):")
    print(f"  k_J ∈ [{LeanConstraints.K_J_MIN}, {LeanConstraints.K_J_MAX}] km/s/Mpc")
    print(f"  eta_prime ∈ [{LeanConstraints.ETA_PRIME_MIN}, {LeanConstraints.ETA_PRIME_MAX}]")
    print(f"  xi ∈ [{LeanConstraints.XI_MIN}, {LeanConstraints.XI_MAX}]")
    print()

    # Fit QFD model
    print(f"FITTING QFD MODEL TO {len(data)} SNe...")
    print("Using V21's static universe physics (D = c*z/k_J)")
    result, best_fit = fit_qfd_model(data, use_lean_constraints=True)

    print()
    print("=" * 80)
    print("V22 BEST-FIT RESULTS")
    print("=" * 80)
    print(f"k_J          = {best_fit['k_J']:.2f} km/s/Mpc")
    print(f"eta_prime    = {best_fit['eta_prime']:.4f}")
    print(f"xi           = {best_fit['xi']:.4f}")
    print(f"χ²           = {best_fit['chi2']:.2f}")
    print(f"DOF          = {best_fit['dof']}")
    print(f"χ²/ν         = {best_fit['chi2_per_dof']:.4f}")
    print(f"RMS          = {best_fit['rms']:.4f} mag")
    print(f"Converged    = {best_fit['converged']}")
    print()

    # Validation
    print("VALIDATION:")
    try:
        LeanConstraints.validate(
            best_fit['k_J'],
            best_fit['eta_prime'],
            best_fit['xi']
        )
        print("✅ All parameters satisfy Lean constraints")
    except ValueError as e:
        print(f"❌ CONSTRAINT VIOLATION: {e}")
    print()

    # Compare to V18 if using V18 data
    if 'v18' in str(data_path).lower():
        print("COMPARISON TO V18 REFERENCE:")
        print("  V18 Stage2 MCMC: k_J_correction=19.94, eta_prime=-5.998, xi=-5.997")
        print("  V18 RMS: 2.18 mag")
        print(f"  V22 RMS: {best_fit['rms']:.2f} mag")
        print()

    # Save results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Supernova_Analysis/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    results_file = output_dir / "v22_proper_fit.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_fit': best_fit,
            'data_source': str(data_path),
            'lean_constraints': {
                'k_J_min': LeanConstraints.K_J_MIN,
                'k_J_max': LeanConstraints.K_J_MAX,
                'eta_prime_min': LeanConstraints.ETA_PRIME_MIN,
                'eta_prime_max': LeanConstraints.ETA_PRIME_MAX,
                'xi_min': LeanConstraints.XI_MIN,
                'xi_max': LeanConstraints.XI_MAX
            },
            'schema_integrated': HAVE_SCHEMA,
            'physics_model': 'static_qfd_v21'
        }, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()

    return data, best_fit


if __name__ == "__main__":
    data, best_fit = run_v22_analysis()
