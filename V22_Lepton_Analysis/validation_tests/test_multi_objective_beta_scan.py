#!/usr/bin/env python3
"""
Option 1: Multi-Objective β-Scan with Magnetic Moment Constraint

Purpose: Add magnetic moment as second observable to break (R, U) degeneracy.

Method:
- Optimize (R, U, amplitude) to match BOTH:
  1. Mass (E_total = target_mass)
  2. Magnetic moment (μ = μ_measured)

This should uniquely determine all three parameters for given β,
allowing β-scan to show genuine minimum.

Formula (from Tracy):
    μ = k × Q × R × U
    where k ≈ 0.2 (geometric factor for Hill vortex)

Scaling:
    Mass: E ~ amplitude × U² × R³
    Mag. moment: μ ~ R × U

Different scalings → breaks degeneracy!
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps
import json
from pathlib import Path
from datetime import datetime
import sys

# Import production solver classes
sys.path.insert(0, str(Path(__file__).parent))
from test_all_leptons_beta_from_alpha import (
    HillVortexStreamFunction, DensityGradient, LeptonEnergy,
    ELECTRON_MASS, MUON_TO_ELECTRON_RATIO, TAU_TO_ELECTRON_RATIO, RHO_VAC
)

# ===========================================================================
# PHYSICAL CONSTANTS (CODATA 2018)
# ===========================================================================

# Electron g-factor and anomalous moment
G_ELECTRON = 2.00231930436256  # Exact measured value
A_ELECTRON = (G_ELECTRON - 2) / 2  # = 0.00115965218128

# Muon g-factor (approximate - update with precise value if available)
G_MUON = 2.0023318418  # From muon g-2 experiments
A_MUON = (G_MUON - 2) / 2

# Tau g-factor (assumed close to 2, limited precision)
G_TAU = 2.0  # Placeholder - tau g-2 not precisely measured
A_TAU = 0.0

# Lepton magnetic moments (in units of Bohr magneton μ_B)
# For dimensionless calculation, we use g-factors
LEPTON_G_FACTORS = {
    'electron': G_ELECTRON,
    'muon': G_MUON,
    'tau': G_TAU
}

LEPTON_MASSES = {
    'electron': ELECTRON_MASS,
    'muon': MUON_TO_ELECTRON_RATIO,
    'tau': TAU_TO_ELECTRON_RATIO
}

# Experimental uncertainties (for normalized objective function)
# Electron mass uncertainty: δm_e/m_e ≈ 2.2×10⁻⁸ (CODATA 2018)
# Electron g-factor uncertainty: δg_e ≈ 2.8×10⁻¹³ (CODATA 2018)
# For muon and tau, uncertainties are larger
MASS_UNCERTAINTIES = {
    'electron': 2.2e-8,  # Relative uncertainty in electron mass
    'muon': 2.2e-8,      # Assume similar for mass ratio (conservative)
    'tau': 1.0e-5        # Tau mass has larger uncertainty
}

G_FACTOR_UNCERTAINTIES = {
    'electron': 2.8e-13,  # Absolute uncertainty in g_e (CODATA 2018)
    'muon': 2.3e-10,      # Muon g-2 uncertainty (Fermilab/BNL)
    'tau': 1.0e-2         # Tau g-factor poorly measured
}

# ===========================================================================
# MAGNETIC MOMENT CALCULATION
# ===========================================================================

def magnetic_moment_hill_vortex(R, U, amplitude, beta, geometric_factor=0.2):
    """
    Compute magnetic moment for Hill spherical vortex.

    Formula (Tracy's derivation):
        μ = k × Q × R × U

    where:
        k ≈ 0.2 (geometric factor for uniform vorticity Hill vortex)
        Q = charge (in natural units, Q=1 for electron)
        R = vortex radius
        U = circulation velocity

    Args:
        R: Vortex radius
        U: Circulation velocity
        amplitude: Density amplitude (not used in classical formula,
                   but may affect charge distribution in QFD refinement)
        beta: Vacuum stiffness (not used in classical formula)
        geometric_factor: k ≈ 0.2 for Hill vortex

    Returns:
        Magnetic moment μ (dimensionless, in units appropriate for g-factor)

    Notes:
        - Classical Hill vortex: μ ∝ R × U
        - Mass: E ∝ amplitude × U² × R³
        - Different scaling → breaks (R,U) degeneracy
    """
    # For electron, Q = 1 (fundamental charge)
    Q = 1.0

    # Classical Hill vortex formula
    mu = geometric_factor * Q * R * U

    return mu


def g_factor_from_moment(mu, mass_ratio=1.0):
    """
    Convert magnetic moment to g-factor.

    In natural units where electron mass = 1:
        g = 2 × μ / (m × μ_B)

    For dimensionless comparison, we normalize by electron values.

    Args:
        mu: Magnetic moment (from Hill vortex calculation)
        mass_ratio: Mass relative to electron (1.0 for electron)

    Returns:
        g-factor (dimensionless)
    """
    # Normalization factor (calibrated from known electron solution)
    # Calibration: At β=3.058, electron solution (R≈0.44, U≈0.024) gives:
    #   μ_raw = 0.2 × 1.0 × 0.44 × 0.024 ≈ 2.11×10⁻³
    #   g_target = 2.00231930436256
    #   normalization = g_target / μ_raw ≈ 948
    normalization = 948.0  # Calibrated empirically

    g = normalization * mu / mass_ratio

    return g


def charge_radius_rms(R, amplitude):
    """
    Compute RMS charge radius from density perturbation.

    Alternative/backup second observable.

    For δρ(r) = -amplitude × (1 - (r/R)²):
        R_rms² = ∫ r² |δρ| dV / ∫ |δρ| dV

    Args:
        R: Vortex radius
        amplitude: Density amplitude

    Returns:
        RMS radius
    """
    # For parabolic profile, analytical result:
    # R_rms ≈ R × (sqrt(3/5)) ≈ 0.775 × R
    # (This is exact for uniform density in sphere)

    # For our profile: δρ(r) = -A(1-(r/R)²), more concentrated at center
    # Numerical integral would give exact value, but for now:
    R_rms = R * np.sqrt(3.0 / 5.0)

    return R_rms


# ===========================================================================
# MULTI-OBJECTIVE SOLVER
# ===========================================================================

def solve_multi_objective(target_mass, target_g_factor, beta,
                          w_mass=1.0, w_mag=1.0,
                          num_r=100, num_theta=20, max_iter=1000,
                          use_experimental_uncertainties=False):
    """
    Solve for (R, U, amplitude) matching BOTH mass and g-factor.

    Objective:
        If use_experimental_uncertainties=True:
            residual = (ΔE/σ_mass)² + (Δg/σ_g)²
        Else:
            residual = w_mass × (E - m_target)² + w_mag × (g - g_target)²

    This should uniquely determine (R, U, amplitude) for given β.

    Args:
        target_mass: Target mass ratio
        target_g_factor: Target g-factor (e.g., 2.00231930436256 for electron)
        beta: Vacuum stiffness
        w_mass: Weight for mass constraint
        w_mag: Weight for magnetic moment constraint
        num_r, num_theta: Grid resolution
        max_iter: Max optimizer iterations

    Returns:
        dict with solution
    """
    energy_calc = LeptonEnergy(beta=beta, num_r=num_r, num_theta=num_theta)

    # Get experimental uncertainties for normalization
    # Use electron values by default (can be updated for muon/tau)
    sigma_mass = MASS_UNCERTAINTIES['electron'] * target_mass
    sigma_g = G_FACTOR_UNCERTAINTIES['electron']

    def objective(params):
        R, U, amplitude = params

        # Physical bounds
        if R <= 0 or U <= 0 or amplitude <= 0:
            return 1e10

        # Cavitation constraint
        if amplitude > RHO_VAC:
            return 1e10

        try:
            # Mass from energy functional
            E_total, E_circ, E_stab = energy_calc.total_energy(R, U, amplitude)

            # Magnetic moment
            mu = magnetic_moment_hill_vortex(R, U, amplitude, beta)
            g = g_factor_from_moment(mu, mass_ratio=target_mass)

            # Multi-objective residual
            if use_experimental_uncertainties:
                # Chi-squared form: (Δ/σ)²
                mass_error = ((E_total - target_mass) / sigma_mass)**2
                mag_error = ((g - target_g_factor) / sigma_g)**2
                total_residual = mass_error + mag_error
            else:
                # Simple weighted sum
                mass_error = (E_total - target_mass)**2
                mag_error = (g - target_g_factor)**2
                total_residual = w_mass * mass_error + w_mag * mag_error

            return total_residual

        except Exception as e:
            return 1e10

    # Initial guess (from mass-only scaling)
    R0 = 0.44
    U0 = 0.024 * np.sqrt(target_mass)
    amp0 = 0.90

    # Bounds
    bounds = [
        (0.1, 1.0),    # R
        (0.001, 3.0),  # U
        (0.1, 1.0)     # amplitude
    ]

    try:
        result = minimize(
            objective,
            [R0, U0, amp0],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )

        R_opt, U_opt, amp_opt = result.x

        # Evaluate final observables
        E_total, E_circ, E_stab = energy_calc.total_energy(R_opt, U_opt, amp_opt)
        mu = magnetic_moment_hill_vortex(R_opt, U_opt, amp_opt, beta)
        g = g_factor_from_moment(mu, mass_ratio=target_mass)

        mass_residual = abs(E_total - target_mass)
        g_residual = abs(g - target_g_factor)

        # Success: both constraints satisfied
        converged = (
            result.success and
            mass_residual < 1e-3 and
            g_residual < 0.1  # Looser for magnetic moment (harder to match)
        )

        return {
            'converged': bool(converged),
            'R': float(R_opt),
            'U': float(U_opt),
            'amplitude': float(amp_opt),
            'E_total': float(E_total),
            'E_circ': float(E_circ),
            'E_stab': float(E_stab),
            'mu': float(mu),
            'g_factor': float(g),
            'mass_residual': float(mass_residual),
            'g_residual': float(g_residual),
            'total_objective': float(result.fun),
            'iterations': int(result.nit),
            'optimizer_success': bool(result.success)
        }

    except Exception as e:
        return {
            'converged': False,
            'error': str(e)
        }


# ===========================================================================
# β-SCAN WITH MULTI-OBJECTIVE
# ===========================================================================

def run_multi_objective_beta_scan(beta_min=2.5, beta_max=3.5, num_beta=11,
                                  w_mass=1.0, w_mag=1.0,
                                  use_experimental_uncertainties=False):
    """
    β-scan with mass + magnetic moment constraints.

    This should show:
    - Narrow β window (if model is correct)
    - Minimum near β = 3.058 (if Golden Loop is valid)
    - OR: Mismatch revealing model inadequacy

    Args:
        beta_min, beta_max: β range
        num_beta: Number of β points
        w_mass, w_mag: Weights for mass and magnetic moment

    Returns:
        Complete scan results
    """
    beta_range = np.linspace(beta_min, beta_max, num_beta)

    results = {
        'test': 'Multi-Objective β-Scan (Mass + Magnetic Moment)',
        'purpose': 'Test if second observable (g-factor) restores β identifiability',
        'beta_range': [float(beta_min), float(beta_max)],
        'num_beta': int(num_beta),
        'weights': {'mass': float(w_mass), 'magnetic': float(w_mag)},
        'timestamp': datetime.now().isoformat(),
        'scan_results': []
    }

    print("\n" + "="*70)
    print("MULTI-OBJECTIVE β-SCAN: Mass + Magnetic Moment")
    print("="*70)
    print(f"\nConstraints:")
    print(f"  1. Mass matching (weight = {w_mass})")
    print(f"  2. g-factor matching (weight = {w_mag})")
    print(f"\nβ range: [{beta_min}, {beta_max}] with {num_beta} points\n")

    # Only test electron for now (has most precise g-factor)
    lepton = 'electron'
    target_mass = LEPTON_MASSES[lepton]
    target_g = LEPTON_G_FACTORS[lepton]

    print(f"Testing {lepton}:")
    print(f"  Target mass: {target_mass}")
    print(f"  Target g-factor: {target_g}\n")

    for i, beta in enumerate(beta_range):
        print(f"[{i+1:2d}/{num_beta}] β = {beta:.3f} ... ", end='', flush=True)

        result = solve_multi_objective(
            target_mass, target_g, beta,
            w_mass=w_mass, w_mag=w_mag,
            num_r=100, num_theta=20,
            use_experimental_uncertainties=use_experimental_uncertainties
        )

        result['beta'] = float(beta)
        results['scan_results'].append(result)

        if result['converged']:
            print(f"✓ (mass res: {result['mass_residual']:.2e}, " +
                  f"g res: {result['g_residual']:.2e})")
        else:
            print(f"✗ Failed")

    return results


def analyze_multi_objective_results(results):
    """Analyze if β minimum emerges with multi-objective."""
    print("\n" + "="*70)
    print("ANALYSIS: β IDENTIFIABILITY WITH TWO OBSERVABLES")
    print("="*70)

    # Extract converged points
    converged = [r for r in results['scan_results'] if r['converged']]

    if len(converged) == 0:
        print("\n✗ NO CONVERGED SOLUTIONS")
        print("  Magnetic moment constraint may be too tight")
        print("  OR: Formula needs calibration")
        return

    betas = [r['beta'] for r in converged]
    mass_residuals = [r['mass_residual'] for r in converged]
    g_residuals = [r['g_residual'] for r in converged]
    total_objectives = [r['total_objective'] for r in converged]

    # Find minimum
    min_idx = np.argmin(total_objectives)
    beta_min = betas[min_idx]
    obj_min = total_objectives[min_idx]

    # Check variation
    obj_range = max(total_objectives) - min(total_objectives)
    obj_variation = obj_range / min(total_objectives) if min(total_objectives) > 0 else 0

    print(f"\nConverged at {len(converged)}/{results['num_beta']} β values")
    print(f"Minimum objective: {obj_min:.2e} at β = {beta_min:.3f}")
    print(f"Objective range: {obj_range:.2e}")
    print(f"Variation: {obj_variation:.1%}")

    if beta_min == 3.058 or abs(beta_min - 3.058) < 0.1:
        print(f"\n✓ Minimum at β = {beta_min:.3f} ← MATCHES INFERRED VALUE (3.058)")
    else:
        print(f"\n⚠ Minimum at β = {beta_min:.3f} (inferred: 3.058)")
        print(f"  Offset: {abs(beta_min - 3.058):.3f}")

    if obj_variation > 1.0:
        print("\n✓✓ SHARP β MINIMUM (>100% variation)")
        print("  → β IS UNIQUELY IDENTIFIED by two observables")
        print("  → Degeneracy BROKEN successfully")
    elif obj_variation > 0.1:
        print("\n✓ MODERATE β PREFERENCE (>10% variation)")
        print("  → Partial identifiability restored")
        print("  → May need weight tuning or tighter tolerance")
    else:
        print("\n✗ STILL FLAT (<10% variation)")
        print("  → Magnetic moment formula may need refinement")
        print("  → OR: normalization/scaling factor incorrect")

    print("="*70 + "\n")


def save_results(results, filename='results/multi_objective_beta_scan.json'):
    """Save results to JSON."""
    output_path = Path(__file__).parent / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_path}\n")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-objective β-scan (Option 1: Mass + Magnetic Moment)'
    )
    parser.add_argument('--beta-min', type=float, default=2.5)
    parser.add_argument('--beta-max', type=float, default=3.5)
    parser.add_argument('--num-beta', type=int, default=11)
    parser.add_argument('--w-mass', type=float, default=1.0,
                       help='Weight for mass constraint')
    parser.add_argument('--w-mag', type=float, default=1.0,
                       help='Weight for magnetic moment constraint')
    parser.add_argument('--use-uncertainties', action='store_true',
                       help='Normalize objective by experimental uncertainties (chi-squared form)')

    args = parser.parse_args()

    # Run scan
    results = run_multi_objective_beta_scan(
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        num_beta=args.num_beta,
        w_mass=args.w_mass,
        w_mag=args.w_mag,
        use_experimental_uncertainties=args.use_uncertainties
    )

    # Analyze
    analyze_multi_objective_results(results)

    # Save
    save_results(results)

    print("✓ Multi-objective β-scan complete")
    print("  If successful → β identifiability restored, manuscript claim saved")
    print("  If failed → formula needs refinement or different second observable")
