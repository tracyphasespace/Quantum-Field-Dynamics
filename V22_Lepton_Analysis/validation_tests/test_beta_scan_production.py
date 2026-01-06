#!/usr/bin/env python3
"""
β-Scan Falsifiability Test - Production Solver Version

Uses the same LeptonEnergy solver that successfully reproduced all three
lepton masses, to test whether solutions exist across a range of β values.

This is the CRITICAL falsifiability test: if solutions exist for all β,
the model is too flexible. If they exist only near β ≈ 3.058, that's evidence.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps
import json
from pathlib import Path
from datetime import datetime
import sys

# Constants
RHO_VAC = 1.0

# Lepton masses (in electron-mass units)
LEPTON_MASSES = {
    'electron': 1.0,
    'muon': 206.7682826,
    'tau': 3477.228
}

NUM_R = 100  # Production uses 100 (faster), can increase to 400 for final
NUM_THETA = 20

# ============================================================================
# PRODUCTION SOLVER CLASSES (from test_all_leptons_beta_from_alpha.py)
# ============================================================================

class HillVortexStreamFunction:
    """Hill's spherical vortex stream function (Lamb 1932)."""
    def __init__(self, R, U):
        self.R = R
        self.U = U

    def velocity_components(self, r, theta):
        """Compute velocity components from stream function."""
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        v_r = np.zeros_like(r)
        v_theta = np.zeros_like(r)

        mask_internal = r < self.R

        # Internal region (rotational flow)
        if np.any(mask_internal):
            r_int = r[mask_internal]
            dpsi_dr_int = -(3 * self.U / (self.R**2)) * r_int**3 * sin_theta**2
            dpsi_dtheta_int = -(3 * self.U / (2 * self.R**2)) * \
                (self.R**2 - r_int**2) * r_int**2 * 2 * sin_theta * cos_theta

            v_r[mask_internal] = dpsi_dtheta_int / (r_int**2 * sin_theta + 1e-10)
            v_theta[mask_internal] = -dpsi_dr_int / (r_int * sin_theta + 1e-10)

        # External region (potential flow)
        mask_external = ~mask_internal
        if np.any(mask_external):
            r_ext = r[mask_external]
            dpsi_dr_ext = (self.U / 2) * (2*r_ext + self.R**3 / r_ext**2) * sin_theta**2
            dpsi_dtheta_ext = (self.U / 2) * (r_ext**2 - self.R**3 / r_ext) * \
                2 * sin_theta * cos_theta

            v_r[mask_external] = dpsi_dtheta_ext / (r_ext**2 * sin_theta + 1e-10)
            v_theta[mask_external] = -dpsi_dr_ext / (r_ext * sin_theta + 1e-10)

        return v_r, v_theta


class DensityGradient:
    """Density perturbation from vortex core."""
    def __init__(self, R, amplitude, rho_vac=RHO_VAC):
        self.R = R
        self.amplitude = amplitude
        self.rho_vac = rho_vac

    def rho(self, r):
        """Total density ρ(r) = ρ_vac + δρ(r)."""
        rho = np.ones_like(r) * self.rho_vac
        mask = r < self.R
        rho[mask] = self.rho_vac - self.amplitude * (1 - (r[mask] / self.R)**2)
        return rho

    def delta_rho(self, r):
        """Density perturbation δρ(r)."""
        delta = np.zeros_like(r)
        mask = r < self.R
        delta[mask] = -self.amplitude * (1 - (r[mask] / self.R)**2)
        return delta


class LeptonEnergy:
    """
    Production energy functional for Hill vortex leptons.

    E_total = E_circulation - E_stabilization

    where:
        E_circulation = ∫∫ (1/2) ρ |v|² r² sin(θ) dr dθ  (kinetic)
        E_stabilization = ∫∫ β (δρ)² r² sin(θ) dr dθ   (gradient)
    """
    def __init__(self, beta, r_max=10.0, num_r=NUM_R, num_theta=NUM_THETA):
        self.beta = beta
        self.rho_vac = RHO_VAC

        self.r = np.linspace(0.01, r_max, num_r)
        self.theta = np.linspace(0.01, np.pi-0.01, num_theta)
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]

    def total_energy(self, R, U, amplitude):
        """Compute total energy for given parameters."""
        stream = HillVortexStreamFunction(R, U)
        density = DensityGradient(R, amplitude, self.rho_vac)

        # Circulation energy (kinetic)
        E_circ = 0.0
        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)
            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)
            E_circ += simps(integrand, x=self.r) * self.dtheta
        E_circ *= 2 * np.pi

        # Stabilization energy (gradient)
        E_stab = 0.0
        for theta in self.theta:
            delta_rho = density.delta_rho(self.r)
            integrand = self.beta * delta_rho**2 * self.r**2 * np.sin(theta)
            E_stab += simps(integrand, x=self.r) * self.dtheta
        E_stab *= 2 * np.pi

        E_total = E_circ - E_stab
        return E_total, E_circ, E_stab

# ============================================================================
# β-SCAN IMPLEMENTATION
# ============================================================================

def solve_lepton_at_beta(target_mass, beta, initial_guess=None, max_iterations=500, tolerance=1e-4):
    """
    Attempt to find stable Hill vortex solution for given mass and β.

    Args:
        target_mass: Target mass ratio (relative to electron)
        beta: Vacuum stiffness parameter
        initial_guess: [R, U, amplitude] or None for automatic
        max_iterations: Maximum optimizer iterations
        tolerance: Convergence tolerance for residual (default 1e-4, production: 1e-7)

    Returns:
        dict with:
            - converged: bool
            - params: [R, U, amplitude] if converged
            - energy: E_total if converged
            - residual: |E_total - target_mass|
            - E_circ, E_stab: energy components
            - iterations: number of iterations used
    """
    energy = LeptonEnergy(beta=beta, num_r=NUM_R, num_theta=NUM_THETA)

    # Initial guess based on mass scaling
    if initial_guess is None:
        U_guess = 0.024 * np.sqrt(target_mass)
        initial_guess = [0.44, U_guess, 0.90]

    def objective(params):
        R, U, amplitude = params

        # Physical bounds
        if R <= 0 or U <= 0 or amplitude <= 0 or amplitude > 1.0:
            return 1e10

        try:
            E_total, _, _ = energy.total_energy(R, U, amplitude)
            return (E_total - target_mass)**2
        except:
            return 1e10

    # Bounds
    bounds = [
        (0.1, 1.0),    # R
        (0.001, 3.0),  # U (wider for β scan)
        (0.1, 1.0)     # amplitude
    ]

    try:
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iterations,
                'ftol': 1e-12,  # Tighter for production
                'gtol': 1e-10
            }
        )

        # Compute final energy
        R, U, amplitude = result.x
        E_total, E_circ, E_stab = energy.total_energy(R, U, amplitude)
        residual = abs(E_total - target_mass)

        # Success criteria
        converged = (
            result.success and
            residual < tolerance  # User-specified tolerance
        )

        return {
            'converged': bool(converged),  # Ensure Python bool
            'params': result.x.tolist(),
            'R': float(R),
            'U': float(U),
            'amplitude': float(amplitude),
            'energy': float(E_total),
            'E_circ': float(E_circ),
            'E_stab': float(E_stab),
            'residual': float(residual),
            'iterations': int(result.nit),
            'optimizer_success': bool(int(result.success))  # Force Python bool
        }

    except Exception as e:
        return {
            'converged': False,
            'error': str(e)
        }


def run_beta_scan(beta_min=2.5, beta_max=3.5, num_points=21, tolerance=1e-4):
    """
    Scan β parameter to test falsifiability.

    This is the critical test that distinguishes "evidence" from "compatibility":
    - If solutions exist for all β → model too flexible
    - If solutions exist only near β ≈ 3.058 → genuine constraint

    Args:
        beta_min: Minimum β value
        beta_max: Maximum β value
        num_points: Number of β values to test
        tolerance: Convergence tolerance (1e-4 for scan, 1e-7 for production)

    Returns:
        dict with complete scan results
    """
    beta_range = np.linspace(beta_min, beta_max, num_points)

    results = {
        'test': 'β-Scan Falsifiability (Production Solver)',
        'beta_range': [float(beta_min), float(beta_max)],
        'num_points': int(num_points),
        'tolerance': float(tolerance),
        'grid_resolution': {'num_r': NUM_R, 'num_theta': NUM_THETA},
        'timestamp': datetime.now().isoformat(),
        'scan_results': []
    }

    print(f"\n{'='*70}")
    print(f"β-SCAN FALSIFIABILITY TEST (Production Solver)")
    print(f"{'='*70}")
    print(f"β range: [{beta_min}, {beta_max}] with {num_points} points")
    print(f"Grid: {NUM_R}×{NUM_THETA} (r×θ)")
    print(f"Testing all three leptons at each β value\n")

    for i, beta in enumerate(beta_range):
        print(f"[{i+1:2d}/{num_points}] β = {beta:.4f} ... ", end='', flush=True)

        beta_result = {
            'beta': float(beta),
            'leptons': {}
        }

        # Try to solve all three leptons at this β
        converged_count = 0
        total_residual = 0.0

        for lepton_name, target_mass in LEPTON_MASSES.items():
            result = solve_lepton_at_beta(target_mass, beta, max_iterations=500, tolerance=tolerance)
            beta_result['leptons'][lepton_name] = result

            if result['converged']:
                converged_count += 1
                total_residual += result['residual']

        beta_result['num_converged'] = int(converged_count)
        beta_result['all_converged'] = bool(int(converged_count) == 3)  # Force Python bool
        beta_result['total_residual'] = float(total_residual) if converged_count > 0 else None

        results['scan_results'].append(beta_result)

        # Progress indicator
        status = f"{converged_count}/3 converged"
        if converged_count == 3:
            status += f" (residual sum: {total_residual:.2e})"
        print(status)

    # Analysis: find optimal β
    converged_betas = []
    residuals_at_beta = []

    for beta_result in results['scan_results']:
        if beta_result['all_converged']:
            converged_betas.append(beta_result['beta'])
            residuals_at_beta.append(beta_result['total_residual'])

    if len(converged_betas) > 0:
        min_idx = np.argmin(residuals_at_beta)
        min_residual_beta = converged_betas[min_idx]
        min_residual_sum = residuals_at_beta[min_idx]

        beta_window_min = min(converged_betas)
        beta_window_max = max(converged_betas)
        beta_window_width = beta_window_max - beta_window_min
    else:
        min_residual_beta = None
        min_residual_sum = None
        beta_window_min = None
        beta_window_max = None
        beta_window_width = None

    results['analysis'] = {
        'num_beta_with_all_converged': int(len(converged_betas)),
        'fraction_beta_all_converged': float(len(converged_betas) / num_points),
        'min_residual_beta': float(min_residual_beta) if min_residual_beta else None,
        'min_residual_sum': float(min_residual_sum) if min_residual_sum else None,
        'beta_window': {
            'min': float(beta_window_min) if beta_window_min else None,
            'max': float(beta_window_max) if beta_window_max else None,
            'width': float(beta_window_width) if beta_window_width else None,
            'relative_width_percent': float(beta_window_width / 3.058 * 100) if beta_window_width else None
        }
    }

    return results


def save_results(results, filename='results/beta_scan_production.json'):
    """Save results to JSON file."""
    output_path = Path(__file__).parent / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    return output_path


def print_summary(results):
    """Print comprehensive summary of β-scan results."""
    analysis = results['analysis']

    print("\n" + "="*70)
    print("β-SCAN FALSIFIABILITY TEST - SUMMARY")
    print("="*70)

    print(f"\nβ range scanned: [{results['beta_range'][0]}, {results['beta_range'][1]}]")
    print(f"Number of β values tested: {results['num_points']}")
    print(f"Grid resolution: {results['grid_resolution']['num_r']}×{results['grid_resolution']['num_theta']}")

    num_all_converged = analysis['num_beta_with_all_converged']
    fraction_converged = analysis['fraction_beta_all_converged']

    print(f"\n{'='*70}")
    print("KEY RESULTS:")
    print(f"{'='*70}")

    if num_all_converged > 0:
        print(f"✓ Number of β where ALL 3 leptons converge: {num_all_converged}/{results['num_points']}")
        print(f"  ({fraction_converged*100:.1f}% of tested values)")

        window = analysis['beta_window']
        print(f"\n✓ β window where all leptons exist: [{window['min']:.3f}, {window['max']:.3f}]")
        print(f"  Width: Δβ = {window['width']:.3f}")
        print(f"  Relative width: Δβ/β ≈ {window['relative_width_percent']:.1f}%")

        print(f"\n✓ Minimum residual at β = {analysis['min_residual_beta']:.4f}")
        print(f"  Sum of residuals: {analysis['min_residual_sum']:.2e}")
        print(f"  Compare to inferred: β = 3.058 (from α)")

    else:
        print("✗ WARNING: No β value where all three leptons converge!")
        print("  Possible issues:")
        print("    - Grid resolution too coarse")
        print("    - Convergence tolerance too tight")
        print("    - Energy functional may need adjustment")

    print(f"\n{'='*70}")
    print("FALSIFIABILITY ASSESSMENT:")
    print(f"{'='*70}")

    if num_all_converged == 0:
        print("✗ INCONCLUSIVE: No solutions found in scan range")
        print("  Recommendation: Check solver settings, try wider β range")
    elif fraction_converged < 0.3:
        print("✓✓ STRONGLY FALSIFIABLE:")
        print(f"  Solutions exist for only {fraction_converged*100:.1f}% of β values")
        print("  This is EVIDENCE, not 'optimizer can hit targets'")
        print("  The lepton spectrum is a genuine constraint on β")
    elif fraction_converged < 0.6:
        print("✓ FALSIFIABLE:")
        print(f"  Solutions exist for {fraction_converged*100:.1f}% of β values")
        print("  Model has predictive power but some flexibility")
    else:
        print("⚠ WEAK FALSIFIABILITY:")
        print(f"  Solutions exist for {fraction_converged*100:.1f}% of β values")
        print("  Model may be too flexible - review constraints")

    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='β-Scan Falsifiability Test')
    parser.add_argument('--beta-min', type=float, default=2.5,
                       help='Minimum β value (default: 2.5)')
    parser.add_argument('--beta-max', type=float, default=3.5,
                       help='Maximum β value (default: 3.5)')
    parser.add_argument('--num-points', type=int, default=21,
                       help='Number of β points (default: 21, use 51 for publication)')
    parser.add_argument('--num-r', type=int, default=100,
                       help='Radial grid points (default: 100)')
    parser.add_argument('--num-theta', type=int, default=20,
                       help='Angular grid points (default: 20)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Convergence tolerance (default: 1e-4, production: 1e-7)')

    args = parser.parse_args()

    # Update global grid resolution
    NUM_R = args.num_r
    NUM_THETA = args.num_theta

    # Run the scan
    print(f"\nEstimated runtime: ~{args.num_points * 3 * 5 / 60:.1f} minutes")
    print(f"  ({args.num_points} β points × 3 leptons × ~5 sec/lepton)")
    print(f"  Tolerance: {args.tolerance:.0e} ({'PRODUCTION' if args.tolerance <= 1e-6 else 'SCAN'})\n")

    results = run_beta_scan(
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        num_points=args.num_points,
        tolerance=args.tolerance
    )

    # Save results
    output_path = save_results(results)

    # Print summary
    print_summary(results)

    print(f"\n✓ β-scan complete!")
    print(f"  Results: {output_path}")
    print(f"  Use this data to create Figure 6 (Falsifiability)")
