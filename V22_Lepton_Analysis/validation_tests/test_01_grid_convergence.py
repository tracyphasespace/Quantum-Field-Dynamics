#!/usr/bin/env python3
"""
Grid Convergence Test for V22 Hill Vortex Electron Solution
============================================================

PURPOSE: Verify that the fitted electron parameters (R, U, amplitude, E_total)
         converge as the integration grid is refined.

TEST: Run optimization at three grid resolutions:
      - Coarse:   (nr, nθ) = (50, 10)
      - Standard: (nr, nθ) = (100, 20)  [current]
      - Fine:     (nr, nθ) = (200, 40)
      - Very Fine: (nr, nθ) = (400, 80)

SUCCESS CRITERIA:
  - Parameter drift < 1% between Fine and Very Fine
  - Energy drift < 0.1% between Fine and Very Fine
  - Monotonic convergence (errors decrease with refinement)

EXPECTED RUNTIME: ~5-10 minutes
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps
import json
from pathlib import Path
from datetime import datetime

# Constants
RHO_VAC = 1.0
BETA = 3.1
TARGET_MASS = 1.0  # Electron in electron-mass units


class HillVortexStreamFunction:
    def __init__(self, R, U):
        self.R = R
        self.U = U

    def velocity_components(self, r, theta):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        v_r = np.zeros_like(r)
        v_theta = np.zeros_like(r)

        mask_internal = r < self.R

        if np.any(mask_internal):
            r_int = r[mask_internal]
            dpsi_dr_int = -(3 * self.U / (self.R**2)) * r_int**3 * sin_theta**2
            dpsi_dtheta_int = -(3 * self.U / (2 * self.R**2)) * \
                (self.R**2 - r_int**2) * r_int**2 * 2 * sin_theta * cos_theta

            v_r[mask_internal] = dpsi_dtheta_int / (r_int**2 * sin_theta + 1e-10)
            v_theta[mask_internal] = -dpsi_dr_int / (r_int * sin_theta + 1e-10)

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
    def __init__(self, R, amplitude, rho_vac=RHO_VAC):
        self.R = R
        self.amplitude = amplitude
        self.rho_vac = rho_vac

    def rho(self, r):
        rho = np.ones_like(r) * self.rho_vac
        mask = r < self.R
        rho[mask] = self.rho_vac - self.amplitude * (1 - (r[mask] / self.R)**2)
        return rho

    def delta_rho(self, r):
        delta = np.zeros_like(r)
        mask = r < self.R
        delta[mask] = -self.amplitude * (1 - (r[mask] / self.R)**2)
        return delta


class ElectronEnergy:
    def __init__(self, beta=BETA, r_max=10.0, num_r=100, num_theta=20):
        self.beta = beta
        self.rho_vac = RHO_VAC

        self.r = np.linspace(0.01, r_max, num_r)
        self.theta = np.linspace(0.01, np.pi-0.01, num_theta)
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]

    def total_energy(self, R, U, amplitude):
        stream = HillVortexStreamFunction(R, U)
        density = DensityGradient(R, amplitude, self.rho_vac)

        # Circulation energy
        E_circ = 0.0
        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)
            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)
            E_circ += simps(integrand, x=self.r) * self.dtheta
        E_circ *= 2 * np.pi

        # Stabilization energy
        E_stab = 0.0
        for theta in self.theta:
            delta_rho = density.delta_rho(self.r)
            integrand = self.beta * delta_rho**2 * self.r**2 * np.sin(theta)
            E_stab += simps(integrand, x=self.r) * self.dtheta
        E_stab *= 2 * np.pi

        E_total = E_circ - E_stab
        return E_total, E_circ, E_stab


def optimize_at_resolution(num_r, num_theta, initial_guess=None):
    """
    Optimize electron parameters at given grid resolution.

    Returns: dict with parameters, energies, and convergence info
    """
    print(f"\n{'='*80}")
    print(f"Grid Resolution: (nr, nθ) = ({num_r}, {num_theta})")
    print(f"{'='*80}")

    energy = ElectronEnergy(beta=BETA, num_r=num_r, num_theta=num_theta)

    if initial_guess is None:
        initial_guess = [0.44, 0.024, 0.90]  # R, U, amplitude

    def objective(params):
        R, U, amplitude = params
        if R <= 0 or U <= 0 or amplitude <= 0 or amplitude > 1.0:
            return 1e10
        try:
            E_total, _, _ = energy.total_energy(R, U, amplitude)
            return (E_total - TARGET_MASS)**2
        except:
            return 1e10

    print("Starting optimization...")
    result = minimize(
        objective,
        initial_guess,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
    )

    R_opt, U_opt, amp_opt = result.x
    E_total, E_circ, E_stab = energy.total_energy(R_opt, U_opt, amp_opt)

    print(f"\nOptimized Parameters:")
    print(f"  R         = {R_opt:.6f}")
    print(f"  U         = {U_opt:.6f}")
    print(f"  amplitude = {amp_opt:.6f}")
    print(f"\nEnergies:")
    print(f"  E_circulation   = {E_circ:.6f}")
    print(f"  E_stabilization = {E_stab:.6f}")
    print(f"  E_total         = {E_total:.6f}")
    print(f"\nTarget vs Achieved:")
    print(f"  Target   = {TARGET_MASS:.6f}")
    print(f"  Achieved = {E_total:.6f}")
    print(f"  Residual = {abs(E_total - TARGET_MASS):.3e}")
    print(f"\nConvergence:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evals: {result.nfev}")

    return {
        'num_r': num_r,
        'num_theta': num_theta,
        'R': float(R_opt),
        'U': float(U_opt),
        'amplitude': float(amp_opt),
        'E_circulation': float(E_circ),
        'E_stabilization': float(E_stab),
        'E_total': float(E_total),
        'residual': float(abs(E_total - TARGET_MASS)),
        'converged': bool(result.success),
        'iterations': int(result.nit),
        'function_evals': int(result.nfev)
    }


def analyze_convergence(results):
    """
    Analyze parameter and energy convergence across grid resolutions.
    """
    print(f"\n{'='*80}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*80}\n")

    # Reference: finest grid
    finest = results[-1]

    print("Grid Resolution Progression:")
    print("-" * 80)
    print(f"{'Grid':<15} {'R':<12} {'U':<12} {'amplitude':<12} {'E_total':<12} {'Residual':<12}")
    print("-" * 80)

    for res in results:
        grid_str = f"({res['num_r']}, {res['num_theta']})"
        print(f"{grid_str:<15} {res['R']:<12.6f} {res['U']:<12.6f} "
              f"{res['amplitude']:<12.6f} {res['E_total']:<12.6f} {res['residual']:<12.3e}")

    print("\n" + "="*80)
    print("PARAMETER DRIFT from Finest Grid")
    print("="*80 + "\n")

    print(f"{'Grid':<15} {'ΔR (%)':<12} {'ΔU (%)':<12} {'Δamp (%)':<12} {'ΔE_total (%)':<12}")
    print("-" * 80)

    convergence_data = []
    for res in results:
        dR = 100 * abs(res['R'] - finest['R']) / finest['R']
        dU = 100 * abs(res['U'] - finest['U']) / finest['U']
        damp = 100 * abs(res['amplitude'] - finest['amplitude']) / finest['amplitude']
        dE = 100 * abs(res['E_total'] - finest['E_total']) / finest['E_total']

        grid_str = f"({res['num_r']}, {res['num_theta']})"
        print(f"{grid_str:<15} {dR:<12.4f} {dU:<12.4f} {damp:<12.4f} {dE:<12.4f}")

        convergence_data.append({
            'grid': grid_str,
            'drift_R_percent': float(dR),
            'drift_U_percent': float(dU),
            'drift_amplitude_percent': float(damp),
            'drift_E_total_percent': float(dE)
        })

    # Check success criteria
    print("\n" + "="*80)
    print("SUCCESS CRITERIA CHECK")
    print("="*80 + "\n")

    if len(results) >= 2:
        second_finest = results[-2]

        dR_fine = 100 * abs(second_finest['R'] - finest['R']) / finest['R']
        dU_fine = 100 * abs(second_finest['U'] - finest['U']) / finest['U']
        damp_fine = 100 * abs(second_finest['amplitude'] - finest['amplitude']) / finest['amplitude']
        dE_fine = 100 * abs(second_finest['E_total'] - finest['E_total']) / finest['E_total']

        param_pass = max(dR_fine, dU_fine, damp_fine) < 1.0
        energy_pass = dE_fine < 0.1

        print(f"Between ({second_finest['num_r']}, {second_finest['num_theta']}) and ({finest['num_r']}, {finest['num_theta']}):")
        print(f"  Max parameter drift: {max(dR_fine, dU_fine, damp_fine):.4f}% {'✓ PASS' if param_pass else '✗ FAIL'} (< 1%)")
        print(f"  Energy drift:        {dE_fine:.4f}% {'✓ PASS' if energy_pass else '✗ FAIL'} (< 0.1%)")
        print()

        overall_pass = param_pass and energy_pass
        print(f"Overall: {'✓ GRID CONVERGENCE VALIDATED' if overall_pass else '⚠ CONVERGENCE NOT ACHIEVED'}")

        return {
            'convergence_data': convergence_data,
            'success_criteria': {
                'max_parameter_drift_percent': float(max(dR_fine, dU_fine, damp_fine)),
                'energy_drift_percent': float(dE_fine),
                'parameter_converged': bool(param_pass),
                'energy_converged': bool(energy_pass),
                'overall_pass': bool(overall_pass)
            }
        }
    else:
        return {'convergence_data': convergence_data}


if __name__ == "__main__":
    print("="*80)
    print("V22 ELECTRON GRID CONVERGENCE TEST")
    print("="*80)
    print()
    print(f"Target: m_e = {TARGET_MASS} (electron-mass units)")
    print(f"β = {BETA}")
    print()

    grids = [
        (50, 10),    # Coarse
        (100, 20),   # Standard (current)
        (200, 40),   # Fine
        (400, 80)    # Very Fine
    ]

    results = []
    initial = None

    for num_r, num_theta in grids:
        res = optimize_at_resolution(num_r, num_theta, initial_guess=initial)
        results.append(res)
        # Use converged solution as initial guess for next resolution
        initial = [res['R'], res['U'], res['amplitude']]

    # Analyze convergence
    analysis = analyze_convergence(results)

    # Save results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    output = {
        'test': 'Grid Convergence',
        'particle': 'electron',
        'beta': BETA,
        'target_mass': TARGET_MASS,
        'timestamp': datetime.now().isoformat(),
        'grid_resolutions': results,
        'convergence_analysis': analysis
    }

    output_file = output_dir / "grid_convergence_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")
