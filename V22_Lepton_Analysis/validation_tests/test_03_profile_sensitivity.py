#!/usr/bin/env python3
"""
Profile Sensitivity Test for V22 Hill Vortex Electron Solution
===============================================================

PURPOSE: Test whether β = 3.1 is robust to the choice of density profile
         or whether the parabolic form is essential to the physics.

TEST: Run optimization with four different density depression profiles:
      1. Parabolic (current):  δρ = -a(1 - r²/R²)
      2. Quartic core:         δρ = -a(1 - r²/R²)²
      3. Gaussian core:        δρ = -a exp(-r²/R²)
      4. Linear:               δρ = -a(1 - r/R)

For each profile, optimize (R, U, amplitude) with FIXED β = 3.1.

SUCCESS CRITERIA:
  - If β = 3.1 works across all profiles without retuning β → β is robust
  - If only parabolic works → parabolic form is part of the physics
  - Compare residuals and parameter values across profiles

EXPECTED RUNTIME: ~5 minutes (4 profiles × ~1 min each)
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
NUM_R = 100
NUM_THETA = 20


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
    """
    Flexible density gradient supporting multiple profile types.
    """
    def __init__(self, R, amplitude, profile_type='parabolic', rho_vac=RHO_VAC):
        self.R = R
        self.amplitude = amplitude
        self.profile_type = profile_type
        self.rho_vac = rho_vac

    def delta_rho(self, r):
        """Density perturbation δρ(r) = ρ(r) - ρ_vac"""
        delta = np.zeros_like(r)
        mask = r < self.R

        if self.profile_type == 'parabolic':
            # δρ = -a(1 - r²/R²)
            delta[mask] = -self.amplitude * (1 - (r[mask] / self.R)**2)

        elif self.profile_type == 'quartic':
            # δρ = -a(1 - r²/R²)²
            delta[mask] = -self.amplitude * (1 - (r[mask] / self.R)**2)**2

        elif self.profile_type == 'gaussian':
            # δρ = -a exp(-r²/R²)
            delta[mask] = -self.amplitude * np.exp(-(r[mask] / self.R)**2)

        elif self.profile_type == 'linear':
            # δρ = -a(1 - r/R)
            delta[mask] = -self.amplitude * (1 - r[mask] / self.R)

        else:
            raise ValueError(f"Unknown profile type: {self.profile_type}")

        return delta

    def rho(self, r):
        """Total density ρ(r) = ρ_vac + δρ(r)"""
        return self.rho_vac + self.delta_rho(r)


class ElectronEnergy:
    def __init__(self, profile_type='parabolic', beta=BETA, r_max=10.0,
                 num_r=NUM_R, num_theta=NUM_THETA):
        self.profile_type = profile_type
        self.beta = beta
        self.rho_vac = RHO_VAC

        self.r = np.linspace(0.01, r_max, num_r)
        self.theta = np.linspace(0.01, np.pi-0.01, num_theta)
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]

    def total_energy(self, R, U, amplitude):
        stream = HillVortexStreamFunction(R, U)
        density = DensityGradient(R, amplitude, self.profile_type, self.rho_vac)

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


def optimize_profile(profile_type, initial_guess=None):
    """
    Optimize electron parameters for given density profile.

    Returns: dict with parameters, energies, and fit quality
    """
    print(f"\n{'='*80}")
    print(f"PROFILE: {profile_type.upper()}")
    print(f"{'='*80}")

    energy = ElectronEnergy(profile_type=profile_type, beta=BETA,
                            num_r=NUM_R, num_theta=NUM_THETA)

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

    print("Optimizing...")
    result = minimize(
        objective,
        initial_guess,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
    )

    if result.success:
        R_opt, U_opt, amp_opt = result.x
        E_total, E_circ, E_stab = energy.total_energy(R_opt, U_opt, amp_opt)

        residual = abs(E_total - TARGET_MASS)
        accuracy = (1 - residual / TARGET_MASS) * 100

        print(f"\nOptimized Parameters:")
        print(f"  R         = {R_opt:.6f}")
        print(f"  U         = {U_opt:.6f}")
        print(f"  amplitude = {amp_opt:.6f}")
        print(f"\nEnergies:")
        print(f"  E_circulation   = {E_circ:.6f}")
        print(f"  E_stabilization = {E_stab:.6f}")
        print(f"  E_total         = {E_total:.6f}")
        print(f"\nFit Quality:")
        print(f"  Target   = {TARGET_MASS:.6f}")
        print(f"  Achieved = {E_total:.6f}")
        print(f"  Residual = {residual:.3e}")
        print(f"  Accuracy = {accuracy:.4f}%")

        return {
            'profile': profile_type,
            'converged': True,
            'R': float(R_opt),
            'U': float(U_opt),
            'amplitude': float(amp_opt),
            'E_circulation': float(E_circ),
            'E_stabilization': float(E_stab),
            'E_total': float(E_total),
            'residual': float(residual),
            'accuracy_percent': float(accuracy),
            'iterations': int(result.nit)
        }
    else:
        print("\n⚠ Optimization failed to converge")
        return {
            'profile': profile_type,
            'converged': False
        }


def compare_profiles(results):
    """
    Compare performance across different density profiles.
    """
    print(f"\n{'='*80}")
    print("PROFILE COMPARISON")
    print(f"{'='*80}\n")

    converged = [r for r in results if r['converged']]

    if not converged:
        print("⚠ No profiles converged!")
        return None

    # Comparison table
    print(f"{'Profile':<15} {'R':<10} {'U':<10} {'amplitude':<10} "
          f"{'E_stab':<10} {'Residual':<12} {'Accuracy (%)':<12}")
    print("-" * 90)

    for res in results:
        if res['converged']:
            print(f"{res['profile']:<15} {res['R']:<10.4f} {res['U']:<10.4f} "
                  f"{res['amplitude']:<10.4f} {res['E_stabilization']:<10.4f} "
                  f"{res['residual']:<12.3e} {res['accuracy_percent']:<12.4f}")
        else:
            print(f"{res['profile']:<15} {'FAILED':<10}")

    # Reference: parabolic (original)
    parabolic = next((r for r in converged if r['profile'] == 'parabolic'), None)

    if parabolic and len(converged) > 1:
        print(f"\n{'='*80}")
        print("PARAMETER VARIATION FROM PARABOLIC")
        print(f"{'='*80}\n")

        print(f"{'Profile':<15} {'ΔR (%)':<12} {'ΔU (%)':<12} {'Δamp (%)':<12} {'ΔE_stab (%)':<12}")
        print("-" * 80)

        for res in converged:
            if res['profile'] != 'parabolic':
                dR = 100 * abs(res['R'] - parabolic['R']) / parabolic['R']
                dU = 100 * abs(res['U'] - parabolic['U']) / parabolic['U']
                damp = 100 * abs(res['amplitude'] - parabolic['amplitude']) / parabolic['amplitude']
                dE_stab = 100 * abs(res['E_stabilization'] - parabolic['E_stabilization']) / parabolic['E_stabilization']

                print(f"{res['profile']:<15} {dR:<12.2f} {dU:<12.2f} {damp:<12.2f} {dE_stab:<12.2f}")

    # Success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*80}\n")

    all_accurate = all(r['residual'] < 0.01 for r in converged)
    num_working = len(converged)
    num_total = len(results)

    print(f"Profiles converged: {num_working}/{num_total}")
    print(f"All residuals < 0.01: {all_accurate} {'✓' if all_accurate else '✗'}")
    print()

    if num_working == num_total and all_accurate:
        print("✓ β = 3.1 IS ROBUST - Works across all density profiles")
        print("  → β is a universal stiffness parameter, profile shape less critical")
    elif num_working == 1 and converged[0]['profile'] == 'parabolic':
        print("⚠ PARABOLIC PROFILE IS ESSENTIAL - Other profiles fail")
        print("  → Parabolic density gradient is part of the physics")
    else:
        print("⚠ PARTIAL SUCCESS - Some profiles work, others don't")
        print("  → Further investigation needed to identify physical constraint")

    return {
        'num_profiles_tested': num_total,
        'num_converged': num_working,
        'all_accurate': bool(all_accurate),
        'beta_robust': bool(num_working == num_total and all_accurate),
        'parabolic_essential': bool(num_working == 1 and converged[0]['profile'] == 'parabolic')
    }


if __name__ == "__main__":
    print("="*80)
    print("V22 ELECTRON PROFILE SENSITIVITY TEST")
    print("="*80)
    print()
    print(f"Target: m_e = {TARGET_MASS} (electron-mass units)")
    print(f"β = {BETA} (FIXED - not retuned)")
    print(f"Grid: (nr, nθ) = ({NUM_R}, {NUM_THETA})")
    print()

    profiles = [
        'parabolic',  # Current: δρ = -a(1 - r²/R²)
        'quartic',    # Sharper: δρ = -a(1 - r²/R²)²
        'gaussian',   # Smooth:  δρ = -a exp(-r²/R²)
        'linear'      # Gentle:  δρ = -a(1 - r/R)
    ]

    print("Testing density profiles:")
    for profile in profiles:
        print(f"  - {profile}")
    print()

    results = []

    for profile in profiles:
        res = optimize_profile(profile)
        results.append(res)

    # Compare results
    analysis = compare_profiles(results)

    # Save results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    output = {
        'test': 'Profile Sensitivity',
        'particle': 'electron',
        'beta': BETA,
        'beta_fixed': True,
        'target_mass': TARGET_MASS,
        'grid': {'num_r': NUM_R, 'num_theta': NUM_THETA},
        'timestamp': datetime.now().isoformat(),
        'profiles_tested': profiles,
        'results': results,
        'analysis': analysis
    }

    output_file = output_dir / "profile_sensitivity_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")
