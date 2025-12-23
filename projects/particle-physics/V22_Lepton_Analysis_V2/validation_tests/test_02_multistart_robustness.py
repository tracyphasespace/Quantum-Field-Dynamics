#!/usr/bin/env python3
"""
Multi-Start Robustness Test for V22 Hill Vortex Electron Solution
==================================================================

PURPOSE: Determine whether the fitted electron solution is unique or whether
         multiple local minima exist in the parameter space.

TEST: Run optimization from 50 random initial seeds spanning physically
      reasonable parameter ranges:
      - R in [0.2, 0.8]
      - U in [0.01, 0.10]
      - amplitude in [0.5, 1.0]

SUCCESS CRITERIA:
  - If solutions cluster tightly (σ/μ < 1% for each parameter), solution is unique
  - If multiple distinct clusters exist, identify selection principle
  - All converged solutions should have similar residuals (< 0.01% difference)

EXPECTED RUNTIME: ~10-15 minutes (50 independent optimizations)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Constants
RHO_VAC = 1.0
BETA = 3.1
TARGET_MASS = 1.0  # Electron in electron-mass units

# Use standard grid from main analysis
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
    def __init__(self, beta=BETA, r_max=10.0, num_r=NUM_R, num_theta=NUM_THETA):
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


def optimize_from_seed(seed_number, initial_guess, energy):
    """
    Run single optimization from given initial guess.
    """
    def objective(params):
        R, U, amplitude = params
        if R <= 0 or U <= 0 or amplitude <= 0 or amplitude > 1.0:
            return 1e10
        try:
            E_total, _, _ = energy.total_energy(R, U, amplitude)
            return (E_total - TARGET_MASS)**2
        except:
            return 1e10

    result = minimize(
        objective,
        initial_guess,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
    )

    if result.success:
        R_opt, U_opt, amp_opt = result.x
        E_total, E_circ, E_stab = energy.total_energy(R_opt, U_opt, amp_opt)

        return {
            'seed': seed_number,
            'initial_guess': [float(x) for x in initial_guess],
            'converged': True,
            'R': float(R_opt),
            'U': float(U_opt),
            'amplitude': float(amp_opt),
            'E_circulation': float(E_circ),
            'E_stabilization': float(E_stab),
            'E_total': float(E_total),
            'residual': float(abs(E_total - TARGET_MASS)),
            'iterations': int(result.nit),
            'function_evals': int(result.nfev)
        }
    else:
        return {
            'seed': seed_number,
            'initial_guess': [float(x) for x in initial_guess],
            'converged': False
        }


def cluster_solutions(solutions, tolerance=0.01):
    """
    Identify distinct solution clusters.

    tolerance: Relative difference threshold for same cluster (default 1%)
    """
    if not solutions:
        return []

    clusters = []

    for sol in solutions:
        R, U, amp = sol['R'], sol['U'], sol['amplitude']

        # Check if solution belongs to existing cluster
        found_cluster = False
        for cluster in clusters:
            R_ref = cluster['center']['R']
            U_ref = cluster['center']['U']
            amp_ref = cluster['center']['amplitude']

            dR = abs(R - R_ref) / R_ref
            dU = abs(U - U_ref) / U_ref
            damp = abs(amp - amp_ref) / amp_ref

            if max(dR, dU, damp) < tolerance:
                cluster['members'].append(sol)
                found_cluster = True
                break

        if not found_cluster:
            # Create new cluster
            clusters.append({
                'center': {'R': R, 'U': U, 'amplitude': amp},
                'members': [sol]
            })

    # Recompute cluster centers as means
    for cluster in clusters:
        members = cluster['members']
        cluster['center'] = {
            'R': np.mean([m['R'] for m in members]),
            'U': np.mean([m['U'] for m in members]),
            'amplitude': np.mean([m['amplitude'] for m in members]),
            'E_total': np.mean([m['E_total'] for m in members])
        }
        cluster['size'] = len(members)
        cluster['std'] = {
            'R': np.std([m['R'] for m in members]),
            'U': np.std([m['U'] for m in members]),
            'amplitude': np.std([m['amplitude'] for m in members]),
            'E_total': np.std([m['E_total'] for m in members])
        }

    # Sort by cluster size
    clusters.sort(key=lambda c: c['size'], reverse=True)

    return clusters


def analyze_robustness(solutions):
    """
    Analyze solution landscape and uniqueness.
    """
    converged = [s for s in solutions if s['converged']]
    failed = [s for s in solutions if not s['converged']]

    print(f"\n{'='*80}")
    print("MULTI-START ROBUSTNESS ANALYSIS")
    print(f"{'='*80}\n")

    print(f"Total runs: {len(solutions)}")
    print(f"Converged: {len(converged)} ({100*len(converged)/len(solutions):.1f}%)")
    print(f"Failed: {len(failed)} ({100*len(failed)/len(solutions):.1f}%)")

    if not converged:
        print("\n⚠ No converged solutions found!")
        return None

    # Cluster analysis
    clusters = cluster_solutions(converged, tolerance=0.01)

    print(f"\n{'='*80}")
    print(f"SOLUTION CLUSTERS (1% tolerance)")
    print(f"{'='*80}\n")

    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {cluster['size']} solutions ({100*cluster['size']/len(converged):.1f}%)")
        print(f"  Center: R = {cluster['center']['R']:.6f}, "
              f"U = {cluster['center']['U']:.6f}, "
              f"amplitude = {cluster['center']['amplitude']:.6f}")
        print(f"  E_total = {cluster['center']['E_total']:.6f}")
        print(f"  Std Dev: R = {cluster['std']['R']:.6f}, "
              f"U = {cluster['std']['U']:.6f}, "
              f"amplitude = {cluster['std']['amplitude']:.6f}")
        print(f"  Variation: R = {100*cluster['std']['R']/cluster['center']['R']:.3f}%, "
              f"U = {100*cluster['std']['U']/cluster['center']['U']:.3f}%, "
              f"amplitude = {100*cluster['std']['amplitude']/cluster['center']['amplitude']:.3f}%")
        print()

    # Overall statistics
    all_R = [s['R'] for s in converged]
    all_U = [s['U'] for s in converged]
    all_amp = [s['amplitude'] for s in converged]
    all_E = [s['E_total'] for s in converged]
    all_res = [s['residual'] for s in converged]

    print(f"{'='*80}")
    print(f"OVERALL STATISTICS (all converged solutions)")
    print(f"{'='*80}\n")

    print(f"R:         mean = {np.mean(all_R):.6f}, std = {np.std(all_R):.6f}, "
          f"CV = {100*np.std(all_R)/np.mean(all_R):.3f}%")
    print(f"U:         mean = {np.mean(all_U):.6f}, std = {np.std(all_U):.6f}, "
          f"CV = {100*np.std(all_U)/np.mean(all_U):.3f}%")
    print(f"amplitude: mean = {np.mean(all_amp):.6f}, std = {np.std(all_amp):.6f}, "
          f"CV = {100*np.std(all_amp)/np.mean(all_amp):.3f}%")
    print(f"E_total:   mean = {np.mean(all_E):.6f}, std = {np.std(all_E):.6f}, "
          f"CV = {100*np.std(all_E)/np.mean(all_E):.3f}%")
    print(f"Residual:  mean = {np.mean(all_res):.3e}, max = {np.max(all_res):.3e}")

    # Success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*80}\n")

    cv_R = 100 * np.std(all_R) / np.mean(all_R)
    cv_U = 100 * np.std(all_U) / np.mean(all_U)
    cv_amp = 100 * np.std(all_amp) / np.mean(all_amp)
    max_cv = max(cv_R, cv_U, cv_amp)

    unique = max_cv < 1.0
    residual_consistent = np.max(all_res) < 0.01

    print(f"Solution uniqueness (CV < 1%): {max_cv:.3f}% {'✓ PASS' if unique else '✗ FAIL'}")
    print(f"Residual consistency (< 0.01): {np.max(all_res):.3e} {'✓ PASS' if residual_consistent else '✗ FAIL'}")
    print()

    if len(clusters) == 1 and unique:
        print("✓ SOLUTION IS UNIQUE - Single tight cluster found")
    elif len(clusters) > 1:
        print(f"⚠ MULTIPLE SOLUTION CLUSTERS DETECTED ({len(clusters)} clusters)")
        print("  Selection principle needed:")
        print("  - Lowest residual?")
        print("  - Stability analysis (second variation)?")
        print("  - Physical constraints (e.g., amplitude closest to cavitation)?")
    else:
        print(f"⚠ SOLUTION SPREAD DETECTED - CV up to {max_cv:.3f}%")
        print("  Consider tighter convergence tolerances or deeper investigation")

    return {
        'num_runs': len(solutions),
        'num_converged': len(converged),
        'num_failed': len(failed),
        'convergence_rate': float(len(converged) / len(solutions)),
        'num_clusters': len(clusters),
        'clusters': [{
            'size': c['size'],
            'fraction': float(c['size'] / len(converged)),
            'center': c['center'],
            'std': c['std']
        } for c in clusters],
        'overall_statistics': {
            'R_mean': float(np.mean(all_R)),
            'R_std': float(np.std(all_R)),
            'R_cv_percent': float(cv_R),
            'U_mean': float(np.mean(all_U)),
            'U_std': float(np.std(all_U)),
            'U_cv_percent': float(cv_U),
            'amplitude_mean': float(np.mean(all_amp)),
            'amplitude_std': float(np.std(all_amp)),
            'amplitude_cv_percent': float(cv_amp),
            'E_total_mean': float(np.mean(all_E)),
            'E_total_std': float(np.std(all_E)),
            'residual_mean': float(np.mean(all_res)),
            'residual_max': float(np.max(all_res))
        },
        'success_criteria': {
            'solution_unique': bool(unique),
            'residuals_consistent': bool(residual_consistent),
            'overall_pass': bool(unique and residual_consistent)
        }
    }


if __name__ == "__main__":
    print("="*80)
    print("V22 ELECTRON MULTI-START ROBUSTNESS TEST")
    print("="*80)
    print()
    print(f"Target: m_e = {TARGET_MASS} (electron-mass units)")
    print(f"β = {BETA}")
    print(f"Grid: (nr, nθ) = ({NUM_R}, {NUM_THETA})")
    print()

    # Generate random initial seeds
    np.random.seed(42)  # Reproducibility
    num_runs = 50

    print(f"Generating {num_runs} random initial seeds...")
    print("  R in [0.2, 0.8]")
    print("  U in [0.01, 0.10]")
    print("  amplitude in [0.5, 1.0]")
    print()

    initial_seeds = []
    for i in range(num_runs):
        R_init = np.random.uniform(0.2, 0.8)
        U_init = np.random.uniform(0.01, 0.10)
        amp_init = np.random.uniform(0.5, 1.0)
        initial_seeds.append([R_init, U_init, amp_init])

    # Create energy calculator
    energy = ElectronEnergy(beta=BETA, num_r=NUM_R, num_theta=NUM_THETA)

    # Run optimizations
    print("Running optimizations...")
    solutions = []

    for i, seed in enumerate(initial_seeds, 1):
        print(f"  Run {i}/{num_runs}...", end='\r')
        result = optimize_from_seed(i, seed, energy)
        solutions.append(result)

    print(f"  Completed {num_runs}/{num_runs} runs.                ")

    # Analyze results
    analysis = analyze_robustness(solutions)

    # Save results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/validation_tests/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    output = {
        'test': 'Multi-Start Robustness',
        'particle': 'electron',
        'beta': BETA,
        'target_mass': TARGET_MASS,
        'grid': {'num_r': NUM_R, 'num_theta': NUM_THETA},
        'num_runs': num_runs,
        'timestamp': datetime.now().isoformat(),
        'solutions': solutions,
        'analysis': analysis
    }

    output_file = output_dir / "multistart_robustness_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")
