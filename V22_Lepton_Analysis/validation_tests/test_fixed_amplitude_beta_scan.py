#!/usr/bin/env python3
"""
Option 2: Fixed-Amplitude β-Scan (Diagnostic Symmetry-Break Test)

Purpose: Test if breaking the amplitude ∝ 1/√β degeneracy restores β identifiability.

Method:
- Fix amplitude at multiple values: A = {0.25, 0.5, 0.75, 0.9}
- For each (A, β), optimize only (R, U) to match lepton mass
- Check if β minimum emerges when amplitude scaling is blocked

Frame as: "Diagnostic gauge-fixing to isolate β-dependence"
NOT: "Physical constraint" (that's Option 3)

Expected outcome:
- If β minimum emerges → degeneracy was in amplitude, can be broken
- If still flat → degeneracy moved to (R, U), need second observable
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

LEPTON_MASSES = {
    'electron': ELECTRON_MASS,
    'muon': MUON_TO_ELECTRON_RATIO,
    'tau': TAU_TO_ELECTRON_RATIO
}

def solve_with_fixed_amplitude(target_mass, beta, amplitude_fixed,
                               num_r=100, num_theta=20, max_iter=500):
    """
    Solve for (R, U) with amplitude FIXED.

    This breaks the amplitude ∝ 1/√β scaling degeneracy.

    Args:
        target_mass: Target mass ratio
        beta: Vacuum stiffness
        amplitude_fixed: Fixed amplitude value (NOT optimized)
        num_r, num_theta: Grid resolution
        max_iter: Max optimizer iterations

    Returns:
        dict with results
    """
    energy_calc = LeptonEnergy(beta=beta, num_r=num_r, num_theta=num_theta)

    def objective(params):
        R, U = params

        # Physical bounds
        if R <= 0 or U <= 0:
            return 1e10

        # Check cavitation (ρ ≥ 0)
        if amplitude_fixed > RHO_VAC:
            return 1e10  # Unphysical

        try:
            E_total, E_circ, E_stab = energy_calc.total_energy(
                R, U, amplitude_fixed
            )
            return (E_total - target_mass)**2
        except:
            return 1e10

    # Initial guess (scaling)
    R0 = 0.44
    U0 = 0.024 * np.sqrt(target_mass)

    # Bounds
    bounds = [
        (0.1, 1.0),   # R
        (0.001, 3.0)  # U
    ]

    try:
        result = minimize(
            objective,
            [R0, U0],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )

        R_opt, U_opt = result.x
        E_total, E_circ, E_stab = energy_calc.total_energy(
            R_opt, U_opt, amplitude_fixed
        )
        residual = abs(E_total - target_mass)

        converged = result.success and residual < 1e-3

        return {
            'converged': bool(converged),
            'R': float(R_opt),
            'U': float(U_opt),
            'amplitude': float(amplitude_fixed),  # Fixed, not optimized
            'E_total': float(E_total),
            'E_circ': float(E_circ),
            'E_stab': float(E_stab),
            'residual': float(residual),
            'iterations': int(result.nit),
            'optimizer_success': bool(result.success)
        }

    except Exception as e:
        return {
            'converged': False,
            'error': str(e),
            'amplitude': float(amplitude_fixed)
        }


def run_fixed_amplitude_scan(amplitude_values=[0.25, 0.5, 0.75, 0.9],
                             beta_min=2.5, beta_max=3.5, num_beta=11):
    """
    Run β-scan with grid of fixed amplitude values.

    For each amplitude:
        For each β:
            Optimize (R, U) only
            Record residual

    Check if β minimum emerges when amplitude scaling is blocked.

    Args:
        amplitude_values: Grid of fixed amplitude values
        beta_min, beta_max: β range
        num_beta: Number of β points

    Returns:
        Complete results dict
    """
    beta_range = np.linspace(beta_min, beta_max, num_beta)

    results = {
        'test': 'Fixed-Amplitude β-Scan (Diagnostic Symmetry Break)',
        'purpose': 'Test if blocking amplitude ∝ 1/√β restores β identifiability',
        'amplitude_values': [float(a) for a in amplitude_values],
        'beta_range': [float(beta_min), float(beta_max)],
        'num_beta': int(num_beta),
        'timestamp': datetime.now().isoformat(),
        'scans': []
    }

    print("\n" + "="*70)
    print("OPTION 2: FIXED-AMPLITUDE β-SCAN (Diagnostic)")
    print("="*70)
    print(f"\nTesting {len(amplitude_values)} fixed amplitude values")
    print(f"β range: [{beta_min}, {beta_max}] with {num_beta} points")
    print(f"\nPurpose: Check if β identifiability is restored when")
    print(f"         amplitude ∝ 1/√β degeneracy is broken\n")

    for amplitude in amplitude_values:
        print(f"\n{'─'*70}")
        print(f"AMPLITUDE = {amplitude:.2f} (FIXED)")
        print(f"{'─'*70}")

        scan_result = {
            'amplitude': float(amplitude),
            'beta_scan': []
        }

        for i, beta in enumerate(beta_range):
            print(f"[{i+1:2d}/{num_beta}] β = {beta:.3f} ... ", end='', flush=True)

            beta_result = {
                'beta': float(beta),
                'leptons': {}
            }

            # Try all three leptons
            converged_count = 0
            total_residual = 0.0

            for lepton_name, target_mass in LEPTON_MASSES.items():
                result = solve_with_fixed_amplitude(
                    target_mass, beta, amplitude,
                    num_r=100, num_theta=20
                )

                beta_result['leptons'][lepton_name] = result

                if result['converged']:
                    converged_count += 1
                    total_residual += result['residual']

            beta_result['num_converged'] = int(converged_count)
            beta_result['all_converged'] = bool(converged_count == 3)
            beta_result['total_residual'] = float(total_residual) if converged_count > 0 else None

            scan_result['beta_scan'].append(beta_result)

            # Progress
            if converged_count == 3:
                print(f"✓ All 3 (residual: {total_residual:.2e})")
            else:
                print(f"✗ {converged_count}/3")

        results['scans'].append(scan_result)

    return results


def analyze_results(results):
    """
    Analyze if β minimum emerges for any amplitude value.
    """
    print("\n" + "="*70)
    print("ANALYSIS: β IDENTIFIABILITY TEST")
    print("="*70)

    for scan in results['scans']:
        amplitude = scan['amplitude']
        beta_scan = scan['beta_scan']

        # Extract converged residuals
        converged_points = [
            (point['beta'], point['total_residual'])
            for point in beta_scan
            if point['all_converged']
        ]

        if len(converged_points) == 0:
            print(f"\nAmplitude = {amplitude:.2f}: NO converged solutions")
            continue

        betas = [p[0] for p in converged_points]
        residuals = [p[1] for p in converged_points]

        # Find minimum
        min_idx = np.argmin(residuals)
        beta_min = betas[min_idx]
        residual_min = residuals[min_idx]

        # Check if flat or peaked
        residual_range = max(residuals) - min(residuals)
        residual_variation = residual_range / min(residuals) if min(residuals) > 0 else 0

        print(f"\nAmplitude = {amplitude:.2f}:")
        print(f"  Converged at {len(converged_points)}/{results['num_beta']} β values")
        print(f"  Minimum residual: {residual_min:.2e} at β = {beta_min:.3f}")
        print(f"  Residual range: {residual_range:.2e}")
        print(f"  Variation: {residual_variation:.1%}")

        if residual_variation < 0.1:  # <10% variation
            print(f"  → FLAT (weak β selectivity)")
        elif residual_variation < 1.0:  # <100% variation
            print(f"  → MODERATE β preference")
        else:
            print(f"  → SHARP β minimum (strong identifiability)")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    # Overall assessment
    has_sharp_minimum = False
    has_moderate_preference = False

    for scan in results['scans']:
        converged_points = [
            (point['beta'], point['total_residual'])
            for point in scan['beta_scan']
            if point['all_converged']
        ]

        if len(converged_points) > 0:
            residuals = [p[1] for p in converged_points]
            variation = (max(residuals) - min(residuals)) / min(residuals)

            if variation > 1.0:
                has_sharp_minimum = True
            elif variation > 0.1:
                has_moderate_preference = True

    if has_sharp_minimum:
        print("\n✓ SYMMETRY BREAK SUCCESSFUL")
        print("  • Fixing amplitude DOES restore β identifiability")
        print("  • Clear β minimum emerges for some amplitude values")
        print("  • Next: Implement Option 3 (positivity constraint)")
    elif has_moderate_preference:
        print("\n~ PARTIAL SUCCESS")
        print("  • Some β preference emerges, but not sharp")
        print("  • May need tighter tolerance or second constraint")
        print("  • Proceed to Option 3 with caution")
    else:
        print("\n✗ SYMMETRY BREAK FAILED")
        print("  • Residuals still flat across β")
        print("  • Degeneracy has moved to (R, U) space")
        print("  • NEED Option 1 (second observable) to proceed")

    print("="*70 + "\n")


def save_results(results, filename='results/fixed_amplitude_beta_scan.json'):
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
        description='Fixed-amplitude β-scan (Option 2 diagnostic)'
    )
    parser.add_argument('--amplitudes', type=float, nargs='+',
                       default=[0.25, 0.5, 0.75, 0.9],
                       help='Fixed amplitude values to test')
    parser.add_argument('--beta-min', type=float, default=2.5)
    parser.add_argument('--beta-max', type=float, default=3.5)
    parser.add_argument('--num-beta', type=int, default=11,
                       help='Number of β points (11 for quick, 21 for thorough)')

    args = parser.parse_args()

    # Run scan
    results = run_fixed_amplitude_scan(
        amplitude_values=args.amplitudes,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        num_beta=args.num_beta
    )

    # Analyze
    analyze_results(results)

    # Save
    save_results(results)

    print("✓ Option 2 diagnostic complete")
    print("  If successful → proceed to Option 3 (positivity constraint)")
    print("  If failed → need Option 1 (magnetic moment)")
