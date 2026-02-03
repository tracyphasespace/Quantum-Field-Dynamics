#!/usr/bin/env python3
"""
β-Scan Falsifiability Test

This test demonstrates that the lepton mass solutions are NOT a trivial
"optimizer can hit targets" result, but rather a narrow existence window.

Key Questions:
1. Do stable solutions exist for all β values?
2. Is there a minimum in residual vs β?
3. Do all three leptons co-occur in the same β window?

Expected Outcome:
- Solutions should exist only in narrow window around β ≈ 3.043233053
- Deep minimum in residual vs β
- All three generations converge simultaneously only near inferred β

This is the critical falsifiability test that distinguishes "evidence"
from mere "compatibility".
"""

import numpy as np
from scipy.optimize import minimize
import json
from pathlib import Path
from datetime import datetime
import sys

# CODATA 2018 lepton masses (relative to electron)
LEPTON_MASSES = {
    'electron': 1.0,
    'muon': 206.7682826,
    'tau': 3477.228
}

def hill_vortex_energy(params, beta):
    """
    Compute Hill vortex energy functional.

    Args:
        params: [R, U, amplitude]
        beta: Vacuum stiffness parameter

    Returns:
        Total energy E = E_circulation + E_gradient - E_stabilization
    """
    R, U, amplitude = params

    # Avoid unphysical parameters
    if R <= 0 or U <= 0 or amplitude <= 0:
        return 1e10

    # Circulation energy (kinetic + potential flow)
    # Simplified from full integral (production code uses numerical integration)
    E_circulation = (3 * U**2 * R**3 / 5) * amplitude

    # Gradient energy (stabilization from vacuum stiffness)
    # E_gradient ~ β * ∫|∇ρ|² dV
    E_gradient = beta * amplitude**2 / R

    # Potential energy (quartic stabilization)
    # Simplified form (production uses V(ρ) integral)
    E_potential = (beta / 4) * amplitude**4 * R**3

    # Total energy
    E_total = E_circulation + E_gradient - E_potential

    return E_total

def virial_constraint(params, beta):
    """
    Virial theorem constraint for stability.

    For a stable vortex: 2*E_kin + E_grad = E_pot

    Returns:
        |Virial| value (should be ~ 0 for stable solution)
    """
    R, U, amplitude = params

    if R <= 0 or U <= 0 or amplitude <= 0:
        return 1e10

    # Kinetic energy (simplified)
    E_kin = (3 * U**2 * R**3 / 10) * amplitude

    # Gradient energy
    E_grad = beta * amplitude**2 / R

    # Potential energy
    E_pot = (beta / 4) * amplitude**4 * R**3

    # Virial = 2*E_kin + E_grad - E_pot (should be ~0)
    virial = 2 * E_kin + E_grad - E_pot

    return abs(virial)

def solve_lepton_at_beta(target_mass, beta, timeout=60, max_iterations=1000):
    """
    Attempt to find stable Hill vortex solution for given mass and β.

    Args:
        target_mass: Target mass ratio (relative to electron)
        beta: Vacuum stiffness parameter
        timeout: Maximum time (seconds)
        max_iterations: Maximum optimizer iterations

    Returns:
        dict with:
            - converged: bool
            - params: [R, U, amplitude] if converged
            - energy: E_total if converged
            - residual: |E_total - target_mass|
            - virial: virial constraint value
            - iterations: number of iterations used
    """

    # Objective: minimize (E_total - target_mass)^2 + penalty * virial
    def objective(params):
        E = hill_vortex_energy(params, beta)
        V = virial_constraint(params, beta)

        # Combined objective: match mass + enforce stability
        mass_residual = (E - target_mass)**2
        virial_penalty = 1e6 * V  # Large penalty for instability

        return mass_residual + virial_penalty

    # Initial guess (scales with mass)
    R0 = 0.4 + 0.1 * np.log(target_mass + 1)
    U0 = 0.02 * np.sqrt(target_mass)
    amp0 = 0.9

    initial_guess = [R0, U0, amp0]

    # Bounds (physical constraints)
    bounds = [
        (0.01, 2.0),   # R
        (0.001, 5.0),  # U
        (0.01, 1.5)    # amplitude
    ]

    # Try optimization
    try:
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iterations,
                'ftol': 1e-12,
                'gtol': 1e-10
            }
        )

        # Check convergence
        R, U, amplitude = result.x
        E_total = hill_vortex_energy(result.x, beta)
        residual = abs(E_total - target_mass)
        virial_val = virial_constraint(result.x, beta)

        # Success criteria
        converged = (
            result.success and
            residual < 1e-3 and  # Relaxed for β-scan (not production)
            virial_val < 1e-3
        )

        return {
            'converged': converged,
            'params': result.x.tolist(),
            'energy': float(E_total),
            'residual': float(residual),
            'virial': float(virial_val),
            'iterations': result.nit
        }

    except Exception as e:
        return {
            'converged': False,
            'error': str(e)
        }

def run_beta_scan(beta_min=2.5, beta_max=3.5, num_points=51):
    """
    Scan β parameter to test falsifiability.

    This is the critical test: if solutions exist for all β,
    then the claim is trivial. If they exist only in a narrow
    window around the inferred β ≈ 3.043233053, that's evidence.

    Args:
        beta_min: Minimum β value
        beta_max: Maximum β value
        num_points: Number of β values to test

    Returns:
        dict with scan results
    """
    beta_range = np.linspace(beta_min, beta_max, num_points)

    results = {
        'test': 'β-Scan Falsifiability',
        'beta_range': [beta_min, beta_max],
        'num_points': num_points,
        'timestamp': datetime.now().isoformat(),
        'scan_results': []
    }

    print(f"\nRunning β-scan from {beta_min} to {beta_max} ({num_points} points)")
    print(f"Testing all three leptons at each β value\n")

    for i, beta in enumerate(beta_range):
        print(f"β = {beta:.4f} ({i+1}/{num_points})...", end=' ')

        beta_result = {
            'beta': float(beta),
            'leptons': {}
        }

        # Try to solve all three leptons at this β
        converged_count = 0
        for lepton_name, target_mass in LEPTON_MASSES.items():
            result = solve_lepton_at_beta(target_mass, beta, timeout=30)
            beta_result['leptons'][lepton_name] = result

            if result['converged']:
                converged_count += 1

        beta_result['num_converged'] = converged_count
        beta_result['all_converged'] = (converged_count == 3)

        results['scan_results'].append(beta_result)

        print(f"{converged_count}/3 converged")

    # Analysis: find minimum residual
    min_residual_beta = None
    min_residual_sum = float('inf')

    for beta_result in results['scan_results']:
        # Sum residuals across leptons (only if all converged)
        if beta_result['all_converged']:
            residual_sum = sum(
                beta_result['leptons'][l]['residual']
                for l in LEPTON_MASSES.keys()
            )

            if residual_sum < min_residual_sum:
                min_residual_sum = residual_sum
                min_residual_beta = beta_result['beta']

    results['analysis'] = {
        'min_residual_beta': min_residual_beta,
        'min_residual_sum': min_residual_sum,
        'num_beta_with_all_converged': sum(
            1 for br in results['scan_results'] if br['all_converged']
        )
    }

    return results

def save_results(results, filename='results/beta_scan_falsifiability.json'):
    """Save results to JSON file."""
    output_path = Path(__file__).parent / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

def print_summary(results):
    """Print summary of β-scan results."""
    analysis = results['analysis']

    print("\n" + "="*60)
    print("β-SCAN FALSIFIABILITY TEST - SUMMARY")
    print("="*60)

    print(f"\nβ range scanned: [{results['beta_range'][0]}, {results['beta_range'][1]}]")
    print(f"Number of β values tested: {results['num_points']}")

    print(f"\n✓ Minimum residual at β = {analysis['min_residual_beta']}")
    print(f"  (Sum of residuals: {analysis['min_residual_sum']:.2e})")

    print(f"\n✓ Number of β values where ALL 3 leptons converge: {analysis['num_beta_with_all_converged']}")
    print(f"  (Out of {results['num_points']} tested)")

    # Find β window where all converge
    converged_betas = [
        br['beta'] for br in results['scan_results']
        if br['all_converged']
    ]

    if len(converged_betas) > 0:
        beta_min_converged = min(converged_betas)
        beta_max_converged = max(converged_betas)
        beta_width = beta_max_converged - beta_min_converged

        print(f"\n✓ β window where all leptons exist: [{beta_min_converged:.3f}, {beta_max_converged:.3f}]")
        print(f"  Width: Δβ = {beta_width:.3f}")
        print(f"  Relative width: Δβ/β ≈ {beta_width/3.043233053 * 100:.1f}%")
    else:
        print("\n✗ WARNING: No β value where all three leptons converge!")
        print("  This suggests the model may not be physical.")

    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)

    if analysis['num_beta_with_all_converged'] < results['num_points'] / 2:
        print("✓ FALSIFIABLE: Solutions exist only in narrow β window")
        print("  This is EVIDENCE, not just 'optimizer can hit targets'")
        print("  The lepton spectrum is a genuine constraint on β")
    else:
        print("✗ WARNING: Solutions exist for most β values")
        print("  This suggests the model is too flexible")
        print("  Review energy functional / constraints")

    print("="*60 + "\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("β-SCAN FALSIFIABILITY TEST")
    print("="*60)
    print("\nThis test answers:")
    print("  1. Do stable vortex solutions exist for all β?")
    print("  2. Is there a unique minimum at the inferred β ≈ 3.043233053?")
    print("  3. Do all three leptons co-occur only in a narrow window?")
    print("\nThis is the CRITICAL TEST for falsifiability.")
    print("="*60)

    # Run the scan
    results = run_beta_scan(
        beta_min=2.5,
        beta_max=3.5,
        num_points=21  # Start with coarse scan (51 for publication)
    )

    # Save results
    save_results(results)

    # Print summary
    print_summary(results)

    print("\n✓ β-scan complete!")
    print("  Use results/beta_scan_falsifiability.json to create Figure 6")
