#!/usr/bin/env python3
"""
Compare results from informed vs uninformed priors runs

This script compares the stage2_simple.py results to demonstrate the fix
for the dataset-dependent priors bug.
"""

import json
from pathlib import Path

def load_results(results_dir):
    """Load best_fit.json from results directory"""
    best_fit_path = Path(results_dir) / "best_fit.json"
    if not best_fit_path.exists():
        return None
    with open(best_fit_path, 'r') as f:
        return json.load(f)

def print_comparison():
    """Print comparison table"""
    # Golden reference (November 5, 2024)
    golden = {
        "k_J": 10.770,
        "eta_prime": -7.988,
        "xi": -6.908,
        "k_J_std": 4.567,
        "eta_prime_std": 1.439,
        "xi_std": 3.746
    }

    # Load results
    uninformed = load_results("../results/v15_clean/stage2_simple_plus_sign")
    informed = load_results("../results/v15_clean/stage2_simple_informed_priors")

    print("=" * 80)
    print("COMPARISON: UNINFORMED vs INFORMED PRIORS")
    print("=" * 80)
    print()

    print("Golden Reference (November 5, 2024):")
    print(f"  k_J   = {golden['k_J']:.3f} ± {golden['k_J_std']:.3f}")
    print(f"  eta'  = {golden['eta_prime']:.3f} ± {golden['eta_prime_std']:.3f}")
    print(f"  xi    = {golden['xi']:.3f} ± {golden['xi_std']:.3f}")
    print()

    if uninformed:
        print("Uninformed Priors (dataset-dependent bug):")
        print(f"  k_J   = {uninformed['k_J']:.3f} ± {uninformed['k_J_std']:.3f}  ({uninformed['k_J']/golden['k_J']*100:.0f}% of target)")
        print(f"  eta'  = {uninformed['eta_prime']:.3f} ± {uninformed['eta_prime_std']:.3f}  ({abs(uninformed['eta_prime'])/abs(golden['eta_prime'])*100:.0f}% of target)")
        print(f"  xi    = {uninformed['xi']:.3f} ± {uninformed['xi_std']:.3f}  ({abs(uninformed['xi'])/abs(golden['xi'])*100:.0f}% of target)")
        print()
    else:
        print("Uninformed priors results not found")
        print()

    if informed:
        print("Informed Priors (fixed - priors on standardized coefficients):")
        print(f"  k_J   = {informed['k_J']:.3f} ± {informed['k_J_std']:.3f}  ({informed['k_J']/golden['k_J']*100:.0f}% of target)")
        print(f"  eta'  = {informed['eta_prime']:.3f} ± {informed['eta_prime_std']:.3f}  ({abs(informed['eta_prime'])/abs(golden['eta_prime'])*100:.0f}% of target)")
        print(f"  xi    = {informed['xi']:.3f} ± {informed['xi_std']:.3f}  ({abs(informed['xi'])/abs(golden['xi'])*100:.0f}% of target)")
        print()

        # Check if within ±30% validation range
        k_valid = abs(informed['k_J'] - golden['k_J']) / golden['k_J'] < 0.3
        eta_valid = abs(abs(informed['eta_prime']) - abs(golden['eta_prime'])) / abs(golden['eta_prime']) < 0.3
        xi_valid = abs(abs(informed['xi']) - abs(golden['xi'])) / abs(golden['xi']) < 0.3

        print("Validation (within ±30% of golden reference):")
        print(f"  k_J:  {'✓ PASS' if k_valid else '✗ FAIL'}")
        print(f"  eta': {'✓ PASS' if eta_valid else '✗ FAIL'}")
        print(f"  xi:   {'✓ PASS' if xi_valid else '✗ FAIL'}")
        print()

        if k_valid and eta_valid and xi_valid:
            print("✓ ALL PARAMETERS WITHIN RANGE - FIX SUCCESSFUL!")
        else:
            print("✗ SOME PARAMETERS OUT OF RANGE")
    else:
        print("Informed priors results not yet available (run still in progress?)")

    print()
    print("=" * 80)

if __name__ == '__main__':
    print_comparison()
