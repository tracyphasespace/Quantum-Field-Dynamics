#!/usr/bin/env python3
"""
Beta Transcendental Tension Explorer

The Conundrum:
- β = 3.043: True root of e^β/β = K, but c₂ prediction degrades
- β = 3.058: Excellent c₂ prediction, but ~0.5% tension in transcendental

This script explores the physics of this tension systematically.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import lambertw
import matplotlib.pyplot as plt

# =============================================================================
# EMPIRICAL CONSTANTS (Independent Measurements)
# =============================================================================

# CODATA 2018
ALPHA_INV = 137.035999084  # Fine structure constant inverse

# NuBase 2020 (from 2,550 nuclei)
C1_SURFACE = 0.496297      # Surface coefficient (MeV)
C2_EMPIRICAL = 0.32704     # Volume coefficient (measured)

# Mathematical constant
PI_SQ = np.pi**2

# =============================================================================
# TRANSCENDENTAL EQUATION ANALYSIS
# =============================================================================

def f_transcendental(beta):
    """f(β) = e^β / β - the transcendental function."""
    return np.exp(beta) / beta

def K_target():
    """K = (α⁻¹ × c₁) / π² - the target constant."""
    return (ALPHA_INV * C1_SURFACE) / PI_SQ

def c2_predicted(beta):
    """c₂ = 1/β - the volume coefficient prediction."""
    return 1.0 / beta

def transcendental_error(beta):
    """Error in transcendental equation: |e^β/β - K|."""
    return abs(f_transcendental(beta) - K_target())

def c2_error(beta):
    """Error in c₂ prediction: |1/β - c₂_empirical|."""
    return abs(c2_predicted(beta) - C2_EMPIRICAL)

def c2_relative_error(beta):
    """Relative error in c₂ prediction."""
    return abs(c2_predicted(beta) - C2_EMPIRICAL) / C2_EMPIRICAL * 100

# =============================================================================
# FIND THE ROOTS
# =============================================================================

def find_transcendental_root():
    """Find β where e^β/β = K using Brent's method."""
    K = K_target()

    # The equation e^β/β = K has two roots for K > e (which is our case)
    # We want the larger root (β > 1)

    # f(β) = e^β/β has minimum at β = 1 where f(1) = e ≈ 2.718
    # For K > e, there are two roots

    def objective(beta):
        return f_transcendental(beta) - K

    # Find root in range [1.5, 10] for the larger root
    try:
        root = brentq(objective, 1.5, 10.0)
        return root
    except ValueError:
        # If no root in that range, try different bounds
        return None

def find_c2_optimal_beta():
    """Find β that exactly predicts c₂ = 0.32704."""
    return 1.0 / C2_EMPIRICAL

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_tension():
    """Comprehensive analysis of the beta tension."""

    print("=" * 70)
    print("BETA TRANSCENDENTAL TENSION ANALYSIS")
    print("=" * 70)
    print()

    # Calculate key values
    K = K_target()
    beta_transcendental = find_transcendental_root()
    beta_c2_optimal = find_c2_optimal_beta()
    beta_golden = 3.058230856  # Current GoldenLoop.lean value

    print("EMPIRICAL INPUTS:")
    print(f"  α⁻¹ = {ALPHA_INV} (CODATA 2018)")
    print(f"  c₁  = {C1_SURFACE} (NuBase 2020)")
    print(f"  c₂  = {C2_EMPIRICAL} (NuBase 2020)")
    print(f"  π²  = {PI_SQ:.10f}")
    print()

    print("DERIVED TARGET:")
    print(f"  K = (α⁻¹ × c₁) / π² = {K:.10f}")
    print()

    print("=" * 70)
    print("THREE CANDIDATE VALUES FOR β")
    print("=" * 70)
    print()

    candidates = [
        ("β_transcendental (true root)", beta_transcendental),
        ("β_c2_optimal (1/c₂)", beta_c2_optimal),
        ("β_golden (current)", beta_golden),
    ]

    print(f"{'Candidate':<35} {'β value':>12} {'e^β/β':>12} {'K error %':>10} {'c₂ pred':>10} {'c₂ error %':>10}")
    print("-" * 95)

    for name, beta in candidates:
        f_val = f_transcendental(beta)
        k_err = abs(f_val - K) / K * 100
        c2_pred = c2_predicted(beta)
        c2_err = c2_relative_error(beta)
        print(f"{name:<35} {beta:>12.6f} {f_val:>12.6f} {k_err:>10.4f} {c2_pred:>10.6f} {c2_err:>10.4f}")

    print()
    print("=" * 70)
    print("THE TENSION")
    print("=" * 70)
    print()

    delta_beta = beta_c2_optimal - beta_transcendental
    print(f"Gap: β_c2_optimal - β_transcendental = {delta_beta:.6f}")
    print(f"Relative gap: {delta_beta/beta_transcendental * 100:.3f}%")
    print()

    # Check if there's a compromise
    print("=" * 70)
    print("SEARCHING FOR COMPROMISE")
    print("=" * 70)
    print()

    # Define combined objective: minimize both errors
    def combined_objective(beta, w_trans=1.0, w_c2=1.0):
        """Weighted combination of errors."""
        trans_err = (transcendental_error(beta) / K) ** 2
        c2_err = (c2_error(beta) / C2_EMPIRICAL) ** 2
        return w_trans * trans_err + w_c2 * c2_err

    # Find optimal for different weightings
    print(f"{'Weight ratio (trans:c2)':<25} {'Optimal β':>12} {'Trans error %':>15} {'c₂ error %':>12}")
    print("-" * 70)

    for w_trans, w_c2 in [(1, 0), (10, 1), (1, 1), (1, 10), (0, 1)]:
        if w_trans == 0:
            opt_beta = beta_c2_optimal
        elif w_c2 == 0:
            opt_beta = beta_transcendental
        else:
            result = minimize_scalar(
                lambda b: combined_objective(b, w_trans, w_c2),
                bounds=(3.0, 3.1),
                method='bounded'
            )
            opt_beta = result.x

        trans_err = transcendental_error(opt_beta) / K * 100
        c2_err_pct = c2_relative_error(opt_beta)
        print(f"{w_trans}:{w_c2:<24} {opt_beta:>12.6f} {trans_err:>15.4f} {c2_err_pct:>12.4f}")

    print()

    return beta_transcendental, beta_c2_optimal, K

def explore_physics_possibilities():
    """Explore possible physics explanations for the tension."""

    print("=" * 70)
    print("PHYSICS POSSIBILITIES")
    print("=" * 70)
    print()

    K = K_target()
    beta_trans = find_transcendental_root()
    beta_c2 = find_c2_optimal_beta()

    # Possibility 1: Modified transcendental equation
    print("POSSIBILITY 1: Modified Transcendental Equation")
    print("-" * 50)
    print()
    print("What if the equation is NOT e^β/β = K but has corrections?")
    print()

    # What correction factor would we need?
    f_at_c2_beta = f_transcendental(beta_c2)
    correction = f_at_c2_beta / K
    print(f"At β = {beta_c2:.6f}:")
    print(f"  e^β/β = {f_at_c2_beta:.6f}")
    print(f"  K = {K:.6f}")
    print(f"  Ratio = {correction:.6f}")
    print()
    print(f"If the equation were: e^β/β = {correction:.4f} × K")
    print(f"Then β = {beta_c2:.6f} would be the exact root.")
    print()

    # Possibility 2: c₁ uncertainty
    print("POSSIBILITY 2: Surface Coefficient Uncertainty")
    print("-" * 50)
    print()

    # What c₁ would make β_trans give correct c₂?
    # We need 1/β_trans = c₂_empirical
    # So β_trans = 1/c₂_empirical ≈ 3.0578
    # But β_trans is determined by K = (α⁻¹ × c₁) / π²

    # At β_trans, e^β_trans/β_trans = K
    # We want new K' such that solution is 1/c₂
    desired_beta = 1/C2_EMPIRICAL
    K_needed = f_transcendental(desired_beta)
    c1_needed = K_needed * PI_SQ / ALPHA_INV

    print(f"Current c₁ = {C1_SURFACE:.6f}")
    print(f"c₁ needed for consistency = {c1_needed:.6f}")
    print(f"Difference = {c1_needed - C1_SURFACE:.6f} ({(c1_needed/C1_SURFACE - 1)*100:.3f}%)")
    print()

    # What's the uncertainty on c₁?
    print("NuBase 2020 doesn't give explicit uncertainty on c₁,")
    print(f"but a {(c1_needed/C1_SURFACE - 1)*100:.2f}% shift would resolve the tension.")
    print()

    # Possibility 3: c₂ uncertainty
    print("POSSIBILITY 3: Volume Coefficient Uncertainty")
    print("-" * 50)
    print()

    c2_from_trans = 1/beta_trans
    print(f"Current c₂ = {C2_EMPIRICAL:.6f}")
    print(f"c₂ from β_trans = {c2_from_trans:.6f}")
    print(f"Difference = {c2_from_trans - C2_EMPIRICAL:.6f} ({(c2_from_trans/C2_EMPIRICAL - 1)*100:.3f}%)")
    print()

    # Possibility 4: Both coefficients have correlated uncertainties
    print("POSSIBILITY 4: Correlated Systematic Shift")
    print("-" * 50)
    print()
    print("The semi-empirical mass formula fits c₁ and c₂ together.")
    print("If there's a systematic bias, both might shift together.")
    print()

    # What simultaneous shift works?
    # We need: 1. K = (α⁻¹ × c₁') / π² gives β such that
    #          2. 1/β = c₂'
    # And we want c₁'/c₁ ≈ c₂'/c₂ (same relative shift)

    # Let's parameterize: c₁' = c₁(1+δ), c₂' = c₂(1+δ)
    # Then K' = K(1+δ)
    # And β' = 1/(c₂(1+δ))
    # We need: e^β'/β' = K(1+δ)

    def find_consistent_shift(delta):
        """Find δ that makes everything consistent."""
        c1_new = C1_SURFACE * (1 + delta)
        c2_new = C2_EMPIRICAL * (1 + delta)
        K_new = (ALPHA_INV * c1_new) / PI_SQ
        beta_new = 1 / c2_new
        f_new = f_transcendental(beta_new)
        return (f_new - K_new)**2

    from scipy.optimize import minimize_scalar
    result = minimize_scalar(find_consistent_shift, bounds=(-0.1, 0.1), method='bounded')
    delta_optimal = result.x

    c1_consistent = C1_SURFACE * (1 + delta_optimal)
    c2_consistent = C2_EMPIRICAL * (1 + delta_optimal)
    K_consistent = (ALPHA_INV * c1_consistent) / PI_SQ
    beta_consistent = 1 / c2_consistent

    print(f"Optimal correlated shift: δ = {delta_optimal*100:.4f}%")
    print(f"  c₁' = {c1_consistent:.6f} (was {C1_SURFACE:.6f})")
    print(f"  c₂' = {c2_consistent:.6f} (was {C2_EMPIRICAL:.6f})")
    print(f"  β' = {beta_consistent:.6f}")
    print(f"  e^β'/β' = {f_transcendental(beta_consistent):.6f}")
    print(f"  K' = {K_consistent:.6f}")
    print(f"  Check: |e^β'/β' - K'| = {abs(f_transcendental(beta_consistent) - K_consistent):.2e}")
    print()

def plot_tension():
    """Visualize the tension."""

    K = K_target()
    beta_trans = find_transcendental_root()
    beta_c2 = find_c2_optimal_beta()
    beta_golden = 3.058230856

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Transcendental function
    ax1 = axes[0]
    beta_range = np.linspace(2.8, 3.2, 1000)
    f_values = f_transcendental(beta_range)

    ax1.plot(beta_range, f_values, 'b-', linewidth=2, label=r'$f(\beta) = e^\beta/\beta$')
    ax1.axhline(y=K, color='r', linestyle='--', linewidth=1.5, label=f'K = {K:.4f}')

    # Mark key points
    ax1.axvline(x=beta_trans, color='g', linestyle=':', alpha=0.7)
    ax1.axvline(x=beta_c2, color='purple', linestyle=':', alpha=0.7)
    ax1.axvline(x=beta_golden, color='orange', linestyle=':', alpha=0.7)

    ax1.plot(beta_trans, f_transcendental(beta_trans), 'go', markersize=10,
             label=f'β_trans = {beta_trans:.4f}')
    ax1.plot(beta_c2, f_transcendental(beta_c2), 'mo', markersize=10,
             label=f'β_c2 = {beta_c2:.4f}')
    ax1.plot(beta_golden, f_transcendental(beta_golden), 'o', color='orange', markersize=10,
             label=f'β_golden = {beta_golden:.4f}')

    ax1.set_xlabel(r'$\beta$', fontsize=12)
    ax1.set_ylabel(r'$f(\beta)$', fontsize=12)
    ax1.set_title('Transcendental Equation: e^β/β = K', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error trade-off
    ax2 = axes[1]

    trans_errors = [transcendental_error(b) / K * 100 for b in beta_range]
    c2_errors = [c2_relative_error(b) for b in beta_range]

    ax2.plot(beta_range, trans_errors, 'b-', linewidth=2, label='Transcendental error %')
    ax2.plot(beta_range, c2_errors, 'r-', linewidth=2, label='c₂ prediction error %')

    # Mark key points
    ax2.axvline(x=beta_trans, color='g', linestyle=':', alpha=0.7, label=f'β_trans')
    ax2.axvline(x=beta_c2, color='purple', linestyle=':', alpha=0.7, label=f'β_c2')
    ax2.axvline(x=beta_golden, color='orange', linestyle=':', alpha=0.7, label=f'β_golden')

    ax2.set_xlabel(r'$\beta$', fontsize=12)
    ax2.set_ylabel('Error %', fontsize=12)
    ax2.set_title('Error Trade-off', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig('beta_tension_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: beta_tension_analysis.png")
    plt.close()

def main():
    beta_trans, beta_c2, K = analyze_tension()
    explore_physics_possibilities()
    plot_tension()

    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("The tension between β_transcendental and β_c2_optimal is real (~0.5%).")
    print()
    print("Three interpretations:")
    print()
    print("1. MEASUREMENT UNCERTAINTY")
    print("   - c₁ and c₂ from semi-empirical mass formula have ~0.5% uncertainty")
    print("   - A small correlated shift could resolve the tension")
    print("   - This is the 'conservative' interpretation")
    print()
    print("2. MODIFIED TRANSCENDENTAL EQUATION")
    print("   - Perhaps e^β/β = K is only approximate")
    print("   - Higher-order corrections might shift the effective equation")
    print("   - This could indicate new physics")
    print()
    print("3. c₂ = 1/β IS THE PRIMARY CONSTRAINT")
    print("   - If nuclear stability is determined by 1/β directly,")
    print("   - then the transcendental equation is approximate")
    print("   - β = 3.058 is 'correct', K derivation needs refinement")
    print()

if __name__ == "__main__":
    main()
