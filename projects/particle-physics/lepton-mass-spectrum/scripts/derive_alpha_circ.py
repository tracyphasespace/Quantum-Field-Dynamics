#!/usr/bin/env python3
"""
Derive α_circ Circulation Coupling from First Principles

GOAL: Find mechanistic derivation of α_circ = 0.431410 (currently fitted to muon)

HYPOTHESES:

1. Spin Constraint:
   L = ℏ/2 locks circulation → α_circ from angular momentum

2. Fine Structure Connection:
   α_circ ~ f(α, β, π) from fundamental constants

3. Geometric Ratio:
   α_circ ~ 2/π, 1/3, etc. from D-flow geometry

4. Energy Partition:
   α_circ from ratio of circulation energy to total energy

5. Magnetic Moment Ratio:
   α_circ from (μ_circulation / μ_total) scaling
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Constants
ALPHA = 1/137.035999177
HBARC = 197.3269804  # MeV·fm
PI = np.pi

# QFD parameters
BETA = 3.058
XI = 1.0

# Known result
ALPHA_CIRC_FITTED = 0.431410

# Lepton parameters
M_ELECTRON = 0.51099895  # MeV
M_MUON = 105.6583755     # MeV
M_TAU = 1776.86          # MeV

R_ELECTRON = HBARC / M_ELECTRON  # 386.16 fm
R_MUON = HBARC / M_MUON          # 1.87 fm
R_TAU = HBARC / M_TAU            # 0.111 fm

# Experimental g-2
V4_ELECTRON = -0.326
V4_MUON = +0.836


def hill_vortex_velocity_azimuthal(r, theta, R, U=0.5):
    """Azimuthal velocity of Hill vortex."""
    if r < R:
        x = r / R
        v_phi = U * np.sin(theta) * (1.5 * x - 0.5 * x**3)
    else:
        v_phi = U * np.sin(theta) * (R / r)**3 / 2
    return v_phi


def density_gradient(r, R):
    """Hill vortex density gradient."""
    if r < R:
        x = r / R
        drho_dr = -8 * x * (1 - x**2) / R
    else:
        drho_dr = 0.0
    return drho_dr


def calculate_circulation_integral(R, U=0.5):
    """
    Circulation integral (same as derive_v4_circulation.py).

    I_circ = ∫ (v_φ)² · (dρ/dr)² dV / (U² R³)
    """
    def integrand(r, theta):
        v_phi = hill_vortex_velocity_azimuthal(r, theta, R, U)
        drho_dr = density_gradient(r, R)
        return (v_phi)**2 * (drho_dr)**2 * r**2 * np.sin(theta)

    def integrate_theta(r_val):
        result, _ = quad(lambda theta: integrand(r_val, theta), 0, np.pi)
        return result

    I_r, _ = quad(integrate_theta, 0, 10 * R, limit=100)
    I_circ = 2 * np.pi * I_r
    I_circ_normalized = I_circ / (U**2 * R**3)

    return I_circ_normalized


def calculate_angular_momentum(R, U=0.5):
    """
    Calculate total angular momentum of Hill vortex.

    L = ∫ ρ(r) · r² · ω(r) · r² sin(θ) dr dθ dφ

    For Hill vortex, ω(r) ~ v_φ/r
    """
    def integrand(r, theta):
        if r < 1e-10:
            return 0.0

        v_phi = hill_vortex_velocity_azimuthal(r, theta, R, U)

        # Density
        if r < R:
            x = r / R
            rho = 1.0 + 2 * (1 - x**2)**2
        else:
            rho = 1.0

        # Angular momentum density: ρ · r · v_φ · sin(θ)
        L_density = rho * r * v_phi * r**2 * np.sin(theta)

        return L_density

    def integrate_theta(r_val):
        result, _ = quad(lambda theta: integrand(r_val, theta), 0, np.pi)
        return result

    I_r, _ = quad(integrate_theta, 0, 10 * R, limit=100)
    L_total = 2 * np.pi * I_r

    return L_total


def test_hypothesis_1_spin_constraint():
    """
    H1: α_circ from spin constraint L = ℏ/2

    If the vortex has intrinsic spin L = ℏ/2, this constrains U.
    Then α_circ might be determined by the energy partition.
    """
    print("="*80)
    print("HYPOTHESIS 1: Spin Constraint L = ℏ/2")
    print("="*80)
    print()

    # For leptons, spin = 1/2 → L = ℏ/2
    L_target = 1.0  # In natural units ℏ = 1

    print(f"Target: L = ℏ/2 = {L_target/2:.1f} (natural units)")
    print()

    results = {}

    for name, R in [("Electron", R_ELECTRON), ("Muon", R_MUON)]:
        print(f"{name} (R = {R:.2f} fm):")

        # Scan U to find value that gives L = ℏ/2
        U_values = np.linspace(0.01, 0.99, 50)
        L_values = []

        for U in U_values:
            L = calculate_angular_momentum(R, U)
            L_values.append(L)

        # Find U that minimizes |L - L_target/2|
        idx_best = np.argmin(np.abs(np.array(L_values) - L_target/2))
        U_best = U_values[idx_best]
        L_best = L_values[idx_best]

        print(f"  Best U = {U_best:.4f} gives L = {L_best:.4f}")
        print(f"  (Target L = {L_target/2:.1f})")

        # Calculate circulation integral at this U
        I_circ = calculate_circulation_integral(R, U_best)

        print(f"  Circulation integral: I_circ = {I_circ:.6f}")
        print()

        results[name] = {
            'R': R,
            'U_best': U_best,
            'L_best': L_best,
            'I_circ': I_circ
        }

    # Test if this gives α_circ
    I_circ_muon = results['Muon']['I_circ']
    V4_comp = -XI / BETA
    V4_target = V4_MUON

    alpha_circ_from_spin = (V4_target - V4_comp) / I_circ_muon

    print(f"Derived α_circ from spin constraint:")
    print(f"  α_circ = (V₄_muon - V₄_comp) / I_circ")
    print(f"         = ({V4_target:.4f} - {V4_comp:.4f}) / {I_circ_muon:.4f}")
    print(f"         = {alpha_circ_from_spin:.6f}")
    print()
    print(f"Compare to fitted value: {ALPHA_CIRC_FITTED:.6f}")
    print(f"Ratio: {alpha_circ_from_spin / ALPHA_CIRC_FITTED:.4f}")
    print()

    return alpha_circ_from_spin


def test_hypothesis_2_fine_structure():
    """
    H2: α_circ from fine structure constant

    Test ratios like:
      - α_circ ~ α
      - α_circ ~ √α
      - α_circ ~ β · α
      - α_circ ~ (2/π) · α
    """
    print("="*80)
    print("HYPOTHESIS 2: Fine Structure Connection")
    print("="*80)
    print()

    candidates = {
        "α": ALPHA,
        "√α": np.sqrt(ALPHA),
        "α²": ALPHA**2,
        "2α": 2 * ALPHA,
        "α/π": ALPHA / PI,
        "β·α": BETA * ALPHA,
        "β·α/π": BETA * ALPHA / PI,
        "(2/π)·α": (2/PI) * ALPHA,
        "α·(1-2/π)": ALPHA * (1 - 2/PI),
        "1/(3·π)": 1 / (3 * PI),
        "2/(3·π)": 2 / (3 * PI),
        "1/β·√α": (1/BETA) * np.sqrt(ALPHA),
    }

    print(f"Fitted α_circ = {ALPHA_CIRC_FITTED:.6f}")
    print()
    print("Candidate expressions:")
    print(f"{'Expression':<20} | {'Value':<12} | {'Ratio to fitted':<15} | {'Error %':<10}")
    print("-"*70)

    best_match = None
    best_error = float('inf')

    for expr, value in sorted(candidates.items(), key=lambda x: abs(x[1] - ALPHA_CIRC_FITTED)):
        ratio = value / ALPHA_CIRC_FITTED
        error_pct = 100 * abs(value - ALPHA_CIRC_FITTED) / ALPHA_CIRC_FITTED

        print(f"{expr:<20} | {value:<12.6f} | {ratio:<15.4f} | {error_pct:<10.1f}")

        if error_pct < best_error:
            best_error = error_pct
            best_match = (expr, value)

    print()
    print(f"Best match: {best_match[0]} = {best_match[1]:.6f} (error {best_error:.1f}%)")
    print()

    return best_match


def test_hypothesis_3_geometric():
    """
    H3: α_circ from geometric ratios

    Test combinations of π, 2, 3, e, φ (golden ratio), etc.
    """
    print("="*80)
    print("HYPOTHESIS 3: Geometric Ratios")
    print("="*80)
    print()

    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    e = np.e

    candidates = {
        "π/2": PI/2,
        "2/π": 2/PI,
        "1/3": 1/3,
        "1/π": 1/PI,
        "√2/π": np.sqrt(2)/PI,
        "1/(2π)": 1/(2*PI),
        "1 - 2/π": 1 - 2/PI,
        "e/2π": e/(2*PI),
        "φ/π": phi/PI,
        "1/φ²": 1/phi**2,
        "2/(3π)": 2/(3*PI),
        "√(2/π)": np.sqrt(2/PI),
        "(π-e)/π": (PI-e)/PI,
    }

    print(f"Fitted α_circ = {ALPHA_CIRC_FITTED:.6f}")
    print()
    print("Candidate geometric ratios:")
    print(f"{'Expression':<20} | {'Value':<12} | {'Ratio to fitted':<15} | {'Error %':<10}")
    print("-"*70)

    best_match = None
    best_error = float('inf')

    for expr, value in sorted(candidates.items(), key=lambda x: abs(x[1] - ALPHA_CIRC_FITTED)):
        ratio = value / ALPHA_CIRC_FITTED
        error_pct = 100 * abs(value - ALPHA_CIRC_FITTED) / ALPHA_CIRC_FITTED

        print(f"{expr:<20} | {value:<12.6f} | {ratio:<15.4f} | {error_pct:<10.1f}")

        if error_pct < best_error:
            best_error = error_pct
            best_match = (expr, value)

    print()
    print(f"Best match: {best_match[0]} = {best_match[1]:.6f} (error {best_error:.1f}%)")
    print()

    return best_match


def test_hypothesis_4_energy_partition():
    """
    H4: α_circ from energy partition

    E_total = E_gradient + E_compression + E_circulation

    α_circ ~ (E_circulation / E_total)
    """
    print("="*80)
    print("HYPOTHESIS 4: Energy Partition")
    print("="*80)
    print()

    print("Calculating energy components for muon...")
    print()

    R = R_MUON
    U = 0.5

    # Gradient energy
    def E_gradient_integrand(r):
        if r < R:
            x = r / R
            drho_dr = -8 * x * (1 - x**2) / R
            return 0.5 * XI * (drho_dr)**2 * 4 * PI * r**2
        else:
            return 0.0

    E_grad, _ = quad(E_gradient_integrand, 0, 10*R)

    # Compression energy
    def E_compression_integrand(r):
        if r < R:
            x = r / R
            rho = 1.0 + 2 * (1 - x**2)**2
            delta_rho = rho - 1.0
            return BETA * (delta_rho)**2 * 4 * PI * r**2
        else:
            return 0.0

    E_comp, _ = quad(E_compression_integrand, 0, 10*R)

    # Circulation energy (kinetic)
    def E_circ_integrand(r, theta):
        v_phi = hill_vortex_velocity_azimuthal(r, theta, R, U)
        if r < R:
            x = r / R
            rho = 1.0 + 2 * (1 - x**2)**2
        else:
            rho = 1.0
        return 0.5 * rho * (v_phi)**2 * r**2 * np.sin(theta)

    def integrate_theta_E(r_val):
        result, _ = quad(lambda theta: E_circ_integrand(r_val, theta), 0, PI)
        return result

    I_r, _ = quad(integrate_theta_E, 0, 10*R, limit=100)
    E_circ = 2 * PI * I_r

    E_total = E_grad + E_comp + E_circ

    print(f"Energy components (R = {R:.2f} fm, U = {U:.2f}):")
    print(f"  E_gradient:    {E_grad:.6e}")
    print(f"  E_compression: {E_comp:.6e}")
    print(f"  E_circulation: {E_circ:.6e}")
    print(f"  E_total:       {E_total:.6e}")
    print()
    print(f"Fractions:")
    print(f"  E_grad / E_total = {E_grad/E_total:.6f}")
    print(f"  E_comp / E_total = {E_comp/E_total:.6f}")
    print(f"  E_circ / E_total = {E_circ/E_total:.6f}")
    print()

    alpha_circ_from_energy = E_circ / E_total

    print(f"Hypothesis: α_circ ~ E_circ / E_total = {alpha_circ_from_energy:.6f}")
    print(f"Fitted value: {ALPHA_CIRC_FITTED:.6f}")
    print(f"Ratio: {alpha_circ_from_energy / ALPHA_CIRC_FITTED:.4f}")
    print()

    return alpha_circ_from_energy


def test_hypothesis_5_v4_scaling():
    """
    H5: α_circ from V₄ scaling law

    Since V₄_electron = -ξ/β and V₄_muon requires circulation,
    maybe α_circ ~ |V₄_muon / V₄_electron| - 1
    """
    print("="*80)
    print("HYPOTHESIS 5: V₄ Scaling Relation")
    print("="*80)
    print()

    V4_comp = -XI / BETA

    print(f"V₄_compression = -ξ/β = {V4_comp:.4f}")
    print(f"V₄_electron (exp) = {V4_ELECTRON:.4f}")
    print(f"V₄_muon (exp) = {V4_MUON:.4f}")
    print()

    # Test various scaling relations
    candidates = {
        "|V₄_μ/V₄_e| - 1": abs(V4_MUON / V4_ELECTRON) - 1,
        "V₄_μ / |V₄_comp|": V4_MUON / abs(V4_comp),
        "(V₄_μ - V₄_comp) / |V₄_comp|": (V4_MUON - V4_comp) / abs(V4_comp),
        "√(|V₄_μ · V₄_e|)": np.sqrt(abs(V4_MUON * V4_ELECTRON)),
        "1 - V₄_e/V₄_μ": 1 - V4_ELECTRON/V4_MUON,
    }

    print("Candidate scaling relations:")
    print(f"{'Expression':<30} | {'Value':<12} | {'Error %':<10}")
    print("-"*60)

    for expr, value in sorted(candidates.items(), key=lambda x: abs(x[1] - ALPHA_CIRC_FITTED)):
        error_pct = 100 * abs(value - ALPHA_CIRC_FITTED) / ALPHA_CIRC_FITTED
        print(f"{expr:<30} | {value:<12.6f} | {error_pct:<10.1f}")

    print()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DERIVING α_circ FROM FIRST PRINCIPLES")
    print("="*80)
    print()
    print(f"Target: α_circ = {ALPHA_CIRC_FITTED:.6f} (fitted to muon g-2)")
    print()

    # Test all hypotheses
    alpha_H1 = test_hypothesis_1_spin_constraint()
    match_H2 = test_hypothesis_2_fine_structure()
    match_H3 = test_hypothesis_3_geometric()
    alpha_H4 = test_hypothesis_4_energy_partition()
    test_hypothesis_5_v4_scaling()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("Derived values:")
    print(f"  H1 (Spin L=ℏ/2):     α_circ = {alpha_H1:.6f} (depends on U calibration)")
    print(f"  H2 (Fine structure): Best = {match_H2[0]} = {match_H2[1]:.6f}")
    print(f"  H3 (Geometric):      Best = {match_H3[0]} = {match_H3[1]:.6f}")
    print(f"  H4 (Energy):         α_circ = {alpha_H4:.6f}")
    print()
    print(f"Fitted value: α_circ = {ALPHA_CIRC_FITTED:.6f}")
    print()
    print("CONCLUSION:")
    print()

    # Check which hypothesis is closest
    errors = {
        "H1": abs(alpha_H1 - ALPHA_CIRC_FITTED),
        "H2": abs(match_H2[1] - ALPHA_CIRC_FITTED),
        "H3": abs(match_H3[1] - ALPHA_CIRC_FITTED),
        "H4": abs(alpha_H4 - ALPHA_CIRC_FITTED),
    }

    best_H = min(errors.items(), key=lambda x: x[1])

    print(f"Best match: {best_H[0]} with error {100*best_H[1]/ALPHA_CIRC_FITTED:.1f}%")
    print()

    if best_H[1] / ALPHA_CIRC_FITTED < 0.1:  # < 10% error
        print("✓ Found mechanistic derivation of α_circ!")
        print()
        if best_H[0] == "H2":
            print(f"  Formula: α_circ = {match_H2[0]}")
        elif best_H[0] == "H3":
            print(f"  Formula: α_circ = {match_H3[0]}")
        elif best_H[0] == "H4":
            print(f"  Formula: α_circ = E_circulation / E_total")
    else:
        print("✗ No simple mechanistic derivation found.")
        print("  α_circ likely involves combination of effects.")
        print()
        print("Recommendation: Accept α_circ as emergent parameter,")
        print("calibrated from muon g-2 measurement.")

    print()
    print("="*80)
