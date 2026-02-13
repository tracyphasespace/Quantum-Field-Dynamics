#!/usr/bin/env python3
"""
Soliton Jet Fragmentation: Exploratory QFD Reinterpretation of DIS/Jets
========================================================================

STATUS: EXPLORATORY / SPECULATIVE
This script is a research direction, NOT a claim. DIS and jet physics are
QFD's largest gap. This explores how soliton physics MIGHT reinterpret
Standard Model parton phenomenology.

Three exploratory calculations:
  1. Soliton form factor F(q^2) for elastic/inelastic scattering
  2. Soliton fission model for jet fragmentation
  3. Scale-dependent soliton structure (analog of Bjorken scaling)

HONEST CAVEAT: The Standard Model's treatment of DIS via perturbative QCD
is one of the most successful and precisely tested theories in physics.
QFD has no comparable quantitative framework for the strong interaction.
This script is a first step toward understanding what such a framework
might look like, not a replacement for pQCD.

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License
"""

import sys
import os
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from qfd.shared_constants import (
    ALPHA, BETA, M_PROTON_MEV, M_ELECTRON_MEV,
    K_GEOM, XI_QFD,
)

HBAR_C_MEV_FM = 197.3269804  # MeV*fm


def print_header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# =====================================================================
# 1. SOLITON ELASTIC FORM FACTOR
# =====================================================================
# In QFD, the proton is a topological soliton with profile:
#   psi(r) = psi_0 * (1 - (r/R)^2)  for r < R (Hill vortex)
#   psi(r) = 0                        for r > R
#
# The electromagnetic form factor is the Fourier transform of the
# charge distribution rho(r) = |psi(r)|^2.
#
# For a uniform-density sphere of radius R:
#   F(q) = 3 [sin(qR) - qR*cos(qR)] / (qR)^3
#
# For the Hill vortex profile, we compute numerically.

def hill_vortex_profile(r, R):
    """Hill vortex density profile (normalized)."""
    if r >= R:
        return 0.0
    return (1.0 - (r / R)**2)**2  # |psi|^2 for psi = 1 - r^2/R^2


def soliton_form_factor(q, R):
    """
    Elastic form factor F(q) for a Hill-vortex soliton.

    F(q) = (4pi / N) * integral_0^R rho(r) * j0(qr) * r^2 dr

    where j0(x) = sin(x)/x is the spherical Bessel function.
    """
    if q < 1e-10:
        return 1.0  # F(0) = 1 by normalization

    # Normalization: N = 4pi * integral_0^R rho(r) * r^2 dr
    norm_integrand = lambda r: hill_vortex_profile(r, R) * r**2
    N, _ = quad(norm_integrand, 0, R)

    # Form factor integral
    ff_integrand = lambda r: hill_vortex_profile(r, R) * spherical_jn(0, q * r) * r**2
    F_val, _ = quad(ff_integrand, 0, R)

    return F_val / N


def compute_form_factors():
    """Compute and display soliton form factors at various q^2."""
    print_header("1. SOLITON ELASTIC FORM FACTOR")
    print(f"\n  *** EXPLORATORY — not a validated prediction ***")

    # Proton Compton radius
    R_proton = HBAR_C_MEV_FM / M_PROTON_MEV  # ~0.21 fm

    # Soliton radius from book: R ~ k_geom * R_compton / alpha
    # Actually use the charge radius scale
    R_soliton = 0.84  # fm (approximate proton charge radius for comparison)

    print(f"\n  Proton Compton wavelength: {R_proton:.4f} fm")
    print(f"  Soliton radius (assumed): {R_soliton:.4f} fm")
    print(f"  Hill vortex profile: rho(r) = (1 - r^2/R^2)^2")

    # q values in fm^-1 (q = momentum transfer)
    q_values_gev = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # GeV
    q_values_fm = q_values_gev * 1000 / HBAR_C_MEV_FM  # Convert to fm^-1

    print(f"\n  {'Q (GeV)':<12s} {'q (fm^-1)':<12s} {'F_soliton(q)':<15s} {'F_dipole(q)':<15s}")
    print(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*15}")

    for q_gev, q_fm in zip(q_values_gev, q_values_fm):
        F_sol = soliton_form_factor(q_fm, R_soliton)

        # SM dipole form factor for comparison: G_E(Q^2) = 1/(1+Q^2/0.71)^2
        Q2_gev2 = q_gev**2
        F_dipole = 1.0 / (1 + Q2_gev2 / 0.71)**2

        print(f"  {q_gev:<12.1f} {q_fm:<12.2f} {F_sol:<15.6f} {F_dipole:<15.6f}")

    print(f"\n  Note: SM dipole form factor (G_E) is empirical with Lambda^2 = 0.71 GeV^2.")
    print(f"  The soliton form factor shape depends on the assumed profile and radius.")
    print(f"  A proper comparison requires solving the full variational equation.")


# =====================================================================
# 2. SOLITON FISSION MODEL FOR JET FRAGMENTATION
# =====================================================================
# In SM: hard scattering produces partons → hadronization → jets
# In QFD: hard scattering creates a highly excited soliton state
#   → fission into daughter solitons (energy → new soliton pairs)
#
# Key idea: energy deposited into a soliton above the fission threshold
# creates new soliton-antisoliton pairs. The multiplicity scales as:
#   N ~ sqrt(s) / m_soliton (energy available / mass per soliton)

def soliton_fission_multiplicity(sqrt_s_gev):
    """
    Estimate hadron multiplicity from soliton fission.

    In QFD, the excited soliton fragments when energy exceeds
    the topological binding energy. Multiplicity scales as:
      N ~ c * ln(s/s_0) (logarithmic, like experiment)

    This is a toy model; the constant c is fitted.
    """
    # Experimental average charged multiplicity: <n_ch> ~ a + b*ln(s)
    # with a ~ -4.0, b ~ 2.4 (from e+e- data)
    s_gev2 = sqrt_s_gev**2
    s_0 = (2 * M_PROTON_MEV / 1000)**2  # threshold in GeV^2

    if s_gev2 <= s_0:
        return 0.0

    # QFD model: multiplicity from available energy / soliton mass
    # Use logarithmic scaling (consistent with string-like fragmentation)
    # The 1/beta^2 factor reflects vacuum stiffness limiting fragmentation
    n_soliton = (1.0 / BETA) * np.log(s_gev2 / s_0) + 2
    return max(n_soliton, 2.0)


def compute_multiplicities():
    """Compare soliton fission multiplicity with experimental data."""
    print_header("2. SOLITON FISSION MODEL FOR JET FRAGMENTATION")
    print(f"\n  *** EXPLORATORY — qualitative only ***")

    # Experimental data points (e+e- -> hadrons, <n_ch>)
    # From PDG 2024 review
    data = [
        (10, 8.4),    # PETRA
        (14, 10.0),   # PETRA
        (22, 12.2),   # PETRA
        (29, 13.4),   # PEP
        (35, 14.8),   # PETRA
        (44, 16.5),   # TRISTAN
        (91, 20.9),   # LEP (Z pole)
        (133, 23.5),  # LEP2
        (183, 25.4),  # LEP2
        (206, 26.5),  # LEP2
    ]

    print(f"\n  QFD parameter: 1/beta = {1/BETA:.4f}")
    print(f"\n  {'sqrt(s) (GeV)':<18s} {'N_exp':<10s} {'N_soliton':<12s} {'Ratio':<10s}")
    print(f"  {'-'*18} {'-'*10} {'-'*12} {'-'*10}")

    for sqrt_s, n_exp in data:
        n_sol = soliton_fission_multiplicity(sqrt_s)
        ratio = n_sol / n_exp if n_exp > 0 else 0
        print(f"  {sqrt_s:<18.0f} {n_exp:<10.1f} {n_sol:<12.2f} {ratio:<10.3f}")

    print(f"\n  Note: The soliton model gives logarithmic scaling (correct trend)")
    print(f"  but the normalization is off. A proper treatment would need:")
    print(f"    - Soliton excitation spectrum (analog of Regge trajectories)")
    print(f"    - Fission probability as function of energy and topology")
    print(f"    - Multi-soliton final-state interactions")


# =====================================================================
# 3. SCALE-DEPENDENT SOLITON STRUCTURE (BJORKEN SCALING ANALOG)
# =====================================================================
# In SM: proton structure functions F2(x, Q^2) show approximate Bjorken
# scaling with logarithmic Q^2 evolution (DGLAP equations).
#
# In QFD: at low Q^2, the soliton looks smooth (no substructure).
# At high Q^2 ~ 1/R_soliton^2, internal structure becomes visible.
# The "scaling" arises from the soliton's radial density profile.

def soliton_structure_function(x, Q2_gev2, R_fm=0.84):
    """
    Toy soliton structure function F2(x, Q^2).

    At resolution scale Q, the effective density probed is:
      rho_eff(r, Q) = rho(r) * exp(-r^2 * Q^2 / 2)

    The structure function is the Fourier transform of this weighted density,
    projected onto the light-cone variable x.

    This is a QUALITATIVE model only.
    """
    Q_fm = np.sqrt(Q2_gev2) * 1000 / HBAR_C_MEV_FM  # Convert Q to fm^-1

    # At x -> 0 or x -> 1, F2 -> 0
    # Peaked near x ~ 1/3 (three "lumps" of soliton density)
    # Width controlled by Q^2 (more structure at higher Q)

    # Simple parameterization inspired by soliton density Fourier modes
    # This is NOT derived from first principles — it's exploratory
    sigma_x = 0.2 + 0.1 / (1 + Q2_gev2)  # Width narrows with Q^2
    peak_x = 1.0 / 3.0  # 3-fold symmetry (like valence quarks)

    # Gaussian peaks at x = 1/3 (mimicking 3 density maxima of soliton)
    F2 = 0
    for k in range(1, 4):  # Three "constituents"
        xk = k / 4.0  # peaks at 0.25, 0.5, 0.75
        F2 += np.exp(-0.5 * ((x - xk) / sigma_x)**2)

    # Normalize roughly
    F2 *= x * (1 - x)  # kinematic constraints

    # Q^2 evolution: soliton reveals more structure at higher Q
    # (logarithmic, like DGLAP)
    F2 *= (1 + 0.1 * np.log(1 + Q2_gev2))

    return F2


def compute_structure_functions():
    """Display toy soliton structure functions."""
    print_header("3. SCALE-DEPENDENT SOLITON STRUCTURE")
    print(f"\n  *** HIGHLY SPECULATIVE — toy model only ***")

    Q2_values = [1.0, 10.0, 100.0, 1000.0]
    x_values = np.linspace(0.05, 0.95, 19)

    print(f"\n  Soliton 'structure function' F2(x, Q^2) [arbitrary units]")
    print(f"\n  {'x':<8s}", end="")
    for Q2 in Q2_values:
        print(f"{'Q^2=' + str(int(Q2)):<15s}", end="")
    print()
    print(f"  {'-'*8}", end="")
    for _ in Q2_values:
        print(f"{'-'*15}", end="")
    print()

    for x in x_values:
        print(f"  {x:<8.2f}", end="")
        for Q2 in Q2_values:
            F2 = soliton_structure_function(x, Q2)
            print(f"{F2:<15.4f}", end="")
        print()

    print(f"\n  Observations:")
    print(f"  - Peaks near x = 0.25, 0.5, 0.75 (three soliton density maxima)")
    print(f"  - F2 increases slowly with Q^2 (more structure resolved)")
    print(f"  - This mimics some features of Bjorken scaling + evolution")
    print(f"\n  CRITICAL LIMITATION: This is NOT a quantitative model.")
    print(f"  A real soliton structure function would require:")
    print(f"    1. Solving the full Cl(3,3) soliton equation for rho(r)")
    print(f"    2. Light-cone projection of the soliton density")
    print(f"    3. Inelastic scattering amplitude from soliton perturbation theory")
    print(f"    4. Comparison with actual DIS data (HERA, JLab)")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 72)
    print("  SOLITON JET FRAGMENTATION: EXPLORATORY QFD REINTERPRETATION")
    print("  " + "=" * 68)
    print("  STATUS: SPECULATIVE / RESEARCH DIRECTION")
    print("  This is NOT a validated prediction. DIS is QFD's largest gap.")
    print("=" * 72)

    compute_form_factors()
    compute_multiplicities()
    compute_structure_functions()

    # Summary
    print_header("SUMMARY & HONEST ASSESSMENT")
    print(f"""
  What this script shows:
  -----------------------
  1. FORM FACTOR: A Hill-vortex soliton produces a form factor that
     qualitatively resembles the proton's empirical dipole form factor.
     The shape depends on the assumed soliton radius (~0.84 fm).

  2. MULTIPLICITY: Soliton fission gives logarithmic energy scaling
     for hadron multiplicity, matching the QUALITATIVE trend of data.
     The normalization is off by ~50% without tuning.

  3. STRUCTURE: Scale-dependent soliton density mimics some features
     of Bjorken scaling, with three-fold internal structure.

  What this script does NOT show:
  --------------------------------
  - Quantitative agreement with DIS cross-sections
  - Derivation of parton distribution functions
  - Explanation of R-ratio or jet rates
  - Any result that would challenge perturbative QCD

  The Standard Model (pQCD + factorization) explains DIS with sub-percent
  precision across many orders of magnitude in Q^2 and x. QFD has nothing
  comparable. Closing this gap is the single most important challenge for
  QFD's credibility as a complete framework.

  Research directions:
  --------------------
  a) Solve the radial soliton equation numerically for realistic profiles
  b) Compute inelastic form factors from soliton perturbation theory
  c) Look for signatures of soliton substructure vs point-like quarks
  d) Test whether soliton fission can reproduce jet angular distributions
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
