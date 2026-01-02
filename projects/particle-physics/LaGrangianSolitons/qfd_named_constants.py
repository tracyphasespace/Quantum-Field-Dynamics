#!/usr/bin/env python3
"""
QFD MASS FORMULA WITH PROPERLY NAMED CONSTANTS
===========================================================================
Clear naming to avoid confusion between fundamental constants and
fitted energy coefficients.

FUNDAMENTAL CONSTANTS (from QFD theory):
    alpha_fine   = 1/137.036  (electromagnetic fine structure constant)
    beta_vacuum  = 1/3.058    (QFD vacuum stiffness parameter)
    lambda_time  = 0.42       (temporal metric scaling parameter)
    M_proton     = 938.272 MeV (proton mass scale)

FITTED ENERGY COEFFICIENTS (from nuclear data):
    E_volume   ≈ 927.6 MeV  (bulk energy per nucleon)
    E_surface  ≈ 10.2 MeV   (surface energy scale)

TARGET FORMULA:
    M(A) = E_volume × A + E_surface × A^(2/3)

QUESTION:
    How do (alpha_fine, beta_vacuum, lambda_time, M_proton)
    combine to give (E_volume, E_surface)?
===========================================================================
"""

import numpy as np

# ============================================================================
# FUNDAMENTAL CONSTANTS (QFD Theory)
# ============================================================================

alpha_fine   = 1.0 / 137.036        # Fine structure constant (dimensionless)
beta_vacuum  = 1.0 / 3.058          # Vacuum stiffness (dimensionless)
lambda_time  = 0.42                 # Temporal metric parameter (dimensionless)
M_proton     = 938.272              # Proton mass in MeV

# ============================================================================
# KNOWN FITTED VALUES (for comparison)
# ============================================================================

E_volume_fitted   = 927.652  # MeV (from qfd_topological_mass_formula.py)
E_surface_fitted  = 10.195   # MeV (from qfd_topological_mass_formula.py)

# ============================================================================
# TEST NUCLEI
# ============================================================================

test_nuclei = [
    ("H-1",   1, 1, 938.272),
    ("He-4",  4, 2, 3727.379),
    ("C-12", 12, 6, 11174.862),
    ("O-16", 16, 8, 14895.079),
    ("Ca-40",40,20, 37211.000),
    ("Fe-56",56,26, 52102.500),
]

# ============================================================================
# CANDIDATE FORMULAS FOR E_volume AND E_surface
# ============================================================================

def formula_A_simple(A, Z):
    """
    Hypothesis: Direct scaling
    E_volume  = M_proton × (1 - lambda_time)
    E_surface = M_proton × beta_vacuum
    """
    E_vol  = M_proton * (1 - lambda_time)
    E_surf = M_proton * beta_vacuum
    return E_vol * A + E_surf * (A ** (2/3))

def formula_B_metric_correction(A, Z):
    """
    Hypothesis: Temporal metric suppression
    E_volume  = M_proton × (1 - lambda_time × beta_vacuum)
    E_surface = M_proton × beta_vacuum
    """
    E_vol  = M_proton * (1 - lambda_time * beta_vacuum)
    E_surf = M_proton * beta_vacuum
    return E_vol * A + E_surf * (A ** (2/3))

def formula_C_exponential_suppression(A, Z):
    """
    Hypothesis: Exponential suppression
    E_volume  = M_proton × exp(-lambda_time)
    E_surface = M_proton × beta_vacuum / (1 + lambda_time)
    """
    E_vol  = M_proton * np.exp(-lambda_time)
    E_surf = M_proton * beta_vacuum / (1 + lambda_time)
    return E_vol * A + E_surf * (A ** (2/3))

def formula_D_fine_structure(A, Z):
    """
    Hypothesis: Fine structure coupling
    E_volume  = M_proton / (1 + alpha_fine × beta_vacuum)
    E_surface = M_proton × alpha_fine / lambda_time
    """
    E_vol  = M_proton / (1 + alpha_fine * beta_vacuum)
    E_surf = M_proton * alpha_fine / lambda_time
    return E_vol * A + E_surf * (A ** (2/3))

def formula_E_inverse_scaling(A, Z):
    """
    Hypothesis: Inverse beta scaling for surface
    E_volume  = M_proton
    E_surface = M_proton × alpha_fine × beta_vacuum
    """
    E_vol  = M_proton
    E_surf = M_proton * alpha_fine * beta_vacuum
    return E_vol * A + E_surf * (A ** (2/3))

def formula_F_square_root(A, Z):
    """
    Hypothesis: Geometric mean scaling
    E_volume  = M_proton × sqrt(1 - lambda_time)
    E_surface = M_proton × sqrt(beta_vacuum) / 10
    """
    E_vol  = M_proton * np.sqrt(1 - lambda_time)
    E_surf = M_proton * np.sqrt(beta_vacuum) / 10
    return E_vol * A + E_surf * (A ** (2/3))

def formula_G_ratio(A, Z):
    """
    Hypothesis: Beta/alpha ratio for surface
    E_volume  = M_proton × (1 - lambda_time)
    E_surface = M_proton × beta_vacuum / (137 * lambda_time)
    """
    E_vol  = M_proton * (1 - lambda_time)
    E_surf = M_proton * beta_vacuum / (137 * lambda_time)
    return E_vol * A + E_surf * (A ** (2/3))

def formula_H_empirical_adjustment(A, Z):
    """
    Hypothesis: Empirical factors to match fitted values
    E_volume  = M_proton × (1 - lambda_time / 10)
    E_surface = M_proton × alpha_fine × 150
    """
    E_vol  = M_proton * (1 - lambda_time / 10)
    E_surf = M_proton * alpha_fine * 150
    return E_vol * A + E_surf * (A ** (2/3))

# The formula that was "almost there" - 6.58% RMS error
# Let's try variations around it
def formula_BEST_from_previous(A, Z):
    """
    Formula 6/7 from previous run (6.58% RMS):
    E = M_proton × A / (1 + lambda_time × beta_vacuum × A^(-1/3))

    This is NOT of the form E = E_vol·A + E_surf·A^(2/3)
    But it was closest. Let's see if we can rewrite it.
    """
    return M_proton * A / (1 + lambda_time * beta_vacuum * (A ** (-1/3)))

def formula_BEST_expanded(A, Z):
    """
    Expand formula_BEST using Taylor series:
    1/(1+x) ≈ 1 - x + x^2 - x^3 + ...

    x = lambda_time × beta_vacuum × A^(-1/3)

    E ≈ M_proton × A × (1 - x)
      = M_proton × A - M_proton × lambda_time × beta_vacuum × A^(2/3)

    Compare to E = E_vol·A + E_surf·A^(2/3):
      E_volume  = M_proton
      E_surface = -M_proton × lambda_time × beta_vacuum
    """
    E_vol  = M_proton
    E_surf = -M_proton * lambda_time * beta_vacuum
    return E_vol * A + E_surf * (A ** (2/3))

def formula_BEST_modified(A, Z):
    """
    Modify BEST to have positive surface term:
    E = M_proton × A / (1 - lambda_time × beta_vacuum × A^(-1/3))

    Note the sign flip: (1 - x) instead of (1 + x)
    """
    return M_proton * A / (1 - lambda_time * beta_vacuum * (A ** (-1/3)))

# ============================================================================
# TESTING
# ============================================================================

formulas = [
    ("A: Direct scaling", formula_A_simple),
    ("B: Metric correction", formula_B_metric_correction),
    ("C: Exponential suppression", formula_C_exponential_suppression),
    ("D: Fine structure coupling", formula_D_fine_structure),
    ("E: Inverse scaling", formula_E_inverse_scaling),
    ("F: Square root", formula_F_square_root),
    ("G: Ratio scaling", formula_G_ratio),
    ("H: Empirical adjustment", formula_H_empirical_adjustment),
    ("BEST (previous 6.58%)", formula_BEST_from_previous),
    ("BEST expanded", formula_BEST_expanded),
    ("BEST modified (sign flip)", formula_BEST_modified),
]

print("="*85)
print("QFD MASS FORMULA - TESTING WITH NAMED CONSTANTS")
print("="*85)
print(f"\nFundamental Constants (QFD Theory):")
print(f"  alpha_fine   = 1/137     = {alpha_fine:.6f}  (EM fine structure)")
print(f"  beta_vacuum  = 1/3.058   = {beta_vacuum:.6f}  (vacuum stiffness)")
print(f"  lambda_time  = {lambda_time}            (temporal metric)")
print(f"  M_proton     = {M_proton} MeV     (proton mass scale)")
print()
print(f"Target Fitted Values (from nuclear data):")
print(f"  E_volume     = {E_volume_fitted:.3f} MeV")
print(f"  E_surface    = {E_surface_fitted:.3f} MeV")
print()

for formula_name, formula_func in formulas:
    print(f"\n{formula_name}")
    print("-"*85)

    # Calculate E_volume and E_surface from this formula (if applicable)
    if "BEST" not in formula_name:
        # For linear formulas, extract coefficients using A=1 and A=8
        E1 = formula_func(1, 1)
        E8 = formula_func(8, 4)
        # E1 = E_vol + E_surf
        # E8 = 8*E_vol + 4*E_surf
        # Solving: E_vol = (E8 - 4*E1) / 4, E_surf = E1 - E_vol
        E_vol_implied = (E8 - 4*E1) / 4
        E_surf_implied = E1 - E_vol_implied
        print(f"Implied E_volume  = {E_vol_implied:>10.3f} MeV  (target: {E_volume_fitted:.3f} MeV)")
        print(f"Implied E_surface = {E_surf_implied:>10.3f} MeV  (target: {E_surface_fitted:.3f} MeV)")
        print()

    print(f"{'Nucleus':<8} {'A':>3} {'Exp(MeV)':>11} {'QFD(MeV)':>11} {'Error':>11} {'%':>9}")
    print("-"*85)

    errors = []
    for name, A, Z, m_exp in test_nuclei:
        m_qfd = formula_func(A, Z)
        error = m_qfd - m_exp
        error_pct = 100 * error / m_exp
        errors.append(abs(error_pct))

        print(f"{name:<8} {A:>3} {m_exp:>11.2f} {m_qfd:>11.2f} "
              f"{error:>+11.2f} {error_pct:>+8.2f}%")

    rms = np.sqrt(np.mean([e**2 for e in errors]))
    print(f"{'':>36} RMS error: {rms:>6.2f}%")

print("\n" + "="*85)
print("SUMMARY")
print("="*85)
print("Looking for formulas where:")
print(f"  E_volume  ≈ {E_volume_fitted:.3f} MeV  (within ±5 MeV)")
print(f"  E_surface ≈ {E_surface_fitted:.3f} MeV  (within ±2 MeV)")
print(f"  RMS error < 1%")
print("="*85)
