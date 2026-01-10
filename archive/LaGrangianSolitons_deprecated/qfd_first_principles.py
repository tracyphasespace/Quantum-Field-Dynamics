#!/usr/bin/env python3
"""
QFD MASS FORMULA FROM FIRST PRINCIPLES
===========================================================================
Using the correct derivation chain from QFD Parameter Table.

FUNDAMENTAL CONSTANTS (locked by Golden Loop):
    alpha_fine   = 1/137.036  (fine structure constant)
    beta_vacuum  = 1/3.058    (vacuum stiffness)

DERIVED CONSTANTS:
    M_proton     = 938.272 MeV (mass scale λ)

NUCLEAR PARAMETERS (derived from fundamentals):
    V₀           = M_p × (1 - α²/β)        (well depth)
    β_nuclear    = M_p × β_vacuum / 2      (nuclear stiffness)

MASS FORMULA:
    M(A,Z) = f(V₀, β_nuclear, A, Z)

Testing if this reproduces the 0.1% accuracy of the fitted formula.
===========================================================================
"""

import numpy as np

# ============================================================================
# FUNDAMENTAL CONSTANTS (from Golden Loop)
# ============================================================================

alpha_fine   = 1.0 / 137.036        # Fine structure constant
beta_vacuum  = 1.0 / 3.058          # Vacuum stiffness
M_proton     = 938.272              # Proton mass (MeV) - the λ mass scale

# ============================================================================
# DERIVED NUCLEAR PARAMETERS
# ============================================================================

# Well depth (nuclear potential depth)
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)

# Nuclear stiffness (surface energy parameter)
beta_nuclear = M_proton * beta_vacuum / 2

# Charge asymmetry parameter
q_charge = np.sqrt(alpha_fine / beta_vacuum)

print("="*85)
print("QFD FIRST-PRINCIPLES MASS FORMULA")
print("="*85)
print(f"\nFundamental Constants:")
print(f"  alpha_fine   = 1/{1/alpha_fine:.3f} = {alpha_fine:.6f}")
print(f"  beta_vacuum  = 1/{1/beta_vacuum:.3f} = {beta_vacuum:.6f}")
print(f"  M_proton     = {M_proton} MeV (mass scale λ)")
print()
print(f"Derived Nuclear Parameters:")
print(f"  V₀ (well depth)      = M_p × (1 - α²/β) = {V_0:.3f} MeV")
print(f"  β_nuclear (stiffness) = M_p × β/2       = {beta_nuclear:.3f} MeV")
print(f"  q (charge fraction)   = sqrt(α/β)       = {q_charge:.6f}")
print()

# ============================================================================
# HYPOTHESIS 1: Direct mapping to E = α·A + β·A^(2/3)
# ============================================================================

def formula_hypothesis_1(A, Z):
    """
    Hypothesis 1: Direct parameter mapping
    E_volume  = V₀ (well depth)
    E_surface = β_nuclear (nuclear stiffness)
    """
    E_vol  = V_0
    E_surf = beta_nuclear
    return E_vol * A + E_surf * (A ** (2/3))

# ============================================================================
# HYPOTHESIS 2: Reduced nucleon mass
# ============================================================================

def formula_hypothesis_2(A, Z):
    """
    Hypothesis 2: Nucleon mass reduced by binding
    E_volume  = V₀ × (1 - some correction)
    E_surface = β_nuclear / (geometric factor)

    From fitted values:
    E_volume ≈ 927.6 MeV vs V₀ = 938.1 MeV
    Ratio: 927.6/938.1 = 0.9888

    E_surface ≈ 10.2 MeV vs β_nuclear = 153.4 MeV
    Ratio: 10.2/153.4 = 0.0665 ≈ 1/15
    """
    # Empirical reduction factors from fitted data
    volume_reduction = 927.652 / V_0
    surface_reduction = 10.195 / beta_nuclear

    E_vol  = V_0 * volume_reduction
    E_surf = beta_nuclear * surface_reduction
    return E_vol * A + E_surf * (A ** (2/3))

# ============================================================================
# HYPOTHESIS 3: Geometric surface factor
# ============================================================================

def formula_hypothesis_3(A, Z):
    """
    Hypothesis 3: Surface energy has geometric factor
    E_volume  = V₀ × (1 - ε) where ε ≈ 0.01
    E_surface = β_nuclear × (4π)^(-2/3) × (3/4π)^(2/3)

    The factor (4π)^(-2/3) × (3/4π)^(2/3) converts from volume to surface
    """
    volume_factor = 1 - (V_0 - 927.652) / V_0  # ≈ 0.9888

    # Geometric factor: surface area to volume ratio scaling
    # For sphere: A = 4πr², V = (4/3)πr³
    # A/V^(2/3) ~ (4π) / (4π/3)^(2/3)
    geometric_factor = (4*np.pi) / ((4*np.pi/3) ** (2/3))

    E_vol  = V_0 * volume_factor
    E_surf = beta_nuclear / geometric_factor
    return E_vol * A + E_surf * (A ** (2/3))

# ============================================================================
# HYPOTHESIS 4: Charge-dependent correction
# ============================================================================

def formula_hypothesis_4(A, Z):
    """
    Hypothesis 4: Include charge asymmetry term
    E = V₀·A + β_nuclear·A^(2/3) + symmetry term

    Symmetry: proportional to (N-Z)²/A
    """
    N = A - Z

    E_core = V_0 * A + beta_nuclear * (A ** (2/3))

    # Symmetry energy (charge poor/rich asymmetry)
    # Coefficient derived from q_charge and beta
    a_sym = beta_vacuum * M_proton  # ≈ 306 MeV
    E_sym = a_sym * ((N - Z)**2) / A if A > 0 else 0

    return E_core + E_sym

# ============================================================================
# TEST NUCLEI
# ============================================================================

test_nuclei = [
    ("H-2",   1, 1, 1875.613),
    ("He-3",  2, 1, 2808.391),
    ("He-4",  2, 2, 3727.379),
    ("Li-6",  3, 3, 5601.518),
    ("Li-7",  3, 4, 6533.833),
    ("C-12",  6, 6, 11174.862),
    ("N-14",  7, 7, 13040.700),
    ("O-16",  8, 8, 14895.079),
    ("Ne-20", 10, 10, 18617.708),
    ("Mg-24", 12, 12, 22341.970),
    ("Si-28", 14, 14, 26059.540),
    ("Ca-40", 20, 20, 37211.000),
    ("Fe-56", 26, 30, 52102.500),
]

# ============================================================================
# TESTING
# ============================================================================

formulas = [
    ("Hypothesis 1: Direct V₀, β_nuclear", formula_hypothesis_1),
    ("Hypothesis 2: Scaled to fitted values", formula_hypothesis_2),
    ("Hypothesis 3: Geometric surface factor", formula_hypothesis_3),
    ("Hypothesis 4: With symmetry energy", formula_hypothesis_4),
]

for formula_name, formula_func in formulas:
    print(f"\n{formula_name}")
    print("-"*85)
    print(f"{'Nucleus':<8} {'A':>3} {'Z':>3} {'Exp(MeV)':<11} {'QFD(MeV)':<11} {'Error':>10} {'%':>7}")
    print("-"*85)

    errors = []
    for name, Z, N, m_exp in test_nuclei:
        A = Z + N
        m_qfd = formula_func(A, Z)
        error = m_qfd - m_exp
        error_pct = 100 * error / m_exp

        errors.append(abs(error_pct))

        print(f"{name:<8} {A:>3} {Z:>3} {m_exp:<11.2f} {m_qfd:<11.2f} "
              f"{error:>+9.2f} {error_pct:>+6.3f}%")

    # Statistics
    errors = np.array(errors)
    print("="*85)
    print(f"Mean |error|:   {np.mean(np.abs(errors)):.4f}%")
    print(f"RMS error:      {np.sqrt(np.mean(errors**2)):.4f}%")
    print("="*85)

print("\n" + "="*85)
print("COMPARISON TO FITTED FORMULA")
print("="*85)
print(f"Fitted values (from qfd_topological_mass_formula.py):")
print(f"  E_volume  = 927.652 MeV")
print(f"  E_surface = 10.195 MeV")
print(f"  RMS error: 0.1043%")
print()
print(f"Derived values (from first principles):")
print(f"  V₀           = {V_0:.3f} MeV  (ratio to fitted: {V_0/927.652:.4f})")
print(f"  β_nuclear    = {beta_nuclear:.3f} MeV  (ratio to fitted: {beta_nuclear/10.195:.4f})")
print()
print(f"Question: What is the correct formula relating V₀, β_nuclear to E_volume, E_surface?")
print("="*85)
