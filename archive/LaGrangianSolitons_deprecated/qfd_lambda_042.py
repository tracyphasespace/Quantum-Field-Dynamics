#!/usr/bin/env python3
"""
QFD MASS FORMULA WITH λ = 0.42
===========================================================================
Testing different combinations with the correct λ value.

Constants:
    α = 1/137      (electromagnetic coupling)
    β = 1/3.043233053    (vacuum stiffness)
    λ = 0.42       (temporal metric parameter)
    M_p = 938.272 MeV (proton mass scale)
===========================================================================
"""

import numpy as np

# QFD Fundamental Constants
ALPHA_EM = 1.0 / 137.036
BETA_QFD = 1.0 / 3.043233053
LAMBDA_METRIC = 0.42
M_PROTON = 938.272  # MeV

def formula_1(A, Z):
    """E = M_p × A × (1 - λ×ρ) where ρ ~ something"""
    # Trying: ρ ~ A^(-1/3) (density decreases with size)
    rho = A ** (-1/3)
    return M_PROTON * A * (1 - LAMBDA_METRIC * rho)

def formula_2(A, Z):
    """E = M_p × A × (1 - λ×β)"""
    return M_PROTON * A * (1 - LAMBDA_METRIC * BETA_QFD)

def formula_3(A, Z):
    """E = M_p × [A + λ×β×A^(2/3)]"""
    return M_PROTON * (A + LAMBDA_METRIC * BETA_QFD * (A ** (2/3)))

def formula_4(A, Z):
    """E = M_p × A - M_p × λ × β × A^(2/3)"""
    return M_PROTON * A - M_PROTON * LAMBDA_METRIC * BETA_QFD * (A ** (2/3))

def formula_5(A, Z):
    """E = M_p × A × (1 - λ×β×A^(-1/3))"""
    return M_PROTON * A * (1 - LAMBDA_METRIC * BETA_QFD * (A ** (-1/3)))

def formula_6(A, Z):
    """E = M_p × A / (1 + λ×β×A^(-1/3))"""
    return M_PROTON * A / (1 + LAMBDA_METRIC * BETA_QFD * (A ** (-1/3)))

def formula_7(A, Z):
    """E = M_p × A / (1 + λ×ρ) with ρ = β×A^(-1/3)"""
    rho = BETA_QFD * (A ** (-1/3))
    return M_PROTON * A / (1 + LAMBDA_METRIC * rho)

def formula_8(A, Z):
    """E = M_p × A × (1 - λ) + M_p × β × A^(2/3)"""
    return M_PROTON * A * (1 - LAMBDA_METRIC) + M_PROTON * BETA_QFD * (A ** (2/3))

def formula_9(A, Z):
    """E = M_p × [(1-λ)×A + β×A^(2/3)]"""
    return M_PROTON * ((1 - LAMBDA_METRIC) * A + BETA_QFD * (A ** (2/3)))

def formula_10(A, Z):
    """E = M_p × A - M_p × λ × A + M_p × β × A^(2/3)
       = M_p × [A(1-λ) + β×A^(2/3)]
    """
    return M_PROTON * (A * (1 - LAMBDA_METRIC) + BETA_QFD * (A ** (2/3)))

# Test cases
test_nuclei = [
    ("H-1",   1, 1, 938.272),
    ("He-4",  4, 2, 3727.379),
    ("C-12", 12, 6, 11174.862),
    ("O-16", 16, 8, 14895.079),
    ("Ca-40",40,20, 37211.000),
    ("Fe-56",56,26, 52102.500),
]

formulas = [
    ("1: M_p·A(1-λ·A^(-1/3))", formula_1),
    ("2: M_p·A(1-λβ)", formula_2),
    ("3: M_p[A + λβA^(2/3)]", formula_3),
    ("4: M_p·A - M_p·λβA^(2/3)", formula_4),
    ("5: M_p·A(1-λβA^(-1/3))", formula_5),
    ("6: M_p·A/(1+λβA^(-1/3))", formula_6),
    ("7: M_p·A/(1+λρ), ρ=βA^(-1/3)", formula_7),
    ("8: M_p[A(1-λ) + βA^(2/3)]", formula_8),
    ("9: M_p[(1-λ)A + βA^(2/3)]", formula_9),
    ("10: M_p[A(1-λ) + βA^(2/3)]", formula_10),
]

print("="*85)
print("QFD MASS FORMULA - TESTING WITH λ = 0.42")
print("="*85)
print(f"\nConstants:")
print(f"  α (EM)     = 1/137 = {ALPHA_EM:.6f}")
print(f"  β (QFD)    = 1/3.043233053 = {BETA_QFD:.6f}")
print(f"  λ (metric) = {LAMBDA_METRIC}")
print(f"  M_proton   = {M_PROTON} MeV")
print()

for formula_name, formula_func in formulas:
    print(f"\n{formula_name}")
    print("-"*85)
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
