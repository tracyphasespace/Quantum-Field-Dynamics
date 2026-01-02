#!/usr/bin/env python3
"""
QFD MASS FORMULA FROM FUNDAMENTAL CONSTANTS
===========================================================================
No free parameters - everything from first principles.

Fundamental Constants:
    α = 1/137     (electromagnetic coupling)
    β = 1/3.058   (vacuum stiffness from QFD)
    λ = 938.272 MeV (proton mass scale)

Question: What is the exact formula combining these?
===========================================================================
"""

import numpy as np

# QFD Fundamental Constants (not fitted!)
ALPHA_EM = 1.0 / 137.036      # Fine structure constant
BETA_QFD = 1.0 / 3.058        # Vacuum stiffness (from He-4 calibration)
LAMBDA_SCALE = 938.272        # Proton mass (MeV)

def mass_formula_attempt_1(A):
    """
    Attempt 1: E = λ × (α × A + β × A^(2/3))

    Dimensional analysis:
        λ has units [Energy]
        α, β dimensionless
        Result has units [Energy] ✓
    """
    return LAMBDA_SCALE * (ALPHA_EM * A + BETA_QFD * (A ** (2/3)))


def mass_formula_attempt_2(A):
    """
    Attempt 2: E = λ × A × (α + β × A^(-1/3))

    Pulls out common factor of A.
    """
    return LAMBDA_SCALE * A * (ALPHA_EM + BETA_QFD * (A ** (-1/3)))


def mass_formula_attempt_3(A):
    """
    Attempt 3: E = λ × A + (λ × β) × A^(2/3)

    Volume term = λ × A
    Surface term = (λ × β) × A^(2/3)
    """
    return LAMBDA_SCALE * A + (LAMBDA_SCALE * BETA_QFD) * (A ** (2/3))


def mass_formula_attempt_4(A):
    """
    Attempt 4: E = λ × A × (1 - α) + (λ × β) × A^(2/3)

    Free nucleon mass reduced by factor (1-α)
    Plus surface term.
    """
    return LAMBDA_SCALE * A * (1 - ALPHA_EM) + (LAMBDA_SCALE * BETA_QFD) * (A ** (2/3))


# Test cases
test_nuclei = [
    ("H-1",   1, 938.272),     # Free proton (should match λ exactly?)
    ("He-4",  4, 3727.379),
    ("C-12", 12, 11174.862),
    ("O-16", 16, 14895.079),
    ("Fe-56",56, 52102.500),
]

print("="*75)
print("QFD MASS FORMULA - TESTING DIFFERENT COMBINATIONS")
print("="*75)
print(f"\nConstants:")
print(f"  α (EM)     = 1/137 = {ALPHA_EM:.6f}")
print(f"  β (QFD)    = 1/3.058 = {BETA_QFD:.6f}")
print(f"  λ (scale)  = {LAMBDA_SCALE} MeV")
print()

formulas = [
    ("E = λ(αA + βA^(2/3))", mass_formula_attempt_1),
    ("E = λA(α + βA^(-1/3))", mass_formula_attempt_2),
    ("E = λA + λβA^(2/3)", mass_formula_attempt_3),
    ("E = λA(1-α) + λβA^(2/3)", mass_formula_attempt_4),
]

for formula_name, formula_func in formulas:
    print(f"\n{formula_name}")
    print("-"*75)
    print(f"{'Nucleus':<8} {'A':>3} {'Exp(MeV)':>10} {'QFD(MeV)':>10} {'Error':>10} {'%':>8}")
    print("-"*75)

    for name, A, m_exp in test_nuclei:
        m_qfd = formula_func(A)
        error = m_qfd - m_exp
        error_pct = 100 * error / m_exp

        print(f"{name:<8} {A:>3} {m_exp:>10.2f} {m_qfd:>10.2f} "
              f"{error:>+10.2f} {error_pct:>+7.2f}%")

print("\n" + "="*75)
print("QUESTION FOR USER:")
print("="*75)
print("Which formula is correct, or is it a different combination?")
print()
print("Need to know:")
print("  1. How do α, β, λ combine in the energy functional?")
print("  2. Is there a (1-α) factor somewhere (nucleon mass reduction)?")
print("  3. Should free proton (A=1) give exactly λ = M_proton?")
print("="*75)
