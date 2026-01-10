#!/usr/bin/env python3
"""
Verify Lepton g-2: Parameter-Free Geometric Derivation
=======================================================

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License

PURPOSE:
--------
This script demonstrates that lepton anomalous magnetic moments (g-2) are
DERIVED from geometry with ZERO free parameters. All constants come from:
- alpha (fine structure constant)
- phi (golden ratio)
- lepton masses (Compton wavelengths)

THE MASTER EQUATION:
--------------------
    V4(R) = [(R_vac - R) / (R_vac + R)] * (xi / beta)

Where:
    R_vac = lambda_e / sqrt(5)     (vacuum correlation length)
    xi = phi^2 = phi + 1           (geometric coupling)
    beta from Golden Loop:         e^beta / beta = (alpha^-1 - 1) / (2*pi^2)
    R = hbar*c / m                 (Compton wavelength from mass)

SIGN FLIP MECHANISM:
--------------------
The Mobius transform (R_vac - R)/(R_vac + R) naturally produces:
- Electron: R_e > R_vac => negative V4 correction
- Muon:     R_mu < R_vac => positive V4 correction

This sign flip is a GEOMETRIC NECESSITY, not a fitting artifact.
It was proven formally in: QFD/Lepton/GeometricG2.lean (11 theorems)

ACCURACY:
---------
- Electron g-2 error: 0.0013%
- Muon g-2 error:     0.0027%

References:
    - projects/Lean4/QFD/Lepton/GeometricG2.lean (formal proof)
    - Schwinger (1948): a = alpha/(2*pi)
    - PDG 2024, Fermilab 2021 (experimental values)
"""

import numpy as np
import sys


def solve_beta_golden_loop(alpha_target: float, max_iter: int = 20) -> float:
    """
    Solve the Golden Loop equation for beta.

    Golden Loop: 1/alpha = 2*pi^2 * (e^beta / beta) + 1

    Rearranging: e^beta / beta = (1/alpha - 1) / (2*pi^2) = K

    Uses Newton-Raphson iteration to solve f(beta) = e^beta/beta - K = 0.

    Args:
        alpha_target: Fine structure constant (~ 1/137.036)
        max_iter: Maximum Newton iterations

    Returns:
        beta: Vacuum stiffness parameter (~ 3.043)
    """
    K = (1.0 / alpha_target - 1.0) / (2 * np.pi**2)

    beta = 3.0  # Initial guess (close to solution)
    for _ in range(max_iter):
        f = np.exp(beta) / beta - K
        df = np.exp(beta) * (beta - 1) / (beta**2)
        if abs(df) < 1e-15:
            break
        beta -= f / df
        if abs(f) < 1e-14:
            break

    return beta


def predict_g2_anomaly(R_lepton: float, R_vac: float, xi: float, beta: float,
                       alpha: float) -> dict:
    """
    Predict lepton g-2 anomaly from geometric parameters.

    The prediction uses:
    1. Leading Schwinger term: a_schwinger = alpha / (2*pi)
    2. Scale factor (Mobius transform): S = (R_vac - R) / (R_vac + R)
    3. V4 coefficient: V4 = S * (xi / beta)
    4. Second-order term: term_2 = V4 * (alpha/pi)^2

    Args:
        R_lepton: Compton wavelength of lepton (in units of R_e)
        R_vac: Vacuum correlation length (in units of R_e)
        xi: Geometric coupling (phi^2)
        beta: Vacuum stiffness from Golden Loop
        alpha: Fine structure constant

    Returns:
        Dictionary with prediction details
    """
    # Leading QED term (Schwinger)
    a_schwinger = alpha / (2 * np.pi)

    # Scale factor from Mobius transform
    S = (R_vac - R_lepton) / (R_vac + R_lepton)

    # V4 coefficient from geometry
    V4 = S * (xi / beta)

    # Second-order term
    alpha_over_pi_sq = (alpha / np.pi)**2
    term_2 = V4 * alpha_over_pi_sq

    # Total anomaly
    a_total = a_schwinger + term_2

    return {
        'a_schwinger': a_schwinger,
        'scale_factor': S,
        'V4': V4,
        'term_2': term_2,
        'a_total': a_total
    }


def run_validation():
    """
    Run complete g-2 validation with parameter-free derivation.
    """
    print("=" * 70)
    print("QFD LEPTON g-2: PARAMETER-FREE GEOMETRIC DERIVATION")
    print("=" * 70)

    # =========================================================================
    # 1. INPUT CONSTANTS (NATURE'S VALUES ONLY)
    # =========================================================================
    print("\n[1] INPUT CONSTANTS (from Nature)")

    ALPHA = 1.0 / 137.035999206  # Fine structure constant (CODATA 2018)
    PHI = (1 + np.sqrt(5)) / 2   # Golden ratio

    # Lepton masses (MeV/c^2) - PDG 2024
    MASS_E = 0.51099895000
    MASS_MU = 105.6583755
    MASS_TAU = 1776.86

    print(f"    alpha = 1/{1/ALPHA:.9f}")
    print(f"    phi   = {PHI:.9f}")
    print(f"    m_e   = {MASS_E:.8f} MeV")
    print(f"    m_mu  = {MASS_MU:.7f} MeV")
    print(f"    m_tau = {MASS_TAU:.2f} MeV")

    # =========================================================================
    # 2. DERIVE ALL PARAMETERS (ZERO FREE PARAMETERS)
    # =========================================================================
    print("\n[2] DERIVED PARAMETERS (Zero Free Parameters)")

    # Beta from Golden Loop
    BETA = solve_beta_golden_loop(ALPHA)

    # Xi from golden ratio
    XI = PHI**2  # = phi + 1

    # Vacuum correlation length
    R_VAC = 1.0 / np.sqrt(5)  # In units of electron Compton wavelength

    # Compton wavelengths (R = hbar*c/m, so R_mu/R_e = m_e/m_mu)
    R_E = 1.0  # Reference scale
    R_MU = MASS_E / MASS_MU
    R_TAU = MASS_E / MASS_TAU

    print(f"    beta (Golden Loop)  = {BETA:.9f}")
    print(f"    xi = phi^2          = {XI:.9f}")
    print(f"    xi/beta (amplitude) = {XI/BETA:.9f}")
    print(f"    R_vac/R_e = 1/sqrt5 = {R_VAC:.9f}")
    print(f"    R_e                 = {R_E:.3f} (reference)")
    print(f"    R_mu/R_e            = {R_MU:.9f}")
    print(f"    R_tau/R_e           = {R_TAU:.9f}")

    # Verify Golden Loop closure
    golden_loop_lhs = 1.0 / ALPHA
    golden_loop_rhs = 2 * np.pi**2 * np.exp(BETA) / BETA + 1
    golden_loop_error = abs(golden_loop_lhs - golden_loop_rhs) / golden_loop_lhs * 100
    print(f"\n    Golden Loop verification:")
    print(f"      1/alpha = {golden_loop_lhs:.9f}")
    print(f"      2pi^2 * e^beta/beta + 1 = {golden_loop_rhs:.9f}")
    print(f"      Closure error: {golden_loop_error:.2e}%")

    # Verify xi = phi + 1
    xi_check = PHI + 1
    xi_error = abs(XI - xi_check) / XI * 100
    print(f"\n    Golden ratio identity: phi^2 = phi + 1")
    print(f"      phi^2   = {XI:.9f}")
    print(f"      phi + 1 = {xi_check:.9f}")
    print(f"      Error: {xi_error:.2e}%")

    # =========================================================================
    # 3. EXPERIMENTAL VALUES
    # =========================================================================
    print("\n[3] EXPERIMENTAL g-2 VALUES")
    print("    Sources: Harvard 2008 (electron), Fermilab Final June 2025 (muon)")

    A_E_EXP = 0.00115965218128   # Electron (Harvard 2008, Fan et al.)
    A_MU_EXP = 0.001165920705    # Muon (Fermilab FINAL June 2025, arXiv:2506.01689)

    print(f"    a_e (experiment)  = {A_E_EXP:.14f}")
    print(f"    a_mu (experiment) = {A_MU_EXP:.14f}")

    # =========================================================================
    # 4. PREDICTIONS
    # =========================================================================
    print("\n[4] PARAMETER-FREE PREDICTIONS")
    print("=" * 70)

    results = {}
    leptons = [
        ('Electron', R_E, A_E_EXP),
        ('Muon', R_MU, A_MU_EXP),
        ('Tau', R_TAU, None)  # No precise experimental value
    ]

    for name, R, a_exp in leptons:
        pred = predict_g2_anomaly(R, R_VAC, XI, BETA, ALPHA)
        results[name] = pred

        print(f"\n--- {name.upper()} (R/R_e = {R:.6f}) ---")
        print(f"    Scale factor S = (R_vac - R)/(R_vac + R) = {pred['scale_factor']:+.6f}")

        if pred['scale_factor'] < 0:
            print(f"      => R > R_vac: Vacuum 'compresses' => NEGATIVE correction")
        else:
            print(f"      => R < R_vac: Vacuum 'inflates' => POSITIVE correction")

        print(f"    V4 coefficient = S * (xi/beta)           = {pred['V4']:+.6f}")
        print(f"    a_schwinger    = alpha/(2*pi)            = {pred['a_schwinger']:.14f}")
        print(f"    term_2         = V4 * (alpha/pi)^2       = {pred['term_2']:+.16f}")
        print(f"    a_predicted    = a_schwinger + term_2    = {pred['a_total']:.14f}")

        if a_exp is not None:
            error_pct = (pred['a_total'] - a_exp) / a_exp * 100
            print(f"    a_experiment                             = {a_exp:.14f}")
            print(f"    ERROR: {error_pct:+.4f}%")
            results[name]['error_pct'] = error_pct
            results[name]['a_exp'] = a_exp

    # =========================================================================
    # 5. SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: PARAMETER-FREE GEOMETRIC g-2")
    print("=" * 70)

    print(f"""
THE MASTER EQUATION:
    V4(R) = [(R_vac - R) / (R_vac + R)] * (xi/beta)

WHERE ALL PARAMETERS ARE DERIVED:
    beta = {BETA:.6f}     from Golden Loop: e^beta/beta = (alpha^-1 - 1)/(2*pi^2)
    xi   = {XI:.6f}     from golden ratio: xi = phi^2 = phi + 1
    R_vac = R_e/sqrt(5)   from golden ratio geometry
    R = hbar*c/m          from lepton mass (Compton wavelength)

SIGN FLIP MECHANISM:
    * Electron: R_e = {R_E:.3f} > R_vac = {R_VAC:.3f} => S < 0 => V4 < 0
    * Muon:     R_mu = {R_MU:.4f} < R_vac = {R_VAC:.3f} => S > 0 => V4 > 0

ACCURACY (Zero Free Parameters):
    * Electron g-2 error: {results['Electron']['error_pct']:+.4f}%
    * Muon g-2 error:     {results['Muon']['error_pct']:+.4f}%

The muon g-2 anomaly is a GEOMETRIC NECESSITY, not a fitting artifact.

LEAN PROOF: QFD/Lepton/GeometricG2.lean
    * theorem electron_V4_negative: V4(R_e) < 0
    * theorem muon_V4_positive: V4(R_mu) > 0
    * theorem g2_sign_flip_necessary: electron negative AND muon positive
""")

    # =========================================================================
    # 6. VALIDATION STATUS
    # =========================================================================
    e_pass = abs(results['Electron']['error_pct']) < 0.01
    mu_pass = abs(results['Muon']['error_pct']) < 0.01

    print("VALIDATION STATUS:")
    print(f"    Electron: {'PASS' if e_pass else 'FAIL'} (error < 0.01%)")
    print(f"    Muon:     {'PASS' if mu_pass else 'FAIL'} (error < 0.01%)")

    if e_pass and mu_pass:
        print("\n    [SUCCESS] Parameter-free g-2 derivation validated!")
        return 0
    else:
        print("\n    [WARNING] Some predictions exceed 0.01% error threshold")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
