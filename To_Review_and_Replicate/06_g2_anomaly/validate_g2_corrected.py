#!/usr/bin/env python3
"""
QFD: Parameter-Free Geometric g-2 Validation
=============================================

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License

PURPOSE:
--------
This script validates the geometric derivation of lepton anomalous magnetic
moments (g-2) with ZERO free parameters. All constants are derived from:
- alpha (fine structure constant)
- phi (golden ratio)
- lepton masses (Compton wavelengths)

THE MASTER EQUATION:
--------------------
    V4(R) = [(R_vac - R) / (R_vac + R)] * (xi / beta)

Where:
    R_vac = lambda_e / sqrt(5)     (vacuum correlation length)
    xi = phi^2 = phi + 1           (geometric coupling from golden ratio)
    beta from Golden Loop:         e^beta / beta = (alpha^-1 - 1) / (2*pi^2)
    R = hbar*c / m                 (Compton wavelength from mass)

SIGN FLIP MECHANISM:
--------------------
The Mobius transform (R_vac - R)/(R_vac + R) naturally produces:
- Electron: R_e > R_vac => negative V4 correction
- Muon:     R_mu < R_vac => positive V4 correction

This sign flip is a GEOMETRIC NECESSITY, not a fitting artifact.

ACCURACY (Zero Free Parameters):
--------------------------------
- Electron g-2 error: 0.0013%
- Muon g-2 error:     0.0027%

References:
    - projects/Lean4/QFD/Lepton/GeometricG2.lean (formal proof)
    - Schwinger (1948): a = alpha/(2*pi)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def solve_beta_golden_loop(alpha: float, max_iter: int = 20) -> float:
    """
    Solve the Golden Loop equation for beta.

    Golden Loop: 1/alpha = 2*pi^2 * (e^beta / beta) + 1

    Uses Newton-Raphson iteration.
    """
    K = (1.0 / alpha - 1.0) / (2 * np.pi**2)
    beta = 3.0  # Initial guess
    for _ in range(max_iter):
        f = np.exp(beta) / beta - K
        df = np.exp(beta) * (beta - 1) / (beta**2)
        if abs(df) < 1e-15:
            break
        beta -= f / df
    return beta


def create_g2_figure(results: dict, output_dir: str = '.'):
    """Create comparison figure for g-2 validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: V4 comparison
    ax1 = axes[0]
    labels = ['Electron\n(Experiment)', 'Electron\n(Predicted)',
              'Muon\n(Experiment)', 'Muon\n(Predicted)']

    V4_e_exp = results['electron']['V4_required']
    V4_e_pred = results['electron']['V4_predicted']
    V4_mu_exp = results['muon']['V4_required']
    V4_mu_pred = results['muon']['V4_predicted']

    values = [V4_e_exp, V4_e_pred, V4_mu_exp, V4_mu_pred]
    colors = ['lightblue', 'steelblue', 'lightcoral', 'coral']

    bars = ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05
        ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=11, fontweight='bold')

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('V4 Coefficient', fontsize=12)
    ax1.set_title('Parameter-Free V4 Prediction', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add error annotation
    error_e = results['electron']['error_pct']
    error_mu = results['muon']['error_pct']
    ax1.text(0.02, 0.98, f'Electron error: {abs(error_e):.4f}%\nMuon error: {abs(error_mu):.4f}%',
            transform=ax1.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Right plot: Scale factor diagram
    ax2 = axes[1]
    R_values = np.logspace(-3, 1, 100)
    R_vac = results['constants']['R_vac']
    S_values = (R_vac - R_values) / (R_vac + R_values)

    ax2.semilogx(R_values, S_values, 'b-', linewidth=2, label='S(R) = (R_vac - R)/(R_vac + R)')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(R_vac, color='green', linestyle='--', linewidth=1.5, label=f'R_vac = 1/sqrt(5) = {R_vac:.3f}')

    # Mark electron and muon positions
    R_e = 1.0
    R_mu = results['constants']['R_mu']
    S_e = results['electron']['scale_factor']
    S_mu = results['muon']['scale_factor']

    ax2.plot(R_e, S_e, 'bo', markersize=12, label=f'Electron (R=1, S={S_e:.3f})')
    ax2.plot(R_mu, S_mu, 'ro', markersize=12, label=f'Muon (R={R_mu:.4f}, S={S_mu:.3f})')

    ax2.set_xlabel('R / R_electron', fontsize=12)
    ax2.set_ylabel('Scale Factor S(R)', fontsize=12)
    ax2.set_title('Mobius Transform: Sign Flip Mechanism', fontsize=14)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'g2_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {output_path}")
    plt.close()


def validate_g2_corrected():
    """
    Validate parameter-free geometric g-2 derivation.

    Returns dict with all results for downstream use.
    """
    print("=" * 70)
    print("QFD: PARAMETER-FREE GEOMETRIC g-2 VALIDATION")
    print("=" * 70)

    # =========================================================================
    # 1. INPUT CONSTANTS (from Nature only)
    # =========================================================================
    print("\n[1] INPUT CONSTANTS (from Nature)")

    ALPHA = 1.0 / 137.035999206  # Fine structure constant (CODATA 2018)
    PHI = (1 + np.sqrt(5)) / 2   # Golden ratio

    # Lepton masses (MeV/c^2) - PDG 2024
    MASS_E = 0.51099895000
    MASS_MU = 105.6583755
    MASS_TAU = 1776.86

    print(f"    alpha = 1/{1/ALPHA:.9f}")
    print(f"    phi (golden ratio) = {PHI:.9f}")
    print(f"    m_e   = {MASS_E:.8f} MeV")
    print(f"    m_mu  = {MASS_MU:.7f} MeV")

    # =========================================================================
    # 2. DERIVE ALL PARAMETERS (Zero Free Parameters)
    # =========================================================================
    print("\n[2] DERIVED PARAMETERS (Zero Free Parameters)")

    # Beta from Golden Loop
    BETA = solve_beta_golden_loop(ALPHA)

    # Xi from golden ratio: xi = phi^2 = phi + 1
    XI = PHI**2

    # Vacuum correlation length: R_vac = R_e / sqrt(5)
    R_VAC = 1.0 / np.sqrt(5)

    # Compton wavelengths (R = hbar*c/m, so R_mu/R_e = m_e/m_mu)
    R_E = 1.0  # Reference scale
    R_MU = MASS_E / MASS_MU
    R_TAU = MASS_E / MASS_TAU

    print(f"    beta (Golden Loop)  = {BETA:.9f}")
    print(f"    xi = phi^2 = phi+1  = {XI:.9f}")
    print(f"    xi/beta (amplitude) = {XI/BETA:.9f}")
    print(f"    R_vac = 1/sqrt(5)   = {R_VAC:.9f}")
    print(f"    R_mu/R_e = m_e/m_mu = {R_MU:.9f}")

    # Verify Golden Loop closure
    golden_lhs = 1.0 / ALPHA
    golden_rhs = 2 * np.pi**2 * np.exp(BETA) / BETA + 1
    print(f"\n    Golden Loop verification:")
    print(f"      1/alpha = {golden_lhs:.6f}")
    print(f"      2pi^2 * e^beta/beta + 1 = {golden_rhs:.6f}")
    print(f"      Match: {'YES' if abs(golden_lhs - golden_rhs) < 1e-6 else 'NO'}")

    # =========================================================================
    # 3. EXPERIMENTAL g-2 VALUES
    # =========================================================================
    print("\n[3] EXPERIMENTAL g-2 VALUES")
    print("    Source: PDG 2024, Fermilab 2021")

    A_E_EXP = 0.00115965218128   # Electron (Harvard 2008)
    A_MU_EXP = 0.00116592059     # Muon (Fermilab 2021 + BNL)

    print(f"    a_e (experiment)  = {A_E_EXP:.14f}")
    print(f"    a_mu (experiment) = {A_MU_EXP:.14f}")

    # =========================================================================
    # 4. SCHWINGER TERM AND V4 EXTRACTION
    # =========================================================================
    print("\n[4] SCHWINGER TERM AND V4 EXTRACTION")

    a_schwinger = ALPHA / (2 * np.pi)
    alpha_over_pi_sq = (ALPHA / np.pi)**2

    print(f"    a_schwinger = alpha/(2*pi) = {a_schwinger:.14f}")
    print(f"    (alpha/pi)^2 = {alpha_over_pi_sq:.10e}")

    # Extract V4 required from experiment
    delta_a_e = A_E_EXP - a_schwinger
    delta_a_mu = A_MU_EXP - a_schwinger
    V4_e_required = delta_a_e / alpha_over_pi_sq
    V4_mu_required = delta_a_mu / alpha_over_pi_sq

    print(f"\n    Electron:")
    print(f"      delta_a = a_exp - a_schwinger = {delta_a_e:.10e}")
    print(f"      V4_required = delta_a / (alpha/pi)^2 = {V4_e_required:.6f}")

    print(f"\n    Muon:")
    print(f"      delta_a = a_exp - a_schwinger = {delta_a_mu:.10e}")
    print(f"      V4_required = delta_a / (alpha/pi)^2 = {V4_mu_required:.6f}")

    # =========================================================================
    # 5. GEOMETRIC PREDICTION (Parameter-Free)
    # =========================================================================
    print("\n[5] GEOMETRIC PREDICTION (Parameter-Free)")
    print("    Master Equation: V4(R) = [(R_vac - R)/(R_vac + R)] * (xi/beta)")
    print("    Where R = Compton wavelength = hbar*c/m")

    def predict_V4(R):
        """Predict V4 from geometric formula."""
        S = (R_VAC - R) / (R_VAC + R)
        return S * (XI / BETA), S

    V4_e_pred, S_e = predict_V4(R_E)
    V4_mu_pred, S_mu = predict_V4(R_MU)
    V4_tau_pred, S_tau = predict_V4(R_TAU)

    print(f"\n    Electron (R = R_e = 1):")
    print(f"      Scale factor S = {S_e:+.6f}")
    print(f"      R_e > R_vac => S < 0 => NEGATIVE correction")
    print(f"      V4_predicted = {V4_e_pred:+.6f}")

    print(f"\n    Muon (R = R_e * {R_MU:.6f}):")
    print(f"      Scale factor S = {S_mu:+.6f}")
    print(f"      R_mu < R_vac => S > 0 => POSITIVE correction")
    print(f"      V4_predicted = {V4_mu_pred:+.6f}")

    # =========================================================================
    # 6. RECONSTRUCT g-2 AND COMPARE
    # =========================================================================
    print("\n[6] RECONSTRUCT g-2 FROM GEOMETRIC V4")

    a_e_pred = a_schwinger + V4_e_pred * alpha_over_pi_sq
    a_mu_pred = a_schwinger + V4_mu_pred * alpha_over_pi_sq

    error_e = (a_e_pred - A_E_EXP) / A_E_EXP * 100
    error_mu = (a_mu_pred - A_MU_EXP) / A_MU_EXP * 100

    print(f"\n    Electron:")
    print(f"      a_predicted  = {a_e_pred:.14f}")
    print(f"      a_experiment = {A_E_EXP:.14f}")
    print(f"      ERROR: {error_e:+.4f}%")

    print(f"\n    Muon:")
    print(f"      a_predicted  = {a_mu_pred:.14f}")
    print(f"      a_experiment = {A_MU_EXP:.14f}")
    print(f"      ERROR: {error_mu:+.4f}%")

    # =========================================================================
    # 7. SUMMARY
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
    R_vac = 1/sqrt(5)     from golden ratio geometry
    R = hbar*c/m          from lepton mass (Compton wavelength)

SIGN FLIP MECHANISM:
    * Electron: R_e = {R_E:.3f} > R_vac = {R_VAC:.3f} => S < 0 => V4 < 0
    * Muon:     R_mu = {R_MU:.4f} < R_vac = {R_VAC:.3f} => S > 0 => V4 > 0

ACCURACY (Zero Free Parameters):
    * Electron g-2 error: {error_e:+.4f}%
    * Muon g-2 error:     {error_mu:+.4f}%

The sign flip between electron and muon g-2 is a GEOMETRIC NECESSITY.
""")

    # Validation status
    e_pass = abs(error_e) < 0.01
    mu_pass = abs(error_mu) < 0.01

    print("VALIDATION STATUS:")
    print(f"    Electron: {'PASS' if e_pass else 'FAIL'} (error < 0.01%)")
    print(f"    Muon:     {'PASS' if mu_pass else 'FAIL'} (error < 0.01%)")

    if e_pass and mu_pass:
        print("\n    [SUCCESS] Parameter-free g-2 derivation validated!")

    # Return results for downstream use
    results = {
        'constants': {
            'alpha': ALPHA,
            'beta': BETA,
            'xi': XI,
            'phi': PHI,
            'R_vac': R_VAC,
            'R_e': R_E,
            'R_mu': R_MU,
        },
        'electron': {
            'a_exp': A_E_EXP,
            'a_pred': a_e_pred,
            'V4_required': V4_e_required,
            'V4_predicted': V4_e_pred,
            'scale_factor': S_e,
            'error_pct': error_e,
        },
        'muon': {
            'a_exp': A_MU_EXP,
            'a_pred': a_mu_pred,
            'V4_required': V4_mu_required,
            'V4_predicted': V4_mu_pred,
            'scale_factor': S_mu,
            'error_pct': error_mu,
        },
        'passed': e_pass and mu_pass
    }

    return results


if __name__ == "__main__":
    results = validate_g2_corrected()
    create_g2_figure(results)
