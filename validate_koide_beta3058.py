#!/usr/bin/env python3
"""
Koide Relation Numerical Validation for beta = 3.058 rad

Quick validation script for lepton mass predictions using geometric Koide formula.

Usage:
    python3 validate_koide_beta3058.py

Or make executable and run directly:
    chmod +x validate_koide_beta3058.py
    ./validate_koide_beta3058.py
"""

import numpy as np

# ============================================================================
# PARAMETERS (UPDATE THESE IF YOU HAVE FITTED VALUES)
# ============================================================================
DELTA = 3.058  # rad - generation phase angle
MU = 1.0       # MeV - PLACEHOLDER! Replace with fitted value if available

# Experimental values (PDG 2024)
M_E_EXP = 0.5109989461     # MeV (electron)
M_MU_EXP = 105.6583745     # MeV (muon)
M_TAU_EXP = 1776.86        # MeV (tau)

# ============================================================================
# GEOMETRIC MASS FORMULA
# ============================================================================

def geometric_mass(k, mu, delta):
    """
    Koide geometric mass formula.

    Args:
        k: Generation index (0=electron, 1=muon, 2=tau)
        mu: Mass scale parameter (MeV)
        delta: Phase angle (radians)

    Returns:
        Mass in MeV
    """
    angle = delta + k * (2 * np.pi / 3)
    term = 1 + np.sqrt(2) * np.cos(angle)
    return mu * term**2


def koide_ratio(m_e, m_mu, m_tau):
    """
    Compute Koide Q ratio.

    Q = (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2

    Should equal 2/3 if Koide relation holds.
    """
    numerator = m_e + m_mu + m_tau
    denominator = (np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau))**2
    return numerator / denominator


# ============================================================================
# VALIDATION
# ============================================================================

def main():
    print("=" * 80)
    print("KOIDE RELATION VALIDATION: beta = 3.058 rad")
    print("=" * 80)
    print()

    # Compute predicted masses
    print(f"Parameters:")
    print(f"  delta = {DELTA:.6f} rad = {DELTA * 180/np.pi:.3f}°")
    print(f"  mu    = {MU:.6f} MeV")
    print(f"  (Note: If mu=1.0, this is a PLACEHOLDER - update with fitted value!)")
    print()

    m_e_pred = geometric_mass(0, MU, DELTA)
    m_mu_pred = geometric_mass(1, MU, DELTA)
    m_tau_pred = geometric_mass(2, MU, DELTA)

    # Compute Q ratio
    Q_pred = koide_ratio(m_e_pred, m_mu_pred, m_tau_pred)
    Q_exp = koide_ratio(M_E_EXP, M_MU_EXP, M_TAU_EXP)
    Q_target = 2.0 / 3.0

    # Display results
    print("Predicted Masses:")
    print(f"  m_e   = {m_e_pred:.6f} MeV")
    print(f"  m_mu  = {m_mu_pred:.6f} MeV")
    print(f"  m_tau = {m_tau_pred:.6f} MeV")
    print()

    print("Experimental Masses:")
    print(f"  m_e   = {M_E_EXP:.6f} MeV")
    print(f"  m_mu  = {M_MU_EXP:.6f} MeV")
    print(f"  m_tau = {M_TAU_EXP:.6f} MeV")
    print()

    print("Residuals (Predicted - Experimental):")
    print(f"  Δm_e   = {m_e_pred - M_E_EXP:+.6f} MeV ({100*(m_e_pred/M_E_EXP - 1):+.3f}%)")
    print(f"  Δm_mu  = {m_mu_pred - M_MU_EXP:+.6f} MeV ({100*(m_mu_pred/M_MU_EXP - 1):+.3f}%)")
    print(f"  Δm_tau = {m_tau_pred - M_TAU_EXP:+.6f} MeV ({100*(m_tau_pred/M_TAU_EXP - 1):+.3f}%)")
    print()

    print("Koide Q Ratio:")
    print(f"  Q_predicted    = {Q_pred:.8f}")
    print(f"  Q_experimental = {Q_exp:.8f}")
    print(f"  Q_target (2/3) = {Q_target:.8f}")
    print()
    print(f"  |Q_pred - 2/3|  = {abs(Q_pred - Q_target):.2e}")
    print(f"  |Q_exp - 2/3|   = {abs(Q_exp - Q_target):.2e}")
    print()

    # Assessment
    print("=" * 80)
    print("ASSESSMENT:")
    print("=" * 80)

    if abs(Q_pred - Q_target) < 1e-6:
        print("✓ Q ratio matches 2/3 to high precision (< 1e-6)")
    elif abs(Q_pred - Q_target) < 1e-4:
        print("✓ Q ratio close to 2/3 (< 1e-4)")
    else:
        print("⚠ Q ratio deviates significantly from 2/3")

    print()

    if MU == 1.0:
        print("⚠ WARNING: mu = 1.0 is a placeholder!")
        print("  - This gives Q ratio but not absolute masses")
        print("  - To fit masses, you need to optimize mu")
        print("  - Try: mu = M_E_EXP / geometric_mass(0, 1.0, DELTA)")
        print()

        # Auto-compute mu from electron mass
        mu_fitted = M_E_EXP / geometric_mass(0, 1.0, DELTA)
        print(f"Suggested mu from electron mass: {mu_fitted:.6f} MeV")
        print()
        print("Recomputing with fitted mu...")
        print()

        m_e_fit = geometric_mass(0, mu_fitted, DELTA)
        m_mu_fit = geometric_mass(1, mu_fitted, DELTA)
        m_tau_fit = geometric_mass(2, mu_fitted, DELTA)

        Q_fit = koide_ratio(m_e_fit, m_mu_fit, m_tau_fit)

        print(f"With mu = {mu_fitted:.6f} MeV:")
        print(f"  m_e   = {m_e_fit:.6f} MeV (Δ = {m_e_fit - M_E_EXP:+.2e} MeV)")
        print(f"  m_mu  = {m_mu_fit:.6f} MeV (Δ = {m_mu_fit - M_MU_EXP:+.2e} MeV)")
        print(f"  m_tau = {m_tau_fit:.6f} MeV (Δ = {m_tau_fit - M_TAU_EXP:+.2e} MeV)")
        print(f"  Q     = {Q_fit:.8f} (Δ = {Q_fit - Q_target:+.2e})")

    print()
    print("=" * 80)
    print("To update mu, edit DELTA and MU at top of this file, then re-run.")
    print("=" * 80)


# ============================================================================
# DELTA SENSITIVITY TEST
# ============================================================================

def sensitivity_test():
    """Test how sensitive predictions are to delta parameter."""
    print()
    print("=" * 80)
    print("DELTA SENSITIVITY TEST")
    print("=" * 80)
    print()

    delta_values = np.linspace(3.00, 3.10, 11)

    print(f"{'delta (rad)':<12} {'Q ratio':<12} {'|Q - 2/3|':<15}")
    print("-" * 40)

    for delta in delta_values:
        m_e = geometric_mass(0, 1.0, delta)
        m_mu = geometric_mass(1, 1.0, delta)
        m_tau = geometric_mass(2, 1.0, delta)
        Q = koide_ratio(m_e, m_mu, m_tau)
        error = abs(Q - 2.0/3.0)
        marker = " ★" if abs(delta - DELTA) < 0.001 else ""
        print(f"{delta:12.6f} {Q:12.8f} {error:15.2e}{marker}")

    print()
    print("★ = Current delta value")
    print()


if __name__ == "__main__":
    main()

    # Uncomment to run sensitivity test
    # sensitivity_test()
