#!/usr/bin/env python3
"""Fe-56 anchored QFD optimization with discrete isomer ladder."""

import numpy as np

ALPHA = 1.0 / 137.036
BETA = 1.0 / 3.058231
LAMBDA = 0.42
M_PROTON = 938.272

V0 = M_PROTON * (1 - (ALPHA**2) / BETA)
BETA_NUC = M_PROTON * BETA / 2

E_VOLUME = V0 * (1 - LAMBDA / (12 * np.pi))
E_SURFACE = BETA_NUC / 15
A_SYM = (BETA * M_PROTON) / 15

HBAR_C = 197.327
R0 = 1.2
F_SHIELD = 0.52
A_DISP = (ALPHA * HBAR_C / R0) * F_SHIELD

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
F_RESONANCE = 0.70
E_RESONANCE = E_SURFACE * F_RESONANCE


def get_resonance_bonus(Z: int, N: int) -> float:
    bonus = 0.0
    if Z in ISOMER_NODES:
        bonus += E_RESONANCE
    if N in ISOMER_NODES:
        bonus += E_RESONANCE
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus


def qfd_energy(A: int, Z: int) -> float:
    N = A - Z
    q = Z / A
    E_bulk = E_VOLUME * A
    E_surf = E_SURFACE * (A ** (2 / 3))
    E_asym = A_SYM * A * ((1 - 2 * q) ** 2)
    E_vac = A_DISP * (Z ** 2) / (A ** (1 / 3))
    E_iso = -get_resonance_bonus(Z, N)
    return E_bulk + E_surf + E_asym + E_vac + E_iso


def find_stable_Z_discrete(A: int) -> int:
    if A <= 2:
        return 1
    best_Z = 1
    best_E = qfd_energy(A, best_Z)
    for Z in range(2, A):
        E = qfd_energy(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z


def main():
    A_fe = 56
    Z_fe_exp = 26
    Z_fe_pred = find_stable_Z_discrete(A_fe)
    M_EXP_FE = 52102.50
    m_qfd_fe = qfd_energy(A_fe, Z_fe_exp)
    mass_err_pct = 100.0 * (m_qfd_fe - M_EXP_FE) / M_EXP_FE

    print("=" * 70)
    print("Fe-56 Calibration Anchor")
    print("-" * 70)
    print(f"Shielding factor (f): {F_SHIELD:.3f}")
    print(f"Isomer bonus factor : {F_RESONANCE:.3f}")
    print(f"Experimental Z      : {Z_fe_exp}")
    print(f"Predicted Z         : {Z_fe_pred} "
          f"({'✓ MATCH' if Z_fe_pred == Z_fe_exp else 'x'})")
    print(f"Mass error          : {mass_err_pct:+.6f}%")
    print("=" * 70)

    # optional: sweep over sample nuclides
    nuclides = [
        ("Ca-40", 20, 40),
        ("Fe-56", 26, 56),
        ("Ni-58", 28, 58),
        ("Sn-120", 50, 120),
        ("Pb-208", 82, 208)
    ]
    print("\nSample comparison (discrete solver):")
    for name, Z_exp, A in nuclides:
        Z_pred = find_stable_Z_discrete(A)
        print(f"  {name:<8} A={A:<3} Z_exp={Z_exp:<3} Z_pred={Z_pred:<3} ΔZ={Z_pred - Z_exp:+}")


if __name__ == "__main__":
    main()
