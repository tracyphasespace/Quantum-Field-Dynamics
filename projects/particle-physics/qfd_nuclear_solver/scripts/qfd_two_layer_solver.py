#!/usr/bin/env python3
"""Two-layer soliton model with vortex shielding."""

import csv
from math import pi

ALPHA = 1.0 / 137.036
BETA = 1.0 / 3.058231
LAMBDA = 0.42
M_PROTON = 938.272

V0 = M_PROTON * (1 - (ALPHA**2) / BETA)
BETA_NUC = M_PROTON * BETA / 2

E_VOLUME = V0 * (1 - LAMBDA / (12 * pi))
E_SURFACE = BETA_NUC / 15
A_SYM = (BETA * M_PROTON) / 15

HBAR_C = 197.327
R0 = 1.2
A_DISP_BASE = (ALPHA * HBAR_C / R0)

CORE_FRACTION = 0.35
CORE_CHARGE_FRACTION = 0.25
CORE_SURFACE_SCALE = 3.0
CORE_DISPLACEMENT_SCALE = 0.10
COUPLING_STIFFNESS = 40.0
VORTEX_STRENGTH = 0.40

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_SCALE = 0.70


def resonance_bonus(Z: int, N: int) -> float:
    bonus = 0.0
    if Z in ISOMER_NODES:
        bonus += E_SURFACE * BONUS_SCALE
    if N in ISOMER_NODES:
        bonus += E_SURFACE * BONUS_SCALE
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus


def two_layer_energy(A: int, Z: int) -> float:
    N = A - Z

    A_core = CORE_FRACTION * A
    A_shell = A - A_core
    Z_core = CORE_CHARGE_FRACTION * Z
    Z_shell = Z - Z_core

    q_shell = Z_shell / A_shell if A_shell > 0 else 0.0
    q_core = Z_core / A_core if A_core > 0 else 0.0

    # shell layer
    E_bulk_shell = E_VOLUME * A_shell
    E_surface_shell = E_SURFACE * (A_shell ** (2/3))
    E_asym_shell = A_SYM * A_shell * ((1 - 2 * q_shell) ** 2)
    shield = 1.0 / (1.0 + VORTEX_STRENGTH * max(0.0, (A - 120.0)) / 120.0)
    E_vac_shell = A_DISP_BASE * shield * (Z_shell ** 2) / (A_shell ** (1/3))

    # core layer
    E_bulk_core = E_VOLUME * A_core
    E_surface_core = CORE_SURFACE_SCALE * E_SURFACE * (A_core ** (2/3))
    E_vac_core = CORE_DISPLACEMENT_SCALE * A_DISP_BASE * (Z_core ** 2) / (A_core ** (1/3))

    # coupling energy between core and shell charge fractions
    E_coupling = COUPLING_STIFFNESS * (q_shell - q_core) ** 2

    # resonance bonus acts on total Z,N
    E_res = -resonance_bonus(Z, N)

    return (E_bulk_shell + E_surface_shell + E_asym_shell + E_vac_shell +
            E_bulk_core + E_surface_core + E_vac_core + E_coupling + E_res)


def load_dataset(path='data/stable_nuclides_qfd.csv'):
    items = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            items.append((row['name'], int(row['Z']), int(row['A'])))
    return items


def main():
    nuclides = load_dataset()
    total = len(nuclides)
    exact = 0
    errors = []

    for name, Z_exp, A in nuclides:
        best_Z = 1
        best_E = two_layer_energy(A, best_Z)
        for Z in range(1, A):
            E = two_layer_energy(A, Z)
            if E < best_E:
                best_E = E
                best_Z = Z
        delta = best_Z - Z_exp
        if delta == 0:
            exact += 1
        errors.append(abs(delta))
        print(f"{name:<8} A={A:<4} Z_exp={Z_exp:<3} Z_pred={best_Z:<3} ΔZ={delta:+}")

    mean_error = sum(errors) / len(errors)
    print("\nSUMMARY")
    print(f"Nuclides analysed: {total}")
    print(f"Exact matches:     {exact}/{total} ({100*exact/total:.1f}%)")
    print(f"Mean |ΔZ|:         {mean_error:.3f}")


if __name__ == '__main__':
    main()
