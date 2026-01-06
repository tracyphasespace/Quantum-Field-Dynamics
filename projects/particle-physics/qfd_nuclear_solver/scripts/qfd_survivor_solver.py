#!/usr/bin/env python3
"""Shape-shifting topological survivor search."""

import numpy as np
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
SHIELD_FACTOR = 0.52
A_DISP = (ALPHA * HBAR_C / R0) * SHIELD_FACTOR

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70


def get_resonance_bonus(Z: int, N: int) -> float:
    bonus = 0.0
    if Z in ISOMER_NODES:
        bonus += E_SURFACE * BONUS_STRENGTH
    if N in ISOMER_NODES:
        bonus += E_SURFACE * BONUS_STRENGTH
    return bonus


def qfd_survivor_energy(A: int, Z: int, ecc: float) -> float:
    N = A - Z
    q = Z / A
    G_surf = 1.0 + (2.0/3.0) * (ecc**2)
    G_disp = 1.0 - 0.2 * (ecc**2)

    E_bulk = E_VOLUME * A
    E_surf = E_SURFACE * (A ** (2 / 3)) * G_surf
    E_asym = A_SYM * A * ((1 - 2 * q)**2)
    E_vac = A_DISP * (Z**2) / (A ** (1 / 3)) * G_disp
    E_iso = -get_resonance_bonus(Z, N)
    return E_bulk + E_surf + E_asym + E_vac + E_iso


def find_survivor_state(A: int):
    best_Z, best_ecc = 1, 0.0
    min_energy = qfd_survivor_energy(A, best_Z, best_ecc)
    for z in range(1, A):
        for ecc in np.linspace(0, 0.25, 11):
            energy = qfd_survivor_energy(A, z, ecc)
            if energy < min_energy:
                min_energy = energy
                best_Z, best_ecc = z, ecc
    return best_Z, best_ecc


def load_nuclides(path):
    nuclides = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            nuclides.append((row['name'], int(row['Z']), int(row['A'])))
    return nuclides


def main():
    nuclides = load_nuclides('data/stable_nuclides_qfd.csv')
    errors = []
    for name, Z_exp, A in nuclides:
        Z_pred, ecc = find_survivor_state(A)
        delta = Z_pred - Z_exp
        errors.append(abs(delta))
        print(f"{name:<8} A={A:<4} Z_exp={Z_exp:<3} Z_pred={Z_pred:<3} ΔZ={delta:+} ecc={ecc:.2f}")
    mean_err = sum(errors)/len(errors)
    exact = sum(1 for e in errors if e == 0)
    print(f"\nMean |ΔZ|={mean_err:.3f} charges; exact matches: {exact}/{len(errors)}")

if __name__ == '__main__':
    main()
