#!/usr/bin/env python3
"""QFD stability sweep with discrete isomer ladder corrections."""

from __future__ import annotations

import csv
import pathlib
import statistics
from dataclasses import dataclass
from math import pi

ALPHA = 1.0 / 137.036
BETA = 1.0 / 3.043233053
LAMBDA = 0.42
M_PROTON = 938.272

V0 = M_PROTON * (1 - (ALPHA**2) / BETA)
BETA_NUC = M_PROTON * BETA / 2

E_VOLUME = V0 * (1 - LAMBDA / (12 * pi))
E_SURFACE = BETA_NUC / 15
A_SYM = (BETA * M_PROTON) / 15

HBAR_C = 197.327
R0 = 1.2
A_DISP = (ALPHA * HBAR_C / R0) * 0.50  # optimal shielding

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
DELTA_ISO = 0.5 * E_SURFACE
DELTA_PAIR = 0.25 * E_SURFACE


@dataclass
class Nuclide:
    name: str
    Z: int
    A: int

    @property
    def N(self) -> int:
        return self.A - self.Z


def isomer_bonus(Z: int, N: int) -> float:
    bonus = 0.0
    if Z in ISOMER_NODES:
        bonus += DELTA_ISO
    if N in ISOMER_NODES:
        bonus += DELTA_ISO
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    if Z % 2 == 0 and N % 2 == 0:
        bonus += DELTA_PAIR
    return -bonus


def soliton_energy(A: int, Z: int) -> float:
    N = A - Z
    q = Z / A
    bulk = E_VOLUME * A
    surf = E_SURFACE * (A ** (2 / 3))
    asym = A_SYM * A * (1 - 2 * q) ** 2
    vac = A_DISP * (Z ** 2) / (A ** (1 / 3))
    iso = isomer_bonus(Z, N)
    return bulk + surf + asym + vac + iso


def stable_charge(A: int) -> int:
    if A <= 2:
        return 1
    best_Z = 1
    best_E = soliton_energy(A, 1)
    for Z in range(1, A):
        E = soliton_energy(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z


def load_nuclides() -> list[Nuclide]:
    data_path = pathlib.Path('data/stable_nuclides_qfd.csv')
    nuclides = []
    with data_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            nuclides.append(Nuclide(row['name'], int(row['Z']), int(row['A'])))
    return nuclides


def summarize(errors, label):
    exact = sum(e == 0 for e in errors)
    return f"{label:<16} N={len(errors):<4} Mean|ΔZ|={statistics.mean(errors):.2f}  " \
           f"Median={statistics.median(errors):.1f}  Max={max(errors):.0f}  Exact={exact}/{len(errors)} ({100*exact/len(errors):.0f}%)"


def main():
    nuclides = load_nuclides()
    results: list[tuple[Nuclide, int]] = []
    for iso in nuclides:
        Z_pred = stable_charge(iso.A)
        delta = Z_pred - iso.Z
        results.append((iso, delta))
        print(f"{iso.name:<8} A={iso.A:<4} Z_exp={iso.Z:<3} Z_pred={Z_pred:<3} ΔZ={delta:+}")

    errors = [abs(delta) for _, delta in results]
    exact = sum(e == 0 for e in errors)
    print("\nSUMMARY")
    print(f"Nuclides analysed: {len(results)}")
    print(f"Mean |ΔZ| = {statistics.mean(errors):.3f} | Median = {statistics.median(errors):.1f} | Max = {max(errors)}")
    print(f"Exact matches: {exact}/{len(results)} ({100*exact/len(results):.1f}%)")

    print("\nBy mass region:")
    print(summarize([abs(d) for iso, d in results if iso.A < 40], "A<40"))
    print(summarize([abs(d) for iso, d in results if 40 <= iso.A < 100], "40≤A<100"))
    print(summarize([abs(d) for iso, d in results if 100 <= iso.A < 200], "100≤A<200"))
    print(summarize([abs(d) for iso, d in results if iso.A >= 200], "A≥200"))

    print("\nBy charge range:")
    print(summarize([abs(d) for iso, d in results if iso.Z <= 10], "Z≤10"))
    print(summarize([abs(d) for iso, d in results if 11 <= iso.Z <= 20], "11≤Z≤20"))
    print(summarize([abs(d) for iso, d in results if 21 <= iso.Z <= 30], "21≤Z≤30"))
    print(summarize([abs(d) for iso, d in results if 31 <= iso.Z <= 50], "31≤Z≤50"))
    print(summarize([abs(d) for iso, d in results if iso.Z > 50], "Z>50"))

if __name__ == '__main__':
    main()
