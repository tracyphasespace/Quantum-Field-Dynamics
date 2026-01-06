#!/usr/bin/env python3
"""Toy simulation of dual-core soliton with matched mass and surface area."""

import csv
import numpy as np

# Simple density/gradient model: inner ball + outer shell

def dual_core_profile(A, Z,
                      core_radius=0.7,
                      total_radius=1.0,
                      core_density_scale=1.0,
                      shell_density_scale=0.5,
                      shell_charge_scale=0.8,
                      bulk_modulus=927.668,
                      surface_coeff=10.228,
                      displacement_coeff=1.200):
    """Return energy contributions for a toy dual-core configuration."""
    A = float(A)
    Z = float(Z)

    # Radii and volumes
    R_core = core_radius * (A ** (1/3))
    R_total = total_radius * (A ** (1/3))
    R_shell = max(R_total - R_core, 1e-6)

    V_core = (4.0 / 3.0) * np.pi * (R_core ** 3)
    V_total = (4.0 / 3.0) * np.pi * (R_total ** 3)
    V_shell = max(V_total - V_core, 1e-6)

    # Distribute mass between core and shell
    A_core = core_density_scale * A
    A_shell = max(A - A_core, 1e-6)

    # Charge fraction in shell (outer atmosphere carries most charge)
    Z_shell = shell_charge_scale * Z
    Z_core = max(Z - Z_shell, 0.0)

    # Bulk energy ~ density * bulk modulus
    E_bulk = bulk_modulus * A

    # Surface energy ~ shell area + core area
    S_core = 4.0 * np.pi * (R_core ** 2)
    S_shell_outer = 4.0 * np.pi * (R_total ** 2)
    E_surface = surface_coeff * (S_shell_outer + 0.3 * S_core) / (A ** (2/3))

    # Displacement/charge energy: mostly from shell, slight from core
    E_displacement = displacement_coeff * ((Z_shell ** 2) / (A_shell ** (1/3))
                                           + 0.1 * (Z_core ** 2) / (A_core ** (1/3)))

    return E_bulk, E_surface, E_displacement


def load_nuclides(path='data/stable_nuclides_qfd.csv'):
    records = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append((row['name'], int(row['Z']), int(row['A'])))
    return records


def main():
    nuclides = load_nuclides()
    for name, Z, A in nuclides[:10]:
        E_bulk, E_surface, E_disp = dual_core_profile(A, Z)
        total_E = E_bulk + E_surface + E_disp
        print(f"{name:<8} A={A:<4} Z={Z:<3} E_total={total_E:>10.2f} (bulk={E_bulk:>10.2f} + surface={E_surface:>7.2f} + disp={E_disp:>7.2f})")


if __name__ == '__main__':
    main()
