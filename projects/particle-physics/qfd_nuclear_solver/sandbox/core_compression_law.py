#!/usr/bin/env python3
"""
Core Compression Law calculator.

Implements the empirical relation:
    Z_pred = 0.529 * A^(2/3) + (1 / 3.058) * A

and compares it against true proton numbers for a set of reference nuclei.
"""

import math

COEFF_SURFACE = 0.529
COEFF_BULK = 1.0 / 3.058

# (A, Z_true, label)
REFERENCE_NUCLEI = [
    (4, 2, "He-4"),
    (8, 4, "O-16/alpha x2"),  # placeholder showing trend
    (12, 6, "C-12"),
    (16, 8, "O-16"),
    (20, 10, "Ne-20"),
    (24, 12, "Mg-24"),
    (28, 14, "Si-28"),
    (32, 16, "S-32"),
    (40, 20, "Ca-40"),
    (56, 26, "Fe-56"),
]


def core_compression_charge(A: float) -> float:
    """Predict charge using the Core Compression Law."""
    return COEFF_SURFACE * (A ** (2.0 / 3.0)) + COEFF_BULK * A


def main():
    print(f"{'Nucleus':<8} {'A':>4} {'Z_true':>7} {'Z_pred':>10} {'ΔZ':>8}")
    print("-" * 40)

    for A, Z_true, label in REFERENCE_NUCLEI:
        Z_pred = core_compression_charge(A)
        delta = Z_pred - Z_true
        print(f"{label:<8} {A:>4} {Z_true:>7.2f} {Z_pred:>10.3f} {delta:>+8.3f}")


if __name__ == "__main__":
    main()
