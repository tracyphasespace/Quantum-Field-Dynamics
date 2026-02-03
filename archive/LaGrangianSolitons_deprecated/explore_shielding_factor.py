#!/usr/bin/env python3
"""
Explore geometric shielding factor to find optimal value.
"""

import numpy as np
from scipy.optimize import minimize_scalar

# Constants
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
lambda_time  = 0.42
M_proton     = 938.272

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15

a_sym = (beta_vacuum * M_proton) / 15

hbar_c = 197.327
r_0 = 1.2
a_c_base = alpha_fine * hbar_c / r_0

# Test solitons
test_solitons = [
    ("H-2",   1, 2), ("H-3",   1, 3), ("He-3",  2, 3), ("He-4",  2, 4),
    ("Li-6",  3, 6), ("Li-7",  3, 7), ("Be-9",  4, 9), ("B-10",  5, 10),
    ("B-11",  5, 11), ("C-12",  6, 12), ("C-13",  6, 13), ("N-14",  7, 14),
    ("N-15",  7, 15), ("O-16",  8, 16), ("O-17",  8, 17), ("O-18",  8, 18),
    ("F-19",  9, 19), ("Ne-20", 10, 20), ("Ne-22", 10, 22), ("Mg-24", 12, 24),
    ("Si-28", 14, 28), ("S-32",  16, 32), ("Ca-40", 20, 40),
    ("Fe-56", 26, 56), ("Ni-58", 28, 58),
]

def test_shielding_factor(shield_factor):
    """Test a specific shielding factor."""
    a_c = a_c_base * shield_factor

    def total_energy(A, Z):
        q = Z / A if A > 0 else 0
        E_bulk = E_volume * A
        E_surf = E_surface * (A ** (2/3))
        E_asym = a_sym * A * ((1 - 2*q)**2) if A > 0 else 0
        E_disp = a_c * (Z**2) / (A ** (1/3)) if A > 0 else 0
        return E_bulk + E_surf + E_asym + E_disp

    def find_stable_Z(A):
        result = minimize_scalar(
            lambda Z: total_energy(A, Z),
            bounds=(1, A-1),
            method='bounded'
        )
        return int(np.round(result.x))

    Z_errors = []
    for name, Z_exp, A in test_solitons:
        Z_pred = find_stable_Z(A)
        Z_errors.append(abs(Z_pred - Z_exp))

    return np.mean(Z_errors)

# Test range of shielding factors
print("Exploring geometric shielding factors...")
print("="*70)
print(f"{'Factor':<10} {'a_c (MeV)':<12} {'Mean |ΔZ|':<12} {'Status':<20}")
print("-"*70)

best_factor = None
best_error = float('inf')

for factor in np.linspace(0.5, 1.0, 51):
    mean_err = test_shielding_factor(factor)
    a_c = a_c_base * factor

    if mean_err < best_error:
        best_error = mean_err
        best_factor = factor

    status = ""
    if mean_err < 0.5:
        status = "✓✓✓ Excellent"
    elif mean_err < 1.0:
        status = "✓✓ Good"
    elif mean_err < 1.5:
        status = "✓ Fair"

    if factor in [0.5, 5/7, 0.75, 0.8, 0.85, 0.9, 1.0] or mean_err < 0.6:
        print(f"{factor:<10.4f} {a_c:<12.3f} {mean_err:<12.3f} {status:<20}")

print("="*70)
print(f"\nBest shielding factor: {best_factor:.4f}")
print(f"  a_c = {a_c_base * best_factor:.3f} MeV")
print(f"  Mean |ΔZ| = {best_error:.3f} charges")
print()

# Check geometric meaning of best factor
print("Geometric interpretations:")
print(f"  5/7 = {5/7:.4f}  (5 active dims out of 7 total)")
print(f"  2/3 = {2/3:.4f}  (spatial dims only)")
print(f"  3/4 = {3/4:.4f}  (3 out of 4 effective)")
print(f"  4/5 = {4/5:.4f}  (4 out of 5 active)")
print(f"  Best = {best_factor:.4f}")
