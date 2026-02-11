"""
Compute the dimensionless gravity–stiffness ratios used in the QFD bridge.

alpha_g = G * mp**2 / (hbar * c)
xi_qfd = alpha_g * (L0 / lp)**2 ≈ 16
"""

import math
import sys
import os

# Import QFD shared constants
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from qfd.shared_constants import BETA as _BETA_SHARED

G = 6.67430e-11
mp = 1.672619e-27
c = 299_792_458
hbar = 1.054571817e-34
L0 = 0.8414e-15
lp = 1.616255e-35
beta = _BETA_SHARED

def compute_alpha_g():
    return G * mp**2 / (hbar * c)

def compute_xi_qfd(alpha_g):
    return alpha_g * (L0 / lp)**2

if __name__ == "__main__":
    alpha_g = compute_alpha_g()
    xi = compute_xi_qfd(alpha_g)
    print(f"alpha_g = {alpha_g:.3e}")
    print(f"xi_qfd = {xi:.3f} (target ≈ 16)")
    print(f"16π/3 = {16*math.pi/3:.3f}")
    print(f"5.24*beta = {5.24*beta:.3f}")
