"""
Estimate the geometric circulation coupling alpha_circ from the Hill vortex D-flow.
The boundary tangential velocity behaves like vθ ∝ sin θ, so the mean absolute
value over half a cycle is 2/π. Multiplying by Euler's number e gives alpha_circ ≈ e/(2π).
"""

import numpy as np
import math

def mean_abs_boundary_speed(samples: int = 10000) -> float:
    thetas = np.linspace(0, math.pi, samples, endpoint=False)
    vals = np.abs(np.sin(thetas))
    return vals.mean()

if __name__ == "__main__":
    mean_abs = mean_abs_boundary_speed()
    alpha_circ_numeric = math.e * mean_abs
    print(f"mean |sin θ| over [0, π] = {mean_abs:.6f} (expected 2/π ≈ {2/math.pi:.6f})")
    print(f"alpha_circ_geo = e * (2/π) = {math.e * 2/math.pi:.6f}")
    print(f"alpha_circ_numeric ≈ {alpha_circ_numeric:.6f}")
