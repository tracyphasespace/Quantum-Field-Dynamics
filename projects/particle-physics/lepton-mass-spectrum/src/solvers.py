"""
Density profile implementations for Hill's spherical vortex model.

Provides the spatial structure used in QFD lepton mass calculations.
"""

import numpy as np
from typing import Tuple


def hill_vortex_profile(r: np.ndarray, R: float, U: float, A: float = 1.0) -> np.ndarray:
    """
    Hill's spherical vortex density profile.

    ρ(r) = ρ_vac + A·f(r/R)  for r < R
    ρ(r) = ρ_vac              for r ≥ R

    where f is the Hill vortex shape function.

    Parameters
    ----------
    r : array
        Radial coordinates
    R : float
        Vortex radius
    U : float
        Circulation velocity
    A : float
        Amplitude normalization

    Returns
    -------
    ρ : array
        Density profile (normalized to ρ_vac = 1.0)
    """
    ρ = np.ones_like(r)  # Start with vacuum density = 1.0

    # Hill vortex interior (r < R)
    mask_interior = r < R
    x = r[mask_interior] / R

    # Hill vortex shape: f(x) = (1 - x²)² for simple model
    # This is approximate - full Hill vortex has more complex profile
    f = (1 - x**2)**2

    ρ[mask_interior] += A * f * (U / 0.5)**2  # Scale with velocity

    return ρ
