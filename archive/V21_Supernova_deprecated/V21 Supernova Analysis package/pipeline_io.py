"""
Pipeline I/O: Shared data structures for the V15 supernova analysis pipeline.

This module defines the canonical data structures used to pass information
between pipeline stages (Stage 1 → Stage 2 → Stage 3). Using named structures
instead of raw arrays prevents parameter ordering bugs and makes the code
self-documenting.

CRITICAL: This is the SINGLE SOURCE OF TRUTH for parameter ordering.
"""

from typing import NamedTuple
import numpy as np


class PerSNParams(NamedTuple):
    """
    Per-supernova parameters optimized in Stage 1.

    This defines the canonical order for per-SN parameters. Stage 1 saves
    parameters in this order, and Stage 2 must load them in this order.

    Parameters:
        t0: Time of explosion (MJD). Model peaks at t0 + t_rise (19 days).
        A_plasma: Plasma veil amplitude [0, 1]. Controls strength of wavelength-dependent dimming.
        beta: Wavelength slope of plasma veil [0, 4]. Controls (λ_B/λ)^β dependence.
        ln_A: Natural log of flux amplitude [-30, 30]. Log-space normalization factor.

    Note: L_peak is frozen at canonical value and not optimized.
    """
    t0: float
    A_plasma: float
    beta: float
    ln_A: float

    def to_array(self) -> np.ndarray:
        """
        Convert to NumPy array for optimization/storage.

        Returns:
            1D array [t0, A_plasma, beta, ln_A] suitable for scipy.optimize.minimize
        """
        return np.array(self, dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PerSNParams':
        """
        Create from NumPy array (e.g., loaded from persn_best.npy).

        Args:
            arr: 1D array with 4 elements in Stage 1 order: [t0, A_plasma, beta, ln_A]

        Returns:
            PerSNParams with named fields

        Raises:
            ValueError: If array doesn't have exactly 4 elements
        """
        if len(arr) != 4:
            raise ValueError(f"Expected 4 parameters, got {len(arr)}")
        return cls(*arr)

    def to_model_order(self) -> tuple:
        """
        Convert to the order expected by v15_model.qfd_lightcurve_model_jax.

        The physics model expects: (t0, ln_A, A_plasma, beta)
        Stage 1 saves as:          [t0, A_plasma, beta, ln_A]

        This method handles the reordering to prevent bugs.

        Returns:
            Tuple (t0, ln_A, A_plasma, beta) for passing to the model
        """
        return (self.t0, self.ln_A, self.A_plasma, self.beta)


class GlobalParams(NamedTuple):
    """
    Global cosmology parameters inferred in Stage 2.

    These are the QFD (Quantum Field Dynamics) cosmology parameters
    that are common to all supernovae.

    Parameters:
        k_J: QFD parameter [km/s/Mpc]. Paper expects ~10.7 ± 4.6
        eta_prime: QFD parameter (dimensionless). Paper expects ~-8.0 ± 1.4
        xi: QFD parameter (dimensionless). Paper expects ~-7.0 ± 3.8
    """
    k_J: float
    eta_prime: float
    xi: float

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array for storage."""
        return np.array(self, dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'GlobalParams':
        """Create from NumPy array."""
        if len(arr) != 3:
            raise ValueError(f"Expected 3 parameters, got {len(arr)}")
        return cls(*arr)


# Physics constants
L_PEAK_CANONICAL = 1.5e43  # erg/s - typical SN Ia peak luminosity (frozen in optimization)


# Example usage and validation
if __name__ == "__main__":
    print("=== Pipeline I/O Data Structures ===\n")

    # Example 1: Stage 1 parameter handling
    print("Example 1: Stage 1 saves parameters")
    stage1_params = PerSNParams(
        t0=57295.21,
        A_plasma=0.12,
        beta=0.57,
        ln_A=18.5
    )
    print(f"  Named params: {stage1_params}")
    print(f"  As array:     {stage1_params.to_array()}")
    print(f"  For model:    {stage1_params.to_model_order()}")
    print()

    # Example 2: Stage 2 loads parameters
    print("Example 2: Stage 2 loads from persn_best.npy")
    loaded_array = np.array([57295.21, 0.12, 0.57, 18.5])
    params = PerSNParams.from_array(loaded_array)
    print(f"  Loaded array: {loaded_array}")
    print(f"  Named params: {params}")
    print(f"  Access by name: t0={params.t0:.2f}, beta={params.beta:.2f}")
    print(f"  For model: {params.to_model_order()}")
    print()

    # Example 3: Global parameters
    print("Example 3: Stage 2 global parameters")
    globals_params = GlobalParams(k_J=10.7, eta_prime=-8.0, xi=-7.0)
    print(f"  Named params: {globals_params}")
    print(f"  As array:     {globals_params.to_array()}")
    print()

    print("✅ All examples passed!")
