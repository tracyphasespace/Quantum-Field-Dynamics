"""
QFD Cosmology Module
===================

Cosmological calculations for QFD redshift analysis.
Implements standard cosmology without dark energy acceleration.
"""

import math
from typing import Sequence, Union

import numpy as np

try:  # pragma: no cover - exercised indirectly via tests
    from scipy.integrate import quad  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for test environment
    def quad(func, a, b, limit=100):
        """Lightweight trapezoidal integration used when SciPy is unavailable."""

        steps = max(10, int(abs(b - a) * 100))
        h = (b - a) / steps
        total = 0.5 * (func(a) + func(b))
        for i in range(1, steps):
            total += func(a + i * h)
        return total * h, None


Number = Union[int, float]
ArrayLike = Union[Number, Sequence[Number], np.ndarray]


def _is_scalar(value: ArrayLike) -> bool:
    return isinstance(value, (int, float))


def _to_array(values: ArrayLike) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return np.array(values)
    raise TypeError("Expected a scalar or a sequence of scalars")


def _vectorize(values: ArrayLike, func):
    if _is_scalar(values):
        return func(float(values))
    arr = _to_array(values)
    return np.array([func(float(v)) for v in arr])


class QFDCosmology:
    """
    QFD cosmological model without dark energy.

    Implements standard Hubble expansion with QFD modifications
    but no cosmological acceleration.
    """

    def __init__(self, hubble_constant: float = 70.0):
        """
        Initialize QFD cosmology.

        Parameters:
        -----------
        hubble_constant : float
            Hubble constant in km/s/Mpc
        """
        self.H0 = hubble_constant
        self.c = 299792.458  # km/s (speed of light)

    def hubble_distance(self) -> float:
        """
        Calculate Hubble distance.

        Returns:
        --------
        float
            Hubble distance in Mpc
        """
        return self.c / self.H0

    def comoving_distance(
        self, redshift: ArrayLike
    ) -> ArrayLike:
        """
        Calculate comoving distance for given redshift.

        Uses simple matter-dominated universe (no dark energy).

        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z

        Returns:
        --------
        float or array
            Comoving distance in Mpc
        """
        def _comoving(z: float) -> float:
            if z < 0.1:
                return (self.c * z) / self.H0
            return (self.c / self.H0) * (z + 0.5 * z**2)

        return _vectorize(redshift, _comoving)

    def angular_diameter_distance(
        self, redshift: ArrayLike
    ) -> ArrayLike:
        """
        Calculate angular diameter distance.

        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z

        Returns:
        --------
        float or array
            Angular diameter distance in Mpc
        """
        if _is_scalar(redshift):
            d_comoving = self.comoving_distance(redshift)
            return d_comoving / (1 + float(redshift))

        redshift_arr = _to_array(redshift)
        d_comoving_arr = _to_array(self.comoving_distance(redshift_arr))
        return d_comoving_arr / (1 + redshift_arr)

    def luminosity_distance(
        self, redshift: ArrayLike
    ) -> ArrayLike:
        """
        Calculate luminosity distance.

        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z

        Returns:
        --------
        float or array
            Luminosity distance in Mpc
        """
        if _is_scalar(redshift):
            d_comoving = self.comoving_distance(redshift)
            return d_comoving * (1 + float(redshift))

        redshift_arr = _to_array(redshift)
        d_comoving_arr = _to_array(self.comoving_distance(redshift_arr))
        return d_comoving_arr * (1 + redshift_arr)

    def distance_modulus(
        self, redshift: ArrayLike
    ) -> ArrayLike:
        """
        Calculate distance modulus.

        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z

        Returns:
        --------
        float or array
            Distance modulus in magnitudes
        """
        if _is_scalar(redshift):
            d_lum = self.luminosity_distance(redshift)
            return 5 * math.log10(d_lum * 1e6 / 10)

        redshift_arr = _to_array(redshift)
        d_lum_arr = _to_array(self.luminosity_distance(redshift_arr))
        return np.array([5 * math.log10(val * 1e6 / 10) for val in d_lum_arr])

    def lambda_cdm_distance(
        self,
        redshift: ArrayLike,
        omega_m: float = 0.3,
        omega_lambda: float = 0.7,
    ) -> ArrayLike:
        """
        Calculate ΛCDM luminosity distance for comparison.

        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
        omega_m : float
            Matter density parameter
        omega_lambda : float
            Dark energy density parameter

        Returns:
        --------
        float or array
            ΛCDM luminosity distance in Mpc
        """

        def integrand(z):
            return 1 / np.sqrt(omega_m * (1 + z) ** 3 + omega_lambda)

        if _is_scalar(redshift):
            z = float(redshift)
            if z < 0.01:
                integral = z
            else:
                integral, _ = quad(integrand, 0, z)

            d_comoving = (self.c / self.H0) * integral
            return d_comoving * (1 + z)

        redshift_arr = _to_array(redshift)
        distances = []
        for z in redshift_arr:
            if z < 0.01:
                integral = z
            else:
                integral, _ = quad(integrand, 0, z)

            d_comoving = (self.c / self.H0) * integral
            distances.append(d_comoving * (1 + z))

        return np.array(distances)

    def lookback_time(
        self, redshift: ArrayLike
    ) -> ArrayLike:
        """
        Calculate lookback time.

        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z

        Returns:
        --------
        float or array
            Lookback time in Gyr
        """
        # Simple approximation for matter-dominated universe
        # t_lookback ≈ (2/3H0) * [1 - (1+z)^(-3/2)]

        hubble_time = 1 / (self.H0 * 1.022e-12)  # Convert to Gyr

        def _lookback(z: float) -> float:
            return (2 / 3) * hubble_time * (1 - (1 + z) ** (-1.5))

        return _vectorize(redshift, _lookback)

    def age_of_universe(
        self, redshift: ArrayLike = 0
    ) -> ArrayLike:
        """
        Calculate age of universe at given redshift.

        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z (default: 0 for present age)

        Returns:
        --------
        float or array
            Age of universe in Gyr
        """
        hubble_time = 1 / (self.H0 * 1.022e-12)  # Convert to Gyr

        def _age(z: float) -> float:
            return (2 / 3) * hubble_time * (1 + z) ** (-1.5)

        return _vectorize(redshift, _age)

    def critical_density(
        self, redshift: ArrayLike = 0
    ) -> ArrayLike:
        """
        Calculate critical density of the universe.

        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z

        Returns:
        --------
        float or array
            Critical density in g/cm³
        """
        # Critical density: ρ_c = 3H²/(8πG)
        G = 6.674e-8  # cm³/g/s² (gravitational constant)

        def _critical(z: float) -> float:
            H_z = self.H0 * 1.022e-12 * (1 + z) ** 1.5
            return 3 * H_z**2 / (8 * math.pi * G)

        return _vectorize(redshift, _critical)

    def sound_horizon(
        self, redshift: ArrayLike
    ) -> ArrayLike:
        """
        Calculate sound horizon at given redshift.

        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z

        Returns:
        --------
        float or array
            Sound horizon in Mpc
        """
        # Simplified calculation for matter-dominated universe
        # r_s ≈ c/(H0 * sqrt(Ω_m)) * 2/sqrt(1+z_eq) *
        # ln[(sqrt(1+z_eq) + sqrt(z_eq))/(1+sqrt(z_eq))]

        # For simplicity, use approximate formula
        z_eq = 3600  # Matter-radiation equality
        omega_m = 1.0  # Matter-dominated

        prefactor = self.c / (self.H0 * math.sqrt(omega_m))

        def _sound(z: float) -> float:
            if z > z_eq:
                return prefactor * 2 / math.sqrt(1 + z_eq)
            return prefactor * math.sqrt(z_eq / (1 + z))

        return _vectorize(redshift, _sound)
