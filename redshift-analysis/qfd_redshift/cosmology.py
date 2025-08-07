"""
QFD Cosmology Module
===================

Cosmological calculations for QFD redshift analysis.
Implements standard cosmology without dark energy acceleration.
"""

import numpy as np
from scipy.integrate import quad
from typing import Union


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
        self, redshift: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        if isinstance(redshift, (int, float)):
            if redshift < 0.1:
                # Linear regime
                return (self.c * redshift) / self.H0
            else:
                # Include relativistic corrections (matter-dominated)
                # Simple approximation: D_C = (c/H0) * (z + 0.5*z^2)
                return (self.c / self.H0) * (redshift + 0.5 * redshift**2)
        else:
            # Array input
            result = np.zeros_like(redshift)
            linear_mask = redshift < 0.1
            nonlinear_mask = ~linear_mask

            # Linear regime
            result[linear_mask] = (self.c * redshift[linear_mask]) / self.H0

            # Nonlinear regime
            z_nl = redshift[nonlinear_mask]
            result[nonlinear_mask] = (self.c / self.H0) * (
                z_nl + 0.5 * z_nl**2
            )

            return result

    def angular_diameter_distance(
        self, redshift: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        d_comoving = self.comoving_distance(redshift)
        return d_comoving / (1 + redshift)

    def luminosity_distance(
        self, redshift: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        d_comoving = self.comoving_distance(redshift)
        return d_comoving * (1 + redshift)

    def distance_modulus(
        self, redshift: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        d_lum = self.luminosity_distance(redshift)
        return 5 * np.log10(d_lum * 1e6 / 10)

    def lambda_cdm_distance(
        self,
        redshift: Union[float, np.ndarray],
        omega_m: float = 0.3,
        omega_lambda: float = 0.7,
    ) -> Union[float, np.ndarray]:
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

        if isinstance(redshift, (int, float)):
            if redshift < 0.01:
                # Linear regime
                integral = redshift
            else:
                integral, _ = quad(integrand, 0, redshift)

            d_comoving = (self.c / self.H0) * integral
            return d_comoving * (1 + redshift)
        else:
            # Array input
            result = np.zeros_like(redshift)

            for i, z in enumerate(redshift):
                if z < 0.01:
                    integral = z
                else:
                    integral, _ = quad(integrand, 0, z)

                d_comoving = (self.c / self.H0) * integral
                result[i] = d_comoving * (1 + z)

            return result

    def lookback_time(
        self, redshift: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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

        if isinstance(redshift, (int, float)):
            factor = 1 - (1 + redshift) ** (-1.5)
            return (2 / 3) * hubble_time * factor
        else:
            factor = 1 - (1 + redshift) ** (-1.5)
            return (2 / 3) * hubble_time * factor

    def age_of_universe(
        self, redshift: Union[float, np.ndarray] = 0
    ) -> Union[float, np.ndarray]:
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

        if isinstance(redshift, (int, float)):
            # Matter-dominated universe age: t = (2/3H0) * (1+z)^(-3/2)
            return (2 / 3) * hubble_time * (1 + redshift) ** (-1.5)
        else:
            return (2 / 3) * hubble_time * (1 + redshift) ** (-1.5)

    def critical_density(
        self, redshift: Union[float, np.ndarray] = 0
    ) -> Union[float, np.ndarray]:
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
        H_z = (
            self.H0 * 1.022e-12 * (1 + redshift) ** 1.5
        )  # H(z) for matter-dominated

        return 3 * H_z**2 / (8 * np.pi * G)

    def sound_horizon(
        self, redshift: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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

        prefactor = self.c / (self.H0 * np.sqrt(omega_m))

        if isinstance(redshift, (int, float)):
            if redshift > z_eq:
                return prefactor * 2 / np.sqrt(1 + z_eq)
            else:
                # Approximate for z < z_eq
                return prefactor * np.sqrt(z_eq / (1 + redshift))
        else:
            result = np.zeros_like(redshift)
            high_z_mask = redshift > z_eq
            low_z_mask = ~high_z_mask

            result[high_z_mask] = prefactor * 2 / np.sqrt(1 + z_eq)
            result[low_z_mask] = prefactor * np.sqrt(
                z_eq / (1 + redshift[low_z_mask])
            )

            return result
