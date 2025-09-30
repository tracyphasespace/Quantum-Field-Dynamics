"""
QVD Scattering Module for QFD Supernova Analysis
=================================================

Simplified implementation of the QVD scattering model for the demo.
This module provides the QVDScattering class, which calculates the
dimming of supernova light due to quantum vacuum dynamics effects.

The physics is based on the more complex model in the `Supernova`
package, but is simplified here to be self-contained for the demo.
"""

import numpy as np
from typing import Dict, Tuple

# Physical constants (simplified for the demo)
SIGMA_T = 6.652e-25  # Thomson scattering cross-section, cm^2

class QVDScattering:
    """
    Calculates QVD scattering effects on supernova light.

    This is a simplified model for the qfd_supernova demo.
    """

    def __init__(self,
                 qfd_coupling: float = 0.85,
                 redshift_power: float = 0.6,
                 wavelength_alpha: float = -0.8,
                 plasma_enhancement: float = 500,
                 temporal_scale: float = 15.0):
        """
        Initialize the QVD scattering model.

        Parameters are passed from the SupernovaAnalyzer.
        """
        self.qfd_coupling = qfd_coupling
        self.redshift_power = redshift_power
        self.wavelength_alpha = wavelength_alpha
        self.plasma_enhancement = plasma_enhancement
        self.temporal_scale = temporal_scale

    def calculate_qvd_dimming(self,
                              redshift: float,
                              wavelength_nm: float,
                              time_days: float,
                              plasma_state: Dict[str, float],
                              host_environment: float = 1.0) -> Tuple[float, float]:
        """
        Calculate the dimming of light due to QVD scattering.

        This is a simplified calculation that captures the essence of the
        more complex model.

        Returns:
        --------
        Tuple[float, float]
            - qvd_dimming_mag: The amount of dimming in magnitudes.
            - tau: The optical depth of the QVD effect.
        """
        if plasma_state is None or plasma_state.get('electron_density_cm3', 0) == 0:
            return 0.0, 0.0

        # --- Simplified QVD Cross-Section Calculation ---

        # 1. Base cross-section (related to Thomson scattering)
        sigma_base = SIGMA_T * self.qfd_coupling

        # 2. Wavelength dependence
        # Use a reference wavelength (e.g., V-band)
        ref_wavelength_nm = 551.0
        wavelength_ratio = wavelength_nm / ref_wavelength_nm
        wavelength_factor = wavelength_ratio ** self.wavelength_alpha

        # 3. Plasma enhancement
        # Simplified dependence on electron density
        density = plasma_state.get('electron_density_cm3', 1e6)
        density_ratio = density / 1e10  # Reference density
        plasma_factor = 1 + self.plasma_enhancement * np.sqrt(density_ratio)

        # 4. Temporal evolution (effect decays as plasma expands)
        temporal_factor = np.exp(-time_days / self.temporal_scale)

        # 5. Redshift dependence
        redshift_factor = (1 + redshift) ** self.redshift_power

        # Combine factors to get the effective cross-section
        sigma_qvd = sigma_base * wavelength_factor * plasma_factor * temporal_factor * redshift_factor * host_environment

        # --- Optical Depth and Dimming ---

        # Path length for scattering (a fraction of the supernova radius)
        path_length_cm = plasma_state.get('radius_cm', 1e14) * 0.1

        # Calculate optical depth (tau)
        tau = sigma_qvd * density * path_length_cm

        # Clamp tau to avoid numerical issues
        tau = np.clip(tau, 0, 10)

        # Calculate transmission
        transmission = np.exp(-tau)

        # Avoid log(0) issues
        if transmission < 1e-10:
            transmission = 1e-10

        # Convert transmission to dimming in magnitudes
        qvd_dimming_mag = -2.5 * np.log10(transmission)

        return qvd_dimming_mag, tau
