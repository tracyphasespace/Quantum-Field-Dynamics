"""
Plasma Physics Module for QFD Supernova Analysis
=================================================

Simplified implementation of the supernova plasma evolution for the demo.
This module provides the SupernovaPlasma class, which models the
physical conditions of the supernova ejecta over time.
"""

import numpy as np
from typing import Dict

class SupernovaPlasma:
    """
    Models the evolution of the supernova plasma.

    This is a simplified model for the qfd_supernova demo.
    """

    def __init__(self):
        """
        Initialize the plasma model with typical Type Ia supernova parameters.
        """
        # Physical parameters for a typical Type Ia supernova
        self.initial_radius_cm = 1e9  # ~White dwarf radius
        self.expansion_velocity_cm_s = 1e9  # ~3000 km/s
        self.initial_electron_density_cm3 = 1e24
        self.initial_temperature_K = 1e8
        self.peak_luminosity_erg_s = 1e43

    def calculate_plasma_evolution(self, time_days: float) -> Dict[str, float]:
        """
        Calculate the physical properties of the supernova plasma at a given time.

        This is a simplified model of homologous expansion and cooling.

        Parameters:
        -----------
        time_days : float
            Time in days since the supernova explosion.

        Returns:
        --------
        Dict[str, float]
            A dictionary containing the plasma properties:
            - radius_cm
            - electron_density_cm3
            - temperature_K
            - luminosity_erg_s
            - intensity_erg_cm2_s
        """
        time_seconds = time_days * 86400.0

        # --- Radius Evolution (homologous expansion) ---
        if time_days < -10: # Before explosion
             radius_cm = self.initial_radius_cm
        else:
             radius_cm = self.initial_radius_cm + self.expansion_velocity_cm_s * max(0, time_seconds)

        # --- Density Evolution (mass conservation in expanding sphere) ---
        radius_ratio = radius_cm / self.initial_radius_cm
        volume_ratio = radius_ratio ** 3
        electron_density_cm3 = self.initial_electron_density_cm3 / volume_ratio

        # --- Temperature Evolution (adiabatic cooling) ---
        temperature_K = self.initial_temperature_K / radius_ratio

        # --- Luminosity and Intensity ---
        # For simplicity, we use a fixed peak luminosity and a simple light curve shape
        # This is a rough approximation. A more detailed model would use a more
        # sophisticated light curve.
        if time_days < -10:
            luminosity_erg_s = 0
        elif time_days < 0:
            luminosity_erg_s = self.peak_luminosity_erg_s * (1 + time_days/10)**2
        elif time_days < 20:
            luminosity_erg_s = self.peak_luminosity_erg_s * np.exp(-time_days / 15.0)
        else:
            luminosity_erg_s = self.peak_luminosity_erg_s * np.exp(-20/15) * np.exp(-(time_days-20)/77)

        luminosity_erg_s = max(0, luminosity_erg_s)

        photosphere_area = 4 * np.pi * radius_cm**2
        intensity_erg_cm2_s = luminosity_erg_s / photosphere_area if photosphere_area > 0 else 0

        # Return a dictionary of the plasma state
        plasma_state = {
            'radius_cm': radius_cm,
            'electron_density_cm3': electron_density_cm3,
            'temperature_K': max(1e3, temperature_K), # Avoid zero temperature
            'luminosity_erg_s': luminosity_erg_s,
            'intensity_erg_cm2_s': intensity_erg_cm2_s,
        }

        return plasma_state
