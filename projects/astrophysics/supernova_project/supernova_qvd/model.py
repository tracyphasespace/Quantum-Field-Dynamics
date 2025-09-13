#!/usr/bin/env python3
"""
Core Model for Supernova QVD Scattering
=======================================

Implements the E144-scaled QVD scattering model for supernova light curves.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
import logging
from typing import Dict, Tuple

from .parameters import E144ExperimentalData, SupernovaParameters
from .numerical_safety import (
    safe_power, safe_log10, safe_exp, safe_divide, validate_finite
)
from .physical_bounds import (
    BoundsEnforcer, create_safe_plasma_state, create_safe_scattering_results
)

logger = logging.getLogger(__name__)

class E144ScaledQVDModel:
    """
    QVD scattering model scaled from SLAC E144 experimental results.

    Implements the physics chain:
    E144 Laboratory → QVD Theory → Supernova Conditions → Luminance Curves
    """

    def __init__(self, e144_data: E144ExperimentalData, sn_params: SupernovaParameters):
        self.e144 = e144_data
        self.sn = sn_params
        self._bounds_enforcer = BoundsEnforcer()

        # Calculate scaling factors
        self.intensity_ratio = self.sn.peak_luminosity_erg_s / (4 * np.pi * self.sn.initial_radius_cm**2)
        self.intensity_ratio /= self.e144.laser_intensity_W_cm2 * 1e-7  # Convert W/cm^2 to erg/cm^2/s

        logger.info(f"E144-to-supernova intensity scaling: {self.intensity_ratio:.2e}")

    def calculate_qvd_cross_section(self, wavelength_nm: float,
                                   plasma_density_cm3: float,
                                   intensity_erg_cm2_s: float,
                                   time_days: float) -> float:
        """
        Calculate QVD scattering cross-section scaled from E144 results.
        """
        safe_wavelength = self._bounds_enforcer.enforce_wavelength(wavelength_nm, "qvd_wavelength")
        safe_density = self._bounds_enforcer.enforce_plasma_density(plasma_density_cm3, "qvd_density")
        safe_intensity = self._bounds_enforcer.enforce_intensity(intensity_erg_cm2_s, "qvd_intensity")
        safe_time = self._bounds_enforcer.enforce_time(time_days, "qvd_time")

        sigma_base = self._bounds_enforcer.enforce_cross_section(
            self.e144.photon_photon_cross_section_cm2, "base_cross_section"
        )

        wavelength_ratio = safe_divide(safe_wavelength, self.e144.laser_wavelength_nm)
        wavelength_factor = safe_power(wavelength_ratio, self.sn.wavelength_dependence_alpha)
        wavelength_factor = self._bounds_enforcer.enforce_scaling_factor(
            wavelength_factor, "wavelength_scaling"
        )

        density_ratio = safe_divide(safe_density, 1e20)
        plasma_factor = safe_power(density_ratio, 0.5)
        plasma_enhancement_raw = 1 + self.sn.plasma_enhancement_factor * plasma_factor
        plasma_enhancement = self._bounds_enforcer.enforce_scaling_factor(
            plasma_enhancement_raw, "plasma_enhancement"
        )

        intensity_threshold = self.e144.nonlinear_threshold_W_cm2 * 1e-7
        intensity_threshold = max(intensity_threshold, 1e-20)
        if safe_intensity > intensity_threshold:
            intensity_ratio = safe_divide(safe_intensity, intensity_threshold)
            intensity_factor = safe_power(intensity_ratio, self.sn.fluence_nonlinearity_gamma)
        else:
            intensity_factor = 1.0
        intensity_factor = self._bounds_enforcer.enforce_scaling_factor(
            intensity_factor, "intensity_scaling"
        )

        time_scaling = safe_divide(safe_time, 100.0)
        expansion_base = 1.0 + time_scaling
        expansion_factor = safe_power(expansion_base, -1.0)
        expansion_factor = self._bounds_enforcer.enforce_scaling_factor(
            expansion_factor, "expansion_scaling"
        )

        sigma_qvd = (sigma_base * wavelength_factor * plasma_enhancement *
                    intensity_factor * expansion_factor)

        sigma_qvd = validate_finite(sigma_qvd, "qvd_cross_section", replace_with=sigma_base)
        sigma_qvd = self._bounds_enforcer.enforce_cross_section(sigma_qvd, "final_qvd_cross_section")

        return float(sigma_qvd)

    def calculate_intrinsic_luminosity(self, time_days: float) -> float:
        """
        Calculates the intrinsic luminosity of the supernova at a given time
        using an empirical Type Ia model.
        """
        if time_days < -10:
            L_intrinsic = 0
        elif time_days < 0:
            L_intrinsic = self.sn.peak_luminosity_erg_s * (1 + time_days/10)**3
        elif time_days < 20:
            L_intrinsic = self.sn.peak_luminosity_erg_s * np.exp(-time_days/15)
        else:
            L_intrinsic = self.sn.peak_luminosity_erg_s * np.exp(-20/15) * np.exp(-(time_days-20)/77)

        return max(L_intrinsic, 0)

    def calculate_plasma_evolution(self, time_days: float, intrinsic_luminosity: float) -> Dict[str, float]:
        """
        Calculate plasma properties as function of time after explosion.
        """
        safe_time_days = self._bounds_enforcer.enforce_time(time_days, "evolution_time")
        time_seconds = safe_time_days * 86400

        if safe_time_days < 0:
            radius_cm = self.sn.initial_radius_cm
        else:
            radius_cm = self.sn.initial_radius_cm + self.sn.expansion_velocity_cm_s * time_seconds
        radius_cm = self._bounds_enforcer.enforce_radius(radius_cm, "plasma_radius")

        radius_ratio = safe_divide(radius_cm, self.sn.initial_radius_cm, min_denominator=1e6)
        volume_ratio = safe_power(radius_ratio, 3.0)
        electron_density_cm3 = safe_divide(
            self.sn.initial_electron_density_cm3, volume_ratio, min_denominator=1e-10
        )
        electron_density_cm3 = self._bounds_enforcer.enforce_plasma_density(
            electron_density_cm3, "electron_density"
        )

        temperature_scaling = safe_power(radius_ratio, -2.0)
        temperature_K = self.sn.initial_temperature_K * temperature_scaling
        temperature_K = self._bounds_enforcer.enforce_temperature(
            temperature_K, "plasma_temperature"
        )

        plasma_state = create_safe_plasma_state(
            radius_cm=radius_cm,
            electron_density_cm3=electron_density_cm3,
            temperature_K=temperature_K,
            luminosity_erg_s=intrinsic_luminosity,
            bounds_enforcer=self._bounds_enforcer
        )

        for key, value in plasma_state.items():
            plasma_state[key] = validate_finite(value, f"plasma_{key}", replace_with=1e10)

        return plasma_state

    def calculate_spectral_scattering(self, wavelength_nm: float,
                                    time_days: float) -> Dict[str, float]:
        """
        Calculate wavelength-dependent scattering at given time.
        """
        safe_wavelength = self._bounds_enforcer.enforce_wavelength(wavelength_nm, "scattering_wavelength")
        safe_time = self._bounds_enforcer.enforce_time(time_days, "scattering_time")

        intrinsic_luminosity = self.calculate_intrinsic_luminosity(safe_time)
        plasma = self.calculate_plasma_evolution(safe_time, intrinsic_luminosity)

        sigma_qvd = self.calculate_qvd_cross_section(
            safe_wavelength,
            plasma['electron_density_cm3'],
            plasma['intensity_erg_cm2_s'],
            safe_time
        )

        path_length_fraction = 0.1
        path_length_cm = path_length_fraction * plasma['radius_cm']
        optical_depth = sigma_qvd * plasma['electron_density_cm3'] * path_length_cm
        optical_depth = self._bounds_enforcer.enforce_optical_depth(
            optical_depth, "spectral_optical_depth"
        )

        transmission = safe_exp(-optical_depth)
        transmission = self._bounds_enforcer.enforce_transmission(
            transmission, "spectral_transmission"
        )

        dimming_magnitudes = -2.5 * safe_log10(transmission)
        dimming_magnitudes = self._bounds_enforcer.enforce_dimming_magnitude(
            dimming_magnitudes, "spectral_dimming"
        )

        scattering_results = create_safe_scattering_results(
            qvd_cross_section_cm2=sigma_qvd,
            optical_depth=optical_depth,
            transmission=transmission,
            dimming_magnitudes=dimming_magnitudes,
            bounds_enforcer=self._bounds_enforcer
        )

        scattering_results['plasma_conditions'] = plasma
        scattering_results['intrinsic_luminosity'] = intrinsic_luminosity

        for key, value in scattering_results.items():
            if key != 'plasma_conditions':
                scattering_results[key] = validate_finite(
                    value, f"scattering_{key}", replace_with=1e-30
                )
        return scattering_results
