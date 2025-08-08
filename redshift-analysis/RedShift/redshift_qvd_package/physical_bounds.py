#!/usr/bin/env python3
"""
Physical Bounds Enforcement System
=================================

Provides physically motivated bounds and constraints for QVD calculations
to prevent extreme values that could cause numerical instabilities.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
from typing import Union, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class PhysicalBounds:
    """
    Physical bounds for various quantities in QVD supernova calculations.

    All bounds are based on physical reasoning and astronomical observations.
    """

    # Optical and scattering bounds
    MAX_OPTICAL_DEPTH: float = 50.0         # Beyond this, transmission ≈ 0
    MAX_DIMMING_MAG: float = 10.0           # Reasonable astronomical limit
    MIN_TRANSMISSION: float = 1e-20         # Minimum meaningful transmission
    MAX_CROSS_SECTION: float = 1e-20        # Upper bound on cross-section (cm²)
    MIN_CROSS_SECTION: float = 1e-50        # Lower bound on cross-section (cm²)

    # Plasma properties bounds
    MIN_PLASMA_DENSITY: float = 1e10        # Minimum meaningful density (cm⁻³)
    MAX_PLASMA_DENSITY: float = 1e30        # Maximum physical density (cm⁻³)
    MIN_TEMPERATURE: float = 100.0          # Minimum temperature (K)
    MAX_TEMPERATURE: float = 1e10           # Maximum temperature (K)
    MIN_RADIUS: float = 1e6                 # Minimum radius (cm)
    MAX_RADIUS: float = 1e18                # Maximum radius (cm)

    # Energy and intensity bounds
    MIN_LUMINOSITY: float = 1e30            # Minimum luminosity (erg/s)
    MAX_LUMINOSITY: float = 1e50            # Maximum luminosity (erg/s)
    MIN_INTENSITY: float = 1e-10            # Minimum intensity (erg/cm²/s)
    MAX_INTENSITY: float = 1e30             # Maximum intensity (erg/cm²/s)

    # Wavelength bounds
    MIN_WAVELENGTH_NM: float = 100.0        # Far UV limit
    MAX_WAVELENGTH_NM: float = 10000.0      # Near IR limit

    # Time bounds
    MIN_TIME_DAYS: float = -100.0           # Pre-explosion limit
    MAX_TIME_DAYS: float = 1000.0           # Late-time limit

    # Scaling factor bounds
    MIN_SCALING_FACTOR: float = 1e-10       # Minimum scaling
    MAX_SCALING_FACTOR: float = 1e10        # Maximum scaling

    # Mathematical operation bounds
    MIN_LOG_ARGUMENT: float = 1e-30         # Minimum for log operations
    MAX_EXP_ARGUMENT: float = 700.0         # Maximum for exp operations
    MIN_EXP_ARGUMENT: float = -700.0        # Minimum for exp operations
    MAX_POWER_EXPONENT: float = 100.0       # Maximum power exponent
    MIN_POWER_BASE: float = 1e-30           # Minimum power base


class BoundsEnforcer:
    """
    Enforces physical bounds on calculations, tracking the frequency and
    magnitude of any necessary interventions.
    """

    def __init__(self, bounds: PhysicalBounds = None):
        """
        Initialize bounds enforcer.

        Args:
            bounds: PhysicalBounds instance (uses default if None)
        """
        self.bounds = bounds or PhysicalBounds()
        self.reset_violation_counts()

    def _enforce_bounds(self, value: Union[float, np.ndarray],
                       min_val: float, max_val: float,
                       name: str) -> Union[float, np.ndarray]:
        """
        Internal method to enforce bounds with detailed violation tracking.
        """
        # Handle non-finite values first
        if not np.all(np.isfinite(value)):
            non_finite_mask = ~np.isfinite(value)
            non_finite_count = np.sum(non_finite_mask)
            self._violations[name]['non_finite_count'] += non_finite_count
            logger.warning(f"Found {non_finite_count} non-finite values in {name}, replacing with average of bounds")

            # Replace non-finite values with the average of the bounds
            replacement_val = (min_val + max_val) / 2
            value = np.where(non_finite_mask, replacement_val, value)

        # Apply bounds and find where clamping occurred
        bounded_value = np.clip(value, min_val, max_val)
        clamped_mask = value != bounded_value

        if np.any(clamped_mask):
            violation_count = np.sum(clamped_mask)
            total_deviation = np.sum(np.abs(value[clamped_mask] - bounded_value[clamped_mask]))

            # Update violation statistics
            stats = self._violations[name]
            stats['count'] += violation_count
            stats['total_deviation'] += total_deviation

            logger.debug(
                f"Clamped {violation_count} values in {name} to range [{min_val:.2e}, {max_val:.2e}]. "
                f"Total deviation: {total_deviation:.2e}"
            )

        return bounded_value

    def get_violation_summary(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Get a summary of all bounds violations that have occurred.
        Returns:
            A nested dictionary with violation counts and total deviations.
        """
        return {k: v for k, v in self._violations.items() if v['count'] > 0 or v['non_finite_count'] > 0}

    def reset_violation_counts(self):
        """Reset all violation counters to zero."""
        from collections import defaultdict
        self._violations = defaultdict(lambda: {'count': 0, 'total_deviation': 0.0, 'non_finite_count': 0})

    def enforce_plasma_density(self, density: Union[float, np.ndarray],
                              name: str = "plasma_density") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            density,
            self.bounds.MIN_PLASMA_DENSITY,
            self.bounds.MAX_PLASMA_DENSITY,
            name
        )

    def enforce_temperature(self, temperature: Union[float, np.ndarray],
                           name: str = "temperature") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            temperature,
            self.bounds.MIN_TEMPERATURE,
            self.bounds.MAX_TEMPERATURE,
            name
        )

    def enforce_radius(self, radius: Union[float, np.ndarray],
                      name: str = "radius") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            radius,
            self.bounds.MIN_RADIUS,
            self.bounds.MAX_RADIUS,
            name
        )

    def enforce_luminosity(self, luminosity: Union[float, np.ndarray],
                          name: str = "luminosity") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            luminosity,
            self.bounds.MIN_LUMINOSITY,
            self.bounds.MAX_LUMINOSITY,
            name
        )

    def enforce_intensity(self, intensity: Union[float, np.ndarray],
                         name: str = "intensity") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            intensity,
            self.bounds.MIN_INTENSITY,
            self.bounds.MAX_INTENSITY,
            name
        )

    def enforce_cross_section(self, cross_section: Union[float, np.ndarray],
                             name: str = "cross_section") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            cross_section,
            self.bounds.MIN_CROSS_SECTION,
            self.bounds.MAX_CROSS_SECTION,
            name
        )

    def enforce_optical_depth(self, optical_depth: Union[float, np.ndarray],
                             name: str = "optical_depth") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            optical_depth,
            0.0,
            self.bounds.MAX_OPTICAL_DEPTH,
            name
        )

    def enforce_transmission(self, transmission: Union[float, np.ndarray],
                            name: str = "transmission") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            transmission,
            self.bounds.MIN_TRANSMISSION,
            1.0,
            name
        )

    def enforce_dimming_magnitude(self, dimming: Union[float, np.ndarray],
                                 name: str = "dimming_magnitude") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            dimming,
            0.0,
            self.bounds.MAX_DIMMING_MAG,
            name
        )

    def enforce_wavelength(self, wavelength: Union[float, np.ndarray],
                          name: str = "wavelength") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            wavelength,
            self.bounds.MIN_WAVELENGTH_NM,
            self.bounds.MAX_WAVELENGTH_NM,
            name
        )

    def enforce_time(self, time: Union[float, np.ndarray],
                    name: str = "time") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            time,
            self.bounds.MIN_TIME_DAYS,
            self.bounds.MAX_TIME_DAYS,
            name
        )

    def enforce_redshift(self, redshift: Union[float, np.ndarray],
                        name: str = "redshift") -> Union[float, np.ndarray]:
        """
        Enforce bounds on redshift.

        Args:
            redshift: Redshift z
            name: Name for logging

        Returns:
            Bounded redshift
        """
        return self._enforce_bounds(
            redshift,
            0.0,
            20.0, # A reasonable upper limit for this model
            name
        )

    def enforce_scaling_factor(self, factor: Union[float, np.ndarray],
                              name: str = "scaling_factor") -> Union[float, np.ndarray]:
        return self._enforce_bounds(
            factor,
            self.bounds.MIN_SCALING_FACTOR,
            self.bounds.MAX_SCALING_FACTOR,
            name
        )
