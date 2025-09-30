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
    Enforces physical bounds on calculations with logging and warnings.
    """

    def __init__(self, bounds: PhysicalBounds = None):
        """
        Initialize bounds enforcer.

        Args:
            bounds: PhysicalBounds instance (uses default if None)
        """
        self.bounds = bounds or PhysicalBounds()
        self._violation_counts = {}

    def enforce_plasma_density(self, density: Union[float, np.ndarray],
                              name: str = "plasma_density") -> Union[float, np.ndarray]:
        """
        Enforce bounds on plasma density.

        Args:
            density: Plasma density in cm⁻³
            name: Name for logging

        Returns:
            Bounded plasma density
        """
        return self._enforce_bounds(
            density,
            self.bounds.MIN_PLASMA_DENSITY,
            self.bounds.MAX_PLASMA_DENSITY,
            name
        )

    def enforce_temperature(self, temperature: Union[float, np.ndarray],
                           name: str = "temperature") -> Union[float, np.ndarray]:
        """
        Enforce bounds on temperature.

        Args:
            temperature: Temperature in K
            name: Name for logging

        Returns:
            Bounded temperature
        """
        return self._enforce_bounds(
            temperature,
            self.bounds.MIN_TEMPERATURE,
            self.bounds.MAX_TEMPERATURE,
            name
        )

    def enforce_radius(self, radius: Union[float, np.ndarray],
                      name: str = "radius") -> Union[float, np.ndarray]:
        """
        Enforce bounds on radius.

        Args:
            radius: Radius in cm
            name: Name for logging

        Returns:
            Bounded radius
        """
        return self._enforce_bounds(
            radius,
            self.bounds.MIN_RADIUS,
            self.bounds.MAX_RADIUS,
            name
        )

    def enforce_luminosity(self, luminosity: Union[float, np.ndarray],
                          name: str = "luminosity") -> Union[float, np.ndarray]:
        """
        Enforce bounds on luminosity.

        Args:
            luminosity: Luminosity in erg/s
            name: Name for logging

        Returns:
            Bounded luminosity
        """
        return self._enforce_bounds(
            luminosity,
            self.bounds.MIN_LUMINOSITY,
            self.bounds.MAX_LUMINOSITY,
            name
        )

    def enforce_intensity(self, intensity: Union[float, np.ndarray],
                         name: str = "intensity") -> Union[float, np.ndarray]:
        """
        Enforce bounds on intensity.

        Args:
            intensity: Intensity in erg/cm²/s
            name: Name for logging

        Returns:
            Bounded intensity
        """
        return self._enforce_bounds(
            intensity,
            self.bounds.MIN_INTENSITY,
            self.bounds.MAX_INTENSITY,
            name
        )

    def enforce_cross_section(self, cross_section: Union[float, np.ndarray],
                             name: str = "cross_section") -> Union[float, np.ndarray]:
        """
        Enforce bounds on scattering cross-section.

        Args:
            cross_section: Cross-section in cm²
            name: Name for logging

        Returns:
            Bounded cross-section
        """
        return self._enforce_bounds(
            cross_section,
            self.bounds.MIN_CROSS_SECTION,
            self.bounds.MAX_CROSS_SECTION,
            name
        )

    def enforce_optical_depth(self, optical_depth: Union[float, np.ndarray],
                             name: str = "optical_depth") -> Union[float, np.ndarray]:
        """
        Enforce bounds on optical depth.

        Args:
            optical_depth: Optical depth (dimensionless)
            name: Name for logging

        Returns:
            Bounded optical depth
        """
        return self._enforce_bounds(
            optical_depth,
            0.0,  # Minimum optical depth
            self.bounds.MAX_OPTICAL_DEPTH,
            name
        )

    def enforce_transmission(self, transmission: Union[float, np.ndarray],
                            name: str = "transmission") -> Union[float, np.ndarray]:
        """
        Enforce bounds on transmission probability.

        Args:
            transmission: Transmission probability [0, 1]
            name: Name for logging

        Returns:
            Bounded transmission
        """
        return self._enforce_bounds(
            transmission,
            self.bounds.MIN_TRANSMISSION,
            1.0,  # Maximum transmission
            name
        )

    def enforce_dimming_magnitude(self, dimming: Union[float, np.ndarray],
                                 name: str = "dimming_magnitude") -> Union[float, np.ndarray]:
        """
        Enforce bounds on dimming magnitude.

        Args:
            dimming: Dimming in magnitudes
            name: Name for logging

        Returns:
            Bounded dimming magnitude
        """
        return self._enforce_bounds(
            dimming,
            0.0,  # Minimum dimming (no dimming)
            self.bounds.MAX_DIMMING_MAG,
            name
        )

    def enforce_wavelength(self, wavelength: Union[float, np.ndarray],
                          name: str = "wavelength") -> Union[float, np.ndarray]:
        """
        Enforce bounds on wavelength.

        Args:
            wavelength: Wavelength in nm
            name: Name for logging

        Returns:
            Bounded wavelength
        """
        return self._enforce_bounds(
            wavelength,
            self.bounds.MIN_WAVELENGTH_NM,
            self.bounds.MAX_WAVELENGTH_NM,
            name
        )

    def enforce_time(self, time: Union[float, np.ndarray],
                    name: str = "time") -> Union[float, np.ndarray]:
        """
        Enforce bounds on time.

        Args:
            time: Time in days
            name: Name for logging

        Returns:
            Bounded time
        """
        return self._enforce_bounds(
            time,
            self.bounds.MIN_TIME_DAYS,
            self.bounds.MAX_TIME_DAYS,
            name
        )

    def enforce_scaling_factor(self, factor: Union[float, np.ndarray],
                              name: str = "scaling_factor") -> Union[float, np.ndarray]:
        """
        Enforce bounds on scaling factors.

        Args:
            factor: Scaling factor (dimensionless)
            name: Name for logging

        Returns:
            Bounded scaling factor
        """
        return self._enforce_bounds(
            factor,
            self.bounds.MIN_SCALING_FACTOR,
            self.bounds.MAX_SCALING_FACTOR,
            name
        )

    def _enforce_bounds(self, value: Union[float, np.ndarray],
                       min_val: float, max_val: float,
                       name: str) -> Union[float, np.ndarray]:
        """
        Internal method to enforce bounds with logging.

        Args:
            value: Value to bound
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for logging

        Returns:
            Bounded value
        """
        # Handle NaN/Inf values first
        if np.any(~np.isfinite(value)):
            logger.warning(f"Non-finite values detected in {name}, replacing with bounds")
            value = np.where(np.isfinite(value), value, (min_val + max_val) / 2)

        # Apply bounds
        bounded_value = np.clip(value, min_val, max_val)

        # Count and log violations
        if np.any(value != bounded_value):
            violation_count = np.sum(value != bounded_value)
            self._violation_counts[name] = self._violation_counts.get(name, 0) + violation_count

            logger.debug(f"Applied bounds to {violation_count} values in {name} "
                        f"[{min_val:.2e}, {max_val:.2e}]")

        return bounded_value

    def get_violation_summary(self) -> Dict[str, int]:
        """
        Get summary of bounds violations.

        Returns:
            Dictionary with violation counts by parameter name
        """
        return self._violation_counts.copy()

    def reset_violation_counts(self):
        """Reset violation counters."""
        self._violation_counts.clear()


@dataclass
class SafePlasmaState:
    """
    Plasma state with numerical safety guarantees and automatic bounds enforcement.

    All values are automatically validated and clamped to physical bounds
    during initialization and any subsequent modifications.
    """
    radius_cm: float
    electron_density_cm3: float
    temperature_K: float
    luminosity_erg_s: float
    intensity_erg_cm2_s: float
    photosphere_area_cm2: float = field(init=False)

    # Private field to store bounds enforcer
    _bounds_enforcer: BoundsEnforcer = field(default_factory=BoundsEnforcer, init=False, repr=False)

    def __post_init__(self):
        """Enforce physical bounds after initialization"""
        # Apply bounds to all parameters
        self.radius_cm = self._bounds_enforcer.enforce_radius(self.radius_cm, "plasma_radius")
        self.electron_density_cm3 = self._bounds_enforcer.enforce_plasma_density(
            self.electron_density_cm3, "electron_density"
        )
        self.temperature_K = self._bounds_enforcer.enforce_temperature(
            self.temperature_K, "plasma_temperature"
        )
        self.luminosity_erg_s = self._bounds_enforcer.enforce_luminosity(
            self.luminosity_erg_s, "plasma_luminosity"
        )
        self.intensity_erg_cm2_s = self._bounds_enforcer.enforce_intensity(
            self.intensity_erg_cm2_s, "plasma_intensity"
        )

        # Calculate derived quantities
        self.photosphere_area_cm2 = 4 * np.pi * self.radius_cm**2

    def update_radius(self, new_radius_cm: float):
        """Safely update radius with bounds checking"""
        self.radius_cm = self._bounds_enforcer.enforce_radius(new_radius_cm, "plasma_radius")
        self.photosphere_area_cm2 = 4 * np.pi * self.radius_cm**2

    def update_density(self, new_density_cm3: float):
        """Safely update density with bounds checking"""
        self.electron_density_cm3 = self._bounds_enforcer.enforce_plasma_density(
            new_density_cm3, "electron_density"
        )

    def update_temperature(self, new_temperature_K: float):
        """Safely update temperature with bounds checking"""
        self.temperature_K = self._bounds_enforcer.enforce_temperature(
            new_temperature_K, "plasma_temperature"
        )

    def is_physically_reasonable(self) -> bool:
        """Check if current state is physically reasonable"""
        return is_physically_reasonable_plasma(
            self.electron_density_cm3, self.temperature_K, self.radius_cm
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {
            'radius_cm': float(self.radius_cm),
            'electron_density_cm3': float(self.electron_density_cm3),
            'temperature_K': float(self.temperature_K),
            'luminosity_erg_s': float(self.luminosity_erg_s),
            'intensity_erg_cm2_s': float(self.intensity_erg_cm2_s),
            'photosphere_area_cm2': float(self.photosphere_area_cm2)
        }


@dataclass
class SafeScatteringResults:
    """
    Scattering results with guaranteed finite values and automatic bounds enforcement.

    All values are automatically validated and clamped to physical bounds
    during initialization and any subsequent modifications.
    """
    qvd_cross_section_cm2: float
    optical_depth: float
    transmission: float
    dimming_magnitudes: float

    # Private field to store bounds enforcer
    _bounds_enforcer: BoundsEnforcer = field(default_factory=BoundsEnforcer, init=False, repr=False)

    def __post_init__(self):
        """Ensure all values are physically reasonable"""
        # Apply bounds checking and validation
        self.qvd_cross_section_cm2 = self._bounds_enforcer.enforce_cross_section(
            self.qvd_cross_section_cm2, "qvd_cross_section"
        )
        self.optical_depth = self._bounds_enforcer.enforce_optical_depth(
            self.optical_depth, "optical_depth"
        )
        self.transmission = self._bounds_enforcer.enforce_transmission(
            self.transmission, "transmission"
        )
        self.dimming_magnitudes = self._bounds_enforcer.enforce_dimming_magnitude(
            self.dimming_magnitudes, "dimming_magnitude"
        )

        # Validate consistency between optical depth and transmission
        self._validate_consistency()

    def _validate_consistency(self):
        """Validate physical consistency between related quantities"""
        # Check that transmission is roughly consistent with optical depth
        expected_transmission = np.exp(-self.optical_depth)
        if abs(self.transmission - expected_transmission) > 0.5:
            logger.debug(f"Transmission-optical depth inconsistency detected: "
                        f"transmission={self.transmission:.3e}, "
                        f"expected={expected_transmission:.3e}")

        # Check that dimming is roughly consistent with transmission
        if self.transmission > 0:
            expected_dimming = -2.5 * np.log10(self.transmission)
            if abs(self.dimming_magnitudes - expected_dimming) > 2.0:
                logger.debug(f"Dimming-transmission inconsistency detected: "
                            f"dimming={self.dimming_magnitudes:.3f}, "
                            f"expected={expected_dimming:.3f}")

    def update_cross_section(self, new_cross_section_cm2: float):
        """Safely update cross-section with bounds checking"""
        self.qvd_cross_section_cm2 = self._bounds_enforcer.enforce_cross_section(
            new_cross_section_cm2, "qvd_cross_section"
        )

    def update_optical_depth(self, new_optical_depth: float):
        """Safely update optical depth with bounds checking"""
        self.optical_depth = self._bounds_enforcer.enforce_optical_depth(
            new_optical_depth, "optical_depth"
        )
        self._validate_consistency()

    def update_transmission(self, new_transmission: float):
        """Safely update transmission with bounds checking"""
        self.transmission = self._bounds_enforcer.enforce_transmission(
            new_transmission, "transmission"
        )
        self._validate_consistency()

    def update_dimming(self, new_dimming_magnitudes: float):
        """Safely update dimming with bounds checking"""
        self.dimming_magnitudes = self._bounds_enforcer.enforce_dimming_magnitude(
            new_dimming_magnitudes, "dimming_magnitude"
        )
        self._validate_consistency()

    def is_physically_reasonable(self) -> bool:
        """Check if current state is physically reasonable"""
        return is_physically_reasonable_scattering(
            self.qvd_cross_section_cm2, self.optical_depth, self.transmission
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {
            'qvd_cross_section_cm2': float(self.qvd_cross_section_cm2),
            'optical_depth': float(self.optical_depth),
            'transmission': float(self.transmission),
            'dimming_magnitudes': float(self.dimming_magnitudes)
        }


def create_safe_plasma_state(radius_cm: float,
                            electron_density_cm3: float,
                            temperature_K: float,
                            luminosity_erg_s: float,
                            bounds_enforcer: BoundsEnforcer = None) -> Dict[str, float]:
    """
    Create a plasma state with enforced physical bounds.

    Args:
        radius_cm: Plasma radius in cm
        electron_density_cm3: Electron density in cm⁻³
        temperature_K: Temperature in K
        luminosity_erg_s: Luminosity in erg/s
        bounds_enforcer: BoundsEnforcer instance (creates default if None)

    Returns:
        Dictionary with bounded plasma properties (for backward compatibility)
    """
    if bounds_enforcer is None:
        bounds_enforcer = BoundsEnforcer()

    # Calculate intensity with safe operations
    safe_radius = bounds_enforcer.enforce_radius(radius_cm, "plasma_radius")
    safe_luminosity = bounds_enforcer.enforce_luminosity(luminosity_erg_s, "plasma_luminosity")
    photosphere_area = 4 * np.pi * safe_radius**2
    safe_intensity = bounds_enforcer.enforce_intensity(
        safe_luminosity / photosphere_area, "plasma_intensity"
    )

    # Create safe plasma state dataclass
    plasma_state = SafePlasmaState(
        radius_cm=radius_cm,
        electron_density_cm3=electron_density_cm3,
        temperature_K=temperature_K,
        luminosity_erg_s=luminosity_erg_s,
        intensity_erg_cm2_s=safe_intensity
    )

    # Return as dictionary for backward compatibility
    return plasma_state.to_dict()


def create_safe_plasma_state_dataclass(radius_cm: float,
                                      electron_density_cm3: float,
                                      temperature_K: float,
                                      luminosity_erg_s: float,
                                      intensity_erg_cm2_s: float) -> SafePlasmaState:
    """
    Create a SafePlasmaState dataclass with enforced physical bounds.

    Args:
        radius_cm: Plasma radius in cm
        electron_density_cm3: Electron density in cm⁻³
        temperature_K: Temperature in K
        luminosity_erg_s: Luminosity in erg/s
        intensity_erg_cm2_s: Intensity in erg/cm²/s

    Returns:
        SafePlasmaState dataclass with automatic bounds enforcement
    """
    return SafePlasmaState(
        radius_cm=radius_cm,
        electron_density_cm3=electron_density_cm3,
        temperature_K=temperature_K,
        luminosity_erg_s=luminosity_erg_s,
        intensity_erg_cm2_s=intensity_erg_cm2_s
    )


def create_safe_scattering_results(qvd_cross_section_cm2: float,
                                  optical_depth: float,
                                  transmission: float,
                                  dimming_magnitudes: float,
                                  bounds_enforcer: BoundsEnforcer = None) -> Dict[str, float]:
    """
    Create scattering results with enforced physical bounds.

    Args:
        qvd_cross_section_cm2: QVD cross-section in cm²
        optical_depth: Optical depth (dimensionless)
        transmission: Transmission probability [0, 1]
        dimming_magnitudes: Dimming in magnitudes
        bounds_enforcer: BoundsEnforcer instance (creates default if None)

    Returns:
        Dictionary with bounded scattering properties (for backward compatibility)
    """
    # Create safe scattering results dataclass
    scattering_results = SafeScatteringResults(
        qvd_cross_section_cm2=qvd_cross_section_cm2,
        optical_depth=optical_depth,
        transmission=transmission,
        dimming_magnitudes=dimming_magnitudes
    )

    # Return as dictionary for backward compatibility
    return scattering_results.to_dict()


def create_safe_scattering_results_dataclass(qvd_cross_section_cm2: float,
                                            optical_depth: float,
                                            transmission: float,
                                            dimming_magnitudes: float) -> SafeScatteringResults:
    """
    Create a SafeScatteringResults dataclass with enforced physical bounds.

    Args:
        qvd_cross_section_cm2: QVD cross-section in cm²
        optical_depth: Optical depth (dimensionless)
        transmission: Transmission probability [0, 1]
        dimming_magnitudes: Dimming in magnitudes

    Returns:
        SafeScatteringResults dataclass with automatic bounds enforcement
    """
    return SafeScatteringResults(
        qvd_cross_section_cm2=qvd_cross_section_cm2,
        optical_depth=optical_depth,
        transmission=transmission,
        dimming_magnitudes=dimming_magnitudes
    )


# Convenience functions for common bounds checking
def is_physically_reasonable_plasma(density_cm3: float,
                                   temperature_K: float,
                                   radius_cm: float) -> bool:
    """
    Check if plasma parameters are physically reasonable.

    Args:
        density_cm3: Electron density in cm⁻³
        temperature_K: Temperature in K
        radius_cm: Radius in cm

    Returns:
        True if parameters are reasonable
    """
    bounds = PhysicalBounds()

    return (bounds.MIN_PLASMA_DENSITY <= density_cm3 <= bounds.MAX_PLASMA_DENSITY and
            bounds.MIN_TEMPERATURE <= temperature_K <= bounds.MAX_TEMPERATURE and
            bounds.MIN_RADIUS <= radius_cm <= bounds.MAX_RADIUS)


def is_physically_reasonable_scattering(cross_section_cm2: float,
                                       optical_depth: float,
                                       transmission: float) -> bool:
    """
    Check if scattering parameters are physically reasonable.

    Args:
        cross_section_cm2: Cross-section in cm²
        optical_depth: Optical depth
        transmission: Transmission probability

    Returns:
        True if parameters are reasonable
    """
    bounds = PhysicalBounds()

    return (bounds.MIN_CROSS_SECTION <= cross_section_cm2 <= bounds.MAX_CROSS_SECTION and
            0.0 <= optical_depth <= bounds.MAX_OPTICAL_DEPTH and
            bounds.MIN_TRANSMISSION <= transmission <= 1.0)
