#!/usr/bin/env python3
"""
Model Parameters for Supernova QVD Scattering
=============================================

Dataclasses for storing physical parameters for the E144 experiment
and the supernova environment.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class E144ExperimentalData:
    """SLAC E144 experimental parameters and results"""

    # Laser parameters
    laser_wavelength_nm: float = 527.0       # Green laser
    laser_power_GW: float = 1.0              # Peak power
    laser_pulse_duration_ps: float = 1.0     # Pulse length
    laser_intensity_W_cm2: float = 1e18      # Peak intensity

    # Electron beam parameters
    electron_energy_GeV: float = 46.6        # Beam energy
    electron_density_cm3: float = 1e10       # Beam density

    # Measured results
    photon_photon_cross_section_cm2: float = 1e-30  # Approximate measured value
    nonlinear_threshold_W_cm2: float = 1e17          # Threshold for nonlinear effects
    enhancement_factor_vacuum: float = 1.0           # No plasma enhancement in E144

    # QED theoretical predictions (validated by E144)
    alpha_fine_structure: float = 1/137              # Fine structure constant
    qed_coupling_strength: float = 1e-4              # Effective QED coupling

@dataclass
class SupernovaParameters:
    """Physical parameters for supernova environment"""

    # Supernova energetics
    peak_luminosity_erg_s: float = 1e43              # Typical Type Ia luminosity
    explosion_energy_erg: float = 1e51               # Kinetic energy
    ejected_mass_g: float = 1.4 * 2e33               # Chandrasekhar mass

    # Plasma evolution parameters
    initial_radius_cm: float = 1e9                   # ~White dwarf radius
    expansion_velocity_cm_s: float = 1e9             # ~3000 km/s
    initial_electron_density_cm3: float = 1e24       # Dense plasma
    initial_temperature_K: float = 1e8               # ~10 keV

    # Spectral properties
    blackbody_temperature_K: float = 1e4             # Photosphere temperature
    wavelength_range_nm: Tuple[float, float] = (300, 800)  # Optical range

    # QVD coupling parameters (scaled from E144)
    qvd_base_coupling: float = 1e-4                  # Base QVD strength
    plasma_enhancement_factor: float = 1e6           # Plasma-mediated enhancement
    wavelength_dependence_alpha: float = -2.0        # λ^α scaling
    fluence_nonlinearity_gamma: float = 0.5          # Nonlinear fluence dependence
