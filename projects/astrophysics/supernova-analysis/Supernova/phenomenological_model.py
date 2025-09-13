#!/usr/bin/env python3
"""
Phenomenological Supernova Model for Comparison
==============================================

Simple phenomenological model for supernova light curves to compare
against the E144-scaled QVD model for regression testing.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PhenomenologicalParameters:
    """Parameters for phenomenological supernova model"""
    
    # Standard Type Ia parameters
    peak_absolute_magnitude: float = -19.3      # Peak absolute magnitude
    rise_time_days: float = 19.0                # Time to peak from explosion
    decline_rate_15_days: float = 1.1           # Δm15 parameter
    
    # Color evolution parameters
    peak_b_minus_v: float = 0.0                 # B-V color at peak
    color_evolution_rate: float = 0.02          # Color change per day
    
    # Wavelength-dependent parameters
    extinction_coefficient: float = 0.1         # Interstellar extinction
    wavelength_dependence: float = -1.0         # λ^α dependence
    
    # Distance-dependent parameters (for comparison with QVD)
    hubble_constant: float = 70.0               # km/s/Mpc
    peculiar_velocity_dispersion: float = 300.0 # km/s


class PhenomenologicalModel:
    """
    Simple phenomenological supernova model for comparison with QVD model.
    
    This model represents the "standard" approach without QVD physics,
    using empirical relationships and standard cosmology.
    """
    
    def __init__(self, params: PhenomenologicalParameters = None):
        """
        Initialize phenomenological model.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params or PhenomenologicalParameters()
    
    def calculate_intrinsic_light_curve(self, time_days: np.ndarray) -> np.ndarray:
        """
        Calculate intrinsic supernova light curve (no distance effects).
        
        Args:
            time_days: Time array in days since explosion
            
        Returns:
            Absolute magnitude array
        """
        # Standard Type Ia light curve template
        magnitudes = np.zeros_like(time_days)
        
        for i, t in enumerate(time_days):
            if t < -10:
                # Pre-explosion (not visible)
                magnitudes[i] = 0.0  # Very faint
            elif t < 0:
                # Rise phase - exponential rise
                rise_factor = np.exp(t / 5.0)  # 5-day e-folding time
                magnitudes[i] = self.params.peak_absolute_magnitude + 5.0 * (1 - rise_factor)
            elif t < self.params.rise_time_days:
                # Continued rise to peak
                phase = t / self.params.rise_time_days
                magnitudes[i] = self.params.peak_absolute_magnitude + 2.0 * (1 - phase)**2
            elif t < 40:
                # Early decline phase
                days_past_peak = t - self.params.rise_time_days
                decline = self.params.decline_rate_15_days * (days_past_peak / 15.0)
                magnitudes[i] = self.params.peak_absolute_magnitude + decline
            else:
                # Late decline (Ni-56 decay)
                days_past_peak = t - self.params.rise_time_days
                early_decline = self.params.decline_rate_15_days * (25.0 / 15.0)  # Decline at day 40
                late_decline = (days_past_peak - 25.0) / 77.0  # Ni-56 decay timescale
                magnitudes[i] = self.params.peak_absolute_magnitude + early_decline + late_decline
        
        return magnitudes
    
    def calculate_wavelength_dependence(self, wavelength_nm: float, 
                                      reference_wavelength_nm: float = 550.0) -> float:
        """
        Calculate wavelength-dependent correction factor.
        
        Args:
            wavelength_nm: Observation wavelength
            reference_wavelength_nm: Reference wavelength (V-band)
            
        Returns:
            Wavelength correction factor in magnitudes
        """
        # Simple power-law wavelength dependence
        wavelength_ratio = wavelength_nm / reference_wavelength_nm
        correction = -2.5 * np.log10(wavelength_ratio**self.params.wavelength_dependence)
        
        # Add extinction correction
        extinction_factor = self.params.extinction_coefficient * (550.0 / wavelength_nm - 1.0)
        
        return correction + extinction_factor
    
    def calculate_distance_modulus(self, distance_Mpc: float) -> float:
        """
        Calculate standard distance modulus.
        
        Args:
            distance_Mpc: Distance in megaparsecs
            
        Returns:
            Distance modulus in magnitudes
        """
        # Standard distance modulus formula
        distance_modulus = 5.0 * np.log10(distance_Mpc * 1e6 / 10.0)
        return distance_modulus
    
    def generate_light_curve(self, distance_Mpc: float,
                            wavelength_nm: float,
                            time_range_days: Tuple[float, float] = (-20, 100),
                            time_resolution_days: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate complete supernova light curve.
        
        Args:
            distance_Mpc: Distance to supernova
            wavelength_nm: Observation wavelength
            time_range_days: Time range for observation
            time_resolution_days: Time step
            
        Returns:
            Dictionary with light curve data
        """
        # Time array
        time_days = np.arange(time_range_days[0], time_range_days[1], time_resolution_days)
        
        # Calculate intrinsic light curve
        absolute_magnitudes = self.calculate_intrinsic_light_curve(time_days)
        
        # Apply wavelength correction
        wavelength_correction = self.calculate_wavelength_dependence(wavelength_nm)
        
        # Apply distance modulus
        distance_modulus = self.calculate_distance_modulus(distance_Mpc)
        
        # Calculate apparent magnitudes
        apparent_magnitudes = absolute_magnitudes + distance_modulus + wavelength_correction
        
        # Convert to luminosity (for comparison with QVD model)
        # L = L0 * 10^(-0.4 * (M - M0))
        reference_luminosity = 1e43  # erg/s
        reference_magnitude = -19.3
        
        luminosity_erg_s = reference_luminosity * 10**(-0.4 * (absolute_magnitudes - reference_magnitude))
        
        return {
            'time_days': time_days,
            'absolute_magnitudes': absolute_magnitudes,
            'apparent_magnitudes': apparent_magnitudes,
            'luminosity_erg_s': luminosity_erg_s,
            'distance_Mpc': distance_Mpc,
            'wavelength_nm': wavelength_nm,
            'distance_modulus': distance_modulus,
            'wavelength_correction': wavelength_correction
        }
    
    def generate_multi_wavelength_curves(self, distance_Mpc: float,
                                       wavelengths_nm: List[float] = [400, 500, 600, 700, 800],
                                       time_range_days: Tuple[float, float] = (-20, 100)) -> Dict:
        """
        Generate multi-wavelength light curves.
        
        Args:
            distance_Mpc: Distance to supernova
            wavelengths_nm: List of wavelengths
            time_range_days: Time range
            
        Returns:
            Dictionary with multi-wavelength curves
        """
        curves = {}
        
        for wavelength in wavelengths_nm:
            curves[f'{wavelength}nm'] = self.generate_light_curve(
                distance_Mpc, wavelength, time_range_days
            )
        
        # Calculate color evolution if B and R bands are present
        if 400 in wavelengths_nm and 700 in wavelengths_nm:
            blue_curve = curves['400nm']
            red_curve = curves['700nm']
            
            color_evolution = (blue_curve['apparent_magnitudes'] - 
                             red_curve['apparent_magnitudes'])
            
            curves['color_evolution'] = {
                'time_days': blue_curve['time_days'],
                'B_minus_R': color_evolution
            }
        
        return curves
    
    def calculate_hubble_diagram_data(self, distances_Mpc: List[float],
                                    wavelength_nm: float = 500.0) -> Dict[str, np.ndarray]:
        """
        Calculate Hubble diagram data for comparison.
        
        Args:
            distances_Mpc: List of distances
            wavelength_nm: Observation wavelength
            
        Returns:
            Dictionary with Hubble diagram data
        """
        distances = np.array(distances_Mpc)
        peak_magnitudes = []
        redshifts = []
        
        for distance in distances:
            # Generate light curve
            curve = self.generate_light_curve(distance, wavelength_nm, (-20, 50))
            
            # Find peak magnitude (most negative)
            peak_idx = np.argmin(curve['apparent_magnitudes'])
            peak_mag = curve['apparent_magnitudes'][peak_idx]
            peak_magnitudes.append(peak_mag)
            
            # Calculate redshift (Hubble law)
            velocity = self.params.hubble_constant * distance  # km/s
            redshift = velocity / 299792.458  # z = v/c (km/s) / (km/s)
            redshifts.append(redshift)
        
        return {
            'distances_Mpc': distances,
            'redshifts': np.array(redshifts),
            'peak_magnitudes': np.array(peak_magnitudes),
            'distance_moduli': 5.0 * np.log10(distances * 1e6 / 10.0)
        }