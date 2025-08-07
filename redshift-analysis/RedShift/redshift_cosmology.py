#!/usr/bin/env python3
"""
Enhanced QVD Cosmology Module
============================

Cosmological calculations for QVD redshift analysis with numerical stability.
Implements standard cosmology without dark energy acceleration with comprehensive
bounds enforcement and error handling.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
from scipy.integrate import quad
from typing import Union, Dict, Tuple, Optional
import logging

from numerical_safety import (
    safe_power, safe_log10, safe_exp, safe_divide, safe_sqrt,
    validate_finite, clamp_to_range
)
from redshift_physics import RedshiftBoundsEnforcer

logger = logging.getLogger(__name__)


class EnhancedQVDCosmology:
    """
    Enhanced QVD cosmological model without dark energy.
    
    Implements standard Hubble expansion with QVD modifications
    but no cosmological acceleration. Includes comprehensive
    numerical stability and bounds enforcement.
    """
    
    def __init__(self, hubble_constant: float = 70.0, enable_bounds_checking: bool = True):
        """
        Initialize enhanced QVD cosmology.
        
        Parameters:
        -----------
        hubble_constant : float
            Hubble constant in km/s/Mpc
        enable_bounds_checking : bool
            Enable comprehensive bounds checking
        """
        # Initialize bounds enforcer
        if enable_bounds_checking:
            self.bounds_enforcer = RedshiftBoundsEnforcer()
        else:
            self.bounds_enforcer = None
        
        # Enforce Hubble constant bounds
        self.H0 = clamp_to_range(hubble_constant, 50.0, 100.0)  # Reasonable range
        if abs(hubble_constant - self.H0) > 1e-6:
            logger.warning(f"Hubble constant {hubble_constant} clamped to {self.H0}")
        
        # Physical constants
        self.c = 299792.458  # km/s (speed of light)
        
        # Cosmological parameters (matter-dominated universe)
        self.omega_m = 1.0  # Matter density (no dark energy)
        self.omega_lambda = 0.0  # Dark energy density (zero)
        self.omega_k = 0.0  # Curvature (flat universe)
        
        logger.info(f"Enhanced QVD Cosmology initialized: H0={self.H0:.1f} km/s/Mpc")
    
    def hubble_distance(self) -> float:
        """
        Calculate Hubble distance with validation.
        
        Returns:
        --------
        float
            Hubble distance in Mpc
        """
        d_H = safe_divide(self.c, self.H0, min_denominator=1.0)
        return validate_finite(d_H, "hubble_distance", replace_with=4000.0)
    
    def comoving_distance(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate comoving distance for given redshift with numerical safety.
        
        Uses matter-dominated universe (no dark energy).
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Comoving distance in Mpc (guaranteed finite and bounded)
        """
        # Enforce redshift bounds
        if self.bounds_enforcer:
            safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "comoving_distance")
        else:
            safe_redshift = redshift
        
        if isinstance(safe_redshift, (int, float)):
            if safe_redshift < 0.1:
                # Linear regime with safe operations
                distance = safe_divide(self.c * safe_redshift, self.H0, min_denominator=1.0)
            else:
                # Include relativistic corrections (matter-dominated) with safe operations
                # Simple approximation: D_C = (c/H0) * (z + 0.5*z^2)
                z_squared = safe_power(safe_redshift, 2.0)
                distance = safe_divide(self.c, self.H0, min_denominator=1.0) * (safe_redshift + 0.5 * z_squared)
        else:
            # Array input with safe operations
            result = np.zeros_like(safe_redshift)
            linear_mask = safe_redshift < 0.1
            nonlinear_mask = ~linear_mask
            
            # Linear regime
            if np.any(linear_mask):
                result[linear_mask] = safe_divide(
                    self.c * safe_redshift[linear_mask], 
                    self.H0, 
                    min_denominator=1.0
                )
            
            # Nonlinear regime
            if np.any(nonlinear_mask):
                z_nl = safe_redshift[nonlinear_mask]
                z_nl_squared = safe_power(z_nl, 2.0)
                result[nonlinear_mask] = safe_divide(
                    self.c, self.H0, min_denominator=1.0
                ) * (z_nl + 0.5 * z_nl_squared)
            
            distance = result
        
        # Validate and bound the result
        distance = validate_finite(distance, "comoving_distance", replace_with=100.0)
        
        if self.bounds_enforcer:
            distance = self.bounds_enforcer.enforce_distance(distance, "comoving_result")
        
        return distance
    
    def angular_diameter_distance(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate angular diameter distance with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Angular diameter distance in Mpc
        """
        # Get comoving distance (already bounds-checked)
        d_comoving = self.comoving_distance(redshift)
        
        # Enforce redshift bounds for division
        if self.bounds_enforcer:
            safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "angular_diameter")
        else:
            safe_redshift = redshift
        
        # Angular diameter distance with safe division
        d_angular = safe_divide(d_comoving, 1 + safe_redshift, min_denominator=1.0)
        
        # Validate result
        d_angular = validate_finite(d_angular, "angular_diameter_distance", replace_with=100.0)
        
        return d_angular
    
    def luminosity_distance(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate luminosity distance with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Luminosity distance in Mpc
        """
        # Get comoving distance (already bounds-checked)
        d_comoving = self.comoving_distance(redshift)
        
        # Enforce redshift bounds
        if self.bounds_enforcer:
            safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "luminosity_distance")
        else:
            safe_redshift = redshift
        
        # Luminosity distance with safe operations
        d_luminosity = d_comoving * (1 + safe_redshift)
        
        # Validate result
        d_luminosity = validate_finite(d_luminosity, "luminosity_distance", replace_with=100.0)
        
        if self.bounds_enforcer:
            d_luminosity = self.bounds_enforcer.enforce_distance(d_luminosity, "luminosity_result")
        
        return d_luminosity
    
    def distance_modulus(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate distance modulus with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Distance modulus in magnitudes
        """
        # Get luminosity distance (already bounds-checked)
        d_lum = self.luminosity_distance(redshift)
        
        # Distance modulus with safe operations
        # μ = 5 * log10(d_L * 10^6 / 10)
        d_lum_cm = d_lum * 1e6  # Convert Mpc to pc
        distance_modulus = 5 * safe_log10(safe_divide(d_lum_cm, 10.0, min_denominator=1.0))
        
        # Validate and bound result
        distance_modulus = validate_finite(distance_modulus, "distance_modulus", replace_with=35.0)
        
        # Apply reasonable bounds to distance modulus
        if isinstance(distance_modulus, (int, float)):
            distance_modulus = clamp_to_range(distance_modulus, 20.0, 50.0)
        else:
            distance_modulus = np.clip(distance_modulus, 20.0, 50.0)
        
        return distance_modulus
    
    def lambda_cdm_distance(self, 
                           redshift: Union[float, np.ndarray],
                           omega_m: float = 0.3,
                           omega_lambda: float = 0.7) -> Union[float, np.ndarray]:
        """
        Calculate ΛCDM luminosity distance for comparison with numerical safety.
        
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
        # Enforce parameter bounds
        omega_m = clamp_to_range(omega_m, 0.1, 1.0)
        omega_lambda = clamp_to_range(omega_lambda, 0.0, 1.0)
        
        # Enforce redshift bounds
        if self.bounds_enforcer:
            safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "lambda_cdm")
        else:
            safe_redshift = redshift
        
        def integrand(z):
            """ΛCDM integrand with numerical safety"""
            denominator = omega_m * safe_power(1 + z, 3.0) + omega_lambda
            return safe_divide(1.0, safe_sqrt(denominator), min_denominator=1e-10)
        
        if isinstance(safe_redshift, (int, float)):
            if safe_redshift < 0.01:
                # Linear regime
                integral = safe_redshift
            else:
                try:
                    integral, _ = quad(integrand, 0, safe_redshift, limit=100)
                    integral = validate_finite(integral, "lambda_cdm_integral", replace_with=safe_redshift)
                except:
                    # Fallback to simple approximation
                    integral = safe_redshift * (1 + 0.5 * safe_redshift)
                    logger.warning("ΛCDM integration failed, using approximation")
            
            d_comoving = safe_divide(self.c, self.H0, min_denominator=1.0) * integral
            d_luminosity = d_comoving * (1 + safe_redshift)
        else:
            # Array input
            result = np.zeros_like(safe_redshift)
            
            for i, z in enumerate(safe_redshift):
                if z < 0.01:
                    integral = z
                else:
                    try:
                        integral, _ = quad(integrand, 0, z, limit=100)
                        integral = validate_finite(integral, f"lambda_cdm_integral_{i}", replace_with=z)
                    except:
                        integral = z * (1 + 0.5 * z)
                
                d_comoving = safe_divide(self.c, self.H0, min_denominator=1.0) * integral
                result[i] = d_comoving * (1 + z)
            
            d_luminosity = result
        
        # Validate and bound result
        d_luminosity = validate_finite(d_luminosity, "lambda_cdm_distance", replace_with=100.0)
        
        if self.bounds_enforcer:
            d_luminosity = self.bounds_enforcer.enforce_distance(d_luminosity, "lambda_cdm_result")
        
        return d_luminosity
    
    def lookback_time(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate lookback time with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Lookback time in Gyr
        """
        # Enforce redshift bounds
        if self.bounds_enforcer:
            safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "lookback_time")
        else:
            safe_redshift = redshift
        
        # Simple approximation for matter-dominated universe with safe operations
        # t_lookback ≈ (2/3H0) * [1 - (1+z)^(-3/2)]
        
        hubble_time = safe_divide(1.0, self.H0 * 1.022e-12, min_denominator=1e-15)  # Convert to Gyr
        
        if isinstance(safe_redshift, (int, float)):
            factor = 1.0 - safe_power(1 + safe_redshift, -1.5)
            lookback = (2.0/3.0) * hubble_time * factor
        else:
            factor = 1.0 - safe_power(1 + safe_redshift, -1.5)
            lookback = (2.0/3.0) * hubble_time * factor
        
        # Validate and bound result
        lookback = validate_finite(lookback, "lookback_time", replace_with=1.0)
        
        # Apply reasonable bounds (0 to age of universe)
        if isinstance(lookback, (int, float)):
            lookback = clamp_to_range(lookback, 0.0, 20.0)  # 0 to 20 Gyr
        else:
            lookback = np.clip(lookback, 0.0, 20.0)
        
        return lookback
    
    def age_of_universe(self, redshift: Union[float, np.ndarray] = 0) -> Union[float, np.ndarray]:
        """
        Calculate age of universe at given redshift with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z (default: 0 for present age)
            
        Returns:
        --------
        float or array
            Age of universe in Gyr
        """
        # Enforce redshift bounds
        if self.bounds_enforcer:
            safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "age_universe")
        else:
            safe_redshift = redshift
        
        # Matter-dominated universe age with safe operations
        # t = (2/3H0) * (1+z)^(-3/2)
        hubble_time = safe_divide(1.0, self.H0 * 1.022e-12, min_denominator=1e-15)  # Convert to Gyr
        
        if isinstance(safe_redshift, (int, float)):
            age = (2.0/3.0) * hubble_time * safe_power(1 + safe_redshift, -1.5)
        else:
            age = (2.0/3.0) * hubble_time * safe_power(1 + safe_redshift, -1.5)
        
        # Validate and bound result
        age = validate_finite(age, "age_universe", replace_with=10.0)
        
        # Apply reasonable bounds
        if isinstance(age, (int, float)):
            age = clamp_to_range(age, 0.1, 20.0)  # 0.1 to 20 Gyr
        else:
            age = np.clip(age, 0.1, 20.0)
        
        return age
    
    def critical_density(self, redshift: Union[float, np.ndarray] = 0) -> Union[float, np.ndarray]:
        """
        Calculate critical density of the universe with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Critical density in g/cm³
        """
        # Enforce redshift bounds
        if self.bounds_enforcer:
            safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "critical_density")
        else:
            safe_redshift = redshift
        
        # Critical density: ρ_c = 3H²/(8πG) with safe operations
        G = 6.674e-8  # cm³/g/s² (gravitational constant)
        H_z = self.H0 * 1.022e-12 * safe_power(1 + safe_redshift, 1.5)  # H(z) for matter-dominated
        
        rho_c = safe_divide(3 * safe_power(H_z, 2.0), 8 * np.pi * G, min_denominator=1e-20)
        
        # Validate and bound result
        rho_c = validate_finite(rho_c, "critical_density", replace_with=1e-29)
        
        # Apply reasonable bounds
        if isinstance(rho_c, (int, float)):
            rho_c = clamp_to_range(rho_c, 1e-35, 1e-25)
        else:
            rho_c = np.clip(rho_c, 1e-35, 1e-25)
        
        return rho_c
    
    def validate_cosmological_parameters(self) -> Dict:
        """
        Validate cosmological parameters and consistency.
        
        Returns:
        --------
        dict
            Validation results
        """
        validation = {
            'hubble_constant': self.H0,
            'hubble_distance_Mpc': self.hubble_distance(),
            'age_universe_Gyr': self.age_of_universe(0),
            'critical_density_g_cm3': self.critical_density(0),
            'omega_total': self.omega_m + self.omega_lambda + self.omega_k,
            'flat_universe': abs(self.omega_k) < 1e-6,
            'matter_dominated': self.omega_m > 0.9,
            'no_dark_energy': self.omega_lambda < 0.1,
            'parameters_valid': True
        }
        
        # Check parameter consistency
        if abs(validation['omega_total'] - 1.0) > 0.1:
            validation['parameters_valid'] = False
            logger.warning("Cosmological parameters may be inconsistent")
        
        if validation['age_universe_Gyr'] < 10.0 or validation['age_universe_Gyr'] > 20.0:
            logger.warning(f"Universe age {validation['age_universe_Gyr']:.1f} Gyr may be unrealistic")
        
        return validation
    
    def get_cosmological_parameters(self) -> Dict:
        """
        Get current cosmological parameters.
        
        Returns:
        --------
        dict
            Dictionary of cosmological parameters
        """
        return {
            'hubble_constant': self.H0,
            'omega_matter': self.omega_m,
            'omega_lambda': self.omega_lambda,
            'omega_curvature': self.omega_k,
            'speed_of_light': self.c,
            'bounds_enforced': self.bounds_enforcer is not None,
            'model_type': 'matter_dominated_no_dark_energy'
        }