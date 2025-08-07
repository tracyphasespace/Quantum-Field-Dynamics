#!/usr/bin/env python3
"""
Enhanced QVD Redshift Physics Module
===================================

Core physics implementation for QVD redshift analysis with numerical stability.
Implements wavelength-independent redshift-dependent dimming with comprehensive
bounds enforcement and error handling.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
import logging
from typing import Union, Dict, Optional
from dataclasses import dataclass

from numerical_safety import (
    safe_power, safe_log10, safe_exp, safe_divide, safe_sqrt,
    validate_finite, clamp_to_range
)
from physical_bounds import PhysicalBounds, BoundsEnforcer
from error_handling import setup_qvd_logging

logger = logging.getLogger(__name__)

@dataclass
class RedshiftBounds:
    """Physical bounds for redshift analysis parameters"""
    
    # Redshift bounds
    MIN_REDSHIFT: float = 1e-6
    MAX_REDSHIFT: float = 10.0
    
    # QVD coupling bounds
    MIN_QVD_COUPLING: float = 1e-6
    MAX_QVD_COUPLING: float = 10.0
    
    # Redshift power bounds
    MIN_REDSHIFT_POWER: float = 0.1
    MAX_REDSHIFT_POWER: float = 2.0
    
    # IGM enhancement bounds
    MIN_IGM_ENHANCEMENT: float = 0.01
    MAX_IGM_ENHANCEMENT: float = 5.0
    
    # Distance bounds (Mpc)
    MIN_DISTANCE_MPC: float = 0.1
    MAX_DISTANCE_MPC: float = 50000.0
    
    # Dimming magnitude bounds
    MIN_DIMMING_MAG: float = 0.0
    MAX_DIMMING_MAG: float = 10.0
    
    # Optical depth bounds
    MIN_OPTICAL_DEPTH: float = 1e-10
    MAX_OPTICAL_DEPTH: float = 50.0


class RedshiftBoundsEnforcer:
    """Bounds enforcement for redshift analysis"""
    
    def __init__(self):
        self.bounds = RedshiftBounds()
        self.warning_count = 0
        
    def enforce_redshift(self, redshift: Union[float, np.ndarray], 
                        context: str = "redshift") -> Union[float, np.ndarray]:
        """Enforce redshift bounds"""
        if isinstance(redshift, (int, float)):
            if redshift < self.bounds.MIN_REDSHIFT or redshift > self.bounds.MAX_REDSHIFT:
                logger.warning(f"Redshift {redshift} out of bounds in {context}, clamping")
                self.warning_count += 1
            return clamp_to_range(redshift, self.bounds.MIN_REDSHIFT, self.bounds.MAX_REDSHIFT)
        else:
            out_of_bounds = ((redshift < self.bounds.MIN_REDSHIFT) | 
                           (redshift > self.bounds.MAX_REDSHIFT))
            if np.any(out_of_bounds):
                logger.warning(f"{np.sum(out_of_bounds)} redshift values out of bounds in {context}")
                self.warning_count += 1
            return np.clip(redshift, self.bounds.MIN_REDSHIFT, self.bounds.MAX_REDSHIFT)
    
    def enforce_qvd_coupling(self, coupling: float, context: str = "qvd_coupling") -> float:
        """Enforce QVD coupling bounds"""
        if coupling < self.bounds.MIN_QVD_COUPLING or coupling > self.bounds.MAX_QVD_COUPLING:
            logger.warning(f"QVD coupling {coupling} out of bounds in {context}, clamping")
            self.warning_count += 1
        return clamp_to_range(coupling, self.bounds.MIN_QVD_COUPLING, self.bounds.MAX_QVD_COUPLING)
    
    def enforce_redshift_power(self, power: float, context: str = "redshift_power") -> float:
        """Enforce redshift power bounds"""
        if power < self.bounds.MIN_REDSHIFT_POWER or power > self.bounds.MAX_REDSHIFT_POWER:
            logger.warning(f"Redshift power {power} out of bounds in {context}, clamping")
            self.warning_count += 1
        return clamp_to_range(power, self.bounds.MIN_REDSHIFT_POWER, self.bounds.MAX_REDSHIFT_POWER)
    
    def enforce_distance(self, distance: Union[float, np.ndarray], 
                        context: str = "distance") -> Union[float, np.ndarray]:
        """Enforce distance bounds"""
        if isinstance(distance, (int, float)):
            if distance < self.bounds.MIN_DISTANCE_MPC or distance > self.bounds.MAX_DISTANCE_MPC:
                logger.warning(f"Distance {distance} Mpc out of bounds in {context}, clamping")
                self.warning_count += 1
            return clamp_to_range(distance, self.bounds.MIN_DISTANCE_MPC, self.bounds.MAX_DISTANCE_MPC)
        else:
            out_of_bounds = ((distance < self.bounds.MIN_DISTANCE_MPC) | 
                           (distance > self.bounds.MAX_DISTANCE_MPC))
            if np.any(out_of_bounds):
                logger.warning(f"{np.sum(out_of_bounds)} distance values out of bounds in {context}")
                self.warning_count += 1
            return np.clip(distance, self.bounds.MIN_DISTANCE_MPC, self.bounds.MAX_DISTANCE_MPC)
    
    def enforce_dimming(self, dimming: Union[float, np.ndarray], 
                       context: str = "dimming") -> Union[float, np.ndarray]:
        """Enforce dimming magnitude bounds"""
        if isinstance(dimming, (int, float)):
            if dimming < self.bounds.MIN_DIMMING_MAG or dimming > self.bounds.MAX_DIMMING_MAG:
                logger.warning(f"Dimming {dimming} mag out of bounds in {context}, clamping")
                self.warning_count += 1
            return clamp_to_range(dimming, self.bounds.MIN_DIMMING_MAG, self.bounds.MAX_DIMMING_MAG)
        else:
            out_of_bounds = ((dimming < self.bounds.MIN_DIMMING_MAG) | 
                           (dimming > self.bounds.MAX_DIMMING_MAG))
            if np.any(out_of_bounds):
                logger.warning(f"{np.sum(out_of_bounds)} dimming values out of bounds in {context}")
                self.warning_count += 1
            return np.clip(dimming, self.bounds.MIN_DIMMING_MAG, self.bounds.MAX_DIMMING_MAG)


class EnhancedQVDPhysics:
    """
    Enhanced QVD physics implementation with numerical stability.
    
    Provides wavelength-independent QVD scattering calculations
    with comprehensive bounds enforcement and error handling.
    """
    
    def __init__(self, 
                 qvd_coupling: float = 0.85, 
                 redshift_power: float = 0.6,
                 igm_enhancement: float = 0.7,
                 enable_logging: bool = True):
        """
        Initialize enhanced QVD physics.
        
        Parameters:
        -----------
        qvd_coupling : float
            Base QVD coupling strength (dimensionless)
        redshift_power : float
            Redshift scaling exponent
        igm_enhancement : float
            IGM enhancement factor
        enable_logging : bool
            Enable comprehensive logging
        """
        # Initialize bounds enforcer
        self.bounds_enforcer = RedshiftBoundsEnforcer()
        
        # Enforce parameter bounds
        self.qvd_coupling = self.bounds_enforcer.enforce_qvd_coupling(qvd_coupling, "init_coupling")
        self.redshift_power = self.bounds_enforcer.enforce_redshift_power(redshift_power, "init_power")
        self.igm_enhancement = clamp_to_range(igm_enhancement, 0.01, 5.0)
        
        # Physical constants
        self.c = 3e10  # cm/s (speed of light)
        self.sigma_thomson = 6.65e-25  # cm² (Thomson scattering cross-section)
        
        # Path length and IGM parameters
        self.path_length_factor = 1.0
        
        # Setup logging if requested
        if enable_logging:
            setup_qvd_logging(level=logging.INFO, enable_warnings=True)
        
        logger.info(f"Enhanced QVD Physics initialized: coupling={self.qvd_coupling:.3f}, "
                   f"power={self.redshift_power:.3f}, igm={self.igm_enhancement:.3f}")
    
    def calculate_redshift_dimming(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate QVD dimming for given redshift with numerical safety.
        
        This is the core QVD calculation that provides alternative to dark energy.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            QVD dimming in magnitudes (guaranteed finite and bounded)
        """
        # Enforce redshift bounds
        safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "primary_dimming")
        
        # Primary redshift-dependent dimming using safe operations
        base_dimming = self.qvd_coupling * safe_power(safe_redshift, self.redshift_power)
        
        # Validate intermediate result
        base_dimming = validate_finite(base_dimming, "base_dimming", replace_with=0.0)
        
        # Intergalactic medium enhancement
        igm_contribution = self._calculate_igm_effects(safe_redshift)
        
        # Combined dimming with safe addition
        total_dimming = base_dimming + igm_contribution
        
        # Ensure physical bounds and finite values
        total_dimming = validate_finite(total_dimming, "total_dimming", replace_with=0.0)
        total_dimming = self.bounds_enforcer.enforce_dimming(total_dimming, "final_dimming")
        
        return total_dimming
    
    def _calculate_igm_effects(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate intergalactic medium QVD enhancement with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z (already bounds-checked)
            
        Returns:
        --------
        float or array
            IGM contribution to dimming in magnitudes
        """
        # IGM density evolution (approximate) with safe operations
        if isinstance(redshift, (int, float)):
            igm_density_factor = safe_power(1 + redshift, 3.0)
            # Safe logarithm for redshift > 0
            log_factor = safe_log10(1 + redshift) if redshift > 0 else 0.0
        else:
            igm_density_factor = safe_power(1 + redshift, 3.0)
            # Handle array case with safe operations
            log_factor = np.zeros_like(redshift)
            positive_mask = redshift > 0
            log_factor[positive_mask] = safe_log10(1 + redshift[positive_mask])
        
        # Path length effects (cosmological distances) with safe operations
        path_enhancement = self.path_length_factor * safe_sqrt(igm_density_factor)
        
        # Combined IGM contribution with bounds enforcement
        igm_contribution = self.igm_enhancement * log_factor * path_enhancement
        
        # Validate and bound the result
        igm_contribution = validate_finite(igm_contribution, "igm_contribution", replace_with=0.0)
        
        # Apply reasonable bounds to IGM contribution
        if isinstance(igm_contribution, (int, float)):
            igm_contribution = clamp_to_range(igm_contribution, 0.0, 2.0)
        else:
            igm_contribution = np.clip(igm_contribution, 0.0, 2.0)
        
        return igm_contribution
    
    def calculate_qvd_cross_section(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate effective QVD scattering cross-section with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Effective QVD cross-section in cm²
        """
        # Enforce redshift bounds
        safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "cross_section")
        
        # Base cross-section scaling
        base_sigma = self.sigma_thomson  # Thomson scattering baseline
        
        # QVD enhancement factor with safe operations
        qvd_enhancement = self.qvd_coupling * safe_power(safe_redshift, self.redshift_power)
        
        # Effective cross-section with safe operations
        sigma_effective = base_sigma * (1 + qvd_enhancement)
        
        # Validate and bound the result
        sigma_effective = validate_finite(sigma_effective, "sigma_effective", replace_with=base_sigma)
        
        # Apply physical bounds to cross-section
        if isinstance(sigma_effective, (int, float)):
            sigma_effective = clamp_to_range(sigma_effective, base_sigma, base_sigma * 1e6)
        else:
            sigma_effective = np.clip(sigma_effective, base_sigma, base_sigma * 1e6)
        
        return sigma_effective
    
    def calculate_optical_depth(self, 
                               redshift: Union[float, np.ndarray],
                               path_length_Mpc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate QVD optical depth through cosmological distances with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
        path_length_Mpc : float or array
            Path length in Mpc
            
        Returns:
        --------
        float or array
            QVD optical depth (dimensionless, bounded)
        """
        # Enforce bounds on inputs
        safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "optical_depth_z")
        safe_path_length = self.bounds_enforcer.enforce_distance(path_length_Mpc, "optical_depth_path")
        
        # Effective cross-section
        sigma_qvd = self.calculate_qvd_cross_section(safe_redshift)
        
        # IGM number density (approximate) with safe operations
        n_igm = 1e-7  # cm⁻³ (typical IGM density)
        density_evolution = safe_power(1 + safe_redshift, 3.0)
        n_effective = n_igm * density_evolution
        
        # Path length in cm with safe conversion
        path_length_cm = safe_path_length * 3.086e24
        
        # Optical depth with safe multiplication
        tau = sigma_qvd * n_effective * path_length_cm
        
        # Validate and bound the result
        tau = validate_finite(tau, "optical_depth", replace_with=1e-10)
        
        # Apply optical depth bounds
        bounds = self.bounds_enforcer.bounds
        if isinstance(tau, (int, float)):
            tau = clamp_to_range(tau, bounds.MIN_OPTICAL_DEPTH, bounds.MAX_OPTICAL_DEPTH)
        else:
            tau = np.clip(tau, bounds.MIN_OPTICAL_DEPTH, bounds.MAX_OPTICAL_DEPTH)
        
        return tau
    
    def calculate_transmission(self, 
                              redshift: Union[float, np.ndarray],
                              path_length_Mpc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate photon transmission through QVD scattering medium with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
        path_length_Mpc : float or array
            Path length in Mpc
            
        Returns:
        --------
        float or array
            Transmission fraction (0-1, guaranteed finite)
        """
        # Calculate optical depth (already bounds-checked)
        tau = self.calculate_optical_depth(redshift, path_length_Mpc)
        
        # Safe exponential for transmission
        transmission = safe_exp(-tau)
        
        # Validate and bound transmission
        transmission = validate_finite(transmission, "transmission", replace_with=1.0)
        
        # Ensure transmission is between 0 and 1
        if isinstance(transmission, (int, float)):
            transmission = clamp_to_range(transmission, 0.0, 1.0)
        else:
            transmission = np.clip(transmission, 0.0, 1.0)
        
        return transmission
    
    def energy_loss_fraction(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate fractional energy loss due to QVD scattering with numerical safety.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Fractional energy loss (dimensionless, 0-1)
        """
        # QVD dimming in magnitudes (already bounds-checked)
        dimming_mag = self.calculate_redshift_dimming(redshift)
        
        # Convert to fractional flux loss with safe operations
        # ΔF/F = 1 - 10^(-0.4 * Δm)
        flux_ratio = safe_power(10.0, -0.4 * dimming_mag)
        energy_loss = 1.0 - flux_ratio
        
        # Validate and bound energy loss
        energy_loss = validate_finite(energy_loss, "energy_loss", replace_with=0.0)
        
        # Ensure energy loss is between 0 and 1
        if isinstance(energy_loss, (int, float)):
            energy_loss = clamp_to_range(energy_loss, 0.0, 1.0)
        else:
            energy_loss = np.clip(energy_loss, 0.0, 1.0)
        
        return energy_loss
    
    def validate_energy_conservation(self, redshift_array: np.ndarray) -> Dict:
        """
        Validate energy conservation in QVD scattering with numerical safety.
        
        Parameters:
        -----------
        redshift_array : np.ndarray
            Array of redshift values for validation
            
        Returns:
        --------
        dict
            Energy conservation validation results
        """
        # Enforce bounds on input array
        safe_redshift_array = self.bounds_enforcer.enforce_redshift(redshift_array, "energy_validation")
        
        validation = {
            'redshifts': safe_redshift_array,
            'energy_loss': [],
            'total_energy_loss': 0,
            'conservation_satisfied': True,
            'max_energy_loss': 0,
            'mean_energy_loss': 0
        }
        
        energy_losses = []
        for z in safe_redshift_array:
            energy_loss = self.energy_loss_fraction(z)
            energy_losses.append(energy_loss)
        
        validation['energy_loss'] = np.array(energy_losses)
        validation['total_energy_loss'] = np.sum(validation['energy_loss'])
        validation['max_energy_loss'] = np.max(validation['energy_loss'])
        validation['mean_energy_loss'] = np.mean(validation['energy_loss'])
        
        # Check if energy loss is reasonable (< 50% total, < 90% max)
        if (validation['total_energy_loss'] > 0.5 or 
            validation['max_energy_loss'] > 0.9):
            validation['conservation_satisfied'] = False
            logger.warning("Energy conservation may be violated")
        
        return validation
    
    def get_model_parameters(self) -> Dict:
        """
        Get current QVD model parameters with validation status.
        
        Returns:
        --------
        dict
            Dictionary of model parameters and validation info
        """
        return {
            'qvd_coupling': self.qvd_coupling,
            'redshift_power': self.redshift_power,
            'igm_enhancement': self.igm_enhancement,
            'path_length_factor': self.path_length_factor,
            'sigma_thomson': self.sigma_thomson,
            'bounds_warnings': self.bounds_enforcer.warning_count,
            'bounds_enforced': True,
            'numerical_safety': True
        }
    
    def update_parameters(self, **kwargs) -> None:
        """
        Update QVD model parameters with bounds enforcement.
        
        Parameters:
        -----------
        **kwargs : dict
            Parameter updates (qvd_coupling, redshift_power, etc.)
        """
        for key, value in kwargs.items():
            if key == 'qvd_coupling':
                self.qvd_coupling = self.bounds_enforcer.enforce_qvd_coupling(value, "update_coupling")
            elif key == 'redshift_power':
                self.redshift_power = self.bounds_enforcer.enforce_redshift_power(value, "update_power")
            elif key == 'igm_enhancement':
                self.igm_enhancement = clamp_to_range(value, 0.01, 5.0)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        logger.info(f"Parameters updated: {kwargs}")
    
    def calculate_e144_scaling(self) -> Dict:
        """
        Calculate scaling from E144 experiment to cosmological regime with validation.
        
        Returns:
        --------
        dict
            E144 scaling analysis with bounds checking
        """
        # E144 experimental parameters
        e144_cross_section = 1e-30  # cm²
        e144_threshold = 1e17  # W/cm²
        
        # Cosmological parameters
        cosmic_intensity = 1e-10  # W/cm² (typical cosmological photon intensity)
        
        # Safe calculations
        intensity_ratio = safe_divide(cosmic_intensity, e144_threshold, min_denominator=1e20)
        scaling_factor = safe_divide(self.qvd_coupling, intensity_ratio, min_denominator=1e-20)
        
        scaling = {
            'e144_cross_section': e144_cross_section,
            'e144_threshold': e144_threshold,
            'cosmic_intensity': cosmic_intensity,
            'intensity_ratio': intensity_ratio,
            'qvd_enhancement': self.qvd_coupling,
            'scaling_factor': scaling_factor,
            'scaling_valid': scaling_factor > 0 and np.isfinite(scaling_factor)
        }
        
        return scaling