"""
Physical Bounds Enforcement for Cosmological QVD
================================================

Provides bounds enforcement and validation for cosmological QVD calculations
to ensure all parameters remain within physically reasonable ranges.
"""

import numpy as np
import warnings
from typing import Union, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CosmologicalBounds:
    """Physical bounds for cosmological QVD parameters"""
    
    # Redshift bounds
    MIN_REDSHIFT: float = 0.0
    MAX_REDSHIFT: float = 10.0
    
    # Distance bounds (Mpc)
    MIN_DISTANCE_MPC: float = 0.1
    MAX_DISTANCE_MPC: float = 50000.0
    
    # Energy bounds (erg)
    MIN_PHOTON_ENERGY_ERG: float = 1e-20
    MAX_PHOTON_ENERGY_ERG: float = 1e-10
    
    # Coupling strength bounds
    MIN_COUPLING_STRENGTH: float = 1e-10
    MAX_COUPLING_STRENGTH: float = 10.0
    
    # Dimming magnitude bounds
    MIN_DIMMING_MAG: float = 0.0
    MAX_DIMMING_MAG: float = 5.0
    
    # Optical depth bounds
    MIN_OPTICAL_DEPTH: float = 1e-10
    MAX_OPTICAL_DEPTH: float = 50.0
    
    # Path length bounds (cm)
    MIN_PATH_LENGTH_CM: float = 1e20
    MAX_PATH_LENGTH_CM: float = 1e30
    
    # Interaction rate bounds (cm⁻¹)
    MIN_INTERACTION_RATE: float = 1e-35
    MAX_INTERACTION_RATE: float = 1e-25
    
    # Temperature bounds (K)
    MIN_TEMPERATURE: float = 0.1
    MAX_TEMPERATURE: float = 1e6
    
    # Temperature change bounds (K)
    MIN_TEMPERATURE_CHANGE: float = -100.0
    MAX_TEMPERATURE_CHANGE: float = 100.0
    
    # Energy rate bounds (erg/s/cm³)
    MIN_ENERGY_RATE: float = 1e-25
    MAX_ENERGY_RATE: float = 1e-10


class BoundsEnforcer:
    """
    Enforces physical bounds on cosmological QVD parameters.
    
    Provides safe parameter enforcement with logging and warnings
    when bounds are applied.
    """
    
    def __init__(self, bounds: CosmologicalBounds = None):
        """
        Initialize bounds enforcer.
        
        Parameters:
        -----------
        bounds : CosmologicalBounds, optional
            Physical bounds to enforce (uses default if None)
        """
        self.bounds = bounds or CosmologicalBounds()
        self.warnings_issued = set()
        
    def _enforce_bounds(self, 
                       value: Union[float, np.ndarray], 
                       min_val: float, 
                       max_val: float,
                       param_name: str,
                       context: str = "") -> Union[float, np.ndarray]:
        """
        Enforce bounds on a parameter with logging.
        
        Parameters:
        -----------
        value : float or array
            Value to bound
        min_val : float
            Minimum allowed value
        max_val : float
            Maximum allowed value
        param_name : str
            Parameter name for logging
        context : str
            Context for logging
            
        Returns:
        --------
        float or array
            Bounded value
        """
        original_value = value
        bounded_value = np.clip(value, min_val, max_val)
        
        # Check if bounds were applied
        if isinstance(value, (int, float)):
            bounds_applied = (value < min_val) or (value > max_val)
        else:
            bounds_applied = np.any((value < min_val) | (value > max_val))
        
        # Issue warning if bounds were applied (only once per parameter)
        warning_key = f"{param_name}_{context}"
        if bounds_applied and warning_key not in self.warnings_issued:
            if isinstance(original_value, (int, float)):
                logger.warning(f"Bounds applied to {param_name} in {context}: "
                             f"{original_value:.3e} → [{min_val:.3e}, {max_val:.3e}]")
            else:
                out_of_bounds = np.sum((original_value < min_val) | (original_value > max_val))
                logger.warning(f"Bounds applied to {param_name} in {context}: "
                             f"{out_of_bounds}/{len(original_value)} values bounded")
            
            self.warnings_issued.add(warning_key)
        
        return bounded_value
    
    def enforce_redshift(self, 
                        redshift: Union[float, np.ndarray],
                        context: str = "redshift") -> Union[float, np.ndarray]:
        """Enforce redshift bounds"""
        return self._enforce_bounds(
            redshift, 
            self.bounds.MIN_REDSHIFT, 
            self.bounds.MAX_REDSHIFT,
            "redshift", 
            context
        )
    
    def enforce_distance(self, 
                        distance_Mpc: Union[float, np.ndarray],
                        context: str = "distance") -> Union[float, np.ndarray]:
        """Enforce distance bounds"""
        return self._enforce_bounds(
            distance_Mpc,
            self.bounds.MIN_DISTANCE_MPC,
            self.bounds.MAX_DISTANCE_MPC,
            "distance_Mpc",
            context
        )
    
    def enforce_photon_energy(self, 
                             energy_erg: Union[float, np.ndarray],
                             context: str = "photon_energy") -> Union[float, np.ndarray]:
        """Enforce photon energy bounds"""
        return self._enforce_bounds(
            energy_erg,
            self.bounds.MIN_PHOTON_ENERGY_ERG,
            self.bounds.MAX_PHOTON_ENERGY_ERG,
            "photon_energy_erg",
            context
        )
    
    def enforce_coupling_strength(self, 
                                 coupling: Union[float, np.ndarray],
                                 context: str = "coupling") -> Union[float, np.ndarray]:
        """Enforce coupling strength bounds"""
        return self._enforce_bounds(
            coupling,
            self.bounds.MIN_COUPLING_STRENGTH,
            self.bounds.MAX_COUPLING_STRENGTH,
            "coupling_strength",
            context
        )
    
    def enforce_dimming_magnitude(self, 
                                 dimming_mag: Union[float, np.ndarray],
                                 context: str = "dimming") -> Union[float, np.ndarray]:
        """Enforce dimming magnitude bounds"""
        return self._enforce_bounds(
            dimming_mag,
            self.bounds.MIN_DIMMING_MAG,
            self.bounds.MAX_DIMMING_MAG,
            "dimming_magnitude",
            context
        )
    
    def enforce_optical_depth(self, 
                             optical_depth: Union[float, np.ndarray],
                             context: str = "optical_depth") -> Union[float, np.ndarray]:
        """Enforce optical depth bounds"""
        return self._enforce_bounds(
            optical_depth,
            self.bounds.MIN_OPTICAL_DEPTH,
            self.bounds.MAX_OPTICAL_DEPTH,
            "optical_depth",
            context
        )
    
    def enforce_path_length(self, 
                           path_length_cm: Union[float, np.ndarray],
                           context: str = "path_length") -> Union[float, np.ndarray]:
        """Enforce path length bounds"""
        return self._enforce_bounds(
            path_length_cm,
            self.bounds.MIN_PATH_LENGTH_CM,
            self.bounds.MAX_PATH_LENGTH_CM,
            "path_length_cm",
            context
        )
    
    def enforce_interaction_rate(self, 
                                rate_cm_inv: Union[float, np.ndarray],
                                context: str = "interaction_rate") -> Union[float, np.ndarray]:
        """Enforce interaction rate bounds"""
        return self._enforce_bounds(
            rate_cm_inv,
            self.bounds.MIN_INTERACTION_RATE,
            self.bounds.MAX_INTERACTION_RATE,
            "interaction_rate",
            context
        )
    
    def enforce_temperature(self, 
                           temperature_K: Union[float, np.ndarray],
                           context: str = "temperature") -> Union[float, np.ndarray]:
        """Enforce temperature bounds"""
        return self._enforce_bounds(
            temperature_K,
            self.bounds.MIN_TEMPERATURE,
            self.bounds.MAX_TEMPERATURE,
            "temperature_K",
            context
        )
    
    def enforce_temperature_change(self, 
                                  delta_T_K: Union[float, np.ndarray],
                                  context: str = "temperature_change") -> Union[float, np.ndarray]:
        """Enforce temperature change bounds"""
        return self._enforce_bounds(
            delta_T_K,
            self.bounds.MIN_TEMPERATURE_CHANGE,
            self.bounds.MAX_TEMPERATURE_CHANGE,
            "temperature_change_K",
            context
        )
    
    def enforce_energy_rate(self, 
                           energy_rate: Union[float, np.ndarray],
                           context: str = "energy_rate") -> Union[float, np.ndarray]:
        """Enforce energy rate bounds"""
        return self._enforce_bounds(
            energy_rate,
            self.bounds.MIN_ENERGY_RATE,
            self.bounds.MAX_ENERGY_RATE,
            "energy_rate",
            context
        )


def create_safe_cosmological_state(redshift: float,
                                  distance_Mpc: float,
                                  photon_energy_erg: float,
                                  coupling_strength: float,
                                  bounds_enforcer: BoundsEnforcer = None) -> Dict[str, float]:
    """
    Create a safe cosmological state with all parameters bounded.
    
    Parameters:
    -----------
    redshift : float
        Cosmological redshift
    distance_Mpc : float
        Distance in Mpc
    photon_energy_erg : float
        Photon energy in erg
    coupling_strength : float
        QVD coupling strength
    bounds_enforcer : BoundsEnforcer, optional
        Bounds enforcer to use
        
    Returns:
    --------
    Dict
        Safe cosmological state with bounded parameters
    """
    if bounds_enforcer is None:
        bounds_enforcer = BoundsEnforcer()
    
    safe_state = {
        'redshift': bounds_enforcer.enforce_redshift(redshift, "cosmological_state"),
        'distance_Mpc': bounds_enforcer.enforce_distance(distance_Mpc, "cosmological_state"),
        'photon_energy_erg': bounds_enforcer.enforce_photon_energy(photon_energy_erg, "cosmological_state"),
        'coupling_strength': bounds_enforcer.enforce_coupling_strength(coupling_strength, "cosmological_state")
    }
    
    return safe_state


def validate_cosmological_parameters(parameters: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate that cosmological parameters are physically reasonable.
    
    Parameters:
    -----------
    parameters : Dict
        Dictionary of parameters to validate
        
    Returns:
    --------
    Dict
        Validation results for each parameter
    """
    bounds = CosmologicalBounds()
    validation = {}
    
    # Check redshift
    if 'redshift' in parameters:
        z = parameters['redshift']
        validation['redshift_valid'] = (bounds.MIN_REDSHIFT <= z <= bounds.MAX_REDSHIFT)
    
    # Check distance
    if 'distance_Mpc' in parameters:
        d = parameters['distance_Mpc']
        validation['distance_valid'] = (bounds.MIN_DISTANCE_MPC <= d <= bounds.MAX_DISTANCE_MPC)
    
    # Check photon energy
    if 'photon_energy_erg' in parameters:
        E = parameters['photon_energy_erg']
        validation['energy_valid'] = (bounds.MIN_PHOTON_ENERGY_ERG <= E <= bounds.MAX_PHOTON_ENERGY_ERG)
    
    # Check coupling strength
    if 'coupling_strength' in parameters:
        g = parameters['coupling_strength']
        validation['coupling_valid'] = (bounds.MIN_COUPLING_STRENGTH <= g <= bounds.MAX_COUPLING_STRENGTH)
    
    # Check dimming magnitude
    if 'dimming_magnitude' in parameters:
        dm = parameters['dimming_magnitude']
        validation['dimming_valid'] = (bounds.MIN_DIMMING_MAG <= dm <= bounds.MAX_DIMMING_MAG)
    
    # Overall validation
    validation['all_valid'] = all(validation.values())
    
    return validation


def check_energy_conservation(energy_input: float,
                             energy_output: float,
                             tolerance: float = 0.01) -> Dict[str, Union[float, bool]]:
    """
    Check energy conservation in QVD interactions.
    
    Parameters:
    -----------
    energy_input : float
        Total input energy
    energy_output : float
        Total output energy
    tolerance : float
        Fractional tolerance for conservation
        
    Returns:
    --------
    Dict
        Energy conservation check results
    """
    if energy_input <= 0:
        return {
            'energy_input': energy_input,
            'energy_output': energy_output,
            'fractional_error': float('inf'),
            'conservation_satisfied': False,
            'error_message': 'Invalid input energy'
        }
    
    fractional_error = abs(energy_input - energy_output) / energy_input
    conservation_satisfied = fractional_error <= tolerance
    
    return {
        'energy_input': energy_input,
        'energy_output': energy_output,
        'fractional_error': fractional_error,
        'conservation_satisfied': conservation_satisfied,
        'tolerance': tolerance
    }


def enforce_physical_consistency(cosmological_state: Dict[str, float]) -> Dict[str, float]:
    """
    Enforce physical consistency between related cosmological parameters.
    
    Parameters:
    -----------
    cosmological_state : Dict
        Dictionary of cosmological parameters
        
    Returns:
    --------
    Dict
        Physically consistent cosmological state
    """
    consistent_state = cosmological_state.copy()
    
    # Ensure distance is consistent with redshift (approximate)
    if 'redshift' in consistent_state and 'distance_Mpc' in consistent_state:
        z = consistent_state['redshift']
        d = consistent_state['distance_Mpc']
        
        # Simple Hubble law check: d ≈ c*z/H0
        H0 = 70.0  # km/s/Mpc
        c_km_s = 299792.458  # km/s
        expected_distance = c_km_s * z / H0
        
        # If distance is very inconsistent, adjust it
        if abs(d - expected_distance) / expected_distance > 0.5:
            logger.warning(f"Distance-redshift inconsistency detected: "
                         f"d={d:.1f} Mpc, z={z:.3f} → expected d≈{expected_distance:.1f} Mpc")
            consistent_state['distance_Mpc'] = expected_distance
    
    # Ensure photon energy is reasonable for optical/UV photons
    if 'photon_energy_erg' in consistent_state:
        E = consistent_state['photon_energy_erg']
        
        # Check if energy corresponds to reasonable wavelength (100 nm - 10 μm)
        h = 6.626e-27  # erg⋅s
        c = 2.998e10   # cm/s
        
        wavelength_cm = h * c / E
        wavelength_nm = wavelength_cm * 1e7
        
        if wavelength_nm < 100 or wavelength_nm > 10000:
            logger.warning(f"Photon energy corresponds to unusual wavelength: "
                         f"E={E:.2e} erg → λ={wavelength_nm:.1f} nm")
    
    return consistent_state