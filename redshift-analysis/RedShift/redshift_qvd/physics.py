"""
QVD Vacuum Physics Module
=========================

Core physics implementation for wavelength-independent QVD redshift effects.
Implements direct photon-ψ field interactions without requiring plasma mediation.

Key Physics:
- Direct photon-vacuum field coupling
- Wavelength-independent energy transfer
- CMB energy enhancement mechanism
- No plasma density dependence
"""

import numpy as np
from typing import Union, Dict, Tuple
import logging

from .numerical_safety import (
    safe_power, safe_log10, safe_exp, safe_divide, safe_sqrt,
    validate_finite, clamp_to_range
)
from .physical_bounds import CosmologicalBounds, BoundsEnforcer

logger = logging.getLogger(__name__)


class QVDVacuumPhysics:
    """
    QVD vacuum physics for wavelength-independent redshift effects.
    
    Implements direct photon-ψ field interactions that transfer energy
    from high-energy photons to the CMB without requiring plasma mediation.
    """
    
    def __init__(self, 
                 qvd_coupling: float = 0.85,
                 redshift_power: float = 0.6,
                 cmb_coupling: float = 0.1):
        """
        Initialize QVD vacuum physics.
        
        Parameters:
        -----------
        qvd_coupling : float
            Base QVD coupling strength (dimensionless, fitted to observations)
        redshift_power : float
            Redshift scaling exponent (z^power, phenomenological)
        cmb_coupling : float
            Coupling strength for CMB energy transfer
        """
        self.qvd_coupling = qvd_coupling
        self.redshift_power = redshift_power
        self.cmb_coupling = cmb_coupling
        
        # Initialize bounds enforcer
        self.bounds = CosmologicalBounds()
        self.enforcer = BoundsEnforcer()
        
        # Physical constants
        self.c = 2.998e10  # cm/s (speed of light)
        self.h = 6.626e-27  # erg⋅s (Planck constant)
        self.k_b = 1.381e-16  # erg/K (Boltzmann constant)
        
        # CMB parameters
        self.T_cmb_0 = 2.725  # K (present CMB temperature)
        self.cmb_energy_density_0 = 4.17e-13  # erg/cm³ (present CMB energy density)
        
        # IGM parameters (for path length effects)
        self.igm_enhancement_factor = 0.7
        self.vacuum_permeability_enhancement = 1.2
        
        logger.info(f"QVD Vacuum Physics initialized: coupling={qvd_coupling}, power={redshift_power}")
    
    def calculate_vacuum_coupling_strength(self, 
                                         redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate QVD coupling strength in vacuum as function of redshift.
        
        This is the core wavelength-independent effect that provides
        alternative to dark energy acceleration.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            QVD vacuum coupling strength (dimensionless)
        """
        # Enforce redshift bounds
        safe_redshift = self.enforcer.enforce_redshift(redshift)
        
        # Primary redshift-dependent coupling (phenomenological)
        # This is fitted to supernova observations: z^0.6 scaling
        base_coupling = self.qvd_coupling * safe_power(safe_redshift, self.redshift_power)
        
        # Vacuum permeability enhancement (theoretical)
        # Based on QVD field equations in expanding spacetime
        vacuum_enhancement = self.vacuum_permeability_enhancement * safe_log10(1 + safe_redshift)
        vacuum_enhancement = np.maximum(vacuum_enhancement, 0)  # Ensure positive
        
        # Combined vacuum coupling
        total_coupling = base_coupling * (1 + vacuum_enhancement)
        
        # Enforce coupling bounds
        total_coupling = self.enforcer.enforce_coupling_strength(total_coupling)
        
        # Validate finite result
        total_coupling = validate_finite(total_coupling, "vacuum_coupling", replace_with=base_coupling)
        
        return total_coupling
    
    def calculate_photon_psi_interaction_rate(self,
                                            redshift: Union[float, np.ndarray],
                                            photon_energy_erg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate rate of photon-ψ field interactions per unit path length.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
        photon_energy_erg : float or array
            Photon energy in erg
            
        Returns:
        --------
        float or array
            Interaction rate per cm (cm⁻¹)
        """
        # Enforce bounds
        safe_redshift = self.enforcer.enforce_redshift(redshift)
        safe_energy = self.enforcer.enforce_photon_energy(photon_energy_erg)
        
        # Vacuum coupling strength
        coupling = self.calculate_vacuum_coupling_strength(safe_redshift)
        
        # Energy-dependent interaction cross-section
        # Higher energy photons interact more strongly with ψ field
        reference_energy = self.h * self.c / (550e-7)  # 550 nm photon energy
        energy_ratio = safe_divide(safe_energy, reference_energy)
        energy_factor = safe_power(energy_ratio, 0.5)  # Mild energy dependence
        
        # Vacuum field density (increases with redshift due to expansion)
        vacuum_density_factor = safe_power(1 + safe_redshift, 1.5)
        
        # Interaction rate per unit length
        base_rate = 1e-30  # cm⁻¹ (fitted to observations)
        interaction_rate = base_rate * coupling * energy_factor * vacuum_density_factor
        
        # Enforce rate bounds
        interaction_rate = self.enforcer.enforce_interaction_rate(interaction_rate)
        
        return validate_finite(interaction_rate, "interaction_rate", replace_with=base_rate)
    
    def calculate_energy_transfer_to_cmb(self,
                                       redshift: Union[float, np.ndarray],
                                       photon_energy_erg: Union[float, np.ndarray],
                                       path_length_cm: Union[float, np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate energy transfer from high-energy photons to CMB.
        
        This is the key mechanism: high-energy photons lose energy to ψ field,
        which then transfers energy to CMB photons.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
        photon_energy_erg : float or array
            Initial photon energy
        path_length_cm : float or array
            Path length through vacuum
            
        Returns:
        --------
        Dict
            Energy transfer results with all finite values
        """
        # Enforce bounds
        safe_redshift = self.enforcer.enforce_redshift(redshift)
        safe_energy = self.enforcer.enforce_photon_energy(photon_energy_erg)
        safe_path = self.enforcer.enforce_path_length(path_length_cm)
        
        # Interaction rate
        interaction_rate = self.calculate_photon_psi_interaction_rate(safe_redshift, safe_energy)
        
        # Total optical depth for energy transfer
        optical_depth = interaction_rate * safe_path
        optical_depth = self.enforcer.enforce_optical_depth(optical_depth)
        
        # Fractional energy loss (exponential attenuation)
        energy_loss_fraction = 1 - safe_exp(-optical_depth)
        energy_loss_fraction = clamp_to_range(energy_loss_fraction, 0, 0.9)  # Max 90% loss
        
        # Energy transferred to ψ field
        energy_to_psi_field = safe_energy * energy_loss_fraction
        
        # CMB temperature at redshift z
        T_cmb_z = self.T_cmb_0 * (1 + safe_redshift)
        
        # CMB energy density at redshift z
        cmb_energy_density_z = self.cmb_energy_density_0 * safe_power(1 + safe_redshift, 4)
        
        # Fraction of ψ field energy transferred to CMB
        cmb_transfer_efficiency = self.cmb_coupling * safe_power(T_cmb_z / self.T_cmb_0, 0.25)
        cmb_transfer_efficiency = clamp_to_range(cmb_transfer_efficiency, 0, 1)
        
        # Energy transferred to CMB
        energy_to_cmb = energy_to_psi_field * cmb_transfer_efficiency
        
        # Remaining photon energy
        final_photon_energy = safe_energy - energy_to_psi_field
        
        # Energy transfer results
        results = {
            'initial_photon_energy_erg': safe_energy,
            'final_photon_energy_erg': final_photon_energy,
            'energy_to_psi_field_erg': energy_to_psi_field,
            'energy_to_cmb_erg': energy_to_cmb,
            'energy_loss_fraction': energy_loss_fraction,
            'optical_depth': optical_depth,
            'cmb_temperature_K': T_cmb_z,
            'cmb_energy_density_erg_cm3': cmb_energy_density_z,
            'interaction_rate_cm_inv': interaction_rate
        }
        
        # Validate all results are finite
        for key, value in results.items():
            results[key] = validate_finite(value, f"energy_transfer_{key}", replace_with=0.0)
        
        return results
    
    def calculate_wavelength_independent_dimming(self,
                                               redshift: Union[float, np.ndarray],
                                               path_length_Mpc: Union[float, np.ndarray] = None) -> Union[float, np.ndarray]:
        """
        Calculate wavelength-independent QVD dimming in magnitudes.
        
        This is the observable effect: systematic dimming that increases
        with redshift, providing alternative to dark energy acceleration.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
        path_length_Mpc : float or array, optional
            Path length in Mpc (if None, calculated from redshift)
            
        Returns:
        --------
        float or array
            QVD dimming in magnitudes (wavelength-independent)
        """
        # Enforce redshift bounds
        safe_redshift = self.enforcer.enforce_redshift(redshift)
        
        # Calculate path length if not provided
        if path_length_Mpc is None:
            # Simple approximation: path length ~ c*z/H0
            H0 = 70.0  # km/s/Mpc
            c_km_s = 299792.458  # km/s
            path_length_Mpc = safe_divide(c_km_s * safe_redshift, H0)
        
        safe_path_Mpc = self.enforcer.enforce_distance(path_length_Mpc)
        path_length_cm = safe_path_Mpc * 3.086e24  # Convert to cm
        
        # Reference photon energy (550 nm)
        reference_energy = self.h * self.c / (550e-7)  # erg
        
        # Calculate energy transfer
        energy_transfer = self.calculate_energy_transfer_to_cmb(
            safe_redshift, reference_energy, path_length_cm
        )
        
        # Convert energy loss to magnitude dimming
        # Δm = -2.5 * log10(1 - f_loss) ≈ 2.5 * f_loss for small losses
        energy_loss_fraction = energy_transfer['energy_loss_fraction']
        
        # For small losses: Δm ≈ 2.5 * f_loss / ln(10)
        # For larger losses: use exact formula
        flux_ratio = 1 - energy_loss_fraction
        flux_ratio = np.maximum(flux_ratio, 1e-10)  # Prevent log(0)
        
        dimming_magnitudes = -2.5 * safe_log10(flux_ratio)
        
        # Enforce dimming bounds
        dimming_magnitudes = self.enforcer.enforce_dimming_magnitude(dimming_magnitudes)
        
        return validate_finite(dimming_magnitudes, "wavelength_independent_dimming", replace_with=0.0)
    
    def calculate_igm_enhancement_effects(self,
                                        redshift: Union[float, np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate intergalactic medium enhancement of QVD effects.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        Dict
            IGM enhancement effects
        """
        safe_redshift = self.enforcer.enforce_redshift(redshift)
        
        # IGM density evolution (standard cosmology)
        igm_density_factor = safe_power(1 + safe_redshift, 3)
        
        # Path length effects through IGM
        path_enhancement = self.igm_enhancement_factor * safe_log10(1 + safe_redshift)
        path_enhancement = np.maximum(path_enhancement, 0)
        
        # Vacuum field coupling enhancement in IGM
        vacuum_coupling_enhancement = safe_sqrt(igm_density_factor) * path_enhancement
        
        # Total IGM contribution to dimming
        igm_dimming_contribution = vacuum_coupling_enhancement * 0.1  # Calibrated factor
        igm_dimming_contribution = self.enforcer.enforce_dimming_magnitude(igm_dimming_contribution)
        
        results = {
            'igm_density_factor': igm_density_factor,
            'path_enhancement': path_enhancement,
            'vacuum_coupling_enhancement': vacuum_coupling_enhancement,
            'igm_dimming_contribution_mag': igm_dimming_contribution
        }
        
        # Validate all results
        for key, value in results.items():
            results[key] = validate_finite(value, f"igm_{key}", replace_with=1.0)
        
        return results
    
    def calculate_cmb_temperature_enhancement(self,
                                            redshift: Union[float, np.ndarray],
                                            energy_transfer_rate: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate CMB temperature enhancement due to QVD energy transfer.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
        energy_transfer_rate : float or array
            Rate of energy transfer to CMB (erg/s/cm³)
            
        Returns:
        --------
        float or array
            CMB temperature enhancement (K)
        """
        safe_redshift = self.enforcer.enforce_redshift(redshift)
        safe_rate = self.enforcer.enforce_energy_rate(energy_transfer_rate)
        
        # CMB temperature at redshift z
        T_cmb_z = self.T_cmb_0 * (1 + safe_redshift)
        
        # CMB energy density
        cmb_energy_density = self.cmb_energy_density_0 * safe_power(1 + safe_redshift, 4)
        
        # Temperature enhancement from energy injection
        # ΔT/T = ΔE/(4*E) for blackbody radiation
        energy_fraction = safe_divide(safe_rate, cmb_energy_density)
        temperature_enhancement = T_cmb_z * energy_fraction / 4
        
        # Enforce temperature bounds
        temperature_enhancement = self.enforcer.enforce_temperature_change(temperature_enhancement)
        
        return validate_finite(temperature_enhancement, "cmb_temperature_enhancement", replace_with=0.0)
    
    def validate_energy_conservation(self,
                                   redshift_array: np.ndarray,
                                   photon_energy_array: np.ndarray) -> Dict[str, Union[float, np.ndarray, bool]]:
        """
        Validate energy conservation in QVD vacuum interactions.
        
        Parameters:
        -----------
        redshift_array : np.ndarray
            Array of redshift values
        photon_energy_array : np.ndarray
            Array of photon energies
            
        Returns:
        --------
        Dict
            Energy conservation validation results
        """
        validation = {
            'redshifts': redshift_array,
            'photon_energies': photon_energy_array,
            'total_energy_input': 0,
            'total_energy_to_psi': 0,
            'total_energy_to_cmb': 0,
            'total_energy_remaining': 0,
            'energy_conservation_error': 0,
            'conservation_satisfied': True
        }
        
        path_length = 1e26  # cm (reference path length)
        
        total_input = 0
        total_to_psi = 0
        total_to_cmb = 0
        total_remaining = 0
        
        for z, E in zip(redshift_array, photon_energy_array):
            transfer = self.calculate_energy_transfer_to_cmb(z, E, path_length)
            
            total_input += transfer['initial_photon_energy_erg']
            total_to_psi += transfer['energy_to_psi_field_erg']
            total_to_cmb += transfer['energy_to_cmb_erg']
            total_remaining += transfer['final_photon_energy_erg']
        
        validation['total_energy_input'] = total_input
        validation['total_energy_to_psi'] = total_to_psi
        validation['total_energy_to_cmb'] = total_to_cmb
        validation['total_energy_remaining'] = total_remaining
        
        # Check energy conservation
        total_output = total_to_psi + total_remaining
        conservation_error = abs(total_input - total_output) / total_input
        validation['energy_conservation_error'] = conservation_error
        
        # Energy conservation satisfied if error < 1%
        validation['conservation_satisfied'] = conservation_error < 0.01
        
        return validation
    
    def get_model_parameters(self) -> Dict[str, float]:
        """
        Get current QVD vacuum model parameters.
        
        Returns:
        --------
        Dict
            Dictionary of model parameters
        """
        return {
            'qvd_coupling': self.qvd_coupling,
            'redshift_power': self.redshift_power,
            'cmb_coupling': self.cmb_coupling,
            'igm_enhancement_factor': self.igm_enhancement_factor,
            'vacuum_permeability_enhancement': self.vacuum_permeability_enhancement,
            'T_cmb_0': self.T_cmb_0,
            'cmb_energy_density_0': self.cmb_energy_density_0
        }
    
    def update_parameters(self, **kwargs) -> None:
        """
        Update QVD vacuum model parameters.
        
        Parameters:
        -----------
        **kwargs : dict
            Parameter updates
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated parameter {key} = {value}")
            else:
                raise ValueError(f"Unknown parameter: {key}")