"""
QFD Physics Module
=================

Core physics implementation for QFD redshift analysis.
Implements wavelength-independent redshift-dependent dimming.
"""

import numpy as np
from typing import Union


class QFDPhysics:
    """
    QFD physics implementation for redshift analysis.
    
    Provides wavelength-independent QFD scattering calculations
    based on phenomenological parameters fitted to observations.
    """
    
    def __init__(self, qfd_coupling: float = 0.85, redshift_power: float = 0.6):
        """
        Initialize QFD physics.
        
        Parameters:
        -----------
        qfd_coupling : float
            Base QFD coupling strength (dimensionless)
        redshift_power : float
            Redshift scaling exponent
        """
        self.qfd_coupling = qfd_coupling
        self.redshift_power = redshift_power
        
        # Physical constants and parameters
        self.c = 3e10  # cm/s (speed of light)
        self.sigma_thomson = 6.65e-25  # cm² (Thomson scattering cross-section)
        
        # IGM and path length parameters
        self.igm_enhancement = 0.7
        self.path_length_factor = 1.0
        
    def calculate_redshift_dimming(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate QFD dimming for given redshift (wavelength independent).
        
        This is the core QFD calculation that provides alternative to dark energy.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            QFD dimming in magnitudes
        """
        # Primary redshift-dependent dimming (phenomenological)
        base_dimming = self.qfd_coupling * (redshift**self.redshift_power)
        
        # Intergalactic medium enhancement
        igm_contribution = self._calculate_igm_effects(redshift)
        
        # Combined dimming
        total_dimming = base_dimming + igm_contribution
        
        # Ensure physical bounds
        if isinstance(redshift, (int, float)):
            return max(total_dimming, 0)
        else:
            return np.maximum(total_dimming, 0)
    
    def _calculate_igm_effects(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate intergalactic medium QFD enhancement.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            IGM contribution to dimming in magnitudes
        """
        # IGM density evolution (approximate)
        if isinstance(redshift, (int, float)):
            igm_density_factor = (1 + redshift)**3
            log_factor = np.log10(1 + redshift) if redshift > 0 else 0
        else:
            igm_density_factor = (1 + redshift)**3
            log_factor = np.log10(1 + redshift)
            log_factor[redshift <= 0] = 0
        
        # Path length effects (cosmological distances)
        path_enhancement = self.path_length_factor * np.sqrt(igm_density_factor)
        
        # Combined IGM contribution
        igm_contribution = self.igm_enhancement * log_factor * path_enhancement
        
        return igm_contribution
    
    def calculate_qfd_cross_section(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate effective QFD scattering cross-section.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Effective QFD cross-section in cm²
        """
        # Base cross-section scaling
        base_sigma = self.sigma_thomson  # Thomson scattering baseline
        
        # QFD enhancement factor
        qfd_enhancement = self.qfd_coupling * (redshift**self.redshift_power)
        
        # Effective cross-section
        sigma_effective = base_sigma * (1 + qfd_enhancement)
        
        return sigma_effective
    
    def calculate_optical_depth(self, 
                               redshift: Union[float, np.ndarray],
                               path_length_Mpc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate QFD optical depth through cosmological distances.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
        path_length_Mpc : float or array
            Path length in Mpc
            
        Returns:
        --------
        float or array
            QFD optical depth (dimensionless)
        """
        # Effective cross-section
        sigma_qfd = self.calculate_qfd_cross_section(redshift)
        
        # IGM number density (approximate)
        n_igm = 1e-7  # cm⁻³ (typical IGM density)
        density_evolution = (1 + redshift)**3
        n_effective = n_igm * density_evolution
        
        # Path length in cm
        path_length_cm = path_length_Mpc * 3.086e24
        
        # Optical depth
        tau = sigma_qfd * n_effective * path_length_cm
        
        return tau
    
    def calculate_transmission(self, 
                              redshift: Union[float, np.ndarray],
                              path_length_Mpc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate photon transmission through QFD scattering medium.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
        path_length_Mpc : float or array
            Path length in Mpc
            
        Returns:
        --------
        float or array
            Transmission fraction (0-1)
        """
        tau = self.calculate_optical_depth(redshift, path_length_Mpc)
        return np.exp(-tau)
    
    def calculate_scattering_rate(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate QFD scattering rate per unit distance.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Scattering rate in Mpc⁻¹
        """
        # Effective cross-section
        sigma_qfd = self.calculate_qfd_cross_section(redshift)
        
        # IGM number density
        n_igm = 1e-7  # cm⁻³
        density_evolution = (1 + redshift)**3
        n_effective = n_igm * density_evolution
        
        # Scattering rate per cm
        rate_per_cm = sigma_qfd * n_effective
        
        # Convert to per Mpc
        rate_per_Mpc = rate_per_cm * 3.086e24
        
        return rate_per_Mpc
    
    def energy_loss_fraction(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate fractional energy loss due to QFD scattering.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Fractional energy loss (dimensionless)
        """
        # QFD dimming in magnitudes
        dimming_mag = self.calculate_redshift_dimming(redshift)
        
        # Convert to fractional flux loss
        # ΔF/F = 1 - 10^(-0.4 * Δm)
        flux_ratio = 10**(-0.4 * dimming_mag)
        energy_loss = 1 - flux_ratio
        
        return energy_loss
    
    def validate_energy_conservation(self, redshift_array: np.ndarray) -> dict:
        """
        Validate energy conservation in QFD scattering.
        
        Parameters:
        -----------
        redshift_array : np.ndarray
            Array of redshift values for validation
            
        Returns:
        --------
        dict
            Energy conservation validation results
        """
        validation = {
            'redshifts': redshift_array,
            'energy_loss': [],
            'total_energy_loss': 0,
            'conservation_satisfied': True
        }
        
        for z in redshift_array:
            energy_loss = self.energy_loss_fraction(z)
            validation['energy_loss'].append(energy_loss)
        
        validation['energy_loss'] = np.array(validation['energy_loss'])
        validation['total_energy_loss'] = np.sum(validation['energy_loss'])
        
        # Check if energy loss is reasonable (< 50% total)
        if validation['total_energy_loss'] > 0.5:
            validation['conservation_satisfied'] = False
        
        return validation
    
    def calculate_temperature_effects(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate temperature-dependent QFD effects.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            Temperature enhancement factor
        """
        # CMB temperature evolution
        T_cmb_0 = 2.725  # K (present CMB temperature)
        T_cmb_z = T_cmb_0 * (1 + redshift)
        
        # Temperature-dependent QFD coupling (weak dependence)
        temp_factor = (T_cmb_z / T_cmb_0)**0.1
        
        return temp_factor
    
    def get_model_parameters(self) -> dict:
        """
        Get current QFD model parameters.
        
        Returns:
        --------
        dict
            Dictionary of model parameters
        """
        return {
            'qfd_coupling': self.qfd_coupling,
            'redshift_power': self.redshift_power,
            'igm_enhancement': self.igm_enhancement,
            'path_length_factor': self.path_length_factor,
            'sigma_thomson': self.sigma_thomson
        }
    
    def update_parameters(self, **kwargs) -> None:
        """
        Update QFD model parameters.
        
        Parameters:
        -----------
        **kwargs : dict
            Parameter updates (qfd_coupling, redshift_power, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
                
    def calculate_e144_scaling(self) -> dict:
        """
        Calculate scaling from E144 experiment to cosmological regime.
        
        Returns:
        --------
        dict
            E144 scaling analysis
        """
        # E144 experimental parameters
        e144_cross_section = 1e-30  # cm²
        e144_threshold = 1e17  # W/cm²
        
        # Cosmological parameters
        cosmic_intensity = 1e-10  # W/cm² (typical cosmological photon intensity)
        
        scaling = {
            'e144_cross_section': e144_cross_section,
            'e144_threshold': e144_threshold,
            'cosmic_intensity': cosmic_intensity,
            'intensity_ratio': cosmic_intensity / e144_threshold,
            'qfd_enhancement': self.qfd_coupling,
            'scaling_factor': self.qfd_coupling / (cosmic_intensity / e144_threshold)
        }
        
        return scaling