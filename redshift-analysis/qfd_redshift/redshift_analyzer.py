"""
QFD Redshift Analyzer
====================

Core class for analyzing cosmological redshift using QFD physics.
Provides wavelength-independent redshift analysis and Hubble diagram generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

from .cosmology import QFDCosmology
from .physics import QFDPhysics
from .visualization import RedshiftPlotter


class RedshiftAnalyzer:
    """
    Main class for QFD redshift analysis.
    
    Provides wavelength-independent analysis of cosmological redshift
    using QFD scattering as alternative to dark energy.
    """
    
    def __init__(self, 
                 qfd_coupling: float = 0.85,
                 redshift_power: float = 0.6,
                 hubble_constant: float = 70.0):
        """
        Initialize QFD redshift analyzer.
        
        Parameters:
        -----------
        qfd_coupling : float
            Base QFD coupling strength (dimensionless)
        redshift_power : float  
            Redshift scaling exponent (z^power)
        hubble_constant : float
            Hubble constant in km/s/Mpc
        """
        self.qfd_coupling = qfd_coupling
        self.redshift_power = redshift_power
        self.hubble_constant = hubble_constant
        
        # Initialize physics modules
        self.cosmology = QFDCosmology(hubble_constant)
        self.physics = QFDPhysics(qfd_coupling, redshift_power)
        self.plotter = RedshiftPlotter()
        
        # Analysis results storage
        self.results = {}
        
    def calculate_qfd_dimming(self, redshift: float) -> float:
        """
        Calculate QFD dimming for given redshift (wavelength independent).
        
        Parameters:
        -----------
        redshift : float
            Cosmological redshift z
            
        Returns:
        --------
        float
            QFD dimming in magnitudes
        """
        return self.physics.calculate_redshift_dimming(redshift)
    
    def generate_hubble_diagram(self, 
                               z_min: float = 0.01, 
                               z_max: float = 0.6,
                               n_points: int = 100) -> Dict:
        """
        Generate Hubble diagram with QFD effects.
        
        Parameters:
        -----------
        z_min, z_max : float
            Redshift range
        n_points : int
            Number of redshift points
            
        Returns:
        --------
        Dict
            Hubble diagram data with QFD predictions
        """
        redshifts = np.logspace(np.log10(z_min), np.log10(z_max), n_points)
        
        hubble_data = {
            'redshifts': redshifts,
            'distances_Mpc': [],
            'magnitudes_standard': [],
            'magnitudes_qfd': [],
            'qfd_dimming': []
        }
        
        # Standard candle absolute magnitude
        M_abs = -19.3
        
        for z in redshifts:
            # Standard cosmological distance (no acceleration)
            distance_Mpc = self.cosmology.luminosity_distance(z)
            distance_modulus = 5 * np.log10(distance_Mpc * 1e6 / 10)
            
            # Standard apparent magnitude
            m_standard = M_abs + distance_modulus
            
            # QFD dimming effect
            qfd_dimming = self.calculate_qfd_dimming(z)
            m_qfd = m_standard + qfd_dimming
            
            hubble_data['distances_Mpc'].append(distance_Mpc)
            hubble_data['magnitudes_standard'].append(m_standard)
            hubble_data['magnitudes_qfd'].append(m_qfd)
            hubble_data['qfd_dimming'].append(qfd_dimming)
        
        # Convert to numpy arrays
        for key in ['distances_Mpc', 'magnitudes_standard', 'magnitudes_qfd', 'qfd_dimming']:
            hubble_data[key] = np.array(hubble_data[key])
        
        self.results['hubble_diagram'] = hubble_data
        return hubble_data
    
    def compare_with_lambda_cdm(self, 
                               omega_m: float = 0.3,
                               omega_lambda: float = 0.7) -> Dict:
        """
        Compare QFD predictions with ΛCDM model.
        
        Parameters:
        -----------
        omega_m : float
            Matter density parameter
        omega_lambda : float
            Dark energy density parameter
            
        Returns:
        --------
        Dict
            Comparison results
        """
        if 'hubble_diagram' not in self.results:
            self.generate_hubble_diagram()
        
        hubble_data = self.results['hubble_diagram']
        redshifts = hubble_data['redshifts']
        
        comparison = {
            'redshifts': redshifts,
            'qfd_magnitudes': hubble_data['magnitudes_qfd'],
            'lambda_cdm_magnitudes': [],
            'magnitude_differences': []
        }
        
        M_abs = -19.3
        
        for z in redshifts:
            # ΛCDM luminosity distance with dark energy
            d_lambda_cdm = self.cosmology.lambda_cdm_distance(z, omega_m, omega_lambda)
            distance_modulus = 5 * np.log10(d_lambda_cdm * 1e6 / 10)
            m_lambda_cdm = M_abs + distance_modulus
            
            comparison['lambda_cdm_magnitudes'].append(m_lambda_cdm)
        
        comparison['lambda_cdm_magnitudes'] = np.array(comparison['lambda_cdm_magnitudes'])
        comparison['magnitude_differences'] = (comparison['qfd_magnitudes'] - 
                                             comparison['lambda_cdm_magnitudes'])
        
        # Calculate RMS difference
        rms_diff = np.sqrt(np.mean(comparison['magnitude_differences']**2))
        comparison['rms_difference'] = rms_diff
        
        self.results['lambda_cdm_comparison'] = comparison
        return comparison
    
    def validate_against_observations(self) -> Dict:
        """
        Validate QFD model against key observational constraints.
        
        Returns:
        --------
        Dict
            Validation results with statistical metrics
        """
        # Key observational points (approximate from supernova surveys)
        obs_redshifts = np.array([0.1, 0.3, 0.5])
        obs_dimming = np.array([0.15, 0.30, 0.45])  # Observed excess dimming
        
        # QFD model predictions
        qfd_predictions = [self.calculate_qfd_dimming(z) for z in obs_redshifts]
        qfd_predictions = np.array(qfd_predictions)
        
        # Calculate validation metrics
        residuals = qfd_predictions - obs_dimming
        rms_error = np.sqrt(np.mean(residuals**2))
        max_error = np.max(np.abs(residuals))
        
        validation = {
            'test_redshifts': obs_redshifts,
            'observed_dimming': obs_dimming,
            'qfd_predictions': qfd_predictions,
            'residuals': residuals,
            'rms_error': rms_error,
            'max_error': max_error,
            'validation_passed': rms_error < 0.2  # 0.2 mag threshold
        }
        
        self.results['validation'] = validation
        return validation
    
    def create_analysis_plots(self, output_dir: str = "results") -> None:
        """
        Create comprehensive analysis plots.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate data if not available
        if 'hubble_diagram' not in self.results:
            self.generate_hubble_diagram()
        if 'lambda_cdm_comparison' not in self.results:
            self.compare_with_lambda_cdm()
        if 'validation' not in self.results:
            self.validate_against_observations()
        
        # Create plots using visualization module
        self.plotter.plot_hubble_diagram(
            self.results['hubble_diagram'], 
            save_path=output_path / "hubble_diagram.png"
        )
        
        self.plotter.plot_qfd_vs_lambda_cdm(
            self.results['lambda_cdm_comparison'],
            save_path=output_path / "qfd_vs_lambda_cdm.png"
        )
        
        self.plotter.plot_validation_results(
            self.results['validation'],
            save_path=output_path / "validation_results.png"
        )
        
        self.plotter.plot_comprehensive_analysis(
            self.results,
            save_path=output_path / "comprehensive_analysis.png"
        )
    
    def save_results(self, output_file: str = "results/qfd_redshift_results.json") -> None:
        """
        Save analysis results to JSON file.
        
        Parameters:
        -----------
        output_file : str
            Output JSON file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_results[key][subkey] = subvalue.tolist()
                    else:
                        json_results[key][subkey] = subvalue
            else:
                json_results[key] = value
        
        # Add model parameters
        json_results['model_parameters'] = {
            'qfd_coupling': self.qfd_coupling,
            'redshift_power': self.redshift_power,
            'hubble_constant': self.hubble_constant
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def run_complete_analysis(self, output_dir: str = "results") -> Dict:
        """
        Run complete QFD redshift analysis pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for results
            
        Returns:
        --------
        Dict
            Complete analysis results
        """
        print("QFD Redshift Analysis")
        print("=" * 50)
        print(f"QFD Coupling: {self.qfd_coupling}")
        print(f"Redshift Power: z^{self.redshift_power}")
        print(f"Hubble Constant: {self.hubble_constant} km/s/Mpc")
        print()
        
        # Run analysis pipeline
        print("1. Generating Hubble diagram...")
        hubble_data = self.generate_hubble_diagram()
        
        print("2. Comparing with ΛCDM model...")
        lambda_cdm_comparison = self.compare_with_lambda_cdm()
        
        print("3. Validating against observations...")
        validation = self.validate_against_observations()
        
        print("4. Creating analysis plots...")
        self.create_analysis_plots(output_dir)
        
        print("5. Saving results...")
        self.save_results(f"{output_dir}/qfd_redshift_results.json")
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"RMS error vs observations: {validation['rms_error']:.3f} mag")
        print(f"Max error: {validation['max_error']:.3f} mag")
        print(f"Validation passed: {validation['validation_passed']}")
        print(f"RMS difference vs ΛCDM: {lambda_cdm_comparison['rms_difference']:.3f} mag")
        print(f"\nResults saved to: {output_dir}/")
        
        return self.results