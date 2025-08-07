#!/usr/bin/env python3
"""
Enhanced QVD Redshift Analyzer
==============================

Core class for analyzing cosmological redshift using QVD physics with
comprehensive numerical stability, bounds enforcement, and error handling.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Union
import logging

from redshift_physics import EnhancedQVDPhysics
from redshift_cosmology import EnhancedQVDCosmology
from numerical_safety import validate_finite, clamp_to_range
from error_handling import setup_qvd_logging, ErrorReporter

logger = logging.getLogger(__name__)


class EnhancedRedshiftAnalyzer:
    """
    Enhanced QVD redshift analyzer with numerical stability.
    
    Provides wavelength-independent analysis of cosmological redshift
    using QVD scattering as alternative to dark energy with comprehensive
    bounds enforcement and error handling.
    """
    
    def __init__(self, 
                 qvd_coupling: float = 0.85,
                 redshift_power: float = 0.6,
                 hubble_constant: float = 70.0,
                 igm_enhancement: float = 0.7,
                 enable_logging: bool = True,
                 enable_bounds_checking: bool = True):
        """
        Initialize enhanced QVD redshift analyzer.
        
        Parameters:
        -----------
        qvd_coupling : float
            Base QVD coupling strength (dimensionless)
        redshift_power : float  
            Redshift scaling exponent (z^power)
        hubble_constant : float
            Hubble constant in km/s/Mpc
        igm_enhancement : float
            IGM enhancement factor
        enable_logging : bool
            Enable comprehensive logging
        enable_bounds_checking : bool
            Enable bounds enforcement
        """
        # Setup logging first
        if enable_logging:
            setup_qvd_logging(level=logging.INFO, enable_warnings=True)
        
        # Initialize error reporter
        self.error_reporter = ErrorReporter()
        
        # Store configuration
        self.enable_bounds_checking = enable_bounds_checking
        
        # Initialize physics and cosmology modules with bounds checking
        self.physics = EnhancedQVDPhysics(
            qvd_coupling=qvd_coupling,
            redshift_power=redshift_power,
            igm_enhancement=igm_enhancement,
            enable_logging=enable_logging
        )
        
        self.cosmology = EnhancedQVDCosmology(
            hubble_constant=hubble_constant,
            enable_bounds_checking=enable_bounds_checking
        )
        
        # Analysis results storage
        self.results = {}
        
        # Performance tracking
        self.performance_metrics = {
            'analysis_start_time': None,
            'analysis_duration': 0,
            'calculations_performed': 0,
            'bounds_violations': 0,
            'numerical_warnings': 0
        }
        
        logger.info(f"Enhanced RedShift Analyzer initialized: "
                   f"coupling={qvd_coupling:.3f}, power={redshift_power:.3f}, "
                   f"H0={hubble_constant:.1f}, bounds={enable_bounds_checking}")
    
    def calculate_qvd_dimming(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate QVD dimming for given redshift with full safety checks.
        
        Parameters:
        -----------
        redshift : float or array
            Cosmological redshift z
            
        Returns:
        --------
        float or array
            QVD dimming in magnitudes (guaranteed finite and bounded)
        """
        try:
            # Use physics module (already has bounds enforcement)
            dimming = self.physics.calculate_redshift_dimming(redshift)
            
            # Additional validation
            dimming = validate_finite(dimming, "qvd_dimming_final", replace_with=0.0)
            
            # Track performance
            self.performance_metrics['calculations_performed'] += 1
            
            return dimming
            
        except Exception as e:
            self.error_reporter.add_error("QVDDimmingError", str(e), {
                'redshift': redshift if isinstance(redshift, (int, float)) else 'array',
                'function': 'calculate_qvd_dimming'
            })
            logger.error(f"QVD dimming calculation failed: {e}")
            
            # Return safe fallback
            if isinstance(redshift, (int, float)):
                return 0.0
            else:
                return np.zeros_like(redshift)
    
    def generate_hubble_diagram(self, 
                               z_min: float = 0.01, 
                               z_max: float = 0.6,
                               n_points: int = 100) -> Dict:
        """
        Generate Hubble diagram with QVD effects and comprehensive validation.
        
        Parameters:
        -----------
        z_min, z_max : float
            Redshift range
        n_points : int
            Number of redshift points
            
        Returns:
        --------
        Dict
            Hubble diagram data with QVD predictions (all values guaranteed finite)
        """
        logger.info(f"Generating Hubble diagram: z={z_min:.3f} to {z_max:.3f}, n={n_points}")
        
        # Validate input parameters
        z_min = clamp_to_range(z_min, 1e-6, 1.0)
        z_max = clamp_to_range(z_max, z_min + 0.01, 10.0)
        n_points = max(10, min(n_points, 1000))  # Reasonable bounds
        
        # Generate redshift array with safe operations
        try:
            redshifts = np.logspace(np.log10(z_min), np.log10(z_max), n_points)
        except:
            # Fallback to linear spacing
            redshifts = np.linspace(z_min, z_max, n_points)
            logger.warning("Using linear redshift spacing due to log spacing failure")
        
        hubble_data = {
            'redshifts': redshifts,
            'distances_Mpc': [],
            'magnitudes_standard': [],
            'magnitudes_qvd': [],
            'qvd_dimming': [],
            'distance_moduli': [],
            'generation_successful': True,
            'points_calculated': 0,
            'points_failed': 0
        }
        
        # Standard candle absolute magnitude
        M_abs = -19.3
        
        for i, z in enumerate(redshifts):
            try:
                # Standard cosmological distance (already bounds-checked in cosmology module)
                distance_Mpc = self.cosmology.luminosity_distance(z)
                distance_modulus = self.cosmology.distance_modulus(z)
                
                # Standard apparent magnitude
                m_standard = M_abs + distance_modulus
                
                # QVD dimming effect (already bounds-checked in physics module)
                qvd_dimming = self.calculate_qvd_dimming(z)
                
                # QVD-modified apparent magnitude
                m_qvd = m_standard + qvd_dimming
                
                # Validate all results
                distance_Mpc = validate_finite(distance_Mpc, f"distance_{i}", replace_with=100.0)
                m_standard = validate_finite(m_standard, f"m_standard_{i}", replace_with=20.0)
                m_qvd = validate_finite(m_qvd, f"m_qvd_{i}", replace_with=20.0)
                qvd_dimming = validate_finite(qvd_dimming, f"qvd_dimming_{i}", replace_with=0.0)
                distance_modulus = validate_finite(distance_modulus, f"distance_modulus_{i}", replace_with=35.0)
                
                # Store results
                hubble_data['distances_Mpc'].append(distance_Mpc)
                hubble_data['magnitudes_standard'].append(m_standard)
                hubble_data['magnitudes_qvd'].append(m_qvd)
                hubble_data['qvd_dimming'].append(qvd_dimming)
                hubble_data['distance_moduli'].append(distance_modulus)
                
                hubble_data['points_calculated'] += 1
                
            except Exception as e:
                self.error_reporter.add_error("HubbleDiagramError", str(e), {
                    'redshift': z,
                    'point_index': i
                })
                logger.warning(f"Failed to calculate point {i} at z={z:.3f}: {e}")
                
                # Use safe fallback values
                hubble_data['distances_Mpc'].append(100.0)
                hubble_data['magnitudes_standard'].append(20.0)
                hubble_data['magnitudes_qvd'].append(20.0)
                hubble_data['qvd_dimming'].append(0.0)
                hubble_data['distance_moduli'].append(35.0)
                
                hubble_data['points_failed'] += 1
        
        # Convert to numpy arrays with validation
        for key in ['distances_Mpc', 'magnitudes_standard', 'magnitudes_qvd', 
                   'qvd_dimming', 'distance_moduli']:
            array_data = np.array(hubble_data[key])
            array_data = validate_finite(array_data, f"hubble_{key}", replace_with=0.0)
            hubble_data[key] = array_data
        
        # Calculate summary statistics
        hubble_data['max_qvd_dimming'] = np.max(hubble_data['qvd_dimming'])
        hubble_data['mean_qvd_dimming'] = np.mean(hubble_data['qvd_dimming'])
        hubble_data['total_redshift_range'] = z_max - z_min
        
        # Check if generation was successful
        if hubble_data['points_failed'] > hubble_data['points_calculated'] * 0.1:
            hubble_data['generation_successful'] = False
            logger.warning(f"Hubble diagram generation had {hubble_data['points_failed']} failures")
        
        self.results['hubble_diagram'] = hubble_data
        logger.info(f"Hubble diagram generated: {hubble_data['points_calculated']} points, "
                   f"{hubble_data['points_failed']} failures")
        
        return hubble_data
    
    def compare_with_lambda_cdm(self, 
                               omega_m: float = 0.3,
                               omega_lambda: float = 0.7) -> Dict:
        """
        Compare QVD predictions with ΛCDM model using numerical safety.
        
        Parameters:
        -----------
        omega_m : float
            Matter density parameter
        omega_lambda : float
            Dark energy density parameter
            
        Returns:
        --------
        Dict
            Comparison results (all values guaranteed finite)
        """
        logger.info(f"Comparing with ΛCDM: Ωₘ={omega_m:.2f}, ΩΛ={omega_lambda:.2f}")
        
        # Validate ΛCDM parameters
        omega_m = clamp_to_range(omega_m, 0.1, 1.0)
        omega_lambda = clamp_to_range(omega_lambda, 0.0, 1.0)
        
        if 'hubble_diagram' not in self.results:
            logger.info("Generating Hubble diagram for ΛCDM comparison")
            self.generate_hubble_diagram()
        
        hubble_data = self.results['hubble_diagram']
        redshifts = hubble_data['redshifts']
        
        comparison = {
            'redshifts': redshifts,
            'qvd_magnitudes': hubble_data['magnitudes_qvd'],
            'lambda_cdm_magnitudes': [],
            'magnitude_differences': [],
            'lambda_cdm_distances': [],
            'comparison_successful': True,
            'points_calculated': 0,
            'points_failed': 0
        }
        
        M_abs = -19.3
        
        for i, z in enumerate(redshifts):
            try:
                # ΛCDM luminosity distance (already bounds-checked in cosmology module)
                d_lambda_cdm = self.cosmology.lambda_cdm_distance(z, omega_m, omega_lambda)
                
                # ΛCDM distance modulus
                distance_modulus = 5 * np.log10(d_lambda_cdm * 1e6 / 10)
                m_lambda_cdm = M_abs + distance_modulus
                
                # Validate results
                d_lambda_cdm = validate_finite(d_lambda_cdm, f"lambda_cdm_dist_{i}", replace_with=100.0)
                m_lambda_cdm = validate_finite(m_lambda_cdm, f"lambda_cdm_mag_{i}", replace_with=20.0)
                
                comparison['lambda_cdm_magnitudes'].append(m_lambda_cdm)
                comparison['lambda_cdm_distances'].append(d_lambda_cdm)
                comparison['points_calculated'] += 1
                
            except Exception as e:
                self.error_reporter.add_error("LambdaCDMError", str(e), {
                    'redshift': z,
                    'point_index': i,
                    'omega_m': omega_m,
                    'omega_lambda': omega_lambda
                })
                logger.warning(f"ΛCDM calculation failed at z={z:.3f}: {e}")
                
                # Use fallback values
                comparison['lambda_cdm_magnitudes'].append(20.0)
                comparison['lambda_cdm_distances'].append(100.0)
                comparison['points_failed'] += 1
        
        # Convert to arrays and calculate differences
        comparison['lambda_cdm_magnitudes'] = np.array(comparison['lambda_cdm_magnitudes'])
        comparison['lambda_cdm_distances'] = np.array(comparison['lambda_cdm_distances'])
        
        # Calculate magnitude differences with validation
        comparison['magnitude_differences'] = (comparison['qvd_magnitudes'] - 
                                             comparison['lambda_cdm_magnitudes'])
        comparison['magnitude_differences'] = validate_finite(
            comparison['magnitude_differences'], "magnitude_differences", replace_with=0.0
        )
        
        # Calculate RMS difference with safe operations
        try:
            rms_diff = np.sqrt(np.mean(comparison['magnitude_differences']**2))
            rms_diff = validate_finite(rms_diff, "rms_difference", replace_with=1.0)
        except:
            rms_diff = 1.0
            logger.warning("RMS calculation failed, using fallback value")
        
        comparison['rms_difference'] = rms_diff
        
        # Calculate additional statistics
        comparison['max_difference'] = np.max(np.abs(comparison['magnitude_differences']))
        comparison['mean_difference'] = np.mean(comparison['magnitude_differences'])
        comparison['std_difference'] = np.std(comparison['magnitude_differences'])
        
        # Check comparison success
        if comparison['points_failed'] > comparison['points_calculated'] * 0.1:
            comparison['comparison_successful'] = False
            logger.warning(f"ΛCDM comparison had {comparison['points_failed']} failures")
        
        self.results['lambda_cdm_comparison'] = comparison
        logger.info(f"ΛCDM comparison completed: RMS difference = {rms_diff:.3f} mag")
        
        return comparison
    
    def validate_against_observations(self, 
                                    obs_redshifts: Optional[np.ndarray] = None,
                                    obs_dimming: Optional[np.ndarray] = None) -> Dict:
        """
        Validate QVD model against observational constraints with numerical safety.
        
        Parameters:
        -----------
        obs_redshifts : np.ndarray, optional
            Observed redshift values
        obs_dimming : np.ndarray, optional
            Observed excess dimming values
            
        Returns:
        --------
        Dict
            Validation results with statistical metrics (all values finite)
        """
        logger.info("Validating QVD model against observations")
        
        # Use default observational points if not provided
        if obs_redshifts is None or obs_dimming is None:
            # Key observational points (approximate from supernova surveys)
            obs_redshifts = np.array([0.1, 0.3, 0.5])
            obs_dimming = np.array([0.15, 0.30, 0.45])  # Observed excess dimming
        
        # Validate observational data
        obs_redshifts = validate_finite(obs_redshifts, "obs_redshifts", replace_with=np.array([0.1, 0.3, 0.5]))
        obs_dimming = validate_finite(obs_dimming, "obs_dimming", replace_with=np.array([0.15, 0.30, 0.45]))
        
        # QVD model predictions
        qvd_predictions = []
        prediction_errors = 0
        
        for i, z in enumerate(obs_redshifts):
            try:
                prediction = self.calculate_qvd_dimming(z)
                prediction = validate_finite(prediction, f"qvd_pred_{i}", replace_with=0.0)
                qvd_predictions.append(prediction)
            except Exception as e:
                self.error_reporter.add_error("ValidationPredictionError", str(e), {
                    'redshift': z,
                    'observation_index': i
                })
                logger.warning(f"QVD prediction failed for z={z:.3f}: {e}")
                qvd_predictions.append(0.0)
                prediction_errors += 1
        
        qvd_predictions = np.array(qvd_predictions)
        
        # Calculate validation metrics with safe operations
        residuals = qvd_predictions - obs_dimming
        residuals = validate_finite(residuals, "validation_residuals", replace_with=np.zeros_like(obs_dimming))
        
        try:
            rms_error = np.sqrt(np.mean(residuals**2))
            rms_error = validate_finite(rms_error, "rms_error", replace_with=1.0)
        except:
            rms_error = 1.0
            logger.warning("RMS error calculation failed")
        
        try:
            max_error = np.max(np.abs(residuals))
            max_error = validate_finite(max_error, "max_error", replace_with=1.0)
        except:
            max_error = 1.0
            logger.warning("Max error calculation failed")
        
        try:
            mean_error = np.mean(residuals)
            mean_error = validate_finite(mean_error, "mean_error", replace_with=0.0)
        except:
            mean_error = 0.0
        
        validation = {
            'test_redshifts': obs_redshifts,
            'observed_dimming': obs_dimming,
            'qvd_predictions': qvd_predictions,
            'residuals': residuals,
            'rms_error': rms_error,
            'max_error': max_error,
            'mean_error': mean_error,
            'std_error': np.std(residuals),
            'validation_passed': rms_error < 0.2,  # 0.2 mag threshold
            'prediction_errors': prediction_errors,
            'validation_successful': prediction_errors == 0
        }
        
        # Additional validation metrics
        validation['relative_rms_error'] = rms_error / np.mean(obs_dimming) if np.mean(obs_dimming) > 0 else 1.0
        validation['correlation_coefficient'] = np.corrcoef(qvd_predictions, obs_dimming)[0, 1] if len(obs_dimming) > 1 else 0.0
        
        # Validate correlation coefficient
        if not np.isfinite(validation['correlation_coefficient']):
            validation['correlation_coefficient'] = 0.0
        
        self.results['validation'] = validation
        logger.info(f"Validation completed: RMS error = {rms_error:.3f} mag, "
                   f"validation passed = {validation['validation_passed']}")
        
        return validation
    
    def run_complete_analysis(self, output_dir: str = "results") -> Dict:
        """
        Run complete QVD redshift analysis pipeline with comprehensive error handling.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for results
            
        Returns:
        --------
        Dict
            Complete analysis results (all values guaranteed finite)
        """
        self.performance_metrics['analysis_start_time'] = time.time()
        
        print("Enhanced QVD Redshift Analysis")
        print("=" * 60)
        print(f"QVD Coupling: {self.physics.qvd_coupling:.3f}")
        print(f"Redshift Power: z^{self.physics.redshift_power:.3f}")
        print(f"Hubble Constant: {self.cosmology.H0:.1f} km/s/Mpc")
        print(f"Bounds Checking: {self.enable_bounds_checking}")
        print()
        
        analysis_success = True
        
        try:
            # Run analysis pipeline with error handling
            print("1. Generating Hubble diagram...")
            hubble_data = self.generate_hubble_diagram()
            if not hubble_data.get('generation_successful', False):
                analysis_success = False
            
            print("2. Comparing with ΛCDM model...")
            lambda_cdm_comparison = self.compare_with_lambda_cdm()
            if not lambda_cdm_comparison.get('comparison_successful', False):
                analysis_success = False
            
            print("3. Validating against observations...")
            validation = self.validate_against_observations()
            if not validation.get('validation_successful', False):
                analysis_success = False
            
            print("4. Creating analysis plots...")
            self.create_analysis_plots(output_dir)
            
            print("5. Saving results...")
            self.save_results(f"{output_dir}/qvd_redshift_results.json")
            
        except Exception as e:
            self.error_reporter.add_error("AnalysisPipelineError", str(e), {
                'stage': 'complete_analysis'
            })
            logger.error(f"Analysis pipeline failed: {e}")
            analysis_success = False
        
        # Calculate performance metrics
        self.performance_metrics['analysis_duration'] = time.time() - self.performance_metrics['analysis_start_time']
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Overall Success: {analysis_success}")
        if 'validation' in self.results:
            print(f"RMS error vs observations: {self.results['validation']['rms_error']:.3f} mag")
            print(f"Max error: {self.results['validation']['max_error']:.3f} mag")
            print(f"Validation passed: {self.results['validation']['validation_passed']}")
        if 'lambda_cdm_comparison' in self.results:
            print(f"RMS difference vs ΛCDM: {self.results['lambda_cdm_comparison']['rms_difference']:.3f} mag")
        
        print(f"Analysis duration: {self.performance_metrics['analysis_duration']:.1f} seconds")
        print(f"Calculations performed: {self.performance_metrics['calculations_performed']}")
        print(f"Errors encountered: {len(self.error_reporter.errors)}")
        print(f"\nResults saved to: {output_dir}/")
        
        # Add performance metrics to results
        self.results['performance_metrics'] = self.performance_metrics
        self.results['analysis_success'] = analysis_success
        self.results['error_summary'] = {
            'total_errors': len(self.error_reporter.errors),
            'error_types': list(set([e['type'] for e in self.error_reporter.errors])),
            'bounds_warnings': getattr(self.physics.bounds_enforcer, 'warning_count', 0)
        }
        
        return self.results
    
    def create_analysis_plots(self, output_dir: str = "results") -> None:
        """
        Create comprehensive analysis plots with error handling.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for plots
        """
        try:
            from redshift_visualization import EnhancedRedshiftPlotter
            
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            plotter = EnhancedRedshiftPlotter()
            
            # Generate data if not available
            if 'hubble_diagram' not in self.results:
                self.generate_hubble_diagram()
            if 'lambda_cdm_comparison' not in self.results:
                self.compare_with_lambda_cdm()
            if 'validation' not in self.results:
                self.validate_against_observations()
            
            # Create plots with error handling
            try:
                plotter.plot_hubble_diagram(
                    self.results['hubble_diagram'], 
                    save_path=output_path / "hubble_diagram.png"
                )
            except Exception as e:
                logger.warning(f"Hubble diagram plot failed: {e}")
            
            try:
                plotter.plot_qvd_vs_lambda_cdm(
                    self.results['lambda_cdm_comparison'],
                    save_path=output_path / "qvd_vs_lambda_cdm.png"
                )
            except Exception as e:
                logger.warning(f"QVD vs ΛCDM plot failed: {e}")
            
            try:
                plotter.plot_validation_results(
                    self.results['validation'],
                    save_path=output_path / "validation_results.png"
                )
            except Exception as e:
                logger.warning(f"Validation plot failed: {e}")
            
            try:
                plotter.plot_comprehensive_analysis(
                    self.results,
                    save_path=output_path / "comprehensive_analysis.png"
                )
            except Exception as e:
                logger.warning(f"Comprehensive analysis plot failed: {e}")
                
        except ImportError:
            logger.warning("Visualization module not available, skipping plots")
        except Exception as e:
            logger.error(f"Plot creation failed: {e}")
    
    def save_results(self, output_file: str = "results/qvd_redshift_results.json") -> None:
        """
        Save analysis results to JSON file with error handling.
        
        Parameters:
        -----------
        output_file : str
            Output JSON file path
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True)
            
            # Prepare results for JSON serialization
            json_results = self._prepare_json_results()
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
                
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.error_reporter.add_error("SaveResultsError", str(e), {
                'output_file': output_file
            })
            logger.error(f"Failed to save results: {e}")
    
    def _prepare_json_results(self) -> Dict:
        """Prepare results for JSON serialization with validation"""
        json_results = {}
        
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        # Validate array before conversion
                        clean_array = validate_finite(subvalue, f"json_{key}_{subkey}", replace_with=0.0)
                        json_results[key][subkey] = clean_array.tolist()
                    elif isinstance(subvalue, (np.integer, np.floating)):
                        # Validate scalar before conversion
                        clean_scalar = validate_finite(float(subvalue), f"json_{key}_{subkey}", replace_with=0.0)
                        json_results[key][subkey] = clean_scalar
                    else:
                        json_results[key][subkey] = subvalue
            else:
                json_results[key] = value
        
        # Add model parameters
        json_results['model_parameters'] = self.physics.get_model_parameters()
        json_results['cosmological_parameters'] = self.cosmology.get_cosmological_parameters()
        
        return json_results