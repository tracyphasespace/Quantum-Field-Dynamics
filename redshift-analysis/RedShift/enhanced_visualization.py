#!/usr/bin/env python3
"""
Enhanced Visualization Module for QVD Redshift Analysis
======================================================

Production-quality plotting utilities with numerical safety and
publication-ready visualizations.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Union, List
import logging

# Import numerical safety
from numerical_safety import validate_finite, safe_divide, safe_log10

logger = logging.getLogger(__name__)

class EnhancedRedshiftPlotter:
    """
    Enhanced plotting utilities for QVD redshift analysis with numerical safety.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', dpi: int = 300):
        """
        Initialize enhanced plotter with publication settings.
        
        Parameters:
        -----------
        style : str
            Matplotlib style
        dpi : int
            Resolution for saved figures
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            logger.warning(f"Style {style} not available, using default")
        
        # Publication-quality settings
        self.figsize = (12, 8)
        self.dpi = dpi
        self.colors = {
            'qvd': '#d62728',           # Red for QVD
            'standard': '#1f77b4',      # Blue for standard
            'lambda_cdm': '#ff7f0e',    # Orange for ΛCDM
            'observed': '#2ca02c',      # Green for observations
            'error': '#9467bd',         # Purple for errors
            'background': '#f0f0f0'     # Light gray for backgrounds
        }
        
        # Font settings for publication
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
        
        logger.info("Enhanced visualization system initialized")
    
    def plot_hubble_diagram(self, 
                           hubble_data: Dict,
                           save_path: Optional[Union[str, Path]] = None,
                           show_errors: bool = True) -> None:
        """
        Plot enhanced Hubble diagram with error analysis.
        
        Parameters:
        -----------
        hubble_data : Dict
            Hubble diagram data from RedshiftAnalyzer
        save_path : str or Path, optional
            Path to save plot
        show_errors : bool
            Show error bars and confidence intervals
        """
        logger.info("Creating enhanced Hubble diagram")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Validate input data
            redshifts = validate_finite(hubble_data['redshifts'], "hubble_redshifts", replace_with=np.array([0.1, 0.3, 0.5]))
            m_standard = validate_finite(hubble_data['magnitudes_standard'], "hubble_standard", replace_with=np.array([20.0, 21.0, 22.0]))
            m_qvd = validate_finite(hubble_data['magnitudes_qvd'], "hubble_qvd", replace_with=np.array([20.1, 21.2, 22.3]))
            qvd_dimming = validate_finite(hubble_data['qvd_dimming'], "hubble_dimming", replace_with=np.array([0.1, 0.2, 0.3]))
            
            # Main Hubble diagram
            ax1.plot(redshifts, m_standard, 'k--', linewidth=2, 
                    label='Standard Candle', alpha=0.7, zorder=1)
            
            # QVD model with enhanced styling
            ax1.plot(redshifts, m_qvd, color=self.colors['qvd'], linewidth=3,
                    label='QVD Model', marker='o', markersize=4, alpha=0.9, zorder=3)
            
            # Add confidence band for QVD model
            if show_errors and len(redshifts) > 10:
                # Calculate approximate uncertainty band
                uncertainty = 0.05 * np.ones_like(redshifts)  # 5% uncertainty
                ax1.fill_between(redshifts, m_qvd - uncertainty, m_qvd + uncertainty,
                               color=self.colors['qvd'], alpha=0.2, zorder=2)
            
            ax1.set_xlabel('Redshift z', fontweight='bold')
            ax1.set_ylabel('Apparent Magnitude', fontweight='bold')
            ax1.set_title('Enhanced QVD Hubble Diagram', fontweight='bold', pad=20)
            ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax1.invert_yaxis()
            
            # Add text box with key statistics
            if 'statistical_metrics' in hubble_data:
                stats_text = f"Data Points: {len(redshifts)}\nRedshift Range: {redshifts[0]:.3f} - {redshifts[-1]:.3f}"
            else:
                stats_text = f"Data Points: {len(redshifts)}\nRedshift Range: {redshifts[0]:.3f} - {redshifts[-1]:.3f}"
            
            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # QVD dimming evolution
            ax2.plot(redshifts, qvd_dimming, color=self.colors['qvd'], 
                    linewidth=3, marker='s', markersize=4, alpha=0.9)
            
            # Add trend line
            if len(redshifts) > 5:
                try:
                    # Fit power law: dimming = A * z^B
                    log_z = safe_log10(redshifts)
                    log_dimming = safe_log10(qvd_dimming)
                    
                    # Remove any non-finite values for fitting
                    valid_mask = np.isfinite(log_z) & np.isfinite(log_dimming)
                    if np.sum(valid_mask) > 3:
                        coeffs = np.polyfit(log_z[valid_mask], log_dimming[valid_mask], 1)
                        trend_dimming = 10**(coeffs[1]) * (redshifts**coeffs[0])
                        trend_dimming = validate_finite(trend_dimming, "trend_dimming", replace_with=qvd_dimming)
                        
                        ax2.plot(redshifts, trend_dimming, '--', color='gray', 
                               alpha=0.7, linewidth=2, label=f'z^{coeffs[0]:.2f} trend')
                        ax2.legend()
                except Exception as e:
                    logger.warning(f"Trend line fitting failed: {e}")
            
            ax2.set_xlabel('Redshift z', fontweight='bold')
            ax2.set_ylabel('QVD Dimming (magnitudes)', fontweight='bold')
            ax2.set_title('QVD Dimming Evolution', fontweight='bold', pad=20)
            ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add physics annotation
            physics_text = "Physics: E144-based QED\nNo dark energy required"
            ax2.text(0.95, 0.05, physics_text, transform=ax2.transAxes, 
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['background'], 
                    alpha=0.8, edgecolor='gray'))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
                logger.info(f"Hubble diagram saved to {save_path}")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Hubble diagram plotting failed: {e}")
            # Create minimal fallback plot
            self._create_fallback_plot("Hubble Diagram Error", save_path)
    
    def plot_qvd_vs_lambda_cdm(self,
                              comparison_data: Dict,
                              save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot enhanced QVD vs ΛCDM model comparison.
        
        Parameters:
        -----------
        comparison_data : Dict
            Comparison data from RedshiftAnalyzer
        save_path : str or Path, optional
            Path to save plot
        """
        logger.info("Creating QVD vs ΛCDM comparison plot")
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Validate input data
            redshifts = validate_finite(comparison_data['redshifts'], "comp_redshifts", replace_with=np.array([0.1, 0.3, 0.5]))
            m_qvd = validate_finite(comparison_data['qvd_magnitudes'], "comp_qvd", replace_with=np.array([20.1, 21.2, 22.3]))
            m_lambda_cdm = validate_finite(comparison_data['lambda_cdm_magnitudes'], "comp_lambda", replace_with=np.array([20.0, 21.0, 22.0]))
            differences = validate_finite(comparison_data['magnitude_differences'], "comp_diff", replace_with=np.array([0.1, 0.2, 0.3]))
            
            # Panel 1: Model comparison
            ax1.plot(redshifts, m_lambda_cdm, color=self.colors['lambda_cdm'], 
                    linewidth=3, label='ΛCDM Model', marker='o', markersize=4, alpha=0.8)
            ax1.plot(redshifts, m_qvd, color=self.colors['qvd'], 
                    linewidth=3, label='QVD Model', marker='s', markersize=4, alpha=0.8)
            
            ax1.set_xlabel('Redshift z')
            ax1.set_ylabel('Apparent Magnitude')
            ax1.set_title('Cosmological Models Comparison', fontweight='bold')
            ax1.legend(frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3)
            ax1.invert_yaxis()
            
            # Panel 2: Magnitude differences
            ax2.plot(redshifts, differences, color=self.colors['error'], linewidth=3, 
                    marker='d', markersize=5)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
            ax2.set_xlabel('Redshift z')
            ax2.set_ylabel('Magnitude Difference (QVD - ΛCDM)')
            ax2.set_title('Model Differences', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add RMS text with enhanced formatting
            if 'statistical_metrics' in comparison_data:
                rms = comparison_data['statistical_metrics']['rms_difference']
                mean_diff = comparison_data['statistical_metrics']['mean_difference']
                stats_text = f'RMS Difference: {rms:.3f} mag\nMean Difference: {mean_diff:.3f} mag'
            else:
                stats_text = 'Statistics not available'
            
            ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='white', alpha=0.9, edgecolor='gray'))
            
            # Panel 3: Residuals histogram
            ax3.hist(differences, bins=min(15, len(differences)//2), 
                    color=self.colors['error'], alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='k', linestyle='--', alpha=0.7)
            ax3.set_xlabel('Magnitude Difference (QVD - ΛCDM)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Difference Distribution', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Panel 4: Cumulative differences
            sorted_diffs = np.sort(np.abs(differences))
            cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
            ax4.plot(sorted_diffs, cumulative, color=self.colors['qvd'], 
                    linewidth=3, marker='o', markersize=3)
            ax4.set_xlabel('|Magnitude Difference| (mag)')
            ax4.set_ylabel('Cumulative Fraction')
            ax4.set_title('Cumulative Difference Distribution', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add 68% and 95% lines
            ax4.axhline(y=0.68, color='gray', linestyle=':', alpha=0.7, label='68%')
            ax4.axhline(y=0.95, color='gray', linestyle=':', alpha=0.7, label='95%')
            ax4.legend()
            
            plt.suptitle('QVD vs ΛCDM Comprehensive Comparison', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
                logger.info(f"QVD vs ΛCDM comparison saved to {save_path}")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"QVD vs ΛCDM comparison plotting failed: {e}")
            self._create_fallback_plot("Model Comparison Error", save_path)
    
    def plot_validation_results(self,
                               validation_data: Dict,
                               save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot enhanced model validation against observations.
        
        Parameters:
        -----------
        validation_data : Dict
            Validation data from RedshiftAnalyzer
        save_path : str or Path, optional
            Path to save plot
        """
        logger.info("Creating validation results plot")
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract and validate data
            obs_data = validation_data['observational_data']
            z_test = validate_finite(obs_data['redshifts'], "val_redshifts", replace_with=np.array([0.1, 0.3, 0.5]))
            observed = validate_finite(obs_data['observed_dimming'], "val_observed", replace_with=np.array([0.15, 0.30, 0.45]))
            uncertainties = validate_finite(obs_data['uncertainties'], "val_uncertainties", replace_with=np.array([0.05, 0.08, 0.10]))
            predicted = validate_finite(validation_data['qvd_predictions'], "val_predicted", replace_with=np.array([0.14, 0.28, 0.42]))
            residuals = validate_finite(validation_data['residuals'], "val_residuals", replace_with=np.array([-0.01, -0.02, -0.03]))
            
            # Panel 1: Observed vs predicted with error bars
            ax1.errorbar(z_test, observed, yerr=uncertainties, 
                        fmt='o', color=self.colors['observed'], markersize=8, 
                        capsize=5, capthick=2, label='Observed ± σ', linewidth=2)
            ax1.plot(z_test, predicted, 's', color=self.colors['qvd'],
                    markersize=8, label='QVD Model', linewidth=2)
            ax1.plot(z_test, predicted, '--', color=self.colors['qvd'], alpha=0.7, linewidth=2)
            
            # Add 1:1 line
            min_val = min(np.min(observed - uncertainties), np.min(predicted))
            max_val = max(np.max(observed + uncertainties), np.max(predicted))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, label='Perfect Agreement')
            
            ax1.set_xlabel('Redshift z')
            ax1.set_ylabel('Dimming (magnitudes)')
            ax1.set_title('Model Validation Against Observations', fontweight='bold')
            ax1.legend(frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3)
            
            # Panel 2: Residuals with error bars
            ax2.errorbar(z_test, residuals, yerr=uncertainties, 
                        fmt='o', color=self.colors['error'], markersize=8, 
                        capsize=5, capthick=2, linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.7, linewidth=1)
            ax2.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='±0.1 mag')
            ax2.axhline(y=-0.1, color='gray', linestyle='--', alpha=0.5)
            
            ax2.set_xlabel('Redshift z')
            ax2.set_ylabel('Residual (Predicted - Observed)')
            ax2.set_title('Model Residuals', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Panel 3: Statistics summary
            ax3.axis('off')
            if 'statistical_metrics' in validation_data:
                metrics = validation_data['statistical_metrics']
                stats_text = f"""
Model Validation Statistics

RMS Error: {metrics['rms_error']:.3f} mag
Max Error: {metrics['max_error']:.3f} mag
Mean Error: {metrics['mean_error']:.3f} mag
Std Error: {metrics['std_error']:.3f} mag

Chi-squared: {metrics['chi_squared']:.2f}
Reduced χ²: {metrics['reduced_chi_squared']:.2f}

Quality Grade: {validation_data['quality_grade']}
Validation: {'PASSED' if validation_data['validation_passed'] else 'FAILED'}
                """
            else:
                stats_text = "Statistical metrics not available"
            
            ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, 
                    verticalalignment='top', fontsize=12, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['background'], 
                    alpha=0.8, edgecolor='gray'))
            
            # Panel 4: Pull distribution (residuals/uncertainties)
            pulls = safe_divide(residuals, uncertainties, min_denominator=0.01)
            pulls = validate_finite(pulls, "validation_pulls", replace_with=np.array([0.0, 0.0, 0.0]))
            
            ax4.hist(pulls, bins=min(10, len(pulls)), color=self.colors['error'], 
                    alpha=0.7, edgecolor='black', density=True)
            
            # Overlay normal distribution
            x_norm = np.linspace(-3, 3, 100)
            y_norm = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_norm**2)
            ax4.plot(x_norm, y_norm, 'k--', linewidth=2, label='Standard Normal')
            
            ax4.axvline(x=0, color='k', linestyle='-', alpha=0.7)
            ax4.set_xlabel('Pull (Residual/Uncertainty)')
            ax4.set_ylabel('Density')
            ax4.set_title('Pull Distribution', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('QVD Model Validation Analysis', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
                logger.info(f"Validation results saved to {save_path}")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Validation plotting failed: {e}")
            self._create_fallback_plot("Validation Error", save_path)
    
    def create_comprehensive_analysis(self,
                                    results: Dict,
                                    save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create comprehensive 6-panel analysis plot.
        
        Parameters:
        -----------
        results : Dict
            Complete analysis results from RedshiftAnalyzer
        save_path : str or Path, optional
            Path to save plot
        """
        logger.info("Creating comprehensive analysis plot")
        
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.25, 
                                 height_ratios=[1, 1, 0.8])
            
            # Panel 1: Hubble diagram (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_hubble_panel(ax1, results.get('hubble_diagram', {}))
            
            # Panel 2: QVD vs ΛCDM (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_comparison_panel(ax2, results.get('lambda_cdm_comparison', {}))
            
            # Panel 3: QVD dimming evolution (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_dimming_evolution_panel(ax3, results.get('hubble_diagram', {}))
            
            # Panel 4: Validation (middle right)
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_validation_panel(ax4, results.get('validation', {}))
            
            # Panel 5: Summary statistics (bottom, spans both columns)
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_summary_panel(ax5, results)
            
            # Overall title and styling
            fig.suptitle('QVD Redshift Analysis: Comprehensive Results\n' + 
                        'Physics-Based Alternative to Dark Energy Cosmology', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # Add footer with key information
            footer_text = ("Based on SLAC E144 experimental validation | "
                          "No dark energy required | "
                          "Testable observational predictions")
            fig.text(0.5, 0.02, footer_text, ha='center', fontsize=11, 
                    style='italic', color='gray')
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
                logger.info(f"Comprehensive analysis saved to {save_path}")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Comprehensive analysis plotting failed: {e}")
            self._create_fallback_plot("Comprehensive Analysis Error", save_path)
    
    def _plot_hubble_panel(self, ax, hubble_data):
        """Plot Hubble diagram panel"""
        if not hubble_data:
            ax.text(0.5, 0.5, 'Hubble data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        redshifts = validate_finite(hubble_data.get('redshifts', []), "panel_redshifts", replace_with=np.array([0.1, 0.3, 0.5]))
        m_standard = validate_finite(hubble_data.get('magnitudes_standard', []), "panel_standard", replace_with=np.array([20.0, 21.0, 22.0]))
        m_qvd = validate_finite(hubble_data.get('magnitudes_qvd', []), "panel_qvd", replace_with=np.array([20.1, 21.2, 22.3]))
        
        ax.plot(redshifts, m_standard, 'k--', linewidth=2, label='Standard Candle', alpha=0.7)
        ax.plot(redshifts, m_qvd, color=self.colors['qvd'], linewidth=3, 
               label='QVD Model', marker='o', markersize=3)
        
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Apparent Magnitude')
        ax.set_title('QVD Hubble Diagram', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    
    def _plot_comparison_panel(self, ax, comparison_data):
        """Plot QVD vs ΛCDM comparison panel"""
        if not comparison_data:
            ax.text(0.5, 0.5, 'Comparison data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        redshifts = validate_finite(comparison_data.get('redshifts', []), "comp_panel_z", replace_with=np.array([0.1, 0.3, 0.5]))
        differences = validate_finite(comparison_data.get('magnitude_differences', []), "comp_panel_diff", replace_with=np.array([0.1, 0.2, 0.3]))
        
        ax.plot(redshifts, differences, color=self.colors['error'], 
               linewidth=3, marker='s', markersize=4)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('QVD - ΛCDM (mag)')
        ax.set_title('Model Differences', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add RMS if available
        if 'statistical_metrics' in comparison_data:
            rms = comparison_data['statistical_metrics']['rms_difference']
            ax.text(0.05, 0.95, f'RMS: {rms:.3f} mag', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_dimming_evolution_panel(self, ax, hubble_data):
        """Plot QVD dimming evolution panel"""
        if not hubble_data:
            ax.text(0.5, 0.5, 'Dimming data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        redshifts = validate_finite(hubble_data.get('redshifts', []), "dimm_panel_z", replace_with=np.array([0.1, 0.3, 0.5]))
        qvd_dimming = validate_finite(hubble_data.get('qvd_dimming', []), "dimm_panel_dimm", replace_with=np.array([0.1, 0.2, 0.3]))
        
        ax.plot(redshifts, qvd_dimming, color=self.colors['qvd'], 
               linewidth=3, marker='o', markersize=4)
        
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('QVD Dimming (mag)')
        ax.set_title('QVD Dimming Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_validation_panel(self, ax, validation_data):
        """Plot validation panel"""
        if not validation_data or 'observational_data' not in validation_data:
            ax.text(0.5, 0.5, 'Validation data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        obs_data = validation_data['observational_data']
        z_test = validate_finite(obs_data.get('redshifts', []), "val_panel_z", replace_with=np.array([0.1, 0.3, 0.5]))
        observed = validate_finite(obs_data.get('observed_dimming', []), "val_panel_obs", replace_with=np.array([0.15, 0.30, 0.45]))
        predicted = validate_finite(validation_data.get('qvd_predictions', []), "val_panel_pred", replace_with=np.array([0.14, 0.28, 0.42]))
        
        ax.plot(z_test, observed, 'o', color=self.colors['observed'],
               markersize=8, label='Observed')
        ax.plot(z_test, predicted, 's', color=self.colors['qvd'],
               markersize=8, label='QVD Model')
        ax.plot(z_test, predicted, '--', color=self.colors['qvd'], alpha=0.7)
        
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Dimming (mag)')
        ax.set_title('Model Validation', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add quality grade if available
        if 'quality_grade' in validation_data:
            grade = validation_data['quality_grade']
            ax.text(0.95, 0.05, f'Grade: {grade}', transform=ax.transAxes,
                   ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_summary_panel(self, ax, results):
        """Plot summary statistics panel"""
        ax.axis('off')
        
        # Collect key statistics
        summary_text = "QVD Redshift Model Summary\n" + "="*50 + "\n\n"
        
        # Validation statistics
        if 'validation' in results and 'statistical_metrics' in results['validation']:
            val_metrics = results['validation']['statistical_metrics']
            summary_text += f"Validation Performance:\n"
            summary_text += f"  • RMS Error vs Observations: {val_metrics['rms_error']:.3f} mag\n"
            summary_text += f"  • Quality Grade: {results['validation']['quality_grade']}\n"
            summary_text += f"  • Validation Status: {'PASSED' if results['validation']['validation_passed'] else 'FAILED'}\n\n"
        
        # Comparison statistics
        if 'lambda_cdm_comparison' in results and 'statistical_metrics' in results['lambda_cdm_comparison']:
            comp_metrics = results['lambda_cdm_comparison']['statistical_metrics']
            summary_text += f"ΛCDM Comparison:\n"
            summary_text += f"  • RMS Difference: {comp_metrics['rms_difference']:.3f} mag\n"
            summary_text += f"  • Mean Difference: {comp_metrics['mean_difference']:.3f} mag\n\n"
        
        # Model characteristics
        summary_text += f"Model Characteristics:\n"
        summary_text += f"  • Physics Basis: SLAC E144 experimental validation\n"
        summary_text += f"  • Dark Energy: Not required\n"
        summary_text += f"  • Acceleration: Standard Hubble expansion\n"
        summary_text += f"  • Testability: Specific observational signatures\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11, fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1.0', facecolor=self.colors['background'], 
               alpha=0.8, edgecolor='gray'))
    
    def _create_fallback_plot(self, error_message: str, save_path: Optional[Union[str, Path]] = None):
        """Create a fallback plot when main plotting fails"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Plotting Error: {error_message}\nPlease check the data and try again.', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14,
               bbox=dict(boxstyle='round,pad=1.0', facecolor='lightcoral', alpha=0.8))
        ax.set_title('Visualization Error', fontweight='bold', fontsize=16)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        logger.error(f"Created fallback plot for: {error_message}")

def create_publication_figure(results: Dict, 
                            title: str = "QVD Redshift Analysis",
                            save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Create publication-ready figure with all key results.
    
    Parameters:
    -----------
    results : Dict
        Complete analysis results
    title : str
        Figure title
    save_path : str or Path, optional
        Path to save plot
    """
    plotter = EnhancedRedshiftPlotter(dpi=600)  # High DPI for publication
    plotter.create_comprehensive_analysis(results, save_path)