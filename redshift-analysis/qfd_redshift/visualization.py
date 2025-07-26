"""
QFD Visualization Module
========================

Plotting utilities for QFD redshift analysis.
Creates publication-quality plots for scientific analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Optional, Union


class RedshiftPlotter:
    """
    Plotting utilities for QFD redshift analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize plotter with style settings.
        
        Parameters:
        -----------
        style : str
            Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set default parameters
        self.figsize = (12, 8)
        self.dpi = 300
        self.colors = {
            'qfd': '#d62728',
            'standard': '#1f77b4', 
            'lambda_cdm': '#ff7f0e',
            'observed': '#2ca02c'
        }
    
    def plot_hubble_diagram(self, 
                           hubble_data: Dict,
                           save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot Hubble diagram with QFD effects.
        
        Parameters:
        -----------
        hubble_data : Dict
            Hubble diagram data from RedshiftAnalyzer
        save_path : str or Path, optional
            Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        redshifts = hubble_data['redshifts']
        m_standard = hubble_data['magnitudes_standard']
        m_qfd = hubble_data['magnitudes_qfd']
        qfd_dimming = hubble_data['qfd_dimming']
        
        # Main Hubble diagram
        ax1.plot(redshifts, m_standard, 'k--', linewidth=2, 
                label='Standard Candle', alpha=0.7)
        ax1.plot(redshifts, m_qfd, color=self.colors['qfd'], linewidth=3,
                label='With QFD Dimming', marker='o', markersize=3, alpha=0.8)
        
        ax1.set_xlabel('Redshift z', fontsize=12)
        ax1.set_ylabel('Apparent Magnitude', fontsize=12)
        ax1.set_title('QFD Hubble Diagram', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # QFD dimming vs redshift
        ax2.plot(redshifts, qfd_dimming, color=self.colors['qfd'], 
                linewidth=3, marker='o', markersize=3)
        ax2.set_xlabel('Redshift z', fontsize=12)
        ax2.set_ylabel('QFD Dimming (magnitudes)', fontsize=12)
        ax2.set_title('QFD Dimming Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_qfd_vs_lambda_cdm(self,
                              comparison_data: Dict,
                              save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot QFD vs ΛCDM model comparison.
        
        Parameters:
        -----------
        comparison_data : Dict
            Comparison data from RedshiftAnalyzer
        save_path : str or Path, optional
            Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        redshifts = comparison_data['redshifts']
        m_qfd = comparison_data['qfd_magnitudes']
        m_lambda_cdm = comparison_data['lambda_cdm_magnitudes']
        differences = comparison_data['magnitude_differences']
        
        # Model comparison
        ax1.plot(redshifts, m_lambda_cdm, color=self.colors['lambda_cdm'], 
                linewidth=2, label='ΛCDM Model', alpha=0.8)
        ax1.plot(redshifts, m_qfd, color=self.colors['qfd'], 
                linewidth=3, label='QFD Model', alpha=0.8)
        
        ax1.set_xlabel('Redshift z', fontsize=12)
        ax1.set_ylabel('Apparent Magnitude', fontsize=12)
        ax1.set_title('QFD vs ΛCDM Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # Magnitude differences
        ax2.plot(redshifts, differences, color='purple', linewidth=2, 
                marker='o', markersize=4)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Redshift z', fontsize=12)
        ax2.set_ylabel('Magnitude Difference (QFD - ΛCDM)', fontsize=12)
        ax2.set_title('Model Differences', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add RMS text
        rms = comparison_data['rms_difference']
        ax2.text(0.05, 0.95, f'RMS Difference: {rms:.3f} mag',
                transform=ax2.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_validation_results(self,
                               validation_data: Dict,
                               save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot model validation against observations.
        
        Parameters:
        -----------
        validation_data : Dict
            Validation data from RedshiftAnalyzer
        save_path : str or Path, optional
            Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        z_test = validation_data['test_redshifts']
        observed = validation_data['observed_dimming']
        predicted = validation_data['qfd_predictions']
        residuals = validation_data['residuals']
        
        # Observed vs predicted
        ax1.plot(z_test, observed, 'o', color=self.colors['observed'],
                markersize=8, label='Observed', linewidth=2)
        ax1.plot(z_test, predicted, 's', color=self.colors['qfd'],
                markersize=8, label='QFD Model', linewidth=2)
        ax1.plot(z_test, predicted, '--', color=self.colors['qfd'], alpha=0.7)
        
        ax1.set_xlabel('Redshift z', fontsize=12)
        ax1.set_ylabel('Dimming (magnitudes)', fontsize=12)
        ax1.set_title('Model Validation', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        ax2.bar(z_test, residuals, width=0.05, color=self.colors['qfd'], alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Redshift z', fontsize=12)
        ax2.set_ylabel('Residual (Predicted - Observed)', fontsize=12)
        ax2.set_title('Model Residuals', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        rms = validation_data['rms_error']
        max_err = validation_data['max_error']
        ax2.text(0.05, 0.95, f'RMS Error: {rms:.3f} mag\\nMax Error: {max_err:.3f} mag',
                transform=ax2.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_comprehensive_analysis(self,
                                   results: Dict,
                                   save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create comprehensive 4-panel analysis plot.
        
        Parameters:
        -----------
        results : Dict
            Complete analysis results from RedshiftAnalyzer
        save_path : str or Path, optional
            Path to save plot
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Hubble diagram
        ax1 = fig.add_subplot(gs[0, 0])
        hubble_data = results['hubble_diagram']
        
        redshifts = hubble_data['redshifts']
        m_standard = hubble_data['magnitudes_standard']
        m_qfd = hubble_data['magnitudes_qfd']
        
        ax1.plot(redshifts, m_standard, 'k--', linewidth=2, 
                label='Standard Candle', alpha=0.7)
        ax1.plot(redshifts, m_qfd, color=self.colors['qfd'], linewidth=3,
                label='QFD Model', alpha=0.8)
        
        ax1.set_xlabel('Redshift z')
        ax1.set_ylabel('Apparent Magnitude')
        ax1.set_title('QFD Hubble Diagram', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # Panel 2: QFD dimming evolution
        ax2 = fig.add_subplot(gs[0, 1])
        qfd_dimming = hubble_data['qfd_dimming']
        
        ax2.plot(redshifts, qfd_dimming, color=self.colors['qfd'], 
                linewidth=3, marker='o', markersize=2)
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('QFD Dimming (mag)')
        ax2.set_title('QFD Dimming vs Redshift', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Model comparison
        ax3 = fig.add_subplot(gs[1, 0])
        comparison = results['lambda_cdm_comparison']
        
        ax3.plot(comparison['redshifts'], comparison['lambda_cdm_magnitudes'],
                color=self.colors['lambda_cdm'], linewidth=2, label='ΛCDM', alpha=0.8)
        ax3.plot(comparison['redshifts'], comparison['qfd_magnitudes'],
                color=self.colors['qfd'], linewidth=3, label='QFD', alpha=0.8)
        
        ax3.set_xlabel('Redshift z')
        ax3.set_ylabel('Apparent Magnitude')
        ax3.set_title('QFD vs ΛCDM Models', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()
        
        # Panel 4: Validation
        ax4 = fig.add_subplot(gs[1, 1])
        validation = results['validation']
        
        z_test = validation['test_redshifts']
        observed = validation['observed_dimming']
        predicted = validation['qfd_predictions']
        
        ax4.plot(z_test, observed, 'o', color=self.colors['observed'],
                markersize=8, label='Observed')
        ax4.plot(z_test, predicted, 's', color=self.colors['qfd'],
                markersize=8, label='QFD Model')
        ax4.plot(z_test, predicted, '--', color=self.colors['qfd'], alpha=0.7)
        
        ax4.set_xlabel('Redshift z')
        ax4.set_ylabel('Dimming (mag)')
        ax4.set_title('Model Validation', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle('QFD Redshift Analysis: Alternative to Dark Energy\\n' + 
                    'Wavelength-Independent Effects', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add summary statistics
        rms_error = validation['rms_error']
        rms_lambda_cdm = comparison['rms_difference']
        
        fig.text(0.02, 0.02, 
                f'Model Performance: RMS Error vs Observations = {rms_error:.3f} mag | ' +
                f'RMS Difference vs ΛCDM = {rms_lambda_cdm:.3f} mag',
                fontsize=10, style='italic')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_physics_scaling(self,
                            physics_data: Dict,
                            save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot QFD physics scaling analysis.
        
        Parameters:
        -----------
        physics_data : Dict
            Physics scaling data
        save_path : str or Path, optional
            Path to save plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        redshifts = physics_data['redshifts']
        
        # QFD coupling evolution
        ax1.plot(redshifts, physics_data['qfd_coupling_evolution'],
                color=self.colors['qfd'], linewidth=2)
        ax1.set_xlabel('Redshift z')
        ax1.set_ylabel('QFD Coupling Strength')
        ax1.set_title('QFD Coupling Evolution')
        ax1.grid(True, alpha=0.3)
        
        # IGM density evolution
        ax2.semilogy(redshifts, physics_data['igm_density_evolution'],
                    color='green', linewidth=2)
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('IGM Density Factor')
        ax2.set_title('IGM Density Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Path length effects
        ax3.loglog(redshifts, physics_data['path_length_effects'],
                  color='orange', linewidth=2)
        ax3.set_xlabel('Redshift z')
        ax3.set_ylabel('Path Length (cm)')
        ax3.set_title('Cosmological Path Length')
        ax3.grid(True, alpha=0.3)
        
        # Total optical depth
        ax4.semilogy(redshifts, physics_data['total_optical_depth'],
                    color='purple', linewidth=2)
        ax4.set_xlabel('Redshift z')
        ax4.set_ylabel('QFD Optical Depth')
        ax4.set_title('QFD Optical Depth Evolution')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('QFD Physics Scaling Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_publication_figure(self,
                                 results: Dict,
                                 title: str = "QFD Redshift Analysis",
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
        # Use larger figure for publication
        fig = plt.figure(figsize=(20, 12))
        
        # Create complex grid layout
        gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3,
                              height_ratios=[1, 1, 0.6])
        
        # Main Hubble diagram (large panel)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        
        hubble_data = results['hubble_diagram']
        comparison = results['lambda_cdm_comparison']
        
        # Plot multiple models
        ax_main.plot(hubble_data['redshifts'], hubble_data['magnitudes_standard'],
                    'k--', linewidth=2, label='Standard Candle', alpha=0.7)
        ax_main.plot(comparison['redshifts'], comparison['lambda_cdm_magnitudes'],
                    color=self.colors['lambda_cdm'], linewidth=2, label='ΛCDM Model')
        ax_main.plot(hubble_data['redshifts'], hubble_data['magnitudes_qfd'],
                    color=self.colors['qfd'], linewidth=3, label='QFD Model')
        
        ax_main.set_xlabel('Redshift z', fontsize=14)
        ax_main.set_ylabel('Apparent Magnitude', fontsize=14)
        ax_main.set_title('Cosmological Models Comparison', fontsize=16, fontweight='bold')
        ax_main.legend(fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.invert_yaxis()
        
        # Side panels
        self._add_side_panels(fig, gs, results)
        
        # Overall title and caption
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
        
        # Add physics caption
        caption = ("QFD provides alternative explanation for cosmological dimming "
                  "without requiring dark energy or accelerating expansion. "
                  "Based on E144-validated nonlinear photon interactions.")
        
        fig.text(0.5, 0.02, caption, ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _add_side_panels(self, fig, gs, results):
        """Add side panels to publication figure."""
        
        # QFD dimming panel
        ax_dimming = fig.add_subplot(gs[0, 2])
        hubble_data = results['hubble_diagram']
        
        ax_dimming.plot(hubble_data['redshifts'], hubble_data['qfd_dimming'],
                       color=self.colors['qfd'], linewidth=2)
        ax_dimming.set_xlabel('Redshift z')
        ax_dimming.set_ylabel('QFD Dimming (mag)')
        ax_dimming.set_title('QFD Effect')
        ax_dimming.grid(True, alpha=0.3)
        
        # Validation panel
        ax_validation = fig.add_subplot(gs[1, 2])
        validation = results['validation']
        
        z_test = validation['test_redshifts']
        observed = validation['observed_dimming']
        predicted = validation['qfd_predictions']
        
        ax_validation.plot(z_test, observed, 'o', color=self.colors['observed'],
                          markersize=8, label='Observed')
        ax_validation.plot(z_test, predicted, 's', color=self.colors['qfd'],
                          markersize=8, label='QFD Model')
        
        ax_validation.set_xlabel('Redshift z')
        ax_validation.set_ylabel('Dimming (mag)')
        ax_validation.set_title('Validation')
        ax_validation.legend(fontsize=10)
        ax_validation.grid(True, alpha=0.3)
        
        # Statistics panel
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('off')
        
        # Add key statistics as text
        rms_error = validation['rms_error']
        rms_lambda_cdm = results['lambda_cdm_comparison']['rms_difference']
        
        stats_text = f"""
        Model Performance Summary:
        • RMS Error vs Observations: {rms_error:.3f} magnitudes
        • RMS Difference vs ΛCDM: {rms_lambda_cdm:.3f} magnitudes
        • Validation Passed: {validation['validation_passed']}
        • No Dark Energy Required: Physics-based alternative using E144-validated interactions
        """
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))