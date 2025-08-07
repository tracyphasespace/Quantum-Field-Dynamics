"""
Visualization Module for QFD Supernova Analysis
================================================

Provides the SupernovaPlotter class for creating all analysis plots.
This module uses matplotlib to generate visualizations of the QFD
supernova model results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict

class SupernovaPlotter:
    """
    Creates plots for the QFD supernova analysis.
    """

    def __init__(self):
        """
        Initialize the plotter.
        """
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_multi_wavelength_curves(self, data: Dict, save_path: Path):
        """
        Plots multi-wavelength light curves.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        curves = data.get('curves', {})
        colors = {'U': 'purple', 'B': 'blue', 'V': 'green', 'R': 'red', 'I': 'darkred'}

        for band, curve in curves.items():
            # The 'color_evolution' key is also in this dict, but has a different structure.
            # We skip it in this loop and handle it separately.
            if band == 'color_evolution':
                continue

            if 'time_days' not in curve: continue
            ax1.plot(curve['time_days'], curve['magnitude_observed'],
                     color=colors.get(band, 'black'), label=f'{band}-band Observed')
            ax1.plot(curve['time_days'], curve['magnitude_intrinsic'],
                     '--', color=colors.get(band, 'black'), alpha=0.5, label=f'{band}-band Intrinsic')

        ax1.invert_yaxis()
        ax1.set_ylabel('Apparent Magnitude')
        ax1.set_title(f"Multi-Wavelength Light Curves (Redshift z={data.get('redshift', 'N/A')})")
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        if 'color_evolution' in curves:
            color_data = curves['color_evolution']
            ax2.plot(color_data['time_days'], color_data['bv_observed'], 'b-', label='Observed B-V')
            ax2.plot(color_data['time_days'], color_data['bv_intrinsic'], 'k--', label='Intrinsic B-V')
            ax2.set_ylabel('B-V Color Index')
            ax2.set_xlabel('Time (days)')
            ax2.set_title('B-V Color Evolution')
            ax2.legend()
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_hubble_diagram(self, data: Dict, save_path: Path):
        """
        Plots the Hubble diagram.
        """
        plt.figure(figsize=(8, 6))

        distance_modulus_obs = data['magnitudes_observed'] - (-19.3) # M_abs = -19.3
        distance_modulus_int = data['magnitudes_intrinsic'] - (-19.3)

        plt.plot(data['redshifts'], distance_modulus_int, 'k--', label='Expected (No QVD)')
        plt.plot(data['redshifts'], distance_modulus_obs, 'ro', label='Observed (with QVD)')

        plt.xscale('log')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Modulus (m - M)')
        plt.title('Supernova Hubble Diagram with QVD Effects')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_validation_results(self, data: Dict, save_path: Path):
        """
        Plots the validation results against observations.
        """
        plt.figure(figsize=(8, 6))

        plt.plot(data['test_redshifts'], data['observed_dimming'], 'ko', label='Observed Dimming')
        plt.plot(data['test_redshifts'], data['qvd_predictions'], 'r-', label='QVD Prediction')

        plt.xlabel('Redshift (z)')
        plt.ylabel('Excess Dimming (magnitudes)')
        plt.title('Model Validation vs. Observed Dimming')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        rms_error = data.get('rms_error', 0)
        plt.text(0.05, 0.95, f'RMS Error: {rms_error:.3f} mag',
                 transform=plt.gca().transAxes, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_wavelength_dependence(self, data: Dict, save_path: Path):
        """
        Plots the wavelength dependence of QVD effects.
        """
        plt.figure(figsize=(8, 6))

        plt.plot(data['wavelengths'], data['qvd_dimming'], 'b-')

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('QVD Dimming (magnitudes)')
        plt.title('Wavelength Dependence of QVD Scattering')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_comprehensive_analysis(self, results: Dict, save_path: Path):
        """
        Creates a comprehensive summary plot.
        """
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Comprehensive Supernova Analysis Summary', fontsize=16)

        # Hubble Diagram
        hubble_data = results.get('hubble_diagram', {})
        if hubble_data:
            distance_modulus_obs = hubble_data['magnitudes_observed'] - (-19.3)
            distance_modulus_int = hubble_data['magnitudes_intrinsic'] - (-19.3)
            axs[0, 0].plot(hubble_data['redshifts'], distance_modulus_int, 'k--', label='Expected')
            axs[0, 0].plot(hubble_data['redshifts'], distance_modulus_obs, 'ro', label='QVD')
            axs[0, 0].set_xscale('log')
            axs[0, 0].set_title('Hubble Diagram')
            axs[0, 0].set_xlabel('Redshift (z)')
            axs[0, 0].set_ylabel('Distance Modulus')
            axs[0, 0].legend()

        # Wavelength Dependence
        wave_data = results.get('wavelength_dependence', {})
        if wave_data:
            axs[0, 1].plot(wave_data['wavelengths'], wave_data['qvd_dimming'])
            axs[0, 1].set_title('Wavelength Dependence')
            axs[0, 1].set_xlabel('Wavelength (nm)')
            axs[0, 1].set_ylabel('QVD Dimming (mag)')

        # Validation
        val_data = results.get('validation', {})
        if val_data:
            axs[1, 0].plot(val_data['test_redshifts'], val_data['observed_dimming'], 'ko', label='Observed')
            axs[1, 0].plot(val_data['test_redshifts'], val_data['qvd_predictions'], 'r-', label='QVD')
            axs[1, 0].set_title('Validation')
            axs[1, 0].set_xlabel('Redshift (z)')
            axs[1, 0].set_ylabel('Excess Dimming (mag)')
            axs[1, 0].legend()

        # Multi-wavelength curves (peak dimming)
        multi_wave_data = results.get('multi_wavelength', {})
        if multi_wave_data:
            bands = []
            dimmings = []
            wavelengths = multi_wave_data.get('wavelength_bands', {})
            for band, curve in multi_wave_data.get('curves', {}).items():
                if band == 'color_evolution':
                    continue
                if 'time_days' in curve:
                    bands.append(band)
                    dimmings.append(np.max(curve['qvd_dimming']))

            axs[1, 1].bar(bands, dimmings, color='purple')
            axs[1, 1].set_title('Peak Dimming by Band')
            axs[1, 1].set_xlabel('Photometric Band')
            axs[1, 1].set_ylabel('Max QVD Dimming (mag)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.close()
