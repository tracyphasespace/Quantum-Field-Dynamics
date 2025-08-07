#!/usr/bin/env python3
"""
Multi-wavelength Analysis Example
=================================

Demonstrates multi-wavelength supernova analysis with QVD scattering.
"""

import numpy as np
import matplotlib.pyplot as plt
from supernova_qvd_scattering import E144ExperimentalData, SupernovaParameters, E144ScaledQVDModel

def multi_wavelength_analysis():
    """Analyze QVD effects across multiple wavelengths"""
    
    print("Multi-wavelength QVD Analysis")
    print("=" * 40)
    
    # Create model
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Generate multi-wavelength curves
    distance_Mpc = 100.0
    wavelengths_nm = [400, 500, 600, 700, 800]  # B, V, R, I bands
    
    print(f"Analyzing wavelengths: {wavelengths_nm} nm")
    
    curves_data = model.generate_multi_wavelength_curves(
        distance_Mpc=distance_Mpc,
        wavelengths_nm=wavelengths_nm,
        time_range_days=(-20, 100)
    )
    
    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'green', 'orange', 'red', 'darkred']
    
    # 1. Multi-wavelength light curves
    for wavelength, color in zip(wavelengths_nm, colors):
        key = f'{wavelength}nm'
        if key in curves_data:
            curve = curves_data[key]
            ax1.plot(curve['time_days'], curve['magnitude_observed'], 
                    color=color, linewidth=2, label=f'{wavelength}nm')
    
    ax1.set_xlabel('Days since maximum')
    ax1.set_ylabel('Apparent Magnitude')
    ax1.set_title('Multi-wavelength Light Curves (with QVD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.set_xlim(-20, 100)
    
    # 2. QVD dimming by wavelength
    for wavelength, color in zip(wavelengths_nm, colors):
        key = f'{wavelength}nm'
        if key in curves_data:
            curve = curves_data[key]
            ax2.plot(curve['time_days'], curve['dimming_magnitudes'], 
                    color=color, linewidth=2, label=f'{wavelength}nm')
    
    ax2.set_xlabel('Days since maximum')
    ax2.set_ylabel('QVD Dimming (magnitudes)')
    ax2.set_title('Wavelength-dependent QVD Scattering')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-20, 100)
    
    # 3. Color evolution
    if 'color_evolution' in curves_data:
        color_data = curves_data['color_evolution']
        ax3.plot(color_data['time_days'], color_data['intrinsic_B_minus_R'], 
                'k--', linewidth=2, label='Intrinsic B-R', alpha=0.7)
        ax3.plot(color_data['time_days'], color_data['B_minus_R'], 
                'purple', linewidth=3, label='Observed B-R (QVD)')
        ax3.set_xlabel('Days since maximum')
        ax3.set_ylabel('B - R Color Index')
        ax3.set_title('Color Evolution with QVD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-20, 100)
    
    # 4. Spectral dimming at peak
    peak_time = 0  # Days
    peak_wavelengths = []
    peak_dimming = []
    
    for wavelength in wavelengths_nm:
        key = f'{wavelength}nm'
        if key in curves_data:
            curve = curves_data[key]
            # Find closest time to peak
            peak_idx = np.argmin(np.abs(curve['time_days'] - peak_time))
            peak_wavelengths.append(wavelength)
            peak_dimming.append(curve['dimming_magnitudes'][peak_idx])
    
    ax4.plot(peak_wavelengths, peak_dimming, 'ro-', linewidth=2, markersize=8)
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('QVD Dimming at Peak (mag)')
    ax4.set_title('Spectral Dependence of QVD Scattering')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-wavelength QVD Supernova Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('multi_wavelength_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print analysis results
    print(f"\nAnalysis Results:")
    for wavelength in wavelengths_nm:
        key = f'{wavelength}nm'
        if key in curves_data:
            curve = curves_data[key]
            max_dimming = np.max(curve['dimming_magnitudes'])
            print(f"  {wavelength}nm: Max dimming = {max_dimming:.3f} mag")
    
    return curves_data

if __name__ == "__main__":
    curves = multi_wavelength_analysis()