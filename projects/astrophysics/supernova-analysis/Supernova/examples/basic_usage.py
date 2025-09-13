#!/usr/bin/env python3
"""
Basic Usage Example for Supernova QVD Model
==========================================

Demonstrates basic usage of the E144-scaled QVD supernova model.
"""

import numpy as np
import matplotlib.pyplot as plt
from supernova_qvd_scattering import E144ExperimentalData, SupernovaParameters, E144ScaledQVDModel

def basic_supernova_curve():
    """Generate a basic supernova light curve with QVD scattering"""
    
    print("Basic Supernova QVD Example")
    print("=" * 40)
    
    # Create model with default parameters
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Generate light curve for a typical supernova
    distance_Mpc = 100.0  # 100 megaparsecs
    wavelength_nm = 500.0  # V-band (green)
    
    print(f"Generating curve for:")
    print(f"  Distance: {distance_Mpc} Mpc")
    print(f"  Wavelength: {wavelength_nm} nm")
    
    curve = model.generate_luminance_curve(
        distance_Mpc=distance_Mpc,
        wavelength_nm=wavelength_nm,
        time_range_days=(-20, 100),
        time_resolution_days=1.0
    )
    
    # Verify all values are finite
    finite_check = np.all(np.isfinite(curve['magnitude_observed']))
    print(f"  All values finite: {finite_check}")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Main light curve
    plt.subplot(2, 2, 1)
    plt.plot(curve['time_days'], curve['magnitude_intrinsic'], 'k--', 
             linewidth=2, label='Intrinsic (no QVD)', alpha=0.7)
    plt.plot(curve['time_days'], curve['magnitude_observed'], 'r-', 
             linewidth=2, label='Observed (with QVD)')
    plt.xlabel('Days since maximum')
    plt.ylabel('Apparent Magnitude')
    plt.title('Supernova Light Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Brighter magnitudes up
    
    # QVD dimming effect
    plt.subplot(2, 2, 2)
    plt.plot(curve['time_days'], curve['dimming_magnitudes'], 'b-', linewidth=2)
    plt.xlabel('Days since maximum')
    plt.ylabel('QVD Dimming (mag)')
    plt.title('QVD Scattering Effect')
    plt.grid(True, alpha=0.3)
    
    # Optical depth evolution
    plt.subplot(2, 2, 3)
    plt.semilogy(curve['time_days'], curve['optical_depths'], 'g-', linewidth=2)
    plt.xlabel('Days since maximum')
    plt.ylabel('Optical Depth')
    plt.title('QVD Optical Depth')
    plt.grid(True, alpha=0.3)
    
    # Luminosity evolution
    plt.subplot(2, 2, 4)
    plt.semilogy(curve['time_days'], curve['luminosity_intrinsic_erg_s'], 'k--', 
                 linewidth=2, label='Intrinsic', alpha=0.7)
    plt.semilogy(curve['time_days'], curve['luminosity_observed_erg_s'], 'r-', 
                 linewidth=2, label='Observed')
    plt.xlabel('Days since maximum')
    plt.ylabel('Luminosity (erg/s)')
    plt.title('Luminosity Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Basic QVD Supernova Model Example', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('basic_supernova_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some key results
    max_dimming = np.max(curve['dimming_magnitudes'])
    max_optical_depth = np.max(curve['optical_depths'])
    
    print(f"\nKey Results:")
    print(f"  Maximum QVD dimming: {max_dimming:.3f} magnitudes")
    print(f"  Maximum optical depth: {max_optical_depth:.3e}")
    print(f"  Peak intrinsic magnitude: {curve['magnitude_intrinsic'][0]:.2f}")
    print(f"  Peak observed magnitude: {np.min(curve['magnitude_observed']):.2f}")
    
    return curve

if __name__ == "__main__":
    curve = basic_supernova_curve()