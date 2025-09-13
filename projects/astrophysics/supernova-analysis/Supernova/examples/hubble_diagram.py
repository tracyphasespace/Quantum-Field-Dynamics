#!/usr/bin/env python3
"""
Hubble Diagram Example
=====================

Demonstrates how QVD scattering affects the Hubble diagram and could
provide an alternative explanation to dark energy acceleration.
"""

import numpy as np
import matplotlib.pyplot as plt
from supernova_qvd_scattering import E144ExperimentalData, SupernovaParameters, E144ScaledQVDModel

def generate_hubble_diagram():
    """Generate Hubble diagram showing QVD vs standard cosmology"""
    
    print("QVD Hubble Diagram Analysis")
    print("=" * 40)
    
    # Create model
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Distance range for Hubble diagram
    distances_Mpc = np.logspace(1, 3.5, 20)  # 10 to ~3000 Mpc
    wavelength_nm = 500.0  # V-band
    
    print(f"Generating Hubble diagram for {len(distances_Mpc)} distances")
    print(f"Distance range: {distances_Mpc[0]:.1f} to {distances_Mpc[-1]:.1f} Mpc")
    
    # Arrays to store results
    redshifts = []
    magnitudes_intrinsic = []
    magnitudes_observed = []
    qvd_dimming = []
    
    # Hubble constant (km/s/Mpc)
    H0 = 70.0
    
    for distance in distances_Mpc:
        print(f"  Processing distance: {distance:.1f} Mpc")
        
        # Generate supernova curve
        curve = model.generate_luminance_curve(
            distance_Mpc=distance,
            wavelength_nm=wavelength_nm,
            time_range_days=(-10, 50),
            time_resolution_days=2.0
        )
        
        # Find peak magnitude (brightest point)
        peak_idx = np.argmin(curve['magnitude_observed'])
        mag_obs = curve['magnitude_observed'][peak_idx]
        mag_int = curve['magnitude_intrinsic'][peak_idx]
        dimming = curve['dimming_magnitudes'][peak_idx]
        
        # Calculate redshift (Hubble law: v = H0 * d, z = v/c)
        velocity_km_s = H0 * distance
        redshift = velocity_km_s / 299792.458  # c in km/s
        
        # Store results
        redshifts.append(redshift)
        magnitudes_intrinsic.append(mag_int)
        magnitudes_observed.append(mag_obs)
        qvd_dimming.append(dimming)
    
    # Convert to arrays
    redshifts = np.array(redshifts)
    magnitudes_intrinsic = np.array(magnitudes_intrinsic)
    magnitudes_observed = np.array(magnitudes_observed)
    qvd_dimming = np.array(qvd_dimming)
    
    # Create Hubble diagram plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Classic Hubble diagram
    ax1.plot(redshifts, magnitudes_intrinsic, 'k--', linewidth=2, 
             label='Standard Candle (no QVD)', alpha=0.7)
    ax1.plot(redshifts, magnitudes_observed, 'r-', linewidth=3,
             label='With QVD Scattering', marker='o', markersize=6)
    
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Peak Apparent Magnitude')
    ax1.set_title('Hubble Diagram: QVD vs Standard Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.set_xscale('log')
    
    # 2. Magnitude residuals (QVD effect)
    residuals = magnitudes_observed - magnitudes_intrinsic
    ax2.semilogx(redshifts, residuals, 'ro-', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Magnitude Residual (QVD - Standard)')
    ax2.set_title('QVD-induced Dimming vs Distance')
    ax2.grid(True, alpha=0.3)
    
    # 3. QVD dimming vs distance
    ax3.loglog(distances_Mpc, qvd_dimming, 'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Distance (Mpc)')
    ax3.set_ylabel('QVD Dimming (magnitudes)')
    ax3.set_title('QVD Scattering vs Distance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparison with dark energy
    # Theoretical dark energy dimming (approximate)
    # Based on accelerating universe models
    dark_energy_dimming = 0.2 * np.log10(1 + redshifts) * 2.5  # Rough approximation
    
    ax4.semilogx(redshifts, qvd_dimming, 'r-', linewidth=3, 
                 label='QVD Scattering', marker='o', markersize=6)
    ax4.semilogx(redshifts, dark_energy_dimming, 'b--', linewidth=2, 
                 label='Dark Energy (approx)', alpha=0.7)
    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('Additional Dimming (magnitudes)')
    ax4.set_title('QVD vs Dark Energy Explanation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('QVD Alternative to Dark Energy\n(Hubble Diagram Analysis)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hubble_diagram_qvd.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nHubble Diagram Results:")
    print(f"  Redshift range: {redshifts[0]:.4f} to {redshifts[-1]:.3f}")
    print(f"  QVD dimming range: {np.min(qvd_dimming):.3f} to {np.max(qvd_dimming):.3f} mag")
    print(f"  Average QVD effect: {np.mean(qvd_dimming):.3f} ± {np.std(qvd_dimming):.3f} mag")
    
    # Calculate correlation with distance
    correlation = np.corrcoef(np.log10(distances_Mpc), qvd_dimming)[0, 1]
    print(f"  Correlation with log(distance): {correlation:.3f}")
    
    return {
        'distances_Mpc': distances_Mpc,
        'redshifts': redshifts,
        'magnitudes_intrinsic': magnitudes_intrinsic,
        'magnitudes_observed': magnitudes_observed,
        'qvd_dimming': qvd_dimming
    }

if __name__ == "__main__":
    hubble_data = generate_hubble_diagram()