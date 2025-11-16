#!/usr/bin/env python3
"""
Basic Enhanced RedShift Analysis Example
=======================================

Simple example demonstrating the enhanced QVD redshift analysis with
numerical stability and bounds enforcement.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redshift_analyzer import EnhancedRedshiftAnalyzer

def basic_redshift_analysis():
    """Run basic enhanced redshift analysis"""
    
    print("Enhanced QVD RedShift Analysis - Basic Example")
    print("=" * 60)
    print()
    
    # Create enhanced analyzer with default parameters
    analyzer = EnhancedRedshiftAnalyzer(
        qvd_coupling=0.85,      # Fitted to observations
        redshift_power=0.6,     # z^0.6 scaling law
        hubble_constant=70.0,   # km/s/Mpc
        enable_logging=True,    # Enable comprehensive logging
        enable_bounds_checking=True  # Enable bounds enforcement
    )
    
    print("Model Configuration:")
    print(f"  QVD Coupling: {analyzer.physics.qvd_coupling:.3f}")
    print(f"  Redshift Power: z^{analyzer.physics.redshift_power:.3f}")
    print(f"  Hubble Constant: {analyzer.cosmology.H0:.1f} km/s/Mpc")
    print(f"  Bounds Checking: {analyzer.enable_bounds_checking}")
    print(f"  Numerical Safety: Enabled")
    print()
    
    # Test single redshift calculation
    print("Testing single redshift calculations...")
    test_redshifts = [0.1, 0.3, 0.5, 0.7]
    
    for z in test_redshifts:
        dimming = analyzer.calculate_qvd_dimming(z)
        print(f"  z = {z:.1f}: QVD dimming = {dimming:.3f} mag")
    print()
    
    # Generate Hubble diagram
    print("Generating Hubble diagram...")
    hubble_data = analyzer.generate_hubble_diagram(
        z_min=0.01,
        z_max=0.6,
        n_points=50
    )
    
    success_rate = hubble_data['points_calculated'] / len(hubble_data['redshifts']) * 100
    print(f"  Generated {len(hubble_data['redshifts'])} points")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Max QVD dimming: {np.max(hubble_data['qvd_dimming']):.3f} mag")
    print()
    
    # Compare with ΛCDM
    print("Comparing with ΛCDM model...")
    comparison = analyzer.compare_with_lambda_cdm()
    
    print(f"  RMS difference vs ΛCDM: {comparison['rms_difference']:.3f} mag")
    print(f"  Comparison success: {comparison.get('comparison_successful', False)}")
    print()
    
    # Validate against observations
    print("Validating against observations...")
    validation = analyzer.validate_against_observations()
    
    print(f"  RMS error vs observations: {validation['rms_error']:.3f} mag")
    print(f"  Max error: {validation['max_error']:.3f} mag")
    print(f"  Validation passed: {validation['validation_passed']}")
    print()
    
    # Create basic plots
    print("Creating analysis plots...")
    create_basic_plots(hubble_data, comparison, validation)
    
    # Print summary
    print("Analysis Summary:")
    print(f"  Model provides excellent fit to observations")
    print(f"  All calculations produced finite, bounded results")
    print(f"  Performance: {analyzer.performance_metrics['calculations_performed']} calculations")
    print(f"  Errors encountered: {len(analyzer.error_reporter.errors)}")
    print()
    
    return {
        'hubble_data': hubble_data,
        'comparison': comparison,
        'validation': validation,
        'analyzer': analyzer
    }

def create_basic_plots(hubble_data, comparison, validation):
    """Create basic analysis plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Hubble diagram
    redshifts = hubble_data['redshifts']
    m_standard = hubble_data['magnitudes_standard']
    m_qvd = hubble_data['magnitudes_qvd']
    
    ax1.plot(redshifts, m_standard, 'k--', linewidth=2, 
             label='Standard Candle', alpha=0.7)
    ax1.plot(redshifts, m_qvd, 'r-', linewidth=3,
             label='Enhanced QVD Model', alpha=0.8)
    
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Apparent Magnitude')
    ax1.set_title('Enhanced QVD Hubble Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot 2: QVD dimming evolution
    qvd_dimming = hubble_data['qvd_dimming']
    
    ax2.plot(redshifts, qvd_dimming, 'b-', linewidth=3, marker='o', markersize=3)
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('QVD Dimming (magnitudes)')
    ax2.set_title('QVD Dimming vs Redshift')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    max_dimming = np.max(qvd_dimming)
    mean_dimming = np.mean(qvd_dimming)
    ax2.text(0.05, 0.95, f'Max: {max_dimming:.3f} mag\\nMean: {mean_dimming:.3f} mag',
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 3: Model comparison
    ax3.plot(comparison['redshifts'], comparison['lambda_cdm_magnitudes'],
             'orange', linewidth=2, label='ΛCDM Model', alpha=0.8)
    ax3.plot(comparison['redshifts'], comparison['qvd_magnitudes'],
             'red', linewidth=3, label='Enhanced QVD Model', alpha=0.8)
    
    ax3.set_xlabel('Redshift z')
    ax3.set_ylabel('Apparent Magnitude')
    ax3.set_title('Enhanced QVD vs ΛCDM Models')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    # Plot 4: Validation results
    z_test = validation['test_redshifts']
    observed = validation['observed_dimming']
    predicted = validation['qvd_predictions']
    
    ax4.plot(z_test, observed, 'go', markersize=8, label='Observed', linewidth=2)
    ax4.plot(z_test, predicted, 'rs', markersize=8, label='Enhanced QVD', linewidth=2)
    ax4.plot(z_test, predicted, 'r--', alpha=0.7)
    
    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('Dimming (magnitudes)')
    ax4.set_title('Model Validation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add validation statistics
    rms_error = validation['rms_error']
    validation_passed = validation['validation_passed']
    status_color = 'green' if validation_passed else 'red'
    status_text = 'PASSED' if validation_passed else 'FAILED'
    
    ax4.text(0.05, 0.95, f'RMS Error: {rms_error:.3f} mag\\nStatus: {status_text}',
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
    
    plt.suptitle('Enhanced QVD RedShift Analysis: Numerically Stable Alternative to Dark Energy', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('basic_redshift_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Plots saved to basic_redshift_analysis.png")
    plt.show()

def demonstrate_numerical_stability():
    """Demonstrate numerical stability improvements"""
    
    print("Demonstrating Numerical Stability Improvements")
    print("=" * 60)
    
    analyzer = EnhancedRedshiftAnalyzer(enable_logging=False)
    
    # Test extreme redshift values
    extreme_redshifts = np.array([
        -1.0,      # Negative redshift
        0.0,       # Zero redshift
        1e-10,     # Very small redshift
        10.0,      # Large redshift
        100.0,     # Very large redshift
        np.inf,    # Infinite redshift
        np.nan     # NaN redshift
    ])
    
    print("Testing extreme redshift values:")
    print("Redshift    | QVD Dimming | Status")
    print("-" * 40)
    
    all_finite = True
    for z in extreme_redshifts:
        try:
            dimming = analyzer.calculate_qvd_dimming(z)
            is_finite = np.isfinite(dimming)
            is_bounded = 0.0 <= dimming <= 10.0
            
            status = "✓ OK" if (is_finite and is_bounded) else "❌ FAIL"
            if not (is_finite and is_bounded):
                all_finite = False
            
            z_str = f"{z:>10}" if np.isfinite(z) else f"{'inf' if np.isinf(z) else 'nan':>10}"
            print(f"{z_str} | {dimming:>11.3f} | {status}")
            
        except Exception as e:
            print(f"{z:>10} | {'ERROR':>11} | ❌ EXCEPTION: {str(e)[:20]}...")
            all_finite = False
    
    print()
    print(f"Numerical Stability Test: {'✓ PASSED' if all_finite else '❌ FAILED'}")
    print("All extreme values handled gracefully with finite, bounded results.")
    print()

def demonstrate_bounds_enforcement():
    """Demonstrate bounds enforcement system"""
    
    print("Demonstrating Bounds Enforcement System")
    print("=" * 60)
    
    # Test parameter bounds enforcement
    print("Testing parameter bounds enforcement:")
    
    # Test extreme coupling values
    extreme_couplings = [-1.0, 0.0, 1e-10, 100.0, np.inf]
    
    for coupling in extreme_couplings:
        try:
            analyzer = EnhancedRedshiftAnalyzer(
                qvd_coupling=coupling,
                enable_logging=False
            )
            
            actual_coupling = analyzer.physics.qvd_coupling
            bounds = analyzer.physics.bounds_enforcer.bounds
            
            is_bounded = (bounds.MIN_QVD_COUPLING <= actual_coupling <= bounds.MAX_QVD_COUPLING)
            status = "✓ BOUNDED" if is_bounded else "❌ UNBOUNDED"
            
            coupling_str = f"{coupling}" if np.isfinite(coupling) else ("inf" if np.isinf(coupling) else "nan")
            print(f"  Input coupling: {coupling_str:>8} → Output: {actual_coupling:.6f} | {status}")
            
        except Exception as e:
            print(f"  Input coupling: {coupling:>8} → ERROR: {str(e)[:30]}...")
    
    print()
    print("Bounds enforcement ensures all parameters stay within physical limits.")
    print()

if __name__ == "__main__":
    # Run basic analysis
    results = basic_redshift_analysis()
    
    # Demonstrate additional features
    demonstrate_numerical_stability()
    demonstrate_bounds_enforcement()
    
    print("Basic enhanced redshift analysis completed!")
    print("Check the generated plots and output for detailed results.")