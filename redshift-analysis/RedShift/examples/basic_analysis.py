#!/usr/bin/env python3
"""
Enhanced Basic QVD Redshift Analysis Example
===========================================

Demonstrates the enhanced QVD redshift analysis with numerical safety
and comprehensive validation.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from redshift_qvd_analyzer import EnhancedRedshiftAnalyzer
from enhanced_visualization import EnhancedRedshiftPlotter
import logging

def main():
    """Run enhanced basic QVD redshift analysis."""
    
    print("Enhanced QVD Redshift Analysis - Basic Example")
    print("=" * 60)
    print()
    
    # Create analyzer with enhanced safety
    analyzer = EnhancedRedshiftAnalyzer(
        qvd_coupling=0.85,      # Fitted to observations
        redshift_power=0.6,     # z^0.6 scaling law
        hubble_constant=70.0,   # km/s/Mpc
        enable_logging=True     # Enable comprehensive logging
    )
    
    print("Model Parameters:")
    print(f"  QVD Coupling: {analyzer.model_params['qvd_coupling']}")
    print(f"  Redshift Power: z^{analyzer.model_params['redshift_power']}")
    print(f"  Hubble Constant: {analyzer.model_params['hubble_constant']} km/s/Mpc")
    print(f"  Model Version: {analyzer.model_params['model_version']}")
    print()
    
    try:
        # Run complete analysis with enhanced error handling
        print("Running complete analysis...")
        results = analyzer.run_complete_analysis("basic_results")
        
        # Display key results
        print("\n" + "="*60)
        print("ANALYSIS RESULTS SUMMARY")
        print("="*60)
        
        # Validation results
        if 'validation' in results:
            val_metrics = results['validation']['statistical_metrics']
            print(f"Validation Performance:")
            print(f"  RMS Error vs Observations: {val_metrics['rms_error']:.3f} mag")
            print(f"  Quality Grade: {results['validation']['quality_grade']}")
            print(f"  Validation Status: {'PASSED' if results['validation']['validation_passed'] else 'FAILED'}")
            print()
        
        # ΛCDM comparison
        if 'lambda_cdm_comparison' in results:
            comp_metrics = results['lambda_cdm_comparison']['statistical_metrics']
            print(f"ΛCDM Comparison:")
            print(f"  RMS Difference: {comp_metrics['rms_difference']:.3f} mag")
            print(f"  Mean Difference: {comp_metrics['mean_difference']:.3f} mag")
            print()
        
        # Hubble diagram statistics
        if 'hubble_diagram' in results:
            hubble_data = results['hubble_diagram']
            print(f"Hubble Diagram:")
            print(f"  Redshift Range: {hubble_data['redshifts'][0]:.3f} - {hubble_data['redshifts'][-1]:.3f}")
            print(f"  Data Points: {len(hubble_data['redshifts'])}")
            print(f"  Max QVD Dimming: {max(hubble_data['qvd_dimming']):.3f} mag")
            print()
        
        # Error reporting
        if hasattr(analyzer, 'error_reporter') and analyzer.error_reporter.errors:
            print(f"Errors Encountered: {len(analyzer.error_reporter.errors)}")
            for i, error in enumerate(analyzer.error_reporter.errors[:3]):  # Show first 3
                print(f"  {i+1}. {error['type']}: {error['message']}")
            if len(analyzer.error_reporter.errors) > 3:
                print(f"  ... and {len(analyzer.error_reporter.errors) - 3} more")
        else:
            print("No errors encountered - analysis completed successfully!")
        
        print(f"\nDetailed results saved to: basic_results/")
        print("Files created:")
        print("  • enhanced_qvd_redshift_results.json - Complete analysis data")
        print("  • hubble_diagram.png - Hubble diagram visualization")
        print("  • qvd_vs_lambda_cdm.png - Model comparison")
        print("  • validation_results.png - Validation analysis")
        print("  • comprehensive_analysis.png - Complete summary")
        
        # Demonstrate individual calculations
        print("\n" + "="*60)
        print("INDIVIDUAL CALCULATION EXAMPLES")
        print("="*60)
        
        test_redshifts = [0.1, 0.3, 0.5, 0.7]
        print("QVD Dimming Calculations:")
        for z in test_redshifts:
            dimming = analyzer.calculate_qvd_dimming(z)
            print(f"  z = {z:.1f}: {dimming:.3f} mag")
        
        print("\nCosmological Distance Calculations:")
        for z in test_redshifts:
            distance = analyzer.cosmology.luminosity_distance(z)
            print(f"  z = {z:.1f}: {distance:.1f} Mpc")
        
        return True
        
    except Exception as e:
        print(f"\nAnalysis failed with error: {e}")
        print("This may indicate a problem with the input data or model parameters.")
        
        # Show error details if available
        if hasattr(analyzer, 'error_reporter') and analyzer.error_reporter.errors:
            print(f"\nDetailed errors ({len(analyzer.error_reporter.errors)} total):")
            for error in analyzer.error_reporter.errors[-5:]:  # Show last 5
                print(f"  • {error['type']}: {error['message']}")
        
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Basic analysis completed successfully!")
        print("Try running other examples:")
        print("  • python examples/comprehensive_analysis.py")
        print("  • python examples/parameter_study.py")
        print("  • python examples/observational_comparison.py")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")
    
    sys.exit(0 if success else 1)