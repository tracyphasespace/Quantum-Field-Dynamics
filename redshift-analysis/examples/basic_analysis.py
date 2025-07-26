#!/usr/bin/env python3
"""
Basic QFD Redshift Analysis Example
==================================

Simple example demonstrating QFD redshift analysis.
"""

from qfd_redshift import RedshiftAnalyzer

def main():
    """Run basic QFD redshift analysis."""
    
    print("QFD Redshift Analysis - Basic Example")
    print("=" * 50)
    
    # Create analyzer with default parameters
    analyzer = RedshiftAnalyzer(
        qfd_coupling=0.85,      # Fitted to observations
        redshift_power=0.6,     # z^0.6 scaling
        hubble_constant=70.0    # km/s/Mpc
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis("basic_results")
    
    print("\\nBasic analysis completed!")
    print("Check 'basic_results/' directory for plots and data.")

if __name__ == "__main__":
    main()