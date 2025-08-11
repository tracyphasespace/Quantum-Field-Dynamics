#!/usr/bin/env python3
"""
Integrated realm analysis script.

This script runs the complete QFD realm sequence with coupling constants
analysis, generating comprehensive reports and visualizations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from coupling_constants.integration.workflow_scripts import run_integrated_analysis


def main():
    """Main entry point for integrated realm analysis."""
    parser = argparse.ArgumentParser(
        description="Run QFD realm sequence with coupling constants analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with default settings
  python scripts/integrated_realm_analysis.py
  
  # Run with specific plugins and custom output directory
  python scripts/integrated_realm_analysis.py --output results --plugins vacuum_stability photon_mass
  
  # Run with verbose logging and no visualizations
  python scripts/integrated_realm_analysis.py --verbose --no-visualizations
        """
    )
    
    parser.add_argument('--config', default='qfd_params/defaults.yaml',
                       help='Path to QFD configuration file')
    parser.add_argument('--output', default='integrated_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--plugins', nargs='*', 
                       choices=['photon_mass', 'vacuum_stability', 'cosmological_constant'],
                       help='Enable constraint plugins')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting integrated QFD realm analysis")
    
    try:
        # Run integrated analysis
        report_path = run_integrated_analysis(
            config_path=args.config,
            output_dir=args.output,
            enable_plugins=args.plugins,
            generate_visualizations=not args.no_visualizations,
            verbose=args.verbose
        )
        
        print(f"\n{'='*60}")
        print("INTEGRATED ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Report generated: {report_path}")
        print(f"")
        print("Key files:")
        print(f"  - parameters.json: Complete parameter data")
        print(f"  - realm_execution_log.json: Realm execution details")
        print(f"  - dependency_graph.json: Parameter dependencies")
        print(f"  - README.md: Analysis summary")
        
        if not args.no_visualizations:
            print(f"  - visualizations/: Plots and dashboard")
        
        print(f"\nTo view the dashboard, open:")
        print(f"  {report_path}/visualizations/dashboard/dashboard.html")
        
    except Exception as e:
        logger.error(f"Integrated analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()