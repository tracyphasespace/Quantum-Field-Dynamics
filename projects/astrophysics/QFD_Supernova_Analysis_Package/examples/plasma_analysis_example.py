#!/usr/bin/env python3
"""
QFD Plasma Veil Analysis Example

This example demonstrates how to run the QFD plasma veil fitter on individual
supernova light curves to test Stage 1 QFD physics.
"""

import os
import sys
import subprocess
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_example_cosmology():
    """Create example cosmological parameters file."""
    cosmology = {
        "description": "Example QFD cosmological parameters",
        "eta_prime": 9.55e-4,
        "xi": 0.40,
        "H0": 70.0
    }

    cosmology_file = os.path.join(os.path.dirname(__file__), 'example_cosmology.json')
    with open(cosmology_file, 'w') as f:
        json.dump(cosmology, f, indent=2)

    return cosmology_file

def run_plasma_analysis():
    """Run QFD plasma veil analysis on SN2011fe."""

    # Set up paths
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'example_plasma')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create example cosmology file
    cosmology_file = create_example_cosmology()

    # Run QFD plasma veil fitter
    cmd = [
        'python',
        os.path.join(src_dir, 'qfd_plasma_veil_fitter.py'),
        '--data', os.path.join(data_dir, 'sample_lightcurves', 'lightcurves_osc.csv'),
        '--snid', 'SN2011fe',
        '--redshift', '0.0008',
        '--cosmology', cosmology_file,
        '--outdir', output_dir,
        '--verbose'
    ]

    print("=== Running QFD Plasma Veil Analysis ===")
    print(f"Command: {' '.join(cmd)}")
    print("Analyzing SN2011fe light curve...")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Plasma analysis completed!")
        print(f"Results saved to: {output_dir}")

        # Print summary from output
        if "A_plasma" in result.stdout:
            print("\n--- Plasma Parameters ---")
            for line in result.stdout.split('\n'):
                if any(param in line for param in ['A_plasma', 'τ_decay', 'β', 'χ²']):
                    print(line)

    except subprocess.CalledProcessError as e:
        print(f"❌ Plasma analysis failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

    finally:
        # Clean up temporary cosmology file
        if os.path.exists(cosmology_file):
            os.remove(cosmology_file)

    return True

if __name__ == "__main__":
    success = run_plasma_analysis()

    if success:
        print("\n🎉 Plasma veil analysis completed!")
        print("This demonstrates QFD Stage 1 mechanism testing.")
        print("Check the results directory for diagnostic plots.")
    else:
        print("\n❌ Example failed. Please check the error messages above.")
        sys.exit(1)