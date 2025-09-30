#!/usr/bin/env python3
"""
Basic QFD Cosmology Fitting Example

This example demonstrates how to run a simple QFD cosmological parameter fit
using the Union2.1 dataset.
"""

import os
import sys
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_basic_cosmology_fit():
    """Run a basic QFD cosmology fit."""

    # Set up paths
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'example_cosmology')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run QFD cosmology fitter
    cmd = [
        'python',
        os.path.join(src_dir, 'QFD_Cosmology_Fitter_v5.6.py'),
        '--walkers', '16',        # Reduced for quick example
        '--steps', '500',         # Reduced for quick example
        '--outdir', output_dir,
        '--data', os.path.join(data_dir, 'union2.1_data_with_errors.txt')
    ]

    print("=== Running Basic QFD Cosmology Fit ===")
    print(f"Command: {' '.join(cmd)}")
    print("This may take a few minutes...")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Cosmology fit completed successfully!")
        print(f"Results saved to: {output_dir}")

        # Print summary from output
        if "Best-fit" in result.stdout:
            print("\n--- Fit Results ---")
            for line in result.stdout.split('\n'):
                if 'Best-fit' in line or 'η' in line or 'ξ' in line:
                    print(line)

    except subprocess.CalledProcessError as e:
        print(f"❌ Cosmology fit failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

    return True

if __name__ == "__main__":
    success = run_basic_cosmology_fit()

    if success:
        print("\n🎉 Example completed successfully!")
        print("Check the results directory for output files and plots.")
    else:
        print("\n❌ Example failed. Please check the error messages above.")
        sys.exit(1)