"""
Example script to run the QFD Supernova Analysis Demo.

This script demonstrates the use of the SupernovaAnalyzer to run a
complete analysis, generating data, plots, and a results file.

To run this script, execute it from the `supernova-analysis` directory:
    python examples/run_demo.py
"""

import sys
from pathlib import Path

# Add the `supernova-analysis` directory to the path to find the packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qfd_supernova.supernova_analyzer import SupernovaAnalyzer

def main():
    """
    Run the complete supernova analysis and verification.
    """
    print("Starting QFD Supernova Analysis Demo...")

    # 1. Initialize the analyzer
    analyzer = SupernovaAnalyzer()

    # 2. Run the complete analysis pipeline
    output_dir = "supernova_results"
    results = analyzer.run_complete_analysis(output_dir=output_dir)

    # 3. Print a summary of the verification
    if results and 'validation' in results:
        validation_passed = results['validation'].get('validation_passed', False)
        rms_error = results['validation'].get('rms_error', float('nan'))

        print("\n--- Verification Summary ---")
        print(f"Analysis pipeline completed.")
        print(f"Validation against observations passed: {validation_passed}")
        print(f"RMS Error: {rms_error:.4f} mag")
        print(f"Output plots and results saved to '{output_dir}/' directory.")
        print("--------------------------")

        if validation_passed:
            print("\nDemo completed successfully!")
        else:
            print("\nDemo completed, but validation checks did not pass.")
    else:
        print("\n--- Verification Warning ---")
        print("Could not retrieve validation results from the analysis.")
        print("----------------------------")

if __name__ == "__main__":
    main()
