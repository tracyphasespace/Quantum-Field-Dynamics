#!/usr/bin/env python3
"""
Example usage of QFD CMB sample data utilities.

This script demonstrates how to load and work with the sample datasets
without requiring the full scientific computing stack.
"""

import json
import csv
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def demonstrate_basic_loading():
    """Demonstrate basic data loading without pandas/numpy."""
    print("=" * 60)
    print("Basic Data Loading Example")
    print("=" * 60)
    
    # Load minimal test data
    csv_file = Path("data/sample/minimal_test_spectra.csv")
    print(f"Loading data from: {csv_file}")
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    print(f"Loaded {len(data)} data points")
    print(f"Columns: {list(data[0].keys())}")
    
    # Display first few rows
    print("\nFirst 3 data points:")
    for i in range(min(3, len(data))):
        row = data[i]
        ell = int(row['ell'])
        c_tt = float(row['C_TT'])
        c_ee = float(row['C_EE'])
        c_te = float(row['C_TE'])
        rho = float(row['rho_TE'])
        
        print(f"  ℓ={ell:3d}: C_TT={c_tt:.3e}, C_EE={c_ee:.3e}, C_TE={c_te:.3e}, ρ={rho:.3f}")
    
    return data


def demonstrate_reference_data():
    """Demonstrate loading reference data for validation."""
    print("\n" + "=" * 60)
    print("Reference Data Loading Example")
    print("=" * 60)
    
    ref_file = Path("data/sample/reference_spectra.json")
    print(f"Loading reference data from: {ref_file}")
    
    with open(ref_file, 'r') as f:
        ref_data = json.load(f)
    
    metadata = ref_data['metadata']
    data = ref_data['data']
    
    print(f"Description: {metadata['description']}")
    print(f"Parameters used:")
    for param, value in metadata['parameters'].items():
        print(f"  {param}: {value}")
    
    print(f"\nReference multipoles: {data['ell']}")
    print(f"Number of reference points: {len(data['ell'])}")
    
    # Show tolerances for validation
    tolerances = metadata['tolerances']
    print(f"\nValidation tolerances:")
    print(f"  Relative: {tolerances['relative']}")
    print(f"  Absolute: {tolerances['absolute']}")
    
    return ref_data


def demonstrate_metadata_usage():
    """Demonstrate using metadata to understand datasets."""
    print("\n" + "=" * 60)
    print("Metadata Usage Example")
    print("=" * 60)
    
    metadata_file = Path("data/sample/dataset_metadata.json")
    print(f"Loading metadata from: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    qfd_data = metadata['qfd_cmb_sample_data']
    
    print(f"Sample data version: {qfd_data['version']}")
    print(f"Description: {qfd_data['description']}")
    
    print(f"\nStandard parameters:")
    for param, value in qfd_data['planck_parameters'].items():
        print(f"  {param}: {value}")
    
    print(f"\nAvailable datasets:")
    for dataset_name, info in qfd_data['datasets'].items():
        print(f"  {dataset_name}:")
        print(f"    Description: {info['description']}")
        print(f"    Use case: {info['use_case']}")
        if 'data_points' in info:
            print(f"    Data points: {info['data_points']}")
    
    print(f"\nData format documentation:")
    for column, description in qfd_data['data_format'].items():
        print(f"  {column}: {description}")
    
    return metadata


def demonstrate_simple_analysis():
    """Demonstrate simple analysis without scientific packages."""
    print("\n" + "=" * 60)
    print("Simple Analysis Example")
    print("=" * 60)
    
    # Load data
    csv_file = Path("data/sample/minimal_test_spectra.csv")
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Convert to numbers and compute basic statistics
    ells = [int(row['ell']) for row in data]
    c_tt_values = [float(row['C_TT']) for row in data]
    c_ee_values = [float(row['C_EE']) for row in data]
    rho_values = [float(row['rho_TE']) for row in data]
    
    print(f"Multipole range: {min(ells)} to {max(ells)}")
    print(f"Number of data points: {len(ells)}")
    
    # Basic statistics
    print(f"\nTT Spectrum statistics:")
    print(f"  Min: {min(c_tt_values):.3e}")
    print(f"  Max: {max(c_tt_values):.3e}")
    print(f"  Range: {max(c_tt_values)/min(c_tt_values):.1f}x")
    
    print(f"\nEE Spectrum statistics:")
    print(f"  Min: {min(c_ee_values):.3e}")
    print(f"  Max: {max(c_ee_values):.3e}")
    print(f"  EE/TT ratio range: {min(c_ee_values[i]/c_tt_values[i] for i in range(len(data))):.3f} to {max(c_ee_values[i]/c_tt_values[i] for i in range(len(data))):.3f}")
    
    print(f"\nTE Correlation statistics:")
    print(f"  Min correlation: {min(rho_values):.3f}")
    print(f"  Max correlation: {max(rho_values):.3f}")
    print(f"  Average correlation: {sum(rho_values)/len(rho_values):.3f}")
    
    # Find peak (approximate)
    # Compute D_l = l(l+1)C_l for TT
    dl_values = [ells[i] * (ells[i] + 1) * c_tt_values[i] for i in range(len(data))]
    peak_index = dl_values.index(max(dl_values))
    
    print(f"\nApproximate TT spectrum peak:")
    print(f"  Peak at ℓ ≈ {ells[peak_index]}")
    print(f"  Peak D_l value: {dl_values[peak_index]:.3e}")


def demonstrate_validation_concept():
    """Demonstrate the concept of data validation."""
    print("\n" + "=" * 60)
    print("Data Validation Concept Example")
    print("=" * 60)
    
    # Load reference data
    ref_file = Path("data/sample/reference_spectra.json")
    with open(ref_file, 'r') as f:
        ref_data = json.load(f)
    
    # Simulate some "computed" results (in practice, these would come from your calculations)
    ref_ells = ref_data['data']['ell']
    ref_c_tt = ref_data['data']['C_TT']
    
    # Simulate perfect agreement
    computed_c_tt = ref_c_tt.copy()
    
    print("Simulating validation with perfect agreement:")
    max_diff = max(abs(computed_c_tt[i] - ref_c_tt[i]) for i in range(len(ref_c_tt)))
    print(f"  Maximum absolute difference: {max_diff}")
    
    if max_diff == 0:
        print("  ✅ Perfect agreement!")
    
    # Simulate small numerical differences
    import random
    random.seed(42)  # For reproducibility
    computed_c_tt_noisy = [val * (1 + random.uniform(-1e-12, 1e-12)) for val in ref_c_tt]
    
    print("\nSimulating validation with small numerical differences:")
    abs_diffs = [abs(computed_c_tt_noisy[i] - ref_c_tt[i]) for i in range(len(ref_c_tt))]
    rel_diffs = [abs_diffs[i] / ref_c_tt[i] for i in range(len(ref_c_tt))]
    
    max_abs_diff = max(abs_diffs)
    max_rel_diff = max(rel_diffs)
    
    print(f"  Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"  Maximum relative difference: {max_rel_diff:.2e}")
    
    # Check against tolerances
    tolerances = ref_data['metadata']['tolerances']
    abs_tol = tolerances['absolute']
    rel_tol = tolerances['relative']
    
    print(f"\nValidation against tolerances:")
    print(f"  Absolute tolerance: {abs_tol}")
    print(f"  Relative tolerance: {rel_tol}")
    
    abs_pass = max_abs_diff <= abs_tol
    rel_pass = max_rel_diff <= rel_tol
    
    print(f"  Absolute test: {'✅ PASS' if abs_pass else '❌ FAIL'}")
    print(f"  Relative test: {'✅ PASS' if rel_pass else '❌ FAIL'}")
    print(f"  Overall: {'✅ PASS' if (abs_pass or rel_pass) else '❌ FAIL'}")


def main():
    """Run all demonstration examples."""
    print("QFD CMB Sample Data - Usage Examples")
    print("====================================")
    print()
    print("This script demonstrates how to work with QFD CMB sample data")
    print("using only Python standard library (no numpy/pandas required).")
    print()
    
    try:
        # Run demonstrations
        demonstrate_basic_loading()
        demonstrate_reference_data()
        demonstrate_metadata_usage()
        demonstrate_simple_analysis()
        demonstrate_validation_concept()
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("✅ All examples completed successfully!")
        print()
        print("Key takeaways:")
        print("  • Sample data can be loaded with standard Python libraries")
        print("  • Metadata provides comprehensive dataset documentation")
        print("  • Reference data enables automated validation")
        print("  • Basic analysis can be performed without scientific packages")
        print("  • Data formats are designed for easy integration")
        print()
        print("Next steps:")
        print("  • Install numpy/pandas for advanced analysis")
        print("  • Use the full data_utils.py module for convenience functions")
        print("  • Generate additional datasets with generate_sample_data.py")
        print("  • Integrate sample data into your analysis workflows")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Please check the sample data files and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())