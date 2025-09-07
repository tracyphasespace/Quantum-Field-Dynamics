#!/usr/bin/env python3
"""
Test script for data utilities that works without scientific packages.
This validates the basic functionality of the data loading system.
"""

import json
import csv
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_basic_file_operations():
    """Test basic file operations without external dependencies."""
    print("Testing basic file operations...")
    
    data_dir = Path("data/sample")
    
    # Test directory exists
    assert data_dir.exists(), f"Data directory not found: {data_dir}"
    print(f"✓ Data directory exists: {data_dir}")
    
    # Test metadata file
    metadata_file = data_dir / "dataset_metadata.json"
    assert metadata_file.exists(), f"Metadata file not found: {metadata_file}"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    assert 'qfd_cmb_sample_data' in metadata, "Invalid metadata structure"
    print("✓ Metadata file loaded successfully")
    
    # Test CSV file
    csv_file = data_dir / "minimal_test_spectra.csv"
    assert csv_file.exists(), f"CSV file not found: {csv_file}"
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) > 0, "CSV file is empty"
    assert 'ell' in rows[0], "CSV missing 'ell' column"
    assert 'C_TT' in rows[0], "CSV missing 'C_TT' column"
    print(f"✓ CSV file loaded successfully: {len(rows)} rows")
    
    # Test JSON reference file
    ref_file = data_dir / "reference_spectra.json"
    assert ref_file.exists(), f"Reference file not found: {ref_file}"
    
    with open(ref_file, 'r') as f:
        ref_data = json.load(f)
    
    assert 'metadata' in ref_data, "Reference data missing metadata"
    assert 'data' in ref_data, "Reference data missing data section"
    print("✓ Reference data file loaded successfully")


def test_data_structure_validation():
    """Test that data structures are valid."""
    print("\nTesting data structure validation...")
    
    # Load and validate CSV data
    csv_file = Path("data/sample/minimal_test_spectra.csv")
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        csv_data = list(reader)
    
    # Check required columns
    required_columns = ['ell', 'C_TT', 'C_EE', 'C_TE', 'rho_TE']
    for col in required_columns:
        assert col in csv_data[0], f"Missing required column: {col}"
    print(f"✓ CSV has all required columns: {required_columns}")
    
    # Check data types can be converted
    for row in csv_data:
        try:
            ell = int(row['ell'])
            c_tt = float(row['C_TT'])
            c_ee = float(row['C_EE'])
            c_te = float(row['C_TE'])
            rho = float(row['rho_TE'])
            
            # Basic sanity checks
            assert ell >= 2, f"Invalid multipole: {ell}"
            assert c_tt > 0, f"TT spectrum should be positive: {c_tt}"
            assert c_ee > 0, f"EE spectrum should be positive: {c_ee}"
            assert -1 <= rho <= 1, f"Correlation should be in [-1,1]: {rho}"
            
        except (ValueError, AssertionError) as e:
            raise AssertionError(f"Invalid data in row {row}: {e}")
    
    print(f"✓ All {len(csv_data)} CSV rows have valid data")
    
    # Validate reference data
    ref_file = Path("data/sample/reference_spectra.json")
    with open(ref_file, 'r') as f:
        ref_data = json.load(f)
    
    data_section = ref_data['data']
    ells = data_section['ell']
    c_tt = data_section['C_TT']
    c_ee = data_section['C_EE']
    c_te = data_section['C_TE']
    rho_te = data_section['rho_TE']
    
    # Check all arrays have same length
    lengths = [len(ells), len(c_tt), len(c_ee), len(c_te), len(rho_te)]
    assert all(l == lengths[0] for l in lengths), f"Inconsistent array lengths: {lengths}"
    print(f"✓ Reference data arrays have consistent length: {lengths[0]}")
    
    # Check data validity
    for i in range(len(ells)):
        assert ells[i] >= 2, f"Invalid multipole: {ells[i]}"
        assert c_tt[i] > 0, f"TT spectrum should be positive: {c_tt[i]}"
        assert c_ee[i] > 0, f"EE spectrum should be positive: {c_ee[i]}"
        assert -1 <= rho_te[i] <= 1, f"Correlation should be in [-1,1]: {rho_te[i]}"
    
    print("✓ Reference data values are valid")


def test_metadata_consistency():
    """Test that metadata is consistent with actual data."""
    print("\nTesting metadata consistency...")
    
    # Load metadata
    metadata_file = Path("data/sample/dataset_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    datasets_info = metadata['qfd_cmb_sample_data']['datasets']
    
    # Check minimal test data
    csv_info = datasets_info['minimal_test_spectra.csv']
    csv_file = Path("data/sample/minimal_test_spectra.csv")
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        csv_data = list(reader)
    
    # Verify column names match
    expected_columns = csv_info['columns']
    actual_columns = list(csv_data[0].keys())
    assert actual_columns == expected_columns, f"Column mismatch: {actual_columns} vs {expected_columns}"
    print("✓ CSV columns match metadata")
    
    # Check reference data
    ref_info = datasets_info['reference_spectra.json']
    ref_file = Path("data/sample/reference_spectra.json")
    
    with open(ref_file, 'r') as f:
        ref_data = json.load(f)
    
    expected_points = ref_info['multipole_points']
    actual_points = ref_data['data']['ell']
    assert actual_points == expected_points, f"Multipole points mismatch: {actual_points} vs {expected_points}"
    print("✓ Reference data points match metadata")


def test_file_formats():
    """Test that file formats are correct."""
    print("\nTesting file formats...")
    
    # Test CSV format
    csv_file = Path("data/sample/minimal_test_spectra.csv")
    with open(csv_file, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    
    # Check header
    assert first_line == "ell,C_TT,C_EE,C_TE,rho_TE", f"Unexpected CSV header: {first_line}"
    
    # Check data format (should be comma-separated with scientific notation)
    parts = second_line.split(',')
    assert len(parts) == 5, f"Expected 5 columns, got {len(parts)}"
    
    # Check that scientific notation is used for spectra
    for i in range(1, 4):  # C_TT, C_EE, C_TE columns
        assert 'e-' in parts[i] or 'e+' in parts[i], f"Expected scientific notation in column {i}: {parts[i]}"
    
    print("✓ CSV format is correct")
    
    # Test JSON format
    ref_file = Path("data/sample/reference_spectra.json")
    with open(ref_file, 'r') as f:
        content = f.read()
    
    # Should be valid JSON
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON format: {e}")
    
    # Should be properly formatted (indented)
    assert '  "metadata"' in content, "JSON should be indented"
    print("✓ JSON format is correct")


def main():
    """Run all tests."""
    print("QFD CMB Data Utils - Basic Functionality Tests")
    print("=" * 50)
    
    try:
        test_basic_file_operations()
        test_data_structure_validation()
        test_metadata_consistency()
        test_file_formats()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Sample data system is working correctly.")
        print("\nThe following functionality has been validated:")
        print("  - File structure and existence")
        print("  - Data format correctness")
        print("  - Metadata consistency")
        print("  - Basic data validation")
        print("\nYou can now use the data loading utilities with confidence.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nPlease check the sample data generation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())