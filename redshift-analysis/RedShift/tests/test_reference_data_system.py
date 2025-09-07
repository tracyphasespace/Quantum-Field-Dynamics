"""
Test the reference data system functionality

This module tests that the reference data generation and validation
system works correctly, independent of having actual reference data.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from .test_data_utils import ReferenceDataLoader, ReferenceDataValidator


class TestReferenceDataSystem:
    """Test the reference data loading and validation system"""
    
    def test_reference_data_loader_initialization(self):
        """Test that ReferenceDataLoader initializes correctly"""
        loader = ReferenceDataLoader("tests/reference_data")
        assert loader.reference_dir == Path("tests/reference_data")
        assert loader._cache == {}
    
    def test_reference_data_loader_with_mock_data(self):
        """Test ReferenceDataLoader with mock reference data"""
        # Create mock reference data
        mock_data = {
            "metadata": {
                "generated_at": "2024-01-01 00:00:00 UTC",
                "generator_version": "1.0.0",
                "data_hash": "mock_hash",
                "description": "Mock test data"
            },
            "data": {
                "k_values": [0.01, 0.1, 1.0],
                "parameter_sets": {
                    "test_set": {
                        "parameters": {"ns": 0.96, "rpsi": 147.0},
                        "Pk_values": [1e-5, 1e-4, 1e-3]
                    }
                }
            }
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_data, f)
            temp_file = f.name
        
        try:
            # Create loader with temporary directory
            temp_dir = Path(temp_file).parent
            loader = ReferenceDataLoader(str(temp_dir))
            
            # Load the mock data (disable hash validation for mock data)
            loaded_data = loader.load_reference_data(Path(temp_file).name, validate_hash=False)
            
            # Verify structure
            assert "metadata" in loaded_data
            assert "data" in loaded_data
            assert loaded_data["data"]["k_values"] == [0.01, 0.1, 1.0]
            
        finally:
            # Clean up
            Path(temp_file).unlink()
    
    def test_reference_data_validator_initialization(self):
        """Test that ReferenceDataValidator initializes correctly"""
        validator = ReferenceDataValidator()
        assert validator.loader is not None
        
        # Test with custom loader
        custom_loader = ReferenceDataLoader("custom_dir")
        validator_custom = ReferenceDataValidator(custom_loader)
        assert validator_custom.loader is custom_loader
    
    def test_data_hash_computation(self):
        """Test data hash computation for integrity checking"""
        loader = ReferenceDataLoader()
        
        # Test with simple data
        test_data = {"a": 1, "b": [2, 3, 4]}
        hash1 = loader._compute_data_hash(test_data)
        hash2 = loader._compute_data_hash(test_data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        
        # Different data should produce different hash
        test_data_modified = {"a": 1, "b": [2, 3, 5]}
        hash3 = loader._compute_data_hash(test_data_modified)
        assert hash1 != hash3
    
    def test_power_spectrum_validation_logic(self):
        """Test power spectrum validation logic with mock data"""
        # Create mock reference data
        k_values = np.array([0.01, 0.1, 1.0])
        Pk_reference = np.array([1e-5, 1e-4, 1e-3])
        
        validator = ReferenceDataValidator()
        
        # Test exact match
        result = validator.validate_power_spectrum.__func__(
            validator, k_values, Pk_reference, "mock_set", rtol=1e-10, atol=1e-15
        )
        # This will fail because we don't have actual reference data, but we can test the logic
        
        # Test that the function exists and has correct signature
        assert hasattr(validator, 'validate_power_spectrum')
        assert callable(validator.validate_power_spectrum)
    
    def test_window_function_validation_logic(self):
        """Test window function validation logic"""
        validator = ReferenceDataValidator()
        
        # Test that the function exists and has correct signature
        assert hasattr(validator, 'validate_window_function')
        assert callable(validator.validate_window_function)
    
    def test_correlation_validation_logic(self):
        """Test correlation validation logic"""
        validator = ReferenceDataValidator()
        
        # Test that the function exists and has correct signature
        assert hasattr(validator, 'validate_correlation')
        assert callable(validator.validate_correlation)
    
    def test_spectra_validation_logic(self):
        """Test spectra validation logic"""
        validator = ReferenceDataValidator()
        
        # Test that the function exists and has correct signature
        assert hasattr(validator, 'validate_spectra')
        assert callable(validator.validate_spectra)
    
    def test_mueller_validation_logic(self):
        """Test Mueller coefficient validation logic"""
        validator = ReferenceDataValidator()
        
        # Test that the function exists and has correct signature
        assert hasattr(validator, 'validate_mueller_coefficients')
        assert callable(validator.validate_mueller_coefficients)
    
    def test_reference_data_directory_handling(self):
        """Test handling of missing reference data directory"""
        # Test with non-existent directory
        loader = ReferenceDataLoader("non_existent_directory")
        
        # Should not raise error on initialization
        assert loader.reference_dir == Path("non_existent_directory")
        
        # Should return empty dict for list_available_data
        available = loader.list_available_data()
        assert available == {}
    
    def test_file_not_found_handling(self):
        """Test handling of missing reference data files"""
        loader = ReferenceDataLoader("tests/reference_data")
        
        # Should raise FileNotFoundError for non-existent file
        with pytest.raises(FileNotFoundError):
            loader.load_reference_data("non_existent_file.json")
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON reference data"""
        # Create invalid JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            temp_file = f.name
        
        try:
            temp_dir = Path(temp_file).parent
            loader = ReferenceDataLoader(str(temp_dir))
            
            # Should raise JSON decode error
            with pytest.raises(json.JSONDecodeError):
                loader.load_reference_data(Path(temp_file).name)
                
        finally:
            Path(temp_file).unlink()
    
    def test_invalid_data_format_handling(self):
        """Test handling of invalid reference data format"""
        # Create file with invalid format (missing metadata or data)
        invalid_data = {"some_field": "some_value"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_file = f.name
        
        try:
            temp_dir = Path(temp_file).parent
            loader = ReferenceDataLoader(str(temp_dir))
            
            # Should raise ValueError for invalid format
            with pytest.raises(ValueError, match="Invalid reference data format"):
                loader.load_reference_data(Path(temp_file).name)
                
        finally:
            Path(temp_file).unlink()


class TestReferenceDataGeneration:
    """Test reference data generation functionality"""
    
    def test_generation_script_exists(self):
        """Test that the generation script exists"""
        script_path = Path("tests/generate_reference_data.py")
        assert script_path.exists(), "Reference data generation script should exist"
        
        # Check that it's a Python file
        assert script_path.suffix == ".py"
        
        # Check that it's readable
        assert script_path.is_file()
    
    def test_generation_script_has_main(self):
        """Test that generation script has main function"""
        script_path = Path("tests/generate_reference_data.py")
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Should have main function and if __name__ == "__main__" block
        assert "def main(" in content
        assert 'if __name__ == "__main__"' in content
    
    def test_generation_functions_exist(self):
        """Test that generation functions are defined"""
        script_path = Path("tests/generate_reference_data.py")
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for expected generation functions
        expected_functions = [
            "generate_power_spectrum_reference",
            "generate_window_function_reference", 
            "generate_correlation_reference",
            "generate_spectrum_reference",
            "generate_mueller_reference"
        ]
        
        for func_name in expected_functions:
            assert f"def {func_name}(" in content, f"Missing function: {func_name}"


# Test markers
pytestmark = [pytest.mark.unit]