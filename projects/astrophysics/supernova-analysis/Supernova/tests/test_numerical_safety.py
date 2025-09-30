#!/usr/bin/env python3
"""
Unit Tests for Numerical Safety Utilities
========================================

Comprehensive tests for safe mathematical operations to prevent NaN/Inf values.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
import pytest
import warnings
from numerical_safety import (
    safe_power, safe_log10, safe_exp, safe_divide, safe_sqrt,
    validate_finite, clamp_to_range, safe_scientific_operation
)


class TestSafePower:
    """Test safe_power function with various edge cases"""
    
    def test_normal_operation(self):
        """Test normal power operations"""
        assert safe_power(2.0, 3.0) == 8.0
        assert safe_power(10.0, 0.5) == pytest.approx(np.sqrt(10))
        
    def test_negative_base(self):
        """Test handling of negative base values"""
        result = safe_power(-2.0, 3.0)
        assert np.isfinite(result)
        assert result > 0  # Should use absolute value
        
    def test_zero_base(self):
        """Test handling of zero base"""
        result = safe_power(0.0, 2.0)
        assert np.isfinite(result)
        assert result > 0  # Should use minimum value
        
    def test_very_small_base(self):
        """Test handling of very small base values"""
        result = safe_power(1e-50, 2.0)
        assert np.isfinite(result)
        assert result > 0
        
    def test_large_exponent(self):
        """Test handling of large exponents that could cause overflow"""
        result = safe_power(2.0, 1000.0)
        assert np.isfinite(result)
        
    def test_negative_exponent(self):
        """Test handling of negative exponents"""
        result = safe_power(2.0, -3.0)
        assert np.isfinite(result)
        assert result == pytest.approx(1/8)
        
    def test_array_input(self):
        """Test with numpy arrays"""
        bases = np.array([0.0, -1.0, 2.0, 1e-50])
        exponents = np.array([2.0, 3.0, 4.0, 1.0])
        results = safe_power(bases, exponents)
        
        assert len(results) == len(bases)
        assert np.all(np.isfinite(results))
        assert np.all(results > 0)


class TestSafeLog10:
    """Test safe_log10 function with various edge cases"""
    
    def test_normal_operation(self):
        """Test normal logarithm operations"""
        assert safe_log10(10.0) == pytest.approx(1.0)
        assert safe_log10(100.0) == pytest.approx(2.0)
        
    def test_zero_input(self):
        """Test handling of zero input"""
        result = safe_log10(0.0)
        assert np.isfinite(result)
        assert result < 0  # Should be log of minimum value
        
    def test_negative_input(self):
        """Test handling of negative input"""
        result = safe_log10(-5.0)
        assert np.isfinite(result)
        # Should use absolute value
        
    def test_very_small_input(self):
        """Test handling of very small positive values"""
        result = safe_log10(1e-100)
        assert np.isfinite(result)
        assert result < 0
        
    def test_array_input(self):
        """Test with numpy arrays"""
        values = np.array([0.0, -1.0, 10.0, 1e-50])
        results = safe_log10(values)
        
        assert len(results) == len(values)
        assert np.all(np.isfinite(results))


class TestSafeExp:
    """Test safe_exp function with various edge cases"""
    
    def test_normal_operation(self):
        """Test normal exponential operations"""
        assert safe_exp(0.0) == pytest.approx(1.0)
        assert safe_exp(1.0) == pytest.approx(np.e)
        
    def test_large_exponent(self):
        """Test handling of large exponents that could cause overflow"""
        result = safe_exp(1000.0)
        assert np.isfinite(result)
        assert result > 0
        
    def test_very_negative_exponent(self):
        """Test handling of very negative exponents"""
        result = safe_exp(-1000.0)
        assert np.isfinite(result)
        assert result > 0  # Should not underflow to zero
        
    def test_array_input(self):
        """Test with numpy arrays"""
        exponents = np.array([-1000.0, 0.0, 1.0, 1000.0])
        results = safe_exp(exponents)
        
        assert len(results) == len(exponents)
        assert np.all(np.isfinite(results))
        assert np.all(results > 0)


class TestSafeDivide:
    """Test safe_divide function with various edge cases"""
    
    def test_normal_operation(self):
        """Test normal division operations"""
        assert safe_divide(10.0, 2.0) == pytest.approx(5.0)
        assert safe_divide(1.0, 4.0) == pytest.approx(0.25)
        
    def test_zero_denominator(self):
        """Test handling of zero denominator"""
        result = safe_divide(5.0, 0.0)
        assert np.isfinite(result)
        
    def test_very_small_denominator(self):
        """Test handling of very small denominator"""
        result = safe_divide(1.0, 1e-50)
        assert np.isfinite(result)
        
    def test_array_input(self):
        """Test with numpy arrays"""
        numerators = np.array([1.0, 2.0, 3.0, 4.0])
        denominators = np.array([0.0, 1e-50, 2.0, 4.0])
        results = safe_divide(numerators, denominators)
        
        assert len(results) == len(numerators)
        assert np.all(np.isfinite(results))


class TestSafeSqrt:
    """Test safe_sqrt function with various edge cases"""
    
    def test_normal_operation(self):
        """Test normal square root operations"""
        assert safe_sqrt(4.0) == pytest.approx(2.0)
        assert safe_sqrt(9.0) == pytest.approx(3.0)
        
    def test_zero_input(self):
        """Test handling of zero input"""
        result = safe_sqrt(0.0)
        assert np.isfinite(result)
        assert result == 0.0
        
    def test_negative_input(self):
        """Test handling of negative input"""
        result = safe_sqrt(-4.0)
        assert np.isfinite(result)
        assert result >= 0  # Should handle negative gracefully
        
    def test_array_input(self):
        """Test with numpy arrays"""
        values = np.array([-1.0, 0.0, 4.0, 9.0])
        results = safe_sqrt(values)
        
        assert len(results) == len(values)
        assert np.all(np.isfinite(results))
        assert np.all(results >= 0)


class TestValidateFinite:
    """Test validate_finite function"""
    
    def test_finite_values(self):
        """Test with all finite values"""
        values = np.array([1.0, 2.0, 3.0])
        result = validate_finite(values, "test")
        np.testing.assert_array_equal(result, values)
        
    def test_nan_values(self):
        """Test with NaN values"""
        values = np.array([1.0, np.nan, 3.0])
        
        # Without replacement
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_finite(values, "test")
            assert len(w) == 1
            assert "Non-finite" in str(w[0].message)
        
        # With replacement
        result = validate_finite(values, "test", replace_with=0.0)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)
        
    def test_inf_values(self):
        """Test with infinite values"""
        values = np.array([1.0, np.inf, 3.0])
        result = validate_finite(values, "test", replace_with=999.0)
        expected = np.array([1.0, 999.0, 3.0])
        np.testing.assert_array_equal(result, expected)


class TestClampToRange:
    """Test clamp_to_range function"""
    
    def test_values_in_range(self):
        """Test with values already in range"""
        values = np.array([2.0, 3.0, 4.0])
        result = clamp_to_range(values, 1.0, 5.0, "test")
        np.testing.assert_array_equal(result, values)
        
    def test_values_outside_range(self):
        """Test with values outside range"""
        values = np.array([0.0, 3.0, 6.0])
        result = clamp_to_range(values, 1.0, 5.0, "test")
        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_array_equal(result, expected)
        
    def test_single_value(self):
        """Test with single value"""
        result = clamp_to_range(10.0, 1.0, 5.0, "test")
        assert result == 5.0


class TestSafeScientificOperation:
    """Test safe_scientific_operation dispatcher"""
    
    def test_valid_operations(self):
        """Test all valid operations"""
        assert safe_scientific_operation('power', 2.0, 3.0) == 8.0
        assert safe_scientific_operation('log10', 10.0) == pytest.approx(1.0)
        assert safe_scientific_operation('exp', 0.0) == pytest.approx(1.0)
        assert safe_scientific_operation('divide', 10.0, 2.0) == pytest.approx(5.0)
        assert safe_scientific_operation('sqrt', 4.0) == pytest.approx(2.0)
        
    def test_invalid_operation(self):
        """Test invalid operation name"""
        with pytest.raises(ValueError, match="Unknown operation"):
            safe_scientific_operation('invalid', 1.0)


class TestIntegrationScenarios:
    """Test realistic scenarios that could cause numerical issues"""
    
    def test_plasma_density_evolution(self):
        """Test scenario similar to plasma density calculations"""
        # Simulate expanding plasma with decreasing density
        initial_density = 1e24
        expansion_factors = np.array([1.0, 10.0, 100.0, 1000.0, 1e6])
        
        # This could cause division by very large numbers
        densities = safe_divide(initial_density, expansion_factors**3)
        
        assert np.all(np.isfinite(densities))
        assert np.all(densities > 0)
        assert np.all(densities <= initial_density)
        
    def test_transmission_calculation(self):
        """Test scenario similar to transmission probability calculations"""
        # Simulate optical depth that could become very large
        optical_depths = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        
        # This could cause exp(-very_large_number) = 0
        transmissions = safe_exp(-optical_depths)
        
        assert np.all(np.isfinite(transmissions))
        assert np.all(transmissions > 0)
        assert np.all(transmissions <= 1.0)
        
    def test_magnitude_calculation(self):
        """Test scenario similar to magnitude calculations"""
        # Simulate transmission values that could be zero
        transmissions = np.array([1.0, 0.1, 0.01, 0.0, 1e-50])
        
        # This could cause log10(0) = -inf
        magnitudes = -2.5 * safe_log10(transmissions)
        
        assert np.all(np.isfinite(magnitudes))
        
    def test_wavelength_scaling(self):
        """Test scenario similar to wavelength-dependent scaling"""
        # Simulate wavelength ratios that could be extreme
        wavelengths = np.array([100.0, 500.0, 1000.0, 10000.0])  # nm
        reference_wavelength = 500.0
        alpha = -2.0  # Could make very small wavelengths dominant
        
        # This could cause (very_small_number)^(negative_power) = very_large_number
        scaling_factors = safe_power(wavelengths / reference_wavelength, alpha)
        
        assert np.all(np.isfinite(scaling_factors))
        assert np.all(scaling_factors > 0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])