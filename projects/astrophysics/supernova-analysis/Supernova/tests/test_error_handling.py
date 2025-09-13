#!/usr/bin/env python3
"""
Test Error Handling and Logging System
=====================================

Tests for the comprehensive error handling and logging system.
"""

import numpy as np
import warnings
import logging
import sys
import os
from io import StringIO

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_handling import (
    QVDModelLogger, ErrorReporter, 
    NumericalIssueWarning, PhysicalBoundsWarning, ModelConsistencyWarning,
    handle_numerical_errors, validate_input_parameters,
    log_bounds_violations, log_model_inconsistency,
    setup_qvd_logging, global_error_reporter
)

def test_qvd_model_logger():
    """Test QVDModelLogger functionality"""
    print("Testing QVDModelLogger...")
    
    # Create logger
    logger = QVDModelLogger("TestLogger", logging.DEBUG)
    
    # Test context setting
    logger.set_context(wavelength=500.0, time=10.0)
    
    # Capture log output
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    logger.logger.addHandler(handler)
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Check that messages were logged
    log_output = log_stream.getvalue()
    assert "Debug message" in log_output
    assert "Info message" in log_output
    assert "Warning message" in log_output
    assert "Error message" in log_output
    assert "wavelength=500.0" in log_output
    assert "time=10.0" in log_output
    
    # Test error/warning counting
    logger.warning("Test warning")
    logger.error("Test error")
    
    warning_summary = logger.get_warning_summary()
    error_summary = logger.get_error_summary()
    
    assert "Warning" in warning_summary
    assert "Test" in warning_summary
    assert "Error" in error_summary
    assert "Test" in error_summary
    
    # Test reset
    logger.reset_counts()
    assert len(logger.get_warning_summary()) == 0
    assert len(logger.get_error_summary()) == 0
    
    print("  ✓ QVDModelLogger works correctly")

def test_numerical_error_decorator():
    """Test handle_numerical_errors decorator"""
    print("Testing numerical error decorator...")
    
    @handle_numerical_errors
    def function_with_nan():
        return np.nan
    
    @handle_numerical_errors
    def function_with_inf():
        return np.inf
    
    @handle_numerical_errors
    def function_with_array():
        return np.array([1.0, np.nan, 3.0])
    
    @handle_numerical_errors
    def function_with_dict():
        return {'value': np.nan, 'other': 1.0}
    
    @handle_numerical_errors
    def function_with_zero_division():
        return 1.0 / 0.0
    
    # Test with warnings captured
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        result1 = function_with_nan()
        result2 = function_with_inf()
        result3 = function_with_array()
        result4 = function_with_dict()
        result5 = function_with_zero_division()
        
        # Check that warnings were issued
        assert len(w) >= 4  # At least 4 warnings
        assert any(issubclass(warning.category, NumericalIssueWarning) for warning in w)
    
    # Results should be returned (even if modified)
    assert result1 is not None
    assert result2 is not None
    assert result3 is not None
    assert result4 is not None
    assert result5 is not None
    
    print("  ✓ Numerical error decorator works correctly")

def test_input_validation_decorator():
    """Test validate_input_parameters decorator"""
    print("Testing input validation decorator...")
    
    @validate_input_parameters
    def test_function(x, y, z=1.0):
        return x + y + z
    
    # Test with warnings captured
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test with normal inputs
        result1 = test_function(1.0, 2.0, z=3.0)
        assert result1 == 6.0
        
        # Test with NaN inputs
        result2 = test_function(np.nan, 2.0, z=3.0)
        result3 = test_function(1.0, np.inf, z=3.0)
        result4 = test_function(1.0, 2.0, z=np.nan)
        
        # Check that warnings were issued for non-finite inputs
        nan_warnings = [warning for warning in w if issubclass(warning.category, NumericalIssueWarning)]
        assert len(nan_warnings) >= 3  # At least 3 warnings for NaN/Inf inputs
    
    print("  ✓ Input validation decorator works correctly")

def test_bounds_violation_logging():
    """Test bounds violation logging"""
    print("Testing bounds violation logging...")
    
    # Test with warnings captured
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test scalar bounds violation
        log_bounds_violations(
            "test_parameter", 
            original_value=100.0,
            bounded_value=50.0,
            min_bound=0.0,
            max_bound=50.0
        )
        
        # Test array bounds violation
        original_array = np.array([0.5, 100.0, 200.0])
        bounded_array = np.array([1.0, 50.0, 50.0])
        log_bounds_violations(
            "test_array",
            original_value=original_array,
            bounded_value=bounded_array,
            min_bound=1.0,
            max_bound=50.0
        )
        
        # Check that warnings were issued
        bounds_warnings = [warning for warning in w if issubclass(warning.category, PhysicalBoundsWarning)]
        assert len(bounds_warnings) >= 2  # At least 2 bounds warnings
    
    print("  ✓ Bounds violation logging works correctly")

def test_model_inconsistency_logging():
    """Test model inconsistency logging"""
    print("Testing model inconsistency logging...")
    
    # Test with warnings captured
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        log_model_inconsistency(
            "transmission", 0.1,
            "optical_depth", 1.0,
            "transmission ≈ exp(-optical_depth)",
            0.1
        )
        
        # Check that warning was issued
        consistency_warnings = [warning for warning in w if issubclass(warning.category, ModelConsistencyWarning)]
        assert len(consistency_warnings) >= 1
    
    print("  ✓ Model inconsistency logging works correctly")

def test_error_reporter():
    """Test ErrorReporter functionality"""
    print("Testing ErrorReporter...")
    
    reporter = ErrorReporter()
    
    # Add various types of issues
    reporter.add_error("NumericalError", "Division by zero", {"function": "test_func"})
    reporter.add_warning("BoundsWarning", "Value clamped", {"parameter": "density"})
    reporter.add_bounds_violation("temperature", 1e12, 1e10, 100.0, 1e10)
    reporter.add_numerical_issue("calculate_cross_section", "overflow", "Value too large")
    
    # Generate report
    report = reporter.generate_report()
    
    # Check report structure
    assert 'report_metadata' in report
    assert 'errors' in report
    assert 'warnings' in report
    assert 'bounds_violations' in report
    assert 'numerical_issues' in report
    assert 'summary' in report
    
    # Check counts
    metadata = report['report_metadata']
    assert metadata['total_errors'] == 1
    assert metadata['total_warnings'] == 1
    assert metadata['total_bounds_violations'] == 1
    assert metadata['total_numerical_issues'] == 1
    
    # Check summary
    summary = report['summary']
    assert 'NumericalError' in summary['error_types']
    assert 'BoundsWarning' in summary['warning_types']
    assert 'temperature' in summary['most_violated_bounds']
    assert 'calculate_cross_section' in summary['most_problematic_functions']
    
    print("  ✓ ErrorReporter works correctly")

def test_logging_setup():
    """Test logging setup function"""
    print("Testing logging setup...")
    
    # Test basic setup
    logger = setup_qvd_logging(level=logging.DEBUG, enable_warnings=True)
    
    assert isinstance(logger, QVDModelLogger)
    assert logger.logger.level == logging.DEBUG
    
    # Test that warnings are configured
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        warnings.warn("Test numerical issue", NumericalIssueWarning)
        warnings.warn("Test bounds issue", PhysicalBoundsWarning)
        warnings.warn("Test consistency issue", ModelConsistencyWarning)
        
        # All warnings should be captured
        assert len(w) == 3
        assert any(warning.category == NumericalIssueWarning for warning in w)
        assert any(warning.category == PhysicalBoundsWarning for warning in w)
        assert any(warning.category == ModelConsistencyWarning for warning in w)
    
    print("  ✓ Logging setup works correctly")

def test_integration_with_model():
    """Test integration with actual model functions"""
    print("Testing integration with model functions...")
    
    # Create a test function that might have numerical issues
    @handle_numerical_errors
    @validate_input_parameters
    def test_model_function(wavelength, density, intensity):
        """Test function that simulates model calculations"""
        if density <= 0:
            return np.nan  # This should trigger error handling
        
        # Simulate some calculation that could overflow
        result = (wavelength ** -2) * (density ** 0.5) * intensity
        
        if result > 1e20:
            return np.inf  # This should trigger error handling
        
        return result
    
    # Test with various inputs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Normal case
        result1 = test_model_function(500.0, 1e20, 1e15)
        assert np.isfinite(result1)
        
        # Case that should trigger NaN handling
        result2 = test_model_function(500.0, 0.0, 1e15)
        # Should return fallback value, not NaN
        
        # Case with non-finite inputs
        result3 = test_model_function(np.nan, 1e20, 1e15)
        
        # Case that might cause overflow
        result4 = test_model_function(100.0, 1e30, 1e20)
        
        # Check that warnings were generated
        assert len(w) > 0
    
    print("  ✓ Integration with model functions works correctly")

def run_all_tests():
    """Run all error handling tests"""
    print("="*60)
    print("ERROR HANDLING AND LOGGING TEST SUITE")
    print("="*60)
    
    try:
        test_qvd_model_logger()
        test_numerical_error_decorator()
        test_input_validation_decorator()
        test_bounds_violation_logging()
        test_model_inconsistency_logging()
        test_error_reporter()
        test_logging_setup()
        test_integration_with_model()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("Error handling and logging system is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_all_tests()