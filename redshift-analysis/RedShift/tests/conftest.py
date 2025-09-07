"""
Shared fixtures and test utilities for QFD CMB tests

This module provides common fixtures, test data, and utilities
used across multiple test modules.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_k_values():
    """Sample k values for testing power spectrum functions"""
    return np.logspace(-4, 1, 50)  # k from 1e-4 to 10 Mpc^-1


@pytest.fixture
def sample_ell_values():
    """Sample ell values for testing CMB spectra"""
    return np.arange(2, 2001, 50)  # ell from 2 to 2000


@pytest.fixture
def sample_chi_grid():
    """Sample comoving distance grid for testing"""
    return np.linspace(100, 15000, 100)  # chi from 100 to 15000 Mpc


@pytest.fixture
def sample_eta_grid():
    """Sample conformal time grid for testing"""
    return np.linspace(-15000, 0, 100)  # eta from -15000 to 0 Mpc


@pytest.fixture
def planck_parameters():
    """Standard Planck-like cosmological parameters for testing"""
    return {
        'lA': 301.0,
        'rpsi': 147.0,
        'chi_star': 14065.0,
        'sigma_chi': 250.0,
        'A': 1.0,
        'ns': 0.96,
        'Aosc': 0.55,
        'sigma_osc': 0.025
    }


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_power_spectrum():
    """Mock power spectrum function for testing projections"""
    def Pk_func(k):
        k = np.asarray(k)
        return 1e-9 * (k / 0.05)**(-0.96) * np.exp(-(k * 0.1)**2)
    return Pk_func


@pytest.fixture
def mock_window_function(sample_chi_grid):
    """Mock window function for testing projections"""
    chi_star = 14065.0
    sigma_chi = 250.0
    x = (sample_chi_grid - chi_star) / sigma_chi
    W = np.exp(-0.5 * x**2)
    norm = np.sqrt(np.trapz(W**2, sample_chi_grid))
    return W / (norm + 1e-30)


class NumericalTestHelper:
    """Helper class for numerical testing utilities"""
    
    @staticmethod
    def assert_positive_definite(array, name="array"):
        """Assert that all values in array are positive"""
        assert np.all(array > 0), f"{name} must be positive definite"
    
    @staticmethod
    def assert_normalized(array, grid, name="array", rtol=1e-10):
        """Assert that array is properly normalized"""
        norm_squared = np.trapz(array**2, grid)
        np.testing.assert_allclose(norm_squared, 1.0, rtol=rtol,
                                 err_msg=f"{name} is not properly normalized")
    
    @staticmethod
    def assert_physical_spectrum(ells, Cl, spectrum_type="TT"):
        """Assert that CMB spectrum has physical properties"""
        # Check positivity for TT and EE
        if spectrum_type in ["TT", "EE"]:
            assert np.all(Cl > 0), f"{spectrum_type} spectrum must be positive"
        
        # Check reasonable amplitude scaling
        ell_Cl = ells * (ells + 1) * Cl / (2 * np.pi)
        assert np.max(ell_Cl) < 1e4, f"{spectrum_type} spectrum amplitude too large"
        assert np.max(ell_Cl) > 1e-6, f"{spectrum_type} spectrum amplitude too small"


@pytest.fixture
def numerical_helper():
    """Fixture providing numerical testing utilities"""
    return NumericalTestHelper()


# Test data for regression testing
REFERENCE_VALUES = {
    'oscillatory_psik': {
        'k': [0.01, 0.1, 1.0],
        'expected': [9.77e-05, 0.000977, 0.55],  # Approximate values
        'rtol': 1e-3
    },
    'gaussian_window_chi': {
        'chi_star': 14065.0,
        'sigma_chi': 250.0,
        'peak_value': 0.016,  # Approximate normalized peak
        'rtol': 1e-2
    }
}


@pytest.fixture
def reference_values():
    """Reference values for regression testing"""
    return REFERENCE_VALUES


# Parametrized test cases for edge cases
EDGE_CASE_PARAMETERS = [
    # (description, parameters)
    ("zero_k", {"k": 0.0}),
    ("very_small_k", {"k": 1e-10}),
    ("very_large_k", {"k": 1e3}),
    ("negative_amplitude", {"A": -1.0}),
    ("zero_amplitude", {"A": 0.0}),
]


@pytest.fixture(params=EDGE_CASE_PARAMETERS)
def edge_case_params(request):
    """Parametrized fixture for edge case testing"""
    return request.param


def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line("markers", "integration: Integration tests for workflows")
    config.addinivalue_line("markers", "numerical: Tests for numerical accuracy")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "plotting: Tests that generate plots")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add 'unit' marker to test files starting with 'test_'
        if item.fspath.basename.startswith('test_') and 'integration' not in item.fspath.basename:
            item.add_marker(pytest.mark.unit)
        
        # Add 'integration' marker to integration test files
        if 'integration' in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        
        # Add 'plotting' marker to tests with 'plot' in name
        if 'plot' in item.name.lower():
            item.add_marker(pytest.mark.plotting)
        
        # Add 'slow' marker to tests with 'slow' in name
        if 'slow' in item.name.lower():
            item.add_marker(pytest.mark.slow)