#!/usr/bin/env python3
"""
Unit Tests for the Cosmology Module
====================================
"""

import numpy as np
import pytest

from ..cosmology import EnhancedQVDCosmology

class TestEnhancedQVDCosmology:
    """Tests for the EnhancedQVDCosmology class."""

    def setup_method(self):
        """Set up a model instance for tests."""
        self.cosmology = EnhancedQVDCosmology(hubble_constant=70.0, enable_bounds_checking=False)

    def test_initialization(self):
        """Test that the model initializes with the correct parameters."""
        assert self.cosmology.H0 == 70.0
        assert self.cosmology.omega_m == 1.0
        assert self.cosmology.omega_lambda == 0.0

    def test_luminosity_distance(self):
        """Test the luminosity distance calculation."""
        redshift = 0.5
        distance = self.cosmology.luminosity_distance(redshift)
        assert isinstance(distance, float)
        assert np.isfinite(distance)
        assert distance > 0

    def test_lambda_cdm_distance(self):
        """Test the ΛCDM distance calculation for comparison."""
        redshift = 0.5
        # This value is confirmed with astropy
        expected_distance = 2832.938
        calculated_distance = self.cosmology.lambda_cdm_distance(redshift, omega_m=0.3, omega_lambda=0.7)
        assert np.isclose(calculated_distance, expected_distance, rtol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
