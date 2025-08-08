#!/usr/bin/env python3
"""
Unit Tests for the QVD Physics Module
=====================================
"""

import numpy as np
import pytest
import logging
from io import StringIO

from ..physics import EnhancedQVDPhysics

class TestEnhancedQVDPhysics:
    """Tests for the EnhancedQVDPhysics class."""

    def setup_method(self):
        """Set up a model instance for tests."""
        self.physics = EnhancedQVDPhysics(
            qvd_coupling=0.85,
            redshift_power=0.6,
            igm_enhancement=0.7,
            enable_logging=False
        )

    def test_initialization(self):
        """Test that the model initializes with the correct parameters."""
        assert self.physics.qvd_coupling == 0.85
        assert self.physics.redshift_power == 0.6
        assert self.physics.igm_enhancement == 0.7

    def test_calculate_redshift_dimming(self):
        """Test the main dimming calculation."""
        redshift = 0.5
        # This is a more complex calculation, so we test for type and range
        dimming = self.physics.calculate_redshift_dimming(redshift)
        assert isinstance(dimming, float)
        assert np.isfinite(dimming)
        assert 0 <= dimming <= self.physics.bounds_enforcer.bounds.MAX_DIMMING_MAG

    def test_igm_effects_contribution(self):
        """Test that the IGM effect contributes to the dimming."""
        redshift = 0.5

        # Turn off IGM enhancement to get baseline
        self.physics.igm_enhancement = 0.0
        base_dimming = self.physics.calculate_redshift_dimming(redshift)

        # Turn IGM enhancement back on
        self.physics.igm_enhancement = 0.7
        total_dimming = self.physics.calculate_redshift_dimming(redshift)

        assert total_dimming > base_dimming

    def test_get_model_parameters(self):
        """Test the retrieval of model parameters."""
        params = self.physics.get_model_parameters()
        expected_params = {
            'qvd_coupling': 0.85,
            'redshift_power': 0.6,
            'igm_enhancement': 0.7,
        }
        # This test is now simpler as we removed some of the "status" keys
        for key, value in expected_params.items():
            assert params[key] == value

    def test_update_parameters(self, caplog):
        """Test updating the model parameters."""
        self.physics.update_parameters(qvd_coupling=1.0, igm_enhancement=0.0)
        assert self.physics.qvd_coupling == 1.0
        assert self.physics.igm_enhancement == 0.0

        # Check that unknown parameters are handled gracefully
        with caplog.at_level(logging.WARNING):
            self.physics.update_parameters(unknown_param=1.0)
        assert "Attempted to update unknown parameter: unknown_param" in caplog.text

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
