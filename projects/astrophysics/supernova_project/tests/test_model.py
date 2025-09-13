#!/usr/bin/env python3
"""
Unit Tests for the Supernova QVD Model
=======================================
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supernova_qvd.model import E144ScaledQVDModel
from supernova_qvd.parameters import E144ExperimentalData, SupernovaParameters

class TestE144ScaledQVDModel:
    """Tests for the E144ScaledQVDModel class."""

    def setup_method(self):
        """Set up a model instance for tests."""
        self.e144_data = E144ExperimentalData()
        self.sn_params = SupernovaParameters()
        self.model = E144ScaledQVDModel(self.e144_data, self.sn_params)

    def test_initialization(self):
        """Test that the model initializes correctly."""
        assert self.model.e144 is not None
        assert self.model.sn is not None
        assert self.model.intensity_ratio > 0

    def test_calculate_intrinsic_luminosity(self):
        """Test the intrinsic luminosity calculation at different times."""
        # Test pre-explosion
        assert self.model.calculate_intrinsic_luminosity(-20) == 0
        # Test peak
        assert self.model.calculate_intrinsic_luminosity(0) > 0
        # Test post-peak
        assert self.model.calculate_intrinsic_luminosity(20) > 0

    def test_calculate_plasma_evolution(self):
        """Test the plasma evolution calculation."""
        time_days = 10
        intrinsic_lum = self.model.calculate_intrinsic_luminosity(time_days)
        plasma_state = self.model.calculate_plasma_evolution(time_days, intrinsic_lum)

        assert plasma_state is not None
        assert plasma_state['radius_cm'] > self.sn_params.initial_radius_cm
        assert plasma_state['electron_density_cm3'] < self.sn_params.initial_electron_density_cm3
        assert plasma_state['intensity_erg_cm2_s'] > 0

    def test_calculate_qvd_cross_section(self):
        """Test the QVD cross-section calculation."""
        cross_section = self.model.calculate_qvd_cross_section(500, 1e20, 1e25, 10)
        assert cross_section > 0
        assert np.isfinite(cross_section)

    def test_calculate_spectral_scattering(self):
        """Test the spectral scattering calculation."""
        scattering_data = self.model.calculate_spectral_scattering(500, 10)

        assert scattering_data is not None
        assert 'dimming_magnitudes' in scattering_data
        assert 'optical_depth' in scattering_data
        assert 'transmission' in scattering_data
        assert np.isfinite(scattering_data['dimming_magnitudes'])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
