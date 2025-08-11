"""
Regression tests against manually calculated reference values.
"""

import unittest
import numpy as np
import math

from coupling_constants.registry.parameter_registry import ParameterRegistry


class TestReferenceValues(unittest.TestCase):
    """Test framework calculations against manually computed reference values."""
    
    def setUp(self):
        """Set up test fixtures with reference parameter values."""
        self.registry = ParameterRegistry()
        self.tolerance = 1e-10
        
        # Standard reference values
        reference_params = {
            'alpha_em': 7.2973525693e-3,
            'c': 299792458.0,
            'G': 6.67430e-11,
            'hbar': 1.054571817e-34,
            'k_B': 1.380649e-23,
            'PPN_gamma': 1.0,
            'PPN_beta': 1.0,
            'T_CMB_K': 2.7255,
            'H0': 67.4,
            'n_vac': 1.0,
            'k_J': 1e-15,
            'xi': 2.0,
            'psi_s0': -1.5,
        }
        
        for param_name, value in reference_params.items():
            self.registry.update_parameter(param_name, value, "reference", "Reference value")
    
    def test_fine_structure_constant(self):
        """Test fine structure constant calculations."""
        alpha = self.registry.get_parameter("alpha_em").value
        
        # alpha ≈ 1/137.036
        expected_inverse = 137.035999084
        calculated_inverse = 1.0 / alpha
        
        self.assertAlmostEqual(calculated_inverse, expected_inverse, places=8)
    
    def test_planck_units(self):
        """Test Planck unit calculations."""
        G = self.registry.get_parameter("G").value
        hbar = self.registry.get_parameter("hbar").value
        c = self.registry.get_parameter("c").value
        
        # Planck length: l_P = √(ħG/c³)
        planck_length_expected = 1.616255e-35
        planck_length_calculated = math.sqrt(hbar * G / (c**3))
        
        relative_error = abs(planck_length_calculated - planck_length_expected) / planck_length_expected
        self.assertLess(relative_error, 1e-5)
    
    def test_vacuum_parameters(self):
        """Test vacuum parameter constraints."""
        n_vac = self.registry.get_parameter("n_vac").value
        k_J = self.registry.get_parameter("k_J").value
        
        # Vacuum refractive index should be exactly 1
        self.assertEqual(n_vac, 1.0)
        
        # k_J should be very small
        self.assertLess(abs(k_J), 1e-10)


if __name__ == "__main__":
    unittest.main()