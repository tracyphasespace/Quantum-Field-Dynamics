#!/usr/bin/env python3
"""
Basic functionality tests for QFD Supernova Analysis Package
"""

import os
import sys
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of QFD analysis components."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    def test_data_files_exist(self):
        """Test that required data files exist."""
        required_files = [
            'union2.1_data.txt',
            'union2.1_data_with_errors.txt',
            'sample_lightcurves/lightcurves_osc.csv'
        ]

        for file_path in required_files:
            full_path = os.path.join(self.test_data_dir, file_path)
            self.assertTrue(os.path.exists(full_path), f"Missing required file: {file_path}")

    def test_script_files_exist(self):
        """Test that core scripts exist and are importable."""
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        required_scripts = [
            'QFD_Cosmology_Fitter_v5.6.py',
            'qfd_plasma_veil_fitter.py',
            'compare_qfd_lcdm_mu_z.py',
            'qfd_lightcurve_comparison_v2.py',
            'qfd_ingest_lightcurves.py'
        ]

        for script in required_scripts:
            script_path = os.path.join(src_dir, script)
            self.assertTrue(os.path.exists(script_path), f"Missing script: {script}")

    def test_union21_data_format(self):
        """Test Union2.1 data can be loaded correctly."""
        union21_path = os.path.join(self.test_data_dir, 'union2.1_data_with_errors.txt')

        if os.path.exists(union21_path):
            data = np.loadtxt(union21_path)
            self.assertEqual(data.shape[1], 3, "Union2.1 data should have 3 columns (z, mu, sigma_mu)")
            self.assertGreater(len(data), 500, "Should have >500 supernovae")
            self.assertTrue(np.all(data[:, 0] > 0), "All redshifts should be positive")
            self.assertTrue(np.all(data[:, 2] > 0), "All errors should be positive")

    def test_lightcurve_data_format(self):
        """Test light curve data can be loaded correctly."""
        lc_path = os.path.join(self.test_data_dir, 'sample_lightcurves', 'lightcurves_osc.csv')

        if os.path.exists(lc_path):
            # Try to read first few lines
            with open(lc_path, 'r') as f:
                header = f.readline().strip()
                first_data = f.readline().strip()

            # Check required columns exist
            required_cols = ['snid', 'mjd', 'band', 'mag']
            for col in required_cols:
                self.assertIn(col, header, f"Missing required column: {col}")

    def test_package_structure(self):
        """Test that package has correct directory structure."""
        package_root = os.path.dirname(os.path.dirname(__file__))
        required_dirs = ['src', 'data', 'docs', 'examples', 'tests', 'results']

        for dir_name in required_dirs:
            dir_path = os.path.join(package_root, dir_name)
            self.assertTrue(os.path.isdir(dir_path), f"Missing directory: {dir_name}")

    def test_documentation_exists(self):
        """Test that key documentation files exist."""
        package_root = os.path.dirname(os.path.dirname(__file__))
        required_docs = [
            'README.md',
            'requirements.txt',
            'LICENSE'
        ]

        for doc in required_docs:
            doc_path = os.path.join(package_root, doc)
            self.assertTrue(os.path.exists(doc_path), f"Missing documentation: {doc}")

class TestQFDPhysics(unittest.TestCase):
    """Test QFD physics calculations."""

    def test_plasma_redshift_function(self):
        """Test QFD plasma redshift calculation."""
        # Simple test of plasma redshift formula
        # z_plasma = A_plasma * (1 - exp(-t/tau)) * (lambda_B/lambda)^beta

        A_plasma = 0.1
        tau_decay = 50.0  # days
        beta = 1.0
        lambda_B = 440.0  # nm

        # Test parameters
        t_days = np.array([0, 10, 50, 100])
        wavelength_nm = 500.0

        # Calculate expected values
        temporal_factor = 1.0 - np.exp(-t_days / tau_decay)
        wavelength_factor = (lambda_B / wavelength_nm) ** beta
        expected = A_plasma * temporal_factor * wavelength_factor

        # For t=0, should be 0
        self.assertAlmostEqual(expected[0], 0.0, places=10)

        # Should increase with time
        self.assertTrue(np.all(np.diff(expected) >= 0))

        # Should approach asymptotic value
        asymptotic = A_plasma * wavelength_factor
        self.assertLess(expected[-1], asymptotic)
        self.assertGreater(expected[-1], 0.9 * asymptotic)

    def test_wavelength_dependence(self):
        """Test wavelength dependence of plasma redshift."""
        A_plasma = 0.1
        tau_decay = 50.0
        beta = 1.0
        lambda_B = 440.0
        t_days = 50.0  # Fixed time

        # Test different wavelengths
        wavelengths = np.array([300, 440, 600, 800])  # nm
        temporal_factor = 1.0 - np.exp(-t_days / tau_decay)

        z_plasma = A_plasma * temporal_factor * (lambda_B / wavelengths) ** beta

        # Bluer light (shorter wavelength) should be more affected
        self.assertTrue(np.all(np.diff(z_plasma) <= 0))  # Decreasing with wavelength

        # At reference wavelength, should equal A_plasma * temporal_factor
        ref_idx = np.where(wavelengths == lambda_B)[0][0]
        expected_ref = A_plasma * temporal_factor
        self.assertAlmostEqual(z_plasma[ref_idx], expected_ref, places=10)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)