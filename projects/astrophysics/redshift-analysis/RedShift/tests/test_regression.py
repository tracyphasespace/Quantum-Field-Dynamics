"""
Regression tests using stored reference data

This module contains tests that compare current implementation results
against stored reference data to detect unintended changes in behavior.
"""

import pytest
import numpy as np
import os
from pathlib import Path

# Import QFD CMB modules
from qfd_cmb.ppsi_models import oscillatory_psik
from qfd_cmb.visibility import gaussian_window_chi
from qfd_cmb.kernels import sin2_mueller_coeffs, te_correlation_phase
from qfd_cmb.projector import project_limber

# Import test utilities
from .test_data_utils import reference_loader, reference_validator


class TestPowerSpectrumRegression:
    """Regression tests for power spectrum computations"""
    
    @pytest.mark.parametrize("parameter_set", [
        "planck_fiducial",
        "no_oscillations", 
        "strong_oscillations",
        "red_spectrum",
        "blue_spectrum"
    ])
    def test_power_spectrum_regression(self, parameter_set):
        """Test power spectrum against stored reference values"""
        # Skip if reference data not available
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        # Load reference data
        k_ref, Pk_ref = reference_loader.get_power_spectrum_reference(parameter_set)
        
        # Get parameters for this set
        ref_data = reference_loader.load_reference_data("power_spectrum_reference.json")
        params = ref_data["data"]["parameter_sets"][parameter_set]["parameters"]
        
        # Compute with current implementation
        Pk_computed = oscillatory_psik(k_ref, **params)
        
        # Validate against reference
        is_valid = reference_validator.validate_power_spectrum(
            k_ref, Pk_computed, parameter_set, rtol=1e-12, atol=1e-15
        )
        
        assert is_valid, f"Power spectrum regression test failed for {parameter_set}"
    
    def test_power_spectrum_edge_cases_regression(self):
        """Test power spectrum edge cases against reference"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        # Test specific edge cases
        edge_cases = [
            {"k": 1e-10, "expected_finite": True},
            {"k": 1e10, "expected_finite": True},
            {"k": 0.0, "A": 0.0, "expected_value": 0.0}
        ]
        
        for case in edge_cases:
            k = case.pop("k")
            expected_finite = case.pop("expected_finite", True)
            expected_value = case.pop("expected_value", None)
            
            Pk = oscillatory_psik(k, **case)
            
            if expected_finite:
                assert np.isfinite(Pk), f"Power spectrum not finite for case {case}"
            
            if expected_value is not None:
                np.testing.assert_allclose(Pk, expected_value, rtol=1e-15,
                                         err_msg=f"Power spectrum value incorrect for case {case}")
    
    def _reference_data_available(self):
        """Check if reference data is available"""
        ref_file = Path("tests/reference_data/power_spectrum_reference.json")
        return ref_file.exists()


class TestWindowFunctionRegression:
    """Regression tests for window function computations"""
    
    @pytest.mark.parametrize("configuration", [
        "standard_grid",
        "fine_grid",
        "coarse_grid"
    ])
    def test_window_function_regression(self, configuration):
        """Test window function against stored reference values"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        # Load reference data
        chi_ref, Wchi_ref = reference_loader.get_window_function_reference(configuration)
        
        # Get parameters
        ref_data = reference_loader.load_reference_data("window_function_reference.json")
        chi_star = ref_data["data"]["chi_star"]
        sigma_chi = ref_data["data"]["sigma_chi"]
        
        # Compute with current implementation
        Wchi_computed = gaussian_window_chi(chi_ref, chi_star, sigma_chi)
        
        # Validate against reference
        is_valid = reference_validator.validate_window_function(
            chi_ref, Wchi_computed, configuration, rtol=1e-12, atol=1e-15
        )
        
        assert is_valid, f"Window function regression test failed for {configuration}"
    
    def test_window_function_normalization_regression(self):
        """Test window function normalization against reference"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        ref_data = reference_loader.load_reference_data("window_function_reference.json")
        
        for config_name, config_data in ref_data["data"]["configurations"].items():
            chi_grid = np.array(config_data["chi_grid"])
            Wchi = np.array(config_data["Wchi"])
            expected_norm = config_data["normalization"]
            
            # Compute normalization
            computed_norm = np.trapz(Wchi**2, chi_grid)
            
            np.testing.assert_allclose(computed_norm, expected_norm, rtol=1e-10,
                                     err_msg=f"Window normalization regression failed for {config_name}")
    
    def _reference_data_available(self):
        """Check if reference data is available"""
        ref_file = Path("tests/reference_data/window_function_reference.json")
        return ref_file.exists()


class TestCorrelationRegression:
    """Regression tests for TE correlation computations"""
    
    @pytest.mark.parametrize("model", [
        "standard_model",
        "strong_damping",
        "phase_shifted"
    ])
    def test_correlation_regression(self, model):
        """Test TE correlation against stored reference values"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        # Load reference data
        ells_ref, rho_ref = reference_loader.get_correlation_reference(model)
        
        # Get parameters
        ref_data = reference_loader.load_reference_data("correlation_reference.json")
        chi_star = ref_data["data"]["chi_star"]
        rpsi = ref_data["data"]["rpsi"]
        model_params = ref_data["data"]["models"][model]["parameters"]
        
        # Compute with current implementation
        rho_computed = np.array([
            te_correlation_phase((l+0.5)/chi_star, rpsi, l, chi_star, **model_params)
            for l in ells_ref
        ])
        
        # Validate against reference
        is_valid = reference_validator.validate_correlation(
            ells_ref, rho_computed, model, rtol=1e-12, atol=1e-15
        )
        
        assert is_valid, f"TE correlation regression test failed for {model}"
    
    def test_correlation_bounds_regression(self):
        """Test that correlation bounds are maintained"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        # Test all models
        ref_data = reference_loader.load_reference_data("correlation_reference.json")
        
        for model_name, model_data in ref_data["data"]["models"].items():
            rho_values = np.array(model_data["rho_values"])
            
            # Test bounds
            assert np.all(np.abs(rho_values) <= 1.0), \
                f"Correlation bounds violated in {model_name}: max |rho| = {np.max(np.abs(rho_values))}"
    
    def _reference_data_available(self):
        """Check if reference data is available"""
        ref_file = Path("tests/reference_data/correlation_reference.json")
        return ref_file.exists()


class TestSpectraRegression:
    """Regression tests for CMB spectra computations"""
    
    @pytest.mark.parametrize("ell_range", [
        "low_ell",
        "medium_ell"
        # Skip high_ell for speed in regular testing
    ])
    def test_spectra_regression(self, ell_range):
        """Test CMB spectra against stored reference values"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        # Load reference data
        ref_spectra = reference_loader.get_spectrum_reference(ell_range)
        
        # Get computation parameters
        ref_data = reference_loader.load_reference_data("spectra_reference.json")
        params = ref_data["data"]["parameters"]
        
        # Set up computation
        ells = ref_spectra["ells"]
        chi_star = params["chi_star"]
        sigma_chi = params["sigma_chi"]
        rpsi = params["rpsi"]
        
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 501)
        Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Compute spectra
        Pk = lambda k: oscillatory_psik(k, ns=params["ns"], rpsi=rpsi, 
                                      Aosc=params["Aosc"], sigma_osc=params["sigma_osc"])
        Ctt_computed = project_limber(ells, Pk, Wchi, chi_grid)
        Cee_computed = 0.25 * Ctt_computed
        
        rho_computed = np.array([
            te_correlation_phase((l+0.5)/chi_star, rpsi, l, chi_star) 
            for l in ells
        ])
        Cte_computed = rho_computed * np.sqrt(Ctt_computed * Cee_computed)
        
        # Validate against reference
        results = reference_validator.validate_spectra(
            ells, Ctt_computed, Cee_computed, Cte_computed, 
            ell_range, rtol=1e-10, atol=1e-15
        )
        
        for spectrum, is_valid in results.items():
            assert is_valid, f"{spectrum} spectrum regression test failed for {ell_range}"
    
    @pytest.mark.slow
    def test_high_ell_spectra_regression(self):
        """Test high-ell spectra regression (marked as slow)"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        # This test is marked as slow and may be skipped in regular testing
        self.test_spectra_regression("high_ell")
    
    def test_spectra_physical_properties_regression(self):
        """Test that spectra maintain physical properties"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        ref_data = reference_loader.load_reference_data("spectra_reference.json")
        
        for range_name, spectrum_data in ref_data["data"]["spectra"].items():
            C_TT = np.array(spectrum_data["C_TT"])
            C_EE = np.array(spectrum_data["C_EE"])
            C_TE = np.array(spectrum_data["C_TE"])
            rho = np.array(spectrum_data["rho"])
            
            # Test positivity
            assert np.all(C_TT > 0), f"TT spectrum not positive in {range_name}"
            assert np.all(C_EE > 0), f"EE spectrum not positive in {range_name}"
            
            # Test Cauchy-Schwarz
            max_te = np.sqrt(C_TT * C_EE)
            assert np.all(np.abs(C_TE) <= max_te * (1 + 1e-10)), \
                f"Cauchy-Schwarz violated in {range_name}"
            
            # Test correlation bounds
            assert np.all(np.abs(rho) <= 1.0), f"Correlation bounds violated in {range_name}"
    
    def _reference_data_available(self):
        """Check if reference data is available"""
        ref_file = Path("tests/reference_data/spectra_reference.json")
        return ref_file.exists()


class TestMuellerRegression:
    """Regression tests for Mueller coefficient computations"""
    
    def test_mueller_coefficients_regression(self):
        """Test Mueller coefficients against stored reference values"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        # Load reference data
        mu_ref, w_T_ref, w_E_ref = reference_loader.get_mueller_reference()
        
        # Compute with current implementation
        w_T_computed = []
        w_E_computed = []
        
        for mu in mu_ref:
            w_T, w_E = sin2_mueller_coeffs(mu)
            w_T_computed.append(w_T)
            w_E_computed.append(w_E)
        
        w_T_computed = np.array(w_T_computed)
        w_E_computed = np.array(w_E_computed)
        
        # Validate against reference
        results = reference_validator.validate_mueller_coefficients(
            mu_ref, w_T_computed, w_E_computed, rtol=1e-12, atol=1e-15
        )
        
        for coeff_type, is_valid in results.items():
            assert is_valid, f"Mueller coefficient {coeff_type} regression test failed"
    
    def test_mueller_symmetry_regression(self):
        """Test Mueller coefficient symmetry properties"""
        if not self._reference_data_available():
            pytest.skip("Reference data not available")
        
        mu_ref, w_T_ref, w_E_ref = reference_loader.get_mueller_reference()
        
        # Test symmetry: w(μ) = w(-μ)
        for i, mu in enumerate(mu_ref):
            if mu == 0:
                continue
            
            # Find corresponding negative mu
            neg_mu_idx = np.argmin(np.abs(mu_ref + mu))
            
            if np.abs(mu_ref[neg_mu_idx] + mu) < 1e-10:
                np.testing.assert_allclose(w_T_ref[i], w_T_ref[neg_mu_idx], rtol=1e-12,
                                         err_msg=f"w_T symmetry violated at mu={mu}")
                np.testing.assert_allclose(w_E_ref[i], w_E_ref[neg_mu_idx], rtol=1e-12,
                                         err_msg=f"w_E symmetry violated at mu={mu}")
    
    def _reference_data_available(self):
        """Check if reference data is available"""
        ref_file = Path("tests/reference_data/mueller_reference.json")
        return ref_file.exists()


class TestReferenceDataIntegrity:
    """Tests for reference data integrity and availability"""
    
    def test_reference_data_files_exist(self):
        """Test that expected reference data files exist"""
        expected_files = [
            "power_spectrum_reference.json",
            "window_function_reference.json", 
            "correlation_reference.json",
            "spectra_reference.json",
            "mueller_reference.json"
        ]
        
        ref_dir = Path("tests/reference_data")
        
        if not ref_dir.exists():
            pytest.skip("Reference data directory does not exist")
        
        missing_files = []
        for filename in expected_files:
            if not (ref_dir / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            pytest.skip(f"Missing reference data files: {missing_files}")
    
    def test_reference_data_integrity(self):
        """Test integrity of reference data files"""
        if not Path("tests/reference_data").exists():
            pytest.skip("Reference data directory does not exist")
        
        available_data = reference_loader.list_available_data()
        
        for filename, metadata in available_data.items():
            if "error" in metadata:
                pytest.fail(f"Error loading {filename}: {metadata['error']}")
            
            # Check that metadata contains expected fields
            expected_fields = ["generated_at", "generator_version", "data_hash"]
            for field in expected_fields:
                assert field in metadata, f"Missing metadata field '{field}' in {filename}"
    
    def test_reference_data_generation_script(self):
        """Test that reference data generation script exists and is executable"""
        script_path = Path("tests/generate_reference_data.py")
        assert script_path.exists(), "Reference data generation script not found"
        
        # Check that script is readable
        assert os.access(script_path, os.R_OK), "Reference data generation script not readable"


# Regression test markers
pytestmark = [pytest.mark.unit, pytest.mark.regression]