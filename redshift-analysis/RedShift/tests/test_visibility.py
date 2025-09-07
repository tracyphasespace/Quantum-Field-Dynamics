"""
Unit tests for qfd_cmb.visibility module

Tests the gaussian_visibility and gaussian_window_chi functions for
coordinate conversion helpers, boundary conditions, and normalization.
"""

import pytest
import numpy as np
from qfd_cmb.visibility import gaussian_visibility, gaussian_window_chi


class TestGaussianVisibility:
    """Test suite for gaussian_visibility function"""
    
    def test_basic_functionality(self, sample_eta_grid):
        """Test basic function call with default parameters"""
        eta_star = -7000.0
        sigma_eta = 100.0
        
        result = gaussian_visibility(sample_eta_grid, eta_star, sigma_eta)
        
        # Check output shape and type
        assert result.shape == sample_eta_grid.shape
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_normalization(self, sample_eta_grid, numerical_helper):
        """Test that visibility function is properly L2-normalized"""
        eta_star = -7000.0
        sigma_eta = 100.0
        
        result = gaussian_visibility(sample_eta_grid, eta_star, sigma_eta)
        
        # Check L2 normalization
        numerical_helper.assert_normalized(result, sample_eta_grid, "gaussian_visibility")
    
    def test_peak_location(self):
        """Test that peak occurs at eta_star"""
        eta_grid = np.linspace(-10000, 0, 1000)
        eta_star = -5000.0
        sigma_eta = 200.0
        
        result = gaussian_visibility(eta_grid, eta_star, sigma_eta)
        
        # Find peak location
        peak_idx = np.argmax(result)
        peak_eta = eta_grid[peak_idx]
        
        # Peak should be close to eta_star
        np.testing.assert_allclose(peak_eta, eta_star, atol=eta_grid[1] - eta_grid[0])
    
    def test_gaussian_shape(self):
        """Test that function has proper Gaussian shape"""
        eta_grid = np.linspace(-8000, -6000, 1000)
        eta_star = -7000.0
        sigma_eta = 300.0
        
        result = gaussian_visibility(eta_grid, eta_star, sigma_eta)
        
        # Test symmetry around peak
        peak_idx = np.argmax(result)
        left_wing = result[:peak_idx]
        right_wing = result[peak_idx+1:]
        
        # Should be approximately symmetric (within numerical precision)
        min_len = min(len(left_wing), len(right_wing))
        if min_len > 10:  # Only test if we have enough points
            np.testing.assert_allclose(
                left_wing[-min_len:], 
                right_wing[:min_len][::-1], 
                rtol=0.1
            )
    
    def test_width_scaling(self):
        """Test that sigma_eta controls the width correctly"""
        eta_grid = np.linspace(-8000, -6000, 1000)
        eta_star = -7000.0
        
        # Narrow and wide Gaussians
        sigma_narrow = 100.0
        sigma_wide = 500.0
        
        result_narrow = gaussian_visibility(eta_grid, eta_star, sigma_narrow)
        result_wide = gaussian_visibility(eta_grid, eta_star, sigma_wide)
        
        # Narrow should have higher peak (due to normalization)
        assert np.max(result_narrow) > np.max(result_wide)
        
        # Wide should have more spread (check FWHM approximately)
        # Find half-maximum points
        half_max_narrow = np.max(result_narrow) / 2
        half_max_wide = np.max(result_wide) / 2
        
        above_half_narrow = np.sum(result_narrow > half_max_narrow)
        above_half_wide = np.sum(result_wide > half_max_wide)
        
        assert above_half_wide > above_half_narrow
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        eta_grid = np.linspace(-1000, 0, 100)
        
        # Very small sigma
        result_small = gaussian_visibility(eta_grid, -500.0, 1e-3)
        assert np.all(np.isfinite(result_small))
        assert np.max(result_small) > 0
        
        # Very large sigma
        result_large = gaussian_visibility(eta_grid, -500.0, 1e6)
        assert np.all(np.isfinite(result_large))
        assert np.max(result_large) > 0
        
        # eta_star outside grid
        result_outside = gaussian_visibility(eta_grid, -2000.0, 100.0)
        assert np.all(np.isfinite(result_outside))
        assert np.all(result_outside >= 0)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme parameters"""
        eta_grid = np.linspace(-15000, 0, 100)
        
        # Very narrow Gaussian (potential numerical issues)
        result = gaussian_visibility(eta_grid, -7000.0, 1e-6)
        assert np.all(np.isfinite(result))
        
        # Check normalization doesn't fail
        norm_check = np.trapz(result**2, eta_grid)
        assert np.isfinite(norm_check)
        assert norm_check > 0


class TestGaussianWindowChi:
    """Test suite for gaussian_window_chi function"""
    
    def test_basic_functionality(self, sample_chi_grid):
        """Test basic function call with default parameters"""
        chi_star = 14065.0
        sigma_chi = 250.0
        
        result = gaussian_window_chi(sample_chi_grid, chi_star, sigma_chi)
        
        # Check output shape and type
        assert result.shape == sample_chi_grid.shape
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_normalization(self, sample_chi_grid, numerical_helper):
        """Test that window function is properly L2-normalized"""
        chi_star = 14065.0
        sigma_chi = 250.0
        
        result = gaussian_window_chi(sample_chi_grid, chi_star, sigma_chi)
        
        # Check L2 normalization
        numerical_helper.assert_normalized(result, sample_chi_grid, "gaussian_window_chi")
    
    def test_peak_location(self):
        """Test that peak occurs at chi_star"""
        chi_grid = np.linspace(13000, 15000, 1000)
        chi_star = 14065.0
        sigma_chi = 250.0
        
        result = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Find peak location
        peak_idx = np.argmax(result)
        peak_chi = chi_grid[peak_idx]
        
        # Peak should be close to chi_star
        np.testing.assert_allclose(peak_chi, chi_star, atol=chi_grid[1] - chi_grid[0])
    
    def test_gaussian_shape(self):
        """Test that function has proper Gaussian shape"""
        chi_grid = np.linspace(13500, 14500, 1000)
        chi_star = 14065.0
        sigma_chi = 200.0
        
        result = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Test that it's a proper Gaussian by checking log-derivative
        # d/dx log(G(x)) = -(x-x0)/sigma^2
        x = (chi_grid - chi_star) / sigma_chi
        expected_log_deriv = -x / sigma_chi
        
        # Compute numerical log-derivative (avoiding zeros)
        log_result = np.log(result + 1e-30)
        numerical_log_deriv = np.gradient(log_result, chi_grid)
        
        # Should match in the central region
        central_mask = np.abs(x) < 2  # Within 2 sigma
        if np.sum(central_mask) > 10:
            np.testing.assert_allclose(
                numerical_log_deriv[central_mask],
                expected_log_deriv[central_mask],
                rtol=0.1
            )
    
    def test_width_scaling(self):
        """Test that sigma_chi controls the width correctly"""
        chi_grid = np.linspace(13000, 15000, 1000)
        chi_star = 14065.0
        
        # Narrow and wide windows
        sigma_narrow = 100.0
        sigma_wide = 500.0
        
        result_narrow = gaussian_window_chi(chi_grid, chi_star, sigma_narrow)
        result_wide = gaussian_window_chi(chi_grid, chi_star, sigma_wide)
        
        # Narrow should have higher peak (due to normalization)
        assert np.max(result_narrow) > np.max(result_wide)
        
        # Wide should have more spread
        # Check effective width using second moment
        chi_mean_narrow = np.trapz(chi_grid * result_narrow**2, chi_grid)
        chi_mean_wide = np.trapz(chi_grid * result_wide**2, chi_grid)
        
        chi2_mean_narrow = np.trapz(chi_grid**2 * result_narrow**2, chi_grid)
        chi2_mean_wide = np.trapz(chi_grid**2 * result_wide**2, chi_grid)
        
        var_narrow = chi2_mean_narrow - chi_mean_narrow**2
        var_wide = chi2_mean_wide - chi_mean_wide**2
        
        assert var_wide > var_narrow
    
    def test_planck_parameters(self, planck_parameters):
        """Test with Planck-like parameter values"""
        chi_grid = np.linspace(12000, 16000, 1000)
        chi_star = planck_parameters['chi_star']
        sigma_chi = planck_parameters['sigma_chi']
        
        result = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Check basic properties
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))
        
        # Check normalization
        norm_squared = np.trapz(result**2, chi_grid)
        np.testing.assert_allclose(norm_squared, 1.0, rtol=1e-10)
        
        # Check peak location
        peak_idx = np.argmax(result)
        peak_chi = chi_grid[peak_idx]
        np.testing.assert_allclose(peak_chi, chi_star, atol=chi_grid[1] - chi_grid[0])
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        chi_grid = np.linspace(1000, 20000, 100)
        
        # Very small sigma
        result_small = gaussian_window_chi(chi_grid, 14000.0, 1.0)
        assert np.all(np.isfinite(result_small))
        assert np.max(result_small) > 0
        
        # Very large sigma
        result_large = gaussian_window_chi(chi_grid, 14000.0, 10000.0)
        assert np.all(np.isfinite(result_large))
        assert np.max(result_large) > 0
        
        # chi_star outside grid
        result_outside = gaussian_window_chi(chi_grid, 50000.0, 1000.0)
        assert np.all(np.isfinite(result_outside))
        assert np.all(result_outside >= 0)
    
    def test_coordinate_conversion_consistency(self):
        """Test consistency between eta and chi coordinate systems"""
        # This is a conceptual test - in practice, coordinate conversion
        # would involve cosmological parameters
        
        # Test that both functions give similar normalized shapes
        # when using equivalent coordinate ranges
        
        eta_grid = np.linspace(-15000, -13000, 1000)
        chi_grid = np.linspace(13000, 15000, 1000)  # Roughly equivalent range
        
        eta_result = gaussian_visibility(eta_grid, -14000.0, 200.0)
        chi_result = gaussian_window_chi(chi_grid, 14000.0, 200.0)
        
        # Both should be properly normalized
        eta_norm = np.trapz(eta_result**2, eta_grid)
        chi_norm = np.trapz(chi_result**2, chi_grid)
        
        np.testing.assert_allclose(eta_norm, 1.0, rtol=1e-10)
        np.testing.assert_allclose(chi_norm, 1.0, rtol=1e-10)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme parameters"""
        chi_grid = np.linspace(1, 30000, 100)
        
        # Very narrow window (potential numerical issues)
        result = gaussian_window_chi(chi_grid, 15000.0, 1e-3)
        assert np.all(np.isfinite(result))
        
        # Check normalization doesn't fail
        norm_check = np.trapz(result**2, chi_grid)
        assert np.isfinite(norm_check)
        assert norm_check > 0
        
        # Very wide window
        result_wide = gaussian_window_chi(chi_grid, 15000.0, 1e6)
        assert np.all(np.isfinite(result_wide))
        
        norm_check_wide = np.trapz(result_wide**2, chi_grid)
        assert np.isfinite(norm_check_wide)
        assert norm_check_wide > 0
    
    @pytest.mark.numerical
    def test_regression_values(self, reference_values):
        """Test against reference values for regression testing"""
        ref_data = reference_values['gaussian_window_chi']
        chi_star = ref_data['chi_star']
        sigma_chi = ref_data['sigma_chi']
        expected_peak = ref_data['peak_value']
        rtol = ref_data['rtol']
        
        chi_grid = np.linspace(chi_star - 3*sigma_chi, chi_star + 3*sigma_chi, 1000)
        result = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Check peak value is reasonable
        actual_peak = np.max(result)
        assert actual_peak > expected_peak * 0.5
        assert actual_peak < expected_peak * 2.0
        
        # Check normalization
        norm_squared = np.trapz(result**2, chi_grid)
        np.testing.assert_allclose(norm_squared, 1.0, rtol=1e-8)
    
    def test_boundary_conditions(self):
        """Test behavior at grid boundaries"""
        chi_grid = np.linspace(10000, 20000, 1000)
        
        # Window centered at left boundary
        result_left = gaussian_window_chi(chi_grid, 10000.0, 500.0)
        assert np.all(np.isfinite(result_left))
        assert result_left[0] > result_left[-1]  # Should decay from left
        
        # Window centered at right boundary  
        result_right = gaussian_window_chi(chi_grid, 20000.0, 500.0)
        assert np.all(np.isfinite(result_right))
        assert result_right[-1] > result_right[0]  # Should decay from right
        
        # Window far outside grid
        result_far = gaussian_window_chi(chi_grid, 50000.0, 1000.0)
        assert np.all(np.isfinite(result_far))
        assert np.all(result_far >= 0)
        assert np.max(result_far) < 0.1  # Should be very small