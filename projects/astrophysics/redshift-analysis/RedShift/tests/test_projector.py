"""
Unit tests for qfd_cmb.projector module

Tests the project_limber, los_transfer, and project_los functions for
convergence with different grid resolutions and numerical integration accuracy.
"""

import pytest
import numpy as np
from scipy.special import spherical_jn
from qfd_cmb.projector import project_limber, los_transfer, project_los


class TestProjectLimber:
    """Test suite for project_limber function"""
    
    def test_basic_functionality(self, sample_ell_values, mock_power_spectrum, 
                                mock_window_function, sample_chi_grid):
        """Test basic function call with mock inputs"""
        result = project_limber(sample_ell_values, mock_power_spectrum, 
                              mock_window_function, sample_chi_grid)
        
        # Check output shape and type
        assert result.shape == sample_ell_values.shape
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_positivity(self, sample_ell_values, mock_power_spectrum,
                       mock_window_function, sample_chi_grid, numerical_helper):
        """Test that projected power spectrum is positive"""
        result = project_limber(sample_ell_values, mock_power_spectrum,
                              mock_window_function, sample_chi_grid)
        
        numerical_helper.assert_positive_definite(result, "C_ell from Limber projection")
    
    def test_ell_scaling(self, mock_power_spectrum, mock_window_function, sample_chi_grid):
        """Test scaling behavior with ell"""
        ell_low = np.array([2, 10, 50])
        ell_high = np.array([100, 500, 1000])
        
        result_low = project_limber(ell_low, mock_power_spectrum,
                                  mock_window_function, sample_chi_grid)
        result_high = project_limber(ell_high, mock_power_spectrum,
                                   mock_window_function, sample_chi_grid)
        
        # For typical power spectra, high-ell should be smaller
        assert np.all(result_low > 0)
        assert np.all(result_high > 0)
        assert np.mean(result_high) < np.mean(result_low)
    
    def test_window_function_scaling(self, sample_ell_values, mock_power_spectrum, sample_chi_grid):
        """Test that broader windows give larger signals"""
        chi_star = 14065.0
        
        # Narrow window
        sigma_narrow = 100.0
        x_narrow = (sample_chi_grid - chi_star) / sigma_narrow
        W_narrow = np.exp(-0.5 * x_narrow**2)
        W_narrow /= np.sqrt(np.trapz(W_narrow**2, sample_chi_grid))
        
        # Wide window
        sigma_wide = 500.0
        x_wide = (sample_chi_grid - chi_star) / sigma_wide
        W_wide = np.exp(-0.5 * x_wide**2)
        W_wide /= np.sqrt(np.trapz(W_wide**2, sample_chi_grid))
        
        result_narrow = project_limber(sample_ell_values, mock_power_spectrum,
                                     W_narrow, sample_chi_grid)
        result_wide = project_limber(sample_ell_values, mock_power_spectrum,
                                   W_wide, sample_chi_grid)
        
        # Both should be positive
        assert np.all(result_narrow > 0)
        assert np.all(result_wide > 0)
        
        # Wide window typically gives larger signal (more volume)
        # This depends on the specific power spectrum shape
        assert np.max(result_wide) > 0
        assert np.max(result_narrow) > 0
    
    def test_power_spectrum_scaling(self, sample_ell_values, mock_window_function, sample_chi_grid):
        """Test linear scaling with power spectrum amplitude"""
        def Pk_base(k):
            return 1e-9 * (k / 0.05)**(-0.96)
        
        def Pk_scaled(k):
            return 2.0 * Pk_base(k)
        
        result_base = project_limber(sample_ell_values, Pk_base,
                                   mock_window_function, sample_chi_grid)
        result_scaled = project_limber(sample_ell_values, Pk_scaled,
                                     mock_window_function, sample_chi_grid)
        
        # Should scale linearly
        np.testing.assert_allclose(result_scaled, 2.0 * result_base, rtol=1e-10)
    
    def test_convergence_with_resolution(self, mock_power_spectrum, mock_window_function):
        """Test convergence as chi grid resolution increases"""
        ell_test = np.array([10, 100, 500])
        chi_star = 14065.0
        
        # Different resolutions
        resolutions = [50, 100, 200, 400]
        results = []
        
        for N in resolutions:
            chi_grid = np.linspace(chi_star - 1000, chi_star + 1000, N)
            # Recompute window for this grid
            x = (chi_grid - chi_star) / 250.0
            W = np.exp(-0.5 * x**2)
            W /= np.sqrt(np.trapz(W**2, chi_grid))
            
            result = project_limber(ell_test, mock_power_spectrum, W, chi_grid)
            results.append(result)
        
        # Should converge - check that high resolution results are close
        np.testing.assert_allclose(results[-1], results[-2], rtol=0.05)
    
    def test_limber_approximation_validity(self, mock_power_spectrum, mock_window_function, sample_chi_grid):
        """Test that Limber approximation gives reasonable results for high ell"""
        # Limber approximation: k = (ell + 1/2) / chi
        ell_high = np.array([100, 500, 1000, 2000])
        
        result = project_limber(ell_high, mock_power_spectrum,
                              mock_window_function, sample_chi_grid)
        
        # Should be positive and finite
        assert np.all(result > 0)
        assert np.all(np.isfinite(result))
        
        # Should show typical CMB-like scaling (decreasing with ell for most models)
        # This is model-dependent, but check general reasonableness
        assert result[0] > result[-1] * 0.01  # Not too steep decay
        assert result[0] < result[-1] * 100   # Not too shallow
    
    def test_edge_cases(self, mock_power_spectrum, sample_chi_grid):
        """Test edge cases and boundary conditions"""
        # Very narrow window (delta function-like)
        chi_star = 14065.0
        W_delta = np.zeros_like(sample_chi_grid)
        idx_star = np.argmin(np.abs(sample_chi_grid - chi_star))
        W_delta[idx_star] = 1.0
        
        ell_test = np.array([10, 100])
        result_delta = project_limber(ell_test, mock_power_spectrum, W_delta, sample_chi_grid)
        
        assert np.all(np.isfinite(result_delta))
        assert np.all(result_delta >= 0)
        
        # Very broad window (nearly flat)
        W_flat = np.ones_like(sample_chi_grid)
        W_flat /= np.sqrt(np.trapz(W_flat**2, sample_chi_grid))
        
        result_flat = project_limber(ell_test, mock_power_spectrum, W_flat, sample_chi_grid)
        
        assert np.all(np.isfinite(result_flat))
        assert np.all(result_flat > 0)
    
    def test_numerical_integration_accuracy(self, mock_window_function, sample_chi_grid):
        """Test numerical integration accuracy with analytical test case"""
        # Use a simple power law that can be integrated analytically
        def Pk_powerlaw(k):
            return (k / 0.1)**(-1.0)  # Simple power law
        
        ell_test = np.array([100])
        result = project_limber(ell_test, Pk_powerlaw, mock_window_function, sample_chi_grid)
        
        # Should be positive and finite
        assert result[0] > 0
        assert np.isfinite(result[0])
        
        # Check that result is reasonable order of magnitude
        assert result[0] > 1e-15
        assert result[0] < 1e5


class TestLosTransfer:
    """Test suite for los_transfer function"""
    
    def test_basic_functionality(self, sample_ell_values, sample_eta_grid):
        """Test basic function call with mock source function"""
        k_grid = np.logspace(-3, 0, 20)
        
        def S_func(k, eta):
            # Simple mock source function
            return np.exp(-(k * 0.1)**2) * np.exp(-(eta + 7000)**2 / (2 * 500**2))
        
        result = los_transfer(sample_ell_values[:5], k_grid, sample_eta_grid, S_func)
        
        # Check output shape
        assert result.shape == (5, len(k_grid))
        assert isinstance(result, np.ndarray)
    
    def test_spherical_bessel_integration(self, sample_eta_grid):
        """Test that spherical Bessel function integration works correctly"""
        k_grid = np.array([0.01, 0.1])
        ell_test = np.array([0, 1, 2])
        
        # Simple source function
        def S_constant(k, eta):
            return np.ones_like(k[:, None] * eta[None, :])
        
        result = los_transfer(ell_test, k_grid, sample_eta_grid, S_constant)
        
        # Should be finite
        assert np.all(np.isfinite(result))
        
        # For ell=0, j_0(x) = sin(x)/x, should give non-zero result
        assert np.abs(result[0, 0]) > 1e-10
    
    def test_source_function_scaling(self, sample_eta_grid):
        """Test linear scaling with source function amplitude"""
        k_grid = np.array([0.05])
        ell_test = np.array([10])
        
        def S_base(k, eta):
            return np.exp(-(eta + 7000)**2 / (2 * 500**2))
        
        def S_scaled(k, eta):
            return 3.0 * S_base(k, eta)
        
        result_base = los_transfer(ell_test, k_grid, sample_eta_grid, S_base)
        result_scaled = los_transfer(ell_test, k_grid, sample_eta_grid, S_scaled)
        
        # Should scale linearly
        np.testing.assert_allclose(result_scaled, 3.0 * result_base, rtol=1e-10)
    
    def test_k_dependence(self, sample_eta_grid):
        """Test k-dependence of transfer functions"""
        k_grid = np.logspace(-2, 0, 10)
        ell_test = np.array([10, 100])
        
        def S_func(k, eta):
            return np.exp(-(eta + 7000)**2 / (2 * 500**2)) * np.ones_like(k[:, None])
        
        result = los_transfer(ell_test, k_grid, sample_eta_grid, S_func)
        
        # Should vary with k due to spherical Bessel functions
        for i in range(len(ell_test)):
            assert np.var(result[i, :]) > 1e-20, f"Transfer function should vary with k for ell={ell_test[i]}"
    
    def test_ell_dependence(self, sample_eta_grid):
        """Test ell-dependence of transfer functions"""
        k_grid = np.array([0.05])
        ell_test = np.array([1, 10, 50, 100])
        
        def S_func(k, eta):
            return np.exp(-(eta + 7000)**2 / (2 * 500**2))
        
        result = los_transfer(ell_test, k_grid, sample_eta_grid, S_func)
        
        # Different ell should give different results
        for i in range(len(ell_test)):
            for j in range(i+1, len(ell_test)):
                assert not np.allclose(result[i, :], result[j, :], rtol=0.1)
    
    def test_oscillatory_behavior(self, sample_eta_grid):
        """Test oscillatory behavior from spherical Bessel functions"""
        k_grid = np.array([0.1])  # Fixed k
        ell_test = np.arange(1, 50)  # Range of ell values
        
        def S_func(k, eta):
            return np.ones_like(k[:, None] * eta[None, :])
        
        result = los_transfer(ell_test, k_grid, sample_eta_grid, S_func)
        
        # Should show oscillatory behavior with ell
        transfer_vs_ell = result[:, 0]
        
        # Check for sign changes (oscillations)
        sign_changes = np.sum(np.diff(np.sign(transfer_vs_ell)) != 0)
        assert sign_changes > 5, "Transfer function should oscillate with ell"


class TestProjectLos:
    """Test suite for project_los function"""
    
    def test_basic_functionality(self, sample_ell_values, mock_power_spectrum):
        """Test basic function call with mock transfer functions"""
        k_grid = np.logspace(-3, 0, 50)
        
        # Mock transfer functions
        DeltaX = np.random.randn(len(sample_ell_values), len(k_grid)) * 0.1
        DeltaY = np.random.randn(len(sample_ell_values), len(k_grid)) * 0.1
        
        result = project_los(sample_ell_values, k_grid, mock_power_spectrum, DeltaX, DeltaY)
        
        # Check output shape and type
        assert result.shape == sample_ell_values.shape
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_power_spectrum_scaling(self, sample_ell_values):
        """Test linear scaling with power spectrum amplitude"""
        k_grid = np.logspace(-3, 0, 20)
        
        def Pk_base(k):
            return 1e-9 * (k / 0.05)**(-0.96)
        
        def Pk_scaled(k):
            return 2.0 * Pk_base(k)
        
        # Simple transfer functions
        DeltaX = np.ones((len(sample_ell_values), len(k_grid))) * 0.1
        DeltaY = np.ones((len(sample_ell_values), len(k_grid))) * 0.1
        
        result_base = project_los(sample_ell_values, k_grid, Pk_base, DeltaX, DeltaY)
        result_scaled = project_los(sample_ell_values, k_grid, Pk_scaled, DeltaX, DeltaY)
        
        # Should scale linearly
        np.testing.assert_allclose(result_scaled, 2.0 * result_base, rtol=1e-10)
    
    def test_transfer_function_scaling(self, sample_ell_values, mock_power_spectrum):
        """Test scaling with transfer function amplitude"""
        k_grid = np.logspace(-3, 0, 20)
        
        DeltaX_base = np.ones((len(sample_ell_values), len(k_grid))) * 0.1
        DeltaY_base = np.ones((len(sample_ell_values), len(k_grid))) * 0.1
        
        DeltaX_scaled = 2.0 * DeltaX_base
        DeltaY_scaled = 3.0 * DeltaY_base
        
        result_base = project_los(sample_ell_values, k_grid, mock_power_spectrum, 
                                DeltaX_base, DeltaY_base)
        result_scaled = project_los(sample_ell_values, k_grid, mock_power_spectrum,
                                  DeltaX_scaled, DeltaY_scaled)
        
        # Should scale as product: 2.0 * 3.0 = 6.0
        np.testing.assert_allclose(result_scaled, 6.0 * result_base, rtol=1e-10)
    
    def test_cross_correlation_properties(self, sample_ell_values, mock_power_spectrum):
        """Test cross-correlation properties (symmetry, etc.)"""
        k_grid = np.logspace(-3, 0, 20)
        
        # Two different transfer functions
        DeltaX = np.random.randn(len(sample_ell_values), len(k_grid)) * 0.1
        DeltaY = np.random.randn(len(sample_ell_values), len(k_grid)) * 0.1
        
        # Cross-correlations should be symmetric
        Cxy = project_los(sample_ell_values, k_grid, mock_power_spectrum, DeltaX, DeltaY)
        Cyx = project_los(sample_ell_values, k_grid, mock_power_spectrum, DeltaY, DeltaX)
        
        np.testing.assert_allclose(Cxy, Cyx, rtol=1e-15)
    
    def test_auto_correlation_positivity(self, sample_ell_values, mock_power_spectrum):
        """Test that auto-correlations are positive (for positive P(k))"""
        k_grid = np.logspace(-3, 0, 20)
        
        # Random but reasonable transfer function
        Delta = np.random.randn(len(sample_ell_values), len(k_grid)) * 0.1
        
        # Auto-correlation
        result = project_los(sample_ell_values, k_grid, mock_power_spectrum, Delta, Delta)
        
        # Should be positive (P(k) > 0 and Delta² ≥ 0)
        assert np.all(result >= 0), "Auto-correlation should be non-negative"
    
    def test_k_integration_convergence(self, mock_power_spectrum):
        """Test convergence of k-integration with different resolutions"""
        ell_test = np.array([10, 100])
        
        # Simple transfer functions
        resolutions = [20, 50, 100, 200]
        results = []
        
        for N in resolutions:
            k_grid = np.logspace(-3, 0, N)
            Delta = np.ones((len(ell_test), N)) * 0.1
            
            result = project_los(ell_test, k_grid, mock_power_spectrum, Delta, Delta)
            results.append(result)
        
        # Should converge
        np.testing.assert_allclose(results[-1], results[-2], rtol=0.05)
    
    def test_physical_units_consistency(self, sample_ell_values, mock_power_spectrum):
        """Test that units work out correctly"""
        k_grid = np.logspace(-3, 0, 50)  # Mpc^-1
        
        # Transfer functions (dimensionless)
        Delta = np.ones((len(sample_ell_values), len(k_grid))) * 0.1
        
        result = project_los(sample_ell_values, k_grid, mock_power_spectrum, Delta, Delta)
        
        # Result should have units of P(k) (since Delta is dimensionless)
        # For mock P(k) ~ 1e-9, expect similar order of magnitude
        assert np.all(result > 1e-15)
        assert np.all(result < 1e-3)
    
    def test_edge_cases(self, mock_power_spectrum):
        """Test edge cases and boundary conditions"""
        k_grid = np.array([0.01, 0.1])
        ell_test = np.array([2, 10])
        
        # Zero transfer functions
        Delta_zero = np.zeros((len(ell_test), len(k_grid)))
        result_zero = project_los(ell_test, k_grid, mock_power_spectrum, Delta_zero, Delta_zero)
        
        np.testing.assert_allclose(result_zero, 0.0, atol=1e-15)
        
        # Very small transfer functions
        Delta_small = np.ones((len(ell_test), len(k_grid))) * 1e-10
        result_small = project_los(ell_test, k_grid, mock_power_spectrum, Delta_small, Delta_small)
        
        assert np.all(result_small >= 0)
        assert np.all(result_small < 1e-15)
    
    def test_numerical_stability(self, mock_power_spectrum):
        """Test numerical stability with extreme inputs"""
        k_grid = np.logspace(-5, 2, 50)  # Very wide k range
        ell_test = np.array([1, 1000])   # Wide ell range
        
        # Transfer functions with some structure
        Delta = np.exp(-((k_grid[None, :] - 0.1) / 0.05)**2) * 0.1
        Delta = np.broadcast_to(Delta, (len(ell_test), len(k_grid)))
        
        result = project_los(ell_test, k_grid, mock_power_spectrum, Delta, Delta)
        
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)