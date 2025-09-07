"""
Unit tests for qfd_cmb.kernels module

Tests the sin2_mueller_coeffs and te_correlation_phase functions for
symmetry, physical consistency, and Mueller matrix properties.
"""

import pytest
import numpy as np
from qfd_cmb.kernels import sin2_mueller_coeffs, te_correlation_phase


class TestSin2MuellerCoeffs:
    """Test suite for sin2_mueller_coeffs function"""
    
    def test_basic_functionality(self):
        """Test basic function call with various mu values"""
        mu_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        w_T, w_E = sin2_mueller_coeffs(mu_values)
        
        # Check output shapes and types
        assert w_T.shape == mu_values.shape
        assert w_E.shape == mu_values.shape
        assert isinstance(w_T, np.ndarray)
        assert isinstance(w_E, np.ndarray)
    
    def test_scalar_input(self):
        """Test function with scalar input"""
        mu_scalar = 0.5
        w_T, w_E = sin2_mueller_coeffs(mu_scalar)
        
        assert np.isscalar(w_T)
        assert np.isscalar(w_E)
        assert np.isfinite(w_T)
        assert np.isfinite(w_E)
    
    def test_physical_range(self):
        """Test that weights are in physically reasonable ranges"""
        mu_test = np.linspace(-1, 1, 100)
        w_T, w_E = sin2_mueller_coeffs(mu_test)
        
        # Both weights should be non-negative
        assert np.all(w_T >= 0), "Intensity weights must be non-negative"
        assert np.all(w_E >= 0), "Polarization weights must be non-negative"
        
        # Weights should be bounded by reasonable values
        assert np.all(w_T <= 1.0), "Intensity weights should not exceed 1"
        assert np.all(w_E <= 1.0), "Polarization weights should not exceed 1"
    
    def test_symmetry_properties(self):
        """Test symmetry properties of Mueller coefficients"""
        mu_test = np.array([-0.8, -0.3, 0.0, 0.3, 0.8])
        w_T, w_E = sin2_mueller_coeffs(mu_test)
        
        # Both weights should be symmetric in mu (even functions)
        mu_neg = -mu_test
        w_T_neg, w_E_neg = sin2_mueller_coeffs(mu_neg)
        
        np.testing.assert_allclose(w_T, w_T_neg, rtol=1e-15)
        np.testing.assert_allclose(w_E, w_E_neg, rtol=1e-15)
    
    def test_boundary_values(self):
        """Test behavior at boundary values mu = ±1"""
        # At mu = ±1 (forward/backward scattering)
        w_T_forward, w_E_forward = sin2_mueller_coeffs(1.0)
        w_T_backward, w_E_backward = sin2_mueller_coeffs(-1.0)
        
        # sin²(θ) = 1 - cos²(θ) = 1 - mu²
        # At mu = ±1: sin²(θ) = 0
        np.testing.assert_allclose(w_T_forward, 0.0, atol=1e-15)
        np.testing.assert_allclose(w_T_backward, 0.0, atol=1e-15)
        np.testing.assert_allclose(w_E_forward, 0.0, atol=1e-15)
        np.testing.assert_allclose(w_E_backward, 0.0, atol=1e-15)
        
        # At mu = 0 (perpendicular scattering)
        w_T_perp, w_E_perp = sin2_mueller_coeffs(0.0)
        
        # sin²(θ) = 1 - 0² = 1 (maximum)
        np.testing.assert_allclose(w_T_perp, 1.0, rtol=1e-15)
        np.testing.assert_allclose(w_E_perp, 1.0, rtol=1e-15)
    
    def test_sin2_relationship(self):
        """Test that weights follow sin²(θ) = 1 - cos²(θ) = 1 - mu²"""
        mu_test = np.linspace(-1, 1, 50)
        w_T, w_E = sin2_mueller_coeffs(mu_test)
        
        expected_sin2 = 1.0 - mu_test**2
        
        # Both weights should equal sin²(θ) for this simple kernel
        np.testing.assert_allclose(w_T, expected_sin2, rtol=1e-15)
        np.testing.assert_allclose(w_E, expected_sin2, rtol=1e-15)
    
    def test_mueller_matrix_properties(self):
        """Test properties expected from Mueller matrix formalism"""
        mu_test = np.linspace(-0.99, 0.99, 100)  # Avoid exact ±1
        w_T, w_E = sin2_mueller_coeffs(mu_test)
        
        # For sin² kernel, intensity and polarization weights should be equal
        np.testing.assert_allclose(w_T, w_E, rtol=1e-15)
        
        # Check that weights are maximized at perpendicular scattering
        max_idx = np.argmax(w_T)
        mu_at_max = mu_test[max_idx]
        np.testing.assert_allclose(mu_at_max, 0.0, atol=0.02)  # Should be near mu=0
    
    def test_integration_properties(self):
        """Test integration properties relevant for scattering calculations"""
        mu_test = np.linspace(-1, 1, 1000)
        w_T, w_E = sin2_mueller_coeffs(mu_test)
        
        # Integrate weights over solid angle (factor of 2π from φ integration)
        # ∫ w(μ) dμ should give reasonable cross-section
        integral_T = np.trapz(w_T, mu_test)
        integral_E = np.trapz(w_E, mu_test)
        
        # For sin²(θ) kernel: ∫₋₁¹ (1-μ²) dμ = [μ - μ³/3]₋₁¹ = 4/3
        expected_integral = 4.0/3.0
        
        np.testing.assert_allclose(integral_T, expected_integral, rtol=1e-10)
        np.testing.assert_allclose(integral_E, expected_integral, rtol=1e-10)
    
    def test_array_broadcasting(self):
        """Test that function handles array broadcasting correctly"""
        # 1D array
        mu_1d = np.array([0.0, 0.5, 1.0])
        w_T_1d, w_E_1d = sin2_mueller_coeffs(mu_1d)
        assert w_T_1d.shape == (3,)
        
        # 2D array
        mu_2d = np.array([[0.0, 0.5], [1.0, -0.5]])
        w_T_2d, w_E_2d = sin2_mueller_coeffs(mu_2d)
        assert w_T_2d.shape == (2, 2)
        assert w_E_2d.shape == (2, 2)
    
    def test_edge_cases(self):
        """Test edge cases and numerical stability"""
        # Values slightly outside [-1, 1] (numerical errors)
        mu_edge = np.array([-1.0001, 1.0001])
        w_T, w_E = sin2_mueller_coeffs(mu_edge)
        
        # Should handle gracefully (might give small negative values)
        assert np.all(np.isfinite(w_T))
        assert np.all(np.isfinite(w_E))
        
        # Very close to boundaries
        mu_close = np.array([-0.9999999, 0.9999999])
        w_T_close, w_E_close = sin2_mueller_coeffs(mu_close)
        
        assert np.all(w_T_close >= 0)
        assert np.all(w_E_close >= 0)
        assert np.all(w_T_close < 1e-10)  # Should be very small
        assert np.all(w_E_close < 1e-10)


class TestTECorrelationPhase:
    """Test suite for te_correlation_phase function"""
    
    def test_basic_functionality(self, sample_k_values, sample_ell_values):
        """Test basic function call with default parameters"""
        rpsi = 147.0
        chi_star = 14065.0
        
        result = te_correlation_phase(sample_k_values, rpsi, sample_ell_values[0], chi_star)
        
        # Check output shape and type
        assert result.shape == sample_k_values.shape
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_correlation_bounds(self, sample_k_values):
        """Test that correlation coefficient is bounded by [-1, 1]"""
        rpsi = 147.0
        ell = 100
        chi_star = 14065.0
        
        result = te_correlation_phase(sample_k_values, rpsi, ell, chi_star)
        
        # Correlation coefficient should be bounded
        assert np.all(result >= -1.0), "Correlation coefficient must be >= -1"
        assert np.all(result <= 1.0), "Correlation coefficient must be <= 1"
    
    def test_oscillatory_behavior(self):
        """Test that function exhibits oscillatory behavior"""
        k_fine = np.linspace(0.001, 0.1, 1000)
        rpsi = 147.0
        ell = 200
        chi_star = 14065.0
        
        result = te_correlation_phase(k_fine, rpsi, ell, chi_star)
        
        # Should have oscillations - check for sign changes
        sign_changes = np.sum(np.diff(np.sign(result)) != 0)
        assert sign_changes > 5, "Should have multiple oscillations"
        
        # Should span reasonable range
        assert np.max(result) > 0.1, "Should have positive correlations"
        assert np.min(result) < -0.1, "Should have negative correlations"
    
    def test_rpsi_scaling(self):
        """Test that rpsi parameter controls oscillation frequency"""
        k_test = np.linspace(0.01, 0.1, 500)
        ell = 200
        chi_star = 14065.0
        
        # Different rpsi values
        rpsi_small = 50.0
        rpsi_large = 300.0
        
        result_small = te_correlation_phase(k_test, rpsi_small, ell, chi_star)
        result_large = te_correlation_phase(k_test, rpsi_large, ell, chi_star)
        
        # Count oscillations (zero crossings)
        zeros_small = np.sum(np.diff(np.sign(result_small)) != 0)
        zeros_large = np.sum(np.diff(np.sign(result_large)) != 0)
        
        # Larger rpsi should give more oscillations
        assert zeros_large > zeros_small, "Larger rpsi should give more oscillations"
    
    def test_ell_dependence(self):
        """Test dependence on ell parameter"""
        k_test = np.array([0.01, 0.05, 0.1])
        rpsi = 147.0
        chi_star = 14065.0
        
        ell_values = [10, 100, 1000]
        results = []
        
        for ell in ell_values:
            result = te_correlation_phase(k_test, rpsi, ell, chi_star)
            results.append(result)
            
            # Should be bounded
            assert np.all(np.abs(result) <= 1.0)
        
        # Results should be different for different ell
        assert not np.allclose(results[0], results[1], rtol=0.1)
        assert not np.allclose(results[1], results[2], rtol=0.1)
    
    def test_damping_behavior(self):
        """Test exponential damping with ell"""
        k_test = 0.05
        rpsi = 147.0
        chi_star = 14065.0
        sigma_phase = 0.16
        
        ell_values = np.array([10, 50, 100, 200, 500, 1000])
        results = []
        
        for ell in ell_values:
            result = te_correlation_phase(k_test, rpsi, ell, chi_star, sigma_phase)
            results.append(np.abs(result))  # Take absolute value to see damping
        
        results = np.array(results)
        
        # Should show general decreasing trend (damping)
        # Check that high-ell values are smaller than low-ell values
        assert results[-1] < results[0], "High-ell should be more damped"
        
        # Check exponential-like decay
        log_results = np.log(results + 1e-10)
        ell_scaled = ell_values / 200.0  # Scale factor from function
        
        # Should roughly follow exp(-sigma_phase² * (ell/200)²)
        expected_log = -sigma_phase**2 * ell_scaled**2
        
        # This is approximate due to cosine modulation
        correlation = np.corrcoef(log_results, expected_log)[0, 1]
        assert correlation < -0.3, "Should show negative correlation (decay)"
    
    def test_phase_offset(self):
        """Test phi0 phase offset parameter"""
        k_test = np.array([0.01, 0.05, 0.1])
        rpsi = 147.0
        ell = 200
        chi_star = 14065.0
        
        phi0_values = [0.0, np.pi/4, np.pi/2, np.pi]
        results = []
        
        for phi0 in phi0_values:
            result = te_correlation_phase(k_test, rpsi, ell, chi_star, phi0=phi0)
            results.append(result)
        
        # Different phase offsets should give different results
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                assert not np.allclose(results[i], results[j], rtol=0.1)
        
        # π phase shift should approximately flip sign
        result_0 = te_correlation_phase(k_test, rpsi, ell, chi_star, phi0=0.0)
        result_pi = te_correlation_phase(k_test, rpsi, ell, chi_star, phi0=np.pi)
        
        # Should be approximately opposite (within damping effects)
        np.testing.assert_allclose(result_0, -result_pi, rtol=0.2)
    
    def test_chi_star_dependence(self):
        """Test dependence on chi_star parameter"""
        k_test = np.array([0.01, 0.05, 0.1])
        rpsi = 147.0
        ell = 200
        
        chi_star_values = [10000.0, 14065.0, 18000.0]
        results = []
        
        for chi_star in chi_star_values:
            result = te_correlation_phase(k_test, rpsi, ell, chi_star)
            results.append(result)
        
        # Different chi_star should give different effective k values
        # and thus different oscillation phases
        assert not np.allclose(results[0], results[1], rtol=0.1)
        assert not np.allclose(results[1], results[2], rtol=0.1)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        rpsi = 147.0
        ell = 200
        
        # Zero chi_star (should fall back to k dependence)
        result_zero_chi = te_correlation_phase(0.1, rpsi, ell, 0.0)
        assert np.isfinite(result_zero_chi)
        assert np.abs(result_zero_chi) <= 1.0
        
        # Very small chi_star
        result_small_chi = te_correlation_phase(0.1, rpsi, ell, 1e-6)
        assert np.isfinite(result_small_chi)
        
        # Very large chi_star
        result_large_chi = te_correlation_phase(0.1, rpsi, ell, 1e6)
        assert np.isfinite(result_large_chi)
        
        # Zero k
        result_zero_k = te_correlation_phase(0.0, rpsi, ell, 14065.0)
        assert np.isfinite(result_zero_k)
        assert np.abs(result_zero_k) <= 1.0
    
    def test_array_inputs(self):
        """Test with array inputs for different parameters"""
        # Array k, scalar others
        k_array = np.array([0.01, 0.05, 0.1])
        result = te_correlation_phase(k_array, 147.0, 200, 14065.0)
        assert result.shape == k_array.shape
        
        # Scalar k, array ell
        ell_array = np.array([100, 200, 500])
        result = te_correlation_phase(0.05, 147.0, ell_array, 14065.0)
        assert result.shape == ell_array.shape
        
        # Array k and ell (should broadcast)
        k_2d = k_array[:, None]
        ell_2d = ell_array[None, :]
        result_2d = te_correlation_phase(k_2d, 147.0, ell_2d, 14065.0)
        assert result_2d.shape == (3, 3)
    
    def test_planck_parameters(self, planck_parameters):
        """Test with Planck-like parameter values"""
        k_test = np.logspace(-3, 0, 50)
        ell_test = 200
        
        result = te_correlation_phase(
            k_test, 
            planck_parameters['rpsi'], 
            ell_test, 
            planck_parameters['chi_star']
        )
        
        # Check basic properties
        assert np.all(np.abs(result) <= 1.0)
        assert np.all(np.isfinite(result))
        
        # Should have oscillations
        sign_changes = np.sum(np.diff(np.sign(result)) != 0)
        assert sign_changes > 3, "Should have oscillatory behavior"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme parameters"""
        # Very large rpsi
        result = te_correlation_phase(0.1, 1e6, 200, 14065.0)
        assert np.isfinite(result)
        assert np.abs(result) <= 1.0
        
        # Very small sigma_phase
        result = te_correlation_phase(0.1, 147.0, 200, 14065.0, sigma_phase=1e-10)
        assert np.isfinite(result)
        
        # Very large sigma_phase
        result = te_correlation_phase(0.1, 147.0, 200, 14065.0, sigma_phase=1e3)
        assert np.isfinite(result)
        assert np.abs(result) < 1e-10  # Should be heavily damped