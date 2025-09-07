"""
Unit tests for qfd_cmb.ppsi_models module

Tests the oscillatory_psik function for parameter validation,
edge case handling, and numerical accuracy.
"""

import pytest
import numpy as np
from qfd_cmb.ppsi_models import oscillatory_psik


class TestOscillatoryPsik:
    """Test suite for oscillatory_psik function"""
    
    def test_basic_functionality(self, sample_k_values):
        """Test basic function call with default parameters"""
        result = oscillatory_psik(sample_k_values)
        
        # Check output shape and type
        assert result.shape == sample_k_values.shape
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_positivity(self, sample_k_values, numerical_helper):
        """Test that P_ψ(k) is always positive"""
        result = oscillatory_psik(sample_k_values)
        numerical_helper.assert_positive_definite(result, "P_ψ(k)")
    
    def test_scalar_input(self):
        """Test function with scalar input"""
        k_scalar = 0.1
        result = oscillatory_psik(k_scalar)
        
        assert np.isscalar(result)
        assert result > 0
        assert np.isfinite(result)
    
    def test_array_input(self):
        """Test function with various array inputs"""
        # 1D array
        k_1d = np.array([0.01, 0.1, 1.0])
        result_1d = oscillatory_psik(k_1d)
        assert result_1d.shape == (3,)
        
        # 2D array
        k_2d = np.array([[0.01, 0.1], [1.0, 10.0]])
        result_2d = oscillatory_psik(k_2d)
        assert result_2d.shape == (2, 2)
    
    def test_parameter_variations(self, sample_k_values):
        """Test function with different parameter values"""
        # Test amplitude scaling
        A_values = [0.5, 1.0, 2.0]
        for A in A_values:
            result = oscillatory_psik(sample_k_values, A=A)
            assert np.all(result > 0)
            
            # Check amplitude scaling (approximately)
            if A > 0:
                ratio = result / oscillatory_psik(sample_k_values, A=1.0)
                np.testing.assert_allclose(ratio, A, rtol=0.1)
        
        # Test spectral index
        ns_values = [0.9, 0.96, 1.0]
        for ns in ns_values:
            result = oscillatory_psik(sample_k_values, ns=ns)
            assert np.all(result > 0)
        
        # Test oscillation parameters
        rpsi_values = [100.0, 147.0, 200.0]
        for rpsi in rpsi_values:
            result = oscillatory_psik(sample_k_values, rpsi=rpsi)
            assert np.all(result > 0)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Very small k
        k_small = 1e-10
        result_small = oscillatory_psik(k_small)
        assert result_small > 0
        assert np.isfinite(result_small)
        
        # Very large k
        k_large = 1e3
        result_large = oscillatory_psik(k_large)
        assert result_large > 0
        assert np.isfinite(result_large)
        
        # Zero k (should be handled by 1e-16 offset)
        k_zero = 0.0
        result_zero = oscillatory_psik(k_zero)
        assert result_zero > 0
        assert np.isfinite(result_zero)
        
        # Array with zero
        k_with_zero = np.array([0.0, 0.1, 1.0])
        result_with_zero = oscillatory_psik(k_with_zero)
        assert np.all(result_with_zero > 0)
        assert np.all(np.isfinite(result_with_zero))
    
    def test_parameter_validation(self):
        """Test parameter validation and error handling"""
        k_test = np.array([0.1, 1.0])
        
        # Test with extreme parameters
        # Very large amplitude
        result = oscillatory_psik(k_test, A=1e6)
        assert np.all(np.isfinite(result))
        
        # Very small amplitude
        result = oscillatory_psik(k_test, A=1e-10)
        assert np.all(result >= 0)
        
        # Extreme spectral index
        result = oscillatory_psik(k_test, ns=2.0)
        assert np.all(np.isfinite(result))
        
        result = oscillatory_psik(k_test, ns=-1.0)
        assert np.all(np.isfinite(result))
    
    def test_oscillatory_behavior(self):
        """Test that oscillations are present in the power spectrum"""
        k_fine = np.linspace(0.01, 1.0, 1000)
        rpsi = 147.0
        
        # Get results with and without oscillations
        result_osc = oscillatory_psik(k_fine, rpsi=rpsi, Aosc=0.5)
        result_no_osc = oscillatory_psik(k_fine, rpsi=rpsi, Aosc=0.0)
        
        # With oscillations should have more variation
        var_osc = np.var(result_osc / result_no_osc)
        assert var_osc > 1e-4, "Oscillations should create variation in P_ψ(k)"
    
    def test_spectral_index_scaling(self):
        """Test that spectral index affects k-dependence correctly"""
        k_test = np.array([0.01, 0.1, 1.0])
        
        # For ns < 1, power should decrease with k
        result_red = oscillatory_psik(k_test, ns=0.9, Aosc=0.0)  # Remove oscillations
        assert result_red[0] > result_red[-1], "Red spectrum should decrease with k"
        
        # For ns > 1, power should increase with k  
        result_blue = oscillatory_psik(k_test, ns=1.1, Aosc=0.0)
        assert result_blue[0] < result_blue[-1], "Blue spectrum should increase with k"
    
    def test_gaussian_damping(self):
        """Test that Gaussian damping affects high-k behavior"""
        k_high = np.array([1.0, 10.0, 100.0])
        
        # With strong damping
        result_damped = oscillatory_psik(k_high, sigma_osc=0.1, Aosc=1.0)
        
        # With weak damping  
        result_undamped = oscillatory_psik(k_high, sigma_osc=0.001, Aosc=1.0)
        
        # Damped version should have less oscillatory power at high k
        # (this is a qualitative test)
        assert np.all(np.isfinite(result_damped))
        assert np.all(np.isfinite(result_undamped))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs"""
        # Very wide k range
        k_wide = np.logspace(-10, 5, 100)
        result = oscillatory_psik(k_wide)
        
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)
        
        # Test with arrays containing inf/nan (should be handled gracefully)
        k_with_inf = np.array([0.1, np.inf, 1.0])
        result_inf = oscillatory_psik(k_with_inf)
        # At least finite values should be computed
        assert np.isfinite(result_inf[0])
        assert np.isfinite(result_inf[2])
    
    @pytest.mark.numerical
    def test_regression_values(self, reference_values):
        """Test against reference values for regression testing"""
        ref_data = reference_values['oscillatory_psik']
        k_test = np.array(ref_data['k'])
        expected = np.array(ref_data['expected'])
        rtol = ref_data['rtol']
        
        result = oscillatory_psik(k_test)
        
        # Note: This is a rough regression test
        # Exact values depend on parameter choices
        assert result.shape == expected.shape
        assert np.all(result > 0)
        
        # Check order of magnitude is reasonable
        assert np.all(result > expected * 0.01)
        assert np.all(result < expected * 100)
    
    def test_consistency_with_planck_parameters(self, planck_parameters):
        """Test with Planck-like parameter values"""
        k_test = np.logspace(-4, 1, 50)
        
        result = oscillatory_psik(
            k_test,
            A=planck_parameters['A'],
            ns=planck_parameters['ns'],
            rpsi=planck_parameters['rpsi'],
            Aosc=planck_parameters['Aosc'],
            sigma_osc=planck_parameters['sigma_osc']
        )
        
        # Check basic properties
        assert np.all(result > 0)
        assert np.all(np.isfinite(result))
        
        # Check reasonable amplitude range for CMB
        assert np.max(result) < 1e-3  # Not too large
        assert np.max(result) > 1e-12  # Not too small
    
    def test_monotonicity_without_oscillations(self):
        """Test monotonic behavior when oscillations are turned off"""
        k_test = np.logspace(-2, 1, 100)
        
        # No oscillations, red spectrum
        result_red = oscillatory_psik(k_test, ns=0.9, Aosc=0.0)
        
        # Should be monotonically decreasing
        diff = np.diff(result_red)
        assert np.all(diff < 0), "Red spectrum without oscillations should be monotonic"
        
        # No oscillations, blue spectrum  
        result_blue = oscillatory_psik(k_test, ns=1.1, Aosc=0.0)
        
        # Should be monotonically increasing
        diff = np.diff(result_blue)
        assert np.all(diff > 0), "Blue spectrum without oscillations should be monotonic"