"""
Scientific validation test suite for QFD CMB computations

This module contains tests that validate the scientific accuracy and
physical consistency of the QFD CMB computations, including reference
value tests and physical property checks.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import integrate, interpolate

# Import QFD CMB modules
from qfd_cmb.ppsi_models import oscillatory_psik
from qfd_cmb.visibility import gaussian_window_chi
from qfd_cmb.kernels import sin2_mueller_coeffs, te_correlation_phase
from qfd_cmb.projector import project_limber


class TestPlanckAnchoredValidation:
    """Test computations against known Planck-anchored parameters"""
    
    def test_planck_parameter_consistency(self, planck_parameters):
        """Test that Planck-anchored parameters produce consistent results"""
        lA = planck_parameters['lA']
        rpsi = planck_parameters['rpsi']
        chi_star = planck_parameters['chi_star']
        
        # Verify the fundamental relationship: chi_star = lA * rpsi / π
        expected_chi_star = lA * rpsi / np.pi
        np.testing.assert_allclose(chi_star, expected_chi_star, rtol=1e-10,
                                 err_msg="Planck parameter relationship violated")
    
    def test_power_spectrum_planck_values(self, planck_parameters):
        """Test power spectrum with Planck-anchored parameters"""
        rpsi = planck_parameters['rpsi']
        
        # Test at characteristic scales
        k_values = np.array([0.01, 0.05, 0.1, 0.2])  # Mpc^-1
        
        # Compute power spectrum with Planck parameters
        Pk_values = oscillatory_psik(k_values, ns=0.96, rpsi=rpsi, 
                                   Aosc=0.55, sigma_osc=0.025)
        
        # Validate physical properties
        assert np.all(Pk_values > 0), "Power spectrum must be positive"
        assert np.all(np.isfinite(Pk_values)), "Power spectrum must be finite"
        
        # Check approximate scaling behavior
        # For small k, should follow k^(ns-1) ≈ k^(-0.04)
        k_small = k_values[k_values < 0.1]
        Pk_small = Pk_values[k_values < 0.1]
        
        if len(k_small) > 1:
            # Fit power law to small k values
            log_k = np.log(k_small)
            log_Pk = np.log(Pk_small)
            slope = np.polyfit(log_k, log_Pk, 1)[0]
            
            # Should be close to ns - 1 = -0.04
            expected_slope = 0.96 - 1.0
            np.testing.assert_allclose(slope, expected_slope, atol=0.2,
                                     err_msg=f"Power spectrum slope {slope} inconsistent with ns")
    
    def test_oscillation_scale_consistency(self, planck_parameters):
        """Test that oscillation scale rpsi produces expected features"""
        rpsi = planck_parameters['rpsi']
        
        # Test oscillation period in k-space
        k_range = np.linspace(0.01, 0.5, 1000)
        Pk_values = oscillatory_psik(k_range, rpsi=rpsi, Aosc=0.55, sigma_osc=0.025)
        
        # Find oscillation peaks and troughs
        # Remove overall trend by dividing by smooth component
        Pk_smooth = oscillatory_psik(k_range, rpsi=rpsi, Aosc=0.0, sigma_osc=0.025)
        oscillation_ratio = Pk_values / Pk_smooth
        
        # The oscillation should have period ≈ 2π/rpsi in k
        expected_period = 2 * np.pi / rpsi
        
        # Find peaks in oscillation
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(oscillation_ratio, height=1.1, distance=10)
        
        if len(peaks) > 1:
            # Measure actual periods
            k_peaks = k_range[peaks]
            periods = np.diff(k_peaks)
            mean_period = np.mean(periods)
            
            # Should be close to expected period
            np.testing.assert_allclose(mean_period, expected_period, rtol=0.3,
                                     err_msg=f"Oscillation period {mean_period} inconsistent with rpsi")


class TestSpectrumRelationships:
    """Test relationships between TT, TE, and EE spectra"""
    
    def test_tt_ee_relationship(self, planck_parameters):
        """Test the relationship between TT and EE spectra"""
        # Set up computation
        chi_star = planck_parameters['chi_star']
        sigma_chi = planck_parameters['sigma_chi']
        rpsi = planck_parameters['rpsi']
        
        ells = np.arange(2, 101)
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 201)
        Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Compute TT spectrum
        Pk = lambda k: oscillatory_psik(k, ns=0.96, rpsi=rpsi, Aosc=0.55, sigma_osc=0.025)
        Ctt = project_limber(ells, Pk, Wchi, chi_grid)
        
        # Compute EE spectrum (simplified model: EE = 0.25 * TT)
        Cee = 0.25 * Ctt
        
        # Test relationship
        ratio = Cee / Ctt
        expected_ratio = 0.25
        
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-10,
                                 err_msg="EE/TT ratio inconsistent with model")
        
        # Both should be positive
        assert np.all(Ctt > 0), "TT spectrum must be positive"
        assert np.all(Cee > 0), "EE spectrum must be positive"
    
    def test_te_correlation_bounds(self, planck_parameters):
        """Test that TE correlation coefficient is properly bounded"""
        chi_star = planck_parameters['chi_star']
        rpsi = planck_parameters['rpsi']
        
        ells = np.arange(2, 501)
        
        # Compute correlation coefficients
        rho_values = np.array([
            te_correlation_phase((l+0.5)/chi_star, rpsi, l, chi_star) 
            for l in ells
        ])
        
        # Test bounds: |ρ| ≤ 1
        assert np.all(np.abs(rho_values) <= 1.0), "TE correlation must satisfy |ρ| ≤ 1"
        
        # Test that correlation varies with ell (not constant)
        assert np.std(rho_values) > 0.1, "TE correlation should vary significantly with ell"
        
        # Test oscillatory behavior
        # Should have both positive and negative values
        assert np.any(rho_values > 0), "TE correlation should have positive values"
        assert np.any(rho_values < 0), "TE correlation should have negative values"
    
    def test_te_spectrum_consistency(self, planck_parameters):
        """Test TE spectrum consistency with TT and EE"""
        # Set up computation
        chi_star = planck_parameters['chi_star']
        sigma_chi = planck_parameters['sigma_chi']
        rpsi = planck_parameters['rpsi']
        
        ells = np.arange(2, 101)
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 201)
        Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Compute spectra
        Pk = lambda k: oscillatory_psik(k, ns=0.96, rpsi=rpsi, Aosc=0.55, sigma_osc=0.025)
        Ctt = project_limber(ells, Pk, Wchi, chi_grid)
        Cee = 0.25 * Ctt
        
        # Compute TE with correlation
        rho = np.array([te_correlation_phase((l+0.5)/chi_star, rpsi, l, chi_star) for l in ells])
        Cte = rho * np.sqrt(Ctt * Cee)
        
        # Test Cauchy-Schwarz inequality: |C_TE| ≤ √(C_TT * C_EE)
        max_te = np.sqrt(Ctt * Cee)
        assert np.all(np.abs(Cte) <= max_te * (1 + 1e-10)), "TE spectrum violates Cauchy-Schwarz"
        
        # Test that TE can be both positive and negative
        if np.any(rho > 0.1) and np.any(rho < -0.1):
            assert np.any(Cte > 0), "TE spectrum should have positive values"
            assert np.any(Cte < 0), "TE spectrum should have negative values"


class TestPhysicalConsistency:
    """Test physical consistency of computed spectra"""
    
    def test_positive_definite_spectra(self, planck_parameters):
        """Test that TT and EE spectra are positive definite"""
        # Set up computation
        chi_star = planck_parameters['chi_star']
        sigma_chi = planck_parameters['sigma_chi']
        rpsi = planck_parameters['rpsi']
        
        ells = np.arange(2, 201)
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 201)
        Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Test multiple parameter variations
        parameter_sets = [
            {"ns": 0.95, "Aosc": 0.3},
            {"ns": 0.97, "Aosc": 0.7},
            {"ns": 0.96, "Aosc": 0.0},  # No oscillations
            {"ns": 0.96, "Aosc": 0.9}   # Strong oscillations
        ]
        
        for params in parameter_sets:
            Pk = lambda k: oscillatory_psik(k, ns=params["ns"], rpsi=rpsi, 
                                          Aosc=params["Aosc"], sigma_osc=0.025)
            Ctt = project_limber(ells, Pk, Wchi, chi_grid)
            Cee = 0.25 * Ctt
            
            # Test positivity
            assert np.all(Ctt > 0), f"TT spectrum not positive for params {params}"
            assert np.all(Cee > 0), f"EE spectrum not positive for params {params}"
            
            # Test finite values
            assert np.all(np.isfinite(Ctt)), f"TT spectrum not finite for params {params}"
            assert np.all(np.isfinite(Cee)), f"EE spectrum not finite for params {params}"
    
    def test_spectrum_amplitude_scaling(self, planck_parameters):
        """Test that spectrum amplitudes are in reasonable ranges"""
        # Set up computation
        chi_star = planck_parameters['chi_star']
        sigma_chi = planck_parameters['sigma_chi']
        rpsi = planck_parameters['rpsi']
        
        ells = np.arange(2, 1001)
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 201)
        Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        Pk = lambda k: oscillatory_psik(k, ns=0.96, rpsi=rpsi, Aosc=0.55, sigma_osc=0.025)
        Ctt = project_limber(ells, Pk, Wchi, chi_grid)
        
        # Convert to dimensionless form: ℓ(ℓ+1)C_ℓ/(2π)
        ell_Ctt = ells * (ells + 1) * Ctt / (2 * np.pi)
        
        # Test reasonable amplitude ranges (typical CMB values)
        assert np.max(ell_Ctt) < 1e4, "TT spectrum amplitude too large"
        assert np.max(ell_Ctt) > 1e-6, "TT spectrum amplitude too small"
        
        # Test that spectrum decreases at high ell (roughly)
        high_ell_idx = ells > 500
        low_ell_idx = (ells > 50) & (ells < 200)
        
        if np.any(high_ell_idx) and np.any(low_ell_idx):
            high_ell_mean = np.mean(ell_Ctt[high_ell_idx])
            low_ell_mean = np.mean(ell_Ctt[low_ell_idx])
            
            # High ell should generally be smaller (allowing for oscillations)
            assert high_ell_mean < 10 * low_ell_mean, "Spectrum doesn't decrease at high ell"
    
    def test_window_function_properties(self, planck_parameters):
        """Test physical properties of the window function"""
        chi_star = planck_parameters['chi_star']
        sigma_chi = planck_parameters['sigma_chi']
        
        # Create window function
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 501)
        Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Test positivity
        assert np.all(Wchi >= 0), "Window function must be non-negative"
        
        # Test normalization (approximately)
        norm_squared = np.trapz(Wchi**2, chi_grid)
        np.testing.assert_allclose(norm_squared, 1.0, rtol=1e-3,
                                 err_msg="Window function not properly normalized")
        
        # Test peak location
        peak_idx = np.argmax(Wchi)
        peak_chi = chi_grid[peak_idx]
        np.testing.assert_allclose(peak_chi, chi_star, atol=sigma_chi/10,
                                 err_msg="Window function peak not at chi_star")
        
        # Test width (FWHM should be related to sigma_chi)
        half_max = np.max(Wchi) / 2
        above_half = Wchi > half_max
        if np.any(above_half):
            chi_half = chi_grid[above_half]
            fwhm = chi_half[-1] - chi_half[0]
            expected_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma_chi
            np.testing.assert_allclose(fwhm, expected_fwhm, rtol=0.2,
                                     err_msg="Window function width inconsistent")
    
    def test_mueller_coefficients_bounds(self):
        """Test Mueller coefficient bounds and properties"""
        # Test range of mu values
        mu_values = np.linspace(-1, 1, 101)
        
        for mu in mu_values:
            w_T, w_E = sin2_mueller_coeffs(mu)
            
            # Test bounds
            assert 0 <= w_T <= 1, f"Intensity weight out of bounds at mu={mu}"
            assert 0 <= w_E <= 1, f"Polarization weight out of bounds at mu={mu}"
            
            # Test relationship to sin²θ = 1 - μ²
            expected_w = 1 - mu**2
            np.testing.assert_allclose(w_T, expected_w, rtol=1e-10,
                                     err_msg=f"Intensity weight incorrect at mu={mu}")
            np.testing.assert_allclose(w_E, expected_w, rtol=1e-10,
                                     err_msg=f"Polarization weight incorrect at mu={mu}")
        
        # Test symmetry
        mu_test = 0.5
        w_T_pos, w_E_pos = sin2_mueller_coeffs(mu_test)
        w_T_neg, w_E_neg = sin2_mueller_coeffs(-mu_test)
        
        np.testing.assert_allclose(w_T_pos, w_T_neg, rtol=1e-10,
                                 err_msg="Mueller coefficients not symmetric in mu")
        np.testing.assert_allclose(w_E_pos, w_E_neg, rtol=1e-10,
                                 err_msg="Mueller coefficients not symmetric in mu")


class TestNumericalStability:
    """Test numerical stability and edge cases"""
    
    def test_power_spectrum_edge_cases(self):
        """Test power spectrum behavior at edge cases"""
        # Test very small k
        k_small = np.array([1e-10, 1e-8, 1e-6])
        Pk_small = oscillatory_psik(k_small)
        
        assert np.all(np.isfinite(Pk_small)), "Power spectrum not finite at small k"
        assert np.all(Pk_small > 0), "Power spectrum not positive at small k"
        
        # Test very large k
        k_large = np.array([10, 100, 1000])
        Pk_large = oscillatory_psik(k_large)
        
        assert np.all(np.isfinite(Pk_large)), "Power spectrum not finite at large k"
        assert np.all(Pk_large > 0), "Power spectrum not positive at large k"
        
        # Test zero amplitude
        Pk_zero = oscillatory_psik(0.1, A=0.0)
        assert Pk_zero == 0.0, "Power spectrum not zero when A=0"
        
        # Test zero oscillation amplitude
        k_test = np.logspace(-2, 1, 50)
        Pk_no_osc = oscillatory_psik(k_test, Aosc=0.0)
        
        # Should be smooth power law
        assert np.all(np.isfinite(Pk_no_osc)), "Power spectrum not finite with Aosc=0"
        assert np.all(Pk_no_osc > 0), "Power spectrum not positive with Aosc=0"
        
        # Check smoothness (no oscillations)
        log_k = np.log(k_test)
        log_Pk = np.log(Pk_no_osc)
        
        # Should be approximately linear in log-log space
        slope, intercept = np.polyfit(log_k, log_Pk, 1)
        fit_log_Pk = slope * log_k + intercept
        residuals = log_Pk - fit_log_Pk
        
        assert np.std(residuals) < 0.1, "Power spectrum not smooth with Aosc=0"
    
    def test_projection_convergence(self, planck_parameters):
        """Test convergence of Limber projection with grid resolution"""
        chi_star = planck_parameters['chi_star']
        sigma_chi = planck_parameters['sigma_chi']
        rpsi = planck_parameters['rpsi']
        
        ells = np.array([10, 50, 100])  # Test a few ell values
        
        # Test different grid resolutions
        grid_sizes = [101, 201, 501]
        results = []
        
        for grid_size in grid_sizes:
            chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, grid_size)
            Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
            
            Pk = lambda k: oscillatory_psik(k, ns=0.96, rpsi=rpsi, Aosc=0.55, sigma_osc=0.025)
            Ctt = project_limber(ells, Pk, Wchi, chi_grid)
            
            results.append(Ctt)
        
        # Test convergence: higher resolution should give similar results
        Ctt_coarse, Ctt_medium, Ctt_fine = results
        
        # Medium vs coarse
        np.testing.assert_allclose(Ctt_medium, Ctt_coarse, rtol=0.1,
                                 err_msg="Projection not converged between coarse and medium grid")
        
        # Fine vs medium (should be better converged)
        np.testing.assert_allclose(Ctt_fine, Ctt_medium, rtol=0.05,
                                 err_msg="Projection not converged between medium and fine grid")
    
    def test_extreme_parameter_values(self):
        """Test behavior with extreme but valid parameter values"""
        k_test = np.logspace(-2, 1, 20)
        
        # Test extreme spectral indices
        extreme_params = [
            {"ns": 0.9},   # Very red
            {"ns": 1.05},  # Very blue
            {"rpsi": 50.0},   # Small oscillation scale
            {"rpsi": 500.0},  # Large oscillation scale
            {"Aosc": 0.99},   # Very strong oscillations
            {"sigma_osc": 0.001},  # Very narrow oscillations
            {"sigma_osc": 0.1}     # Very broad oscillations
        ]
        
        for params in extreme_params:
            Pk_values = oscillatory_psik(k_test, **params)
            
            assert np.all(np.isfinite(Pk_values)), f"Non-finite values with params {params}"
            assert np.all(Pk_values > 0), f"Non-positive values with params {params}"
            
            # Check that oscillations are reasonable
            if "Aosc" in params and params["Aosc"] > 0.5:
                # Should have significant variation
                assert np.std(Pk_values) / np.mean(Pk_values) > 0.1, \
                    f"Insufficient oscillation with params {params}"


# Scientific validation test markers
pytestmark = [pytest.mark.unit, pytest.mark.numerical]