#!/usr/bin/env python3
"""
Unit Tests for Enhanced QVD Cosmology
=====================================

Comprehensive tests for cosmological calculations with numerical stability.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redshift_cosmology import EnhancedQVDCosmology


class TestEnhancedQVDCosmology:
    """Test enhanced QVD cosmology calculations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cosmology = EnhancedQVDCosmology(
            hubble_constant=70.0,
            enable_bounds_checking=True
        )
    
    def test_initialization(self):
        """Test cosmology module initialization"""
        assert self.cosmology.H0 == 70.0
        assert self.cosmology.omega_m == 1.0  # Matter-dominated
        assert self.cosmology.omega_lambda == 0.0  # No dark energy
        assert hasattr(self.cosmology, 'bounds_enforcer')
    
    def test_hubble_distance(self):
        """Test Hubble distance calculation"""
        d_H = self.cosmology.hubble_distance()
        
        assert np.isfinite(d_H)
        assert d_H > 0.0
        
        # Should be approximately c/H0
        expected = 299792.458 / 70.0
        assert abs(d_H - expected) < 100.0  # Within 100 Mpc
    
    def test_comoving_distance_scalar(self):
        """Test comoving distance for scalar redshift"""
        # Test low redshift (linear regime)
        z_low = 0.05
        d_low = self.cosmology.comoving_distance(z_low)
        
        assert np.isfinite(d_low)
        assert d_low > 0.0
        
        # Test higher redshift (nonlinear regime)
        z_high = 0.5
        d_high = self.cosmology.comoving_distance(z_high)
        
        assert np.isfinite(d_high)
        assert d_high > d_low  # Should increase with redshift
    
    def test_comoving_distance_array(self):
        """Test comoving distance for array redshift"""
        redshifts = np.array([0.1, 0.3, 0.5, 0.7])
        distances = self.cosmology.comoving_distance(redshifts)
        
        assert len(distances) == len(redshifts)
        assert np.all(np.isfinite(distances))
        assert np.all(distances > 0.0)
        
        # Should be monotonically increasing
        assert np.all(np.diff(distances) > 0)
    
    def test_luminosity_distance(self):
        """Test luminosity distance calculation"""
        redshift = 0.5
        d_lum = self.cosmology.luminosity_distance(redshift)
        d_com = self.cosmology.comoving_distance(redshift)
        
        assert np.isfinite(d_lum)
        assert d_lum > 0.0
        
        # Should satisfy d_L = d_C * (1 + z)
        expected = d_com * (1 + redshift)
        assert abs(d_lum - expected) < 1.0  # Within 1 Mpc
    
    def test_angular_diameter_distance(self):
        """Test angular diameter distance calculation"""
        redshift = 0.5
        d_ang = self.cosmology.angular_diameter_distance(redshift)
        d_com = self.cosmology.comoving_distance(redshift)
        
        assert np.isfinite(d_ang)
        assert d_ang > 0.0
        
        # Should satisfy d_A = d_C / (1 + z)
        expected = d_com / (1 + redshift)
        assert abs(d_ang - expected) < 1.0  # Within 1 Mpc
    
    def test_distance_modulus(self):
        """Test distance modulus calculation"""
        redshift = 0.5
        mu = self.cosmology.distance_modulus(redshift)
        d_lum = self.cosmology.luminosity_distance(redshift)
        
        assert np.isfinite(mu)
        assert mu > 0.0
        
        # Should satisfy μ = 5 * log10(d_L * 10^6 / 10)
        expected = 5 * np.log10(d_lum * 1e6 / 10)
        assert abs(mu - expected) < 0.1  # Within 0.1 mag
    
    def test_lambda_cdm_distance(self):
        """Test ΛCDM distance calculation for comparison"""
        redshift = 0.5
        d_lambda_cdm = self.cosmology.lambda_cdm_distance(
            redshift, omega_m=0.3, omega_lambda=0.7
        )
        
        assert np.isfinite(d_lambda_cdm)
        assert d_lambda_cdm > 0.0
        
        # Should be different from matter-dominated distance
        d_matter = self.cosmology.luminosity_distance(redshift)
        assert abs(d_lambda_cdm - d_matter) > 10.0  # Significant difference
    
    def test_lookback_time(self):
        """Test lookback time calculation"""
        redshift = 0.5
        t_lookback = self.cosmology.lookback_time(redshift)
        
        assert np.isfinite(t_lookback)
        assert t_lookback > 0.0
        assert t_lookback < 20.0  # Less than age of universe
    
    def test_age_of_universe(self):
        """Test age of universe calculation"""
        # Present age (z=0)
        age_present = self.cosmology.age_of_universe(0)
        
        assert np.isfinite(age_present)
        assert 10.0 < age_present < 20.0  # Reasonable range in Gyr
        
        # Age at higher redshift
        age_z1 = self.cosmology.age_of_universe(1.0)
        
        assert np.isfinite(age_z1)
        assert age_z1 < age_present  # Should be younger
    
    def test_critical_density(self):
        """Test critical density calculation"""
        rho_c = self.cosmology.critical_density(0)
        
        assert np.isfinite(rho_c)
        assert rho_c > 0.0
        
        # Should be in reasonable range for critical density
        assert 1e-35 < rho_c < 1e-25  # g/cm³
    
    def test_extreme_redshift_handling(self):
        """Test handling of extreme redshift values"""
        extreme_redshifts = [-1.0, 0.0, 1e-10, 100.0, np.inf, np.nan]
        
        for z in extreme_redshifts:
            # All distance calculations should return finite values
            d_com = self.cosmology.comoving_distance(z)
            d_lum = self.cosmology.luminosity_distance(z)
            d_ang = self.cosmology.angular_diameter_distance(z)
            mu = self.cosmology.distance_modulus(z)
            
            assert np.isfinite(d_com), f"Non-finite comoving distance for z={z}"
            assert np.isfinite(d_lum), f"Non-finite luminosity distance for z={z}"
            assert np.isfinite(d_ang), f"Non-finite angular diameter distance for z={z}"
            assert np.isfinite(mu), f"Non-finite distance modulus for z={z}"
            
            assert d_com > 0, f"Non-positive comoving distance for z={z}"
            assert d_lum > 0, f"Non-positive luminosity distance for z={z}"
            assert d_ang > 0, f"Non-positive angular diameter distance for z={z}"
    
    def test_cosmological_parameter_validation(self):
        """Test cosmological parameter validation"""
        validation = self.cosmology.validate_cosmological_parameters()
        
        assert 'hubble_constant' in validation
        assert 'age_universe_Gyr' in validation
        assert 'parameters_valid' in validation
        assert 'flat_universe' in validation
        assert 'matter_dominated' in validation
        assert 'no_dark_energy' in validation
        
        # Should be valid matter-dominated model
        assert validation['parameters_valid']
        assert validation['flat_universe']
        assert validation['matter_dominated']
        assert validation['no_dark_energy']
    
    def test_parameter_retrieval(self):
        """Test cosmological parameter retrieval"""
        params = self.cosmology.get_cosmological_parameters()
        
        assert 'hubble_constant' in params
        assert 'omega_matter' in params
        assert 'omega_lambda' in params
        assert 'model_type' in params
        
        assert params['hubble_constant'] == 70.0
        assert params['omega_matter'] == 1.0
        assert params['omega_lambda'] == 0.0
        assert params['model_type'] == 'matter_dominated_no_dark_energy'
    
    def test_hubble_constant_bounds(self):
        """Test Hubble constant bounds enforcement"""
        # Test extreme values
        cosmology_low = EnhancedQVDCosmology(hubble_constant=10.0)  # Too low
        cosmology_high = EnhancedQVDCosmology(hubble_constant=200.0)  # Too high
        
        # Should be clamped to reasonable range
        assert 50.0 <= cosmology_low.H0 <= 100.0
        assert 50.0 <= cosmology_high.H0 <= 100.0
    
    def test_lambda_cdm_integration_stability(self):
        """Test numerical stability of ΛCDM integration"""
        # Test range of redshifts
        redshifts = np.logspace(-3, 0, 20)  # z = 0.001 to 1.0
        
        for z in redshifts:
            d_lambda_cdm = self.cosmology.lambda_cdm_distance(z)
            assert np.isfinite(d_lambda_cdm), f"ΛCDM integration failed at z={z}"
            assert d_lambda_cdm > 0, f"Non-positive ΛCDM distance at z={z}"


class TestCosmologyIntegration:
    """Integration tests for cosmology calculations"""
    
    def test_distance_relationships(self):
        """Test relationships between different distance measures"""
        cosmology = EnhancedQVDCosmology(enable_bounds_checking=True)
        
        redshift = 0.5
        
        d_com = cosmology.comoving_distance(redshift)
        d_lum = cosmology.luminosity_distance(redshift)
        d_ang = cosmology.angular_diameter_distance(redshift)
        
        # Test fundamental relationships
        assert abs(d_lum - d_com * (1 + redshift)) < 1e-10
        assert abs(d_ang - d_com / (1 + redshift)) < 1e-10
        assert abs(d_lum * d_ang - d_com**2) < 1e-10
    
    def test_cosmological_consistency(self):
        """Test consistency of cosmological calculations"""
        cosmology = EnhancedQVDCosmology(enable_bounds_checking=True)
        
        # Test that distances increase monotonically with redshift
        redshifts = np.linspace(0.1, 1.0, 10)
        
        d_com_array = cosmology.comoving_distance(redshifts)
        d_lum_array = cosmology.luminosity_distance(redshifts)
        
        # Should be monotonically increasing
        assert np.all(np.diff(d_com_array) > 0)
        assert np.all(np.diff(d_lum_array) > 0)
        
        # Luminosity distance should grow faster than comoving
        for i in range(1, len(redshifts)):
            ratio_lum = d_lum_array[i] / d_lum_array[i-1]
            ratio_com = d_com_array[i] / d_com_array[i-1]
            assert ratio_lum > ratio_com
    
    def test_performance_benchmark(self):
        """Test performance of cosmology calculations"""
        cosmology = EnhancedQVDCosmology(enable_bounds_checking=True)
        
        # Large array for performance testing
        redshifts = np.linspace(0.01, 1.0, 1000)
        
        import time
        start_time = time.time()
        
        distances = cosmology.luminosity_distance(redshifts)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time
        assert duration < 2.0  # Less than 2 seconds for 1000 points
        
        # All results should be finite
        assert np.all(np.isfinite(distances))
        
        # Calculate performance metric
        calculations_per_second = len(redshifts) / duration
        print(f"Cosmology performance: {calculations_per_second:.0f} calculations/second")
        
        # Should be reasonably fast
        assert calculations_per_second > 500


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])