#!/usr/bin/env python3
"""
Unit Tests for Enhanced QVD Redshift Physics
============================================

Comprehensive tests for redshift physics calculations with numerical stability.
"""

import numpy as np
import pytest
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redshift_physics import EnhancedQVDPhysics, RedshiftBounds, RedshiftBoundsEnforcer


class TestRedshiftBoundsEnforcer:
    """Test bounds enforcement for redshift parameters"""
    
    def test_redshift_bounds(self):
        """Test redshift bounds enforcement"""
        enforcer = RedshiftBoundsEnforcer()
        bounds = enforcer.bounds
        
        # Test normal values
        assert enforcer.enforce_redshift(0.5) == 0.5
        
        # Test bounds enforcement
        assert enforcer.enforce_redshift(-1.0) == bounds.MIN_REDSHIFT
        assert enforcer.enforce_redshift(100.0) == bounds.MAX_REDSHIFT
        
        # Test array enforcement
        z_array = np.array([-1.0, 0.5, 100.0])
        z_enforced = enforcer.enforce_redshift(z_array)
        
        assert z_enforced[0] == bounds.MIN_REDSHIFT
        assert z_enforced[1] == 0.5
        assert z_enforced[2] == bounds.MAX_REDSHIFT
    
    def test_qvd_coupling_bounds(self):
        """Test QVD coupling bounds enforcement"""
        enforcer = RedshiftBoundsEnforcer()
        bounds = enforcer.bounds
        
        # Test normal values
        assert enforcer.enforce_qvd_coupling(0.85) == 0.85
        
        # Test bounds enforcement
        assert enforcer.enforce_qvd_coupling(-1.0) == bounds.MIN_QVD_COUPLING
        assert enforcer.enforce_qvd_coupling(100.0) == bounds.MAX_QVD_COUPLING
    
    def test_distance_bounds(self):
        """Test distance bounds enforcement"""
        enforcer = RedshiftBoundsEnforcer()
        bounds = enforcer.bounds
        
        # Test normal values
        assert enforcer.enforce_distance(100.0) == 100.0
        
        # Test bounds enforcement
        assert enforcer.enforce_distance(-10.0) == bounds.MIN_DISTANCE_MPC
        assert enforcer.enforce_distance(1e6) == bounds.MAX_DISTANCE_MPC


class TestEnhancedQVDPhysics:
    """Test enhanced QVD physics calculations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.physics = EnhancedQVDPhysics(
            qvd_coupling=0.85,
            redshift_power=0.6,
            enable_logging=False
        )
    
    def test_initialization(self):
        """Test physics module initialization"""
        assert self.physics.qvd_coupling == 0.85
        assert self.physics.redshift_power == 0.6
        assert hasattr(self.physics, 'bounds_enforcer')
    
    def test_redshift_dimming_scalar(self):
        """Test redshift dimming calculation for scalar input"""
        # Test normal redshift
        dimming = self.physics.calculate_redshift_dimming(0.5)
        
        assert np.isfinite(dimming)
        assert dimming >= 0.0
        assert dimming <= 10.0  # Within reasonable bounds
    
    def test_redshift_dimming_array(self):
        """Test redshift dimming calculation for array input"""
        redshifts = np.array([0.1, 0.3, 0.5, 0.7])
        dimming = self.physics.calculate_redshift_dimming(redshifts)
        
        assert len(dimming) == len(redshifts)
        assert np.all(np.isfinite(dimming))
        assert np.all(dimming >= 0.0)
        assert np.all(dimming <= 10.0)
    
    def test_redshift_dimming_extreme_values(self):
        """Test redshift dimming with extreme input values"""
        # Test extreme redshifts
        extreme_redshifts = np.array([-1.0, 0.0, 1e-10, 100.0])
        dimming = self.physics.calculate_redshift_dimming(extreme_redshifts)
        
        # All results should be finite and bounded
        assert np.all(np.isfinite(dimming))
        assert np.all(dimming >= 0.0)
        assert np.all(dimming <= 10.0)
    
    def test_qvd_cross_section(self):
        """Test QVD cross-section calculation"""
        redshift = 0.5
        cross_section = self.physics.calculate_qvd_cross_section(redshift)
        
        assert np.isfinite(cross_section)
        assert cross_section > 0.0
        assert cross_section >= self.physics.sigma_thomson  # Should be enhanced
    
    def test_optical_depth(self):
        """Test optical depth calculation"""
        redshift = 0.5
        path_length = 1000.0  # Mpc
        
        optical_depth = self.physics.calculate_optical_depth(redshift, path_length)
        
        assert np.isfinite(optical_depth)
        assert optical_depth >= 0.0
        assert optical_depth <= 50.0  # Within bounds
    
    def test_transmission(self):
        """Test transmission calculation"""
        redshift = 0.5
        path_length = 1000.0  # Mpc
        
        transmission = self.physics.calculate_transmission(redshift, path_length)
        
        assert np.isfinite(transmission)
        assert 0.0 <= transmission <= 1.0
    
    def test_energy_loss_fraction(self):
        """Test energy loss fraction calculation"""
        redshift = 0.5
        energy_loss = self.physics.energy_loss_fraction(redshift)
        
        assert np.isfinite(energy_loss)
        assert 0.0 <= energy_loss <= 1.0
    
    def test_energy_conservation_validation(self):
        """Test energy conservation validation"""
        redshift_array = np.linspace(0.1, 1.0, 10)
        validation = self.physics.validate_energy_conservation(redshift_array)
        
        assert 'redshifts' in validation
        assert 'energy_loss' in validation
        assert 'conservation_satisfied' in validation
        assert len(validation['energy_loss']) == len(redshift_array)
        assert np.all(np.isfinite(validation['energy_loss']))
    
    def test_parameter_updates(self):
        """Test parameter update functionality"""
        original_coupling = self.physics.qvd_coupling
        
        # Update parameters
        self.physics.update_parameters(qvd_coupling=1.0, redshift_power=0.8)
        
        assert self.physics.qvd_coupling == 1.0
        assert self.physics.redshift_power == 0.8
        
        # Test invalid parameter
        with pytest.raises(ValueError):
            self.physics.update_parameters(invalid_param=1.0)
    
    def test_e144_scaling(self):
        """Test E144 scaling calculation"""
        scaling = self.physics.calculate_e144_scaling()
        
        assert 'e144_cross_section' in scaling
        assert 'scaling_factor' in scaling
        assert 'scaling_valid' in scaling
        assert np.isfinite(scaling['scaling_factor'])
        assert scaling['scaling_valid']
    
    def test_model_parameters(self):
        """Test model parameters retrieval"""
        params = self.physics.get_model_parameters()
        
        assert 'qvd_coupling' in params
        assert 'redshift_power' in params
        assert 'bounds_enforced' in params
        assert 'numerical_safety' in params
        assert params['bounds_enforced'] is True
        assert params['numerical_safety'] is True
    
    def test_igm_effects(self):
        """Test IGM effects calculation"""
        redshift = 0.5
        
        # Access private method for testing
        igm_contribution = self.physics._calculate_igm_effects(redshift)
        
        assert np.isfinite(igm_contribution)
        assert igm_contribution >= 0.0
    
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge cases"""
        edge_cases = [
            0.0,           # Zero redshift
            1e-10,         # Very small redshift
            1e10,          # Very large redshift (will be bounded)
            np.inf,        # Infinite redshift (will be bounded)
            np.nan,        # NaN redshift (will be bounded)
        ]
        
        for z in edge_cases:
            dimming = self.physics.calculate_redshift_dimming(z)
            assert np.isfinite(dimming), f"Non-finite result for z={z}"
            assert dimming >= 0.0, f"Negative dimming for z={z}"
    
    def test_bounds_warning_tracking(self):
        """Test that bounds violations are tracked"""
        initial_warnings = self.physics.bounds_enforcer.warning_count
        
        # Trigger bounds violations
        self.physics.calculate_redshift_dimming(-1.0)  # Negative redshift
        self.physics.calculate_redshift_dimming(100.0)  # Large redshift
        
        # Should have more warnings now
        assert self.physics.bounds_enforcer.warning_count > initial_warnings


class TestRedshiftPhysicsIntegration:
    """Integration tests for redshift physics"""
    
    def test_physics_consistency(self):
        """Test consistency between different physics calculations"""
        physics = EnhancedQVDPhysics(enable_logging=False)
        
        redshift = 0.5
        path_length = 1000.0
        
        # Calculate related quantities
        dimming = physics.calculate_redshift_dimming(redshift)
        cross_section = physics.calculate_qvd_cross_section(redshift)
        optical_depth = physics.calculate_optical_depth(redshift, path_length)
        transmission = physics.calculate_transmission(redshift, path_length)
        energy_loss = physics.energy_loss_fraction(redshift)
        
        # All should be finite
        quantities = [dimming, cross_section, optical_depth, transmission, energy_loss]
        assert all(np.isfinite(q) for q in quantities)
        
        # Check physical relationships
        assert transmission == np.exp(-optical_depth)  # Basic relationship
        assert cross_section >= physics.sigma_thomson  # Enhanced cross-section
        assert 0 <= energy_loss <= 1  # Energy loss fraction bounds
    
    def test_redshift_scaling_law(self):
        """Test that redshift scaling follows expected power law"""
        physics = EnhancedQVDPhysics(
            qvd_coupling=1.0,  # Simple coupling for testing
            redshift_power=0.6,
            igm_enhancement=0.0,  # Disable IGM for pure power law test
            enable_logging=False
        )
        
        # Test redshifts
        z1, z2 = 0.1, 0.2
        
        dimming1 = physics.calculate_redshift_dimming(z1)
        dimming2 = physics.calculate_redshift_dimming(z2)
        
        # Should follow power law (approximately, given bounds enforcement)
        expected_ratio = (z2 / z1) ** 0.6
        actual_ratio = dimming2 / dimming1 if dimming1 > 0 else 1.0
        
        # Allow some tolerance due to bounds enforcement and IGM effects
        assert abs(actual_ratio - expected_ratio) < 0.5
    
    def test_performance_benchmark(self):
        """Test performance of physics calculations"""
        physics = EnhancedQVDPhysics(enable_logging=False)
        
        # Large array for performance testing
        redshifts = np.linspace(0.01, 1.0, 1000)
        
        import time
        start_time = time.time()
        
        dimming = physics.calculate_redshift_dimming(redshifts)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time (< 1 second for 1000 points)
        assert duration < 1.0
        
        # All results should be finite
        assert np.all(np.isfinite(dimming))
        
        # Calculate performance metric
        calculations_per_second = len(redshifts) / duration
        print(f"Performance: {calculations_per_second:.0f} calculations/second")
        
        # Should be reasonably fast
        assert calculations_per_second > 1000


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])