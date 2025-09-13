#!/usr/bin/env python3
"""
Test Physical Bounds Enforcement System
======================================

Tests for physical bounds and constraints in QVD calculations.
"""

import numpy as np
from physical_bounds import (
    PhysicalBounds, BoundsEnforcer, 
    create_safe_plasma_state, create_safe_scattering_results,
    is_physically_reasonable_plasma, is_physically_reasonable_scattering
)

def test_physical_bounds_dataclass():
    """Test PhysicalBounds dataclass"""
    print("Testing PhysicalBounds dataclass...")
    
    bounds = PhysicalBounds()
    
    # Check that all bounds are reasonable
    assert bounds.MAX_OPTICAL_DEPTH > 0
    assert bounds.MAX_DIMMING_MAG > 0
    assert bounds.MIN_PLASMA_DENSITY > 0
    assert bounds.MAX_PLASMA_DENSITY > bounds.MIN_PLASMA_DENSITY
    assert bounds.MIN_TEMPERATURE > 0
    assert bounds.MAX_TEMPERATURE > bounds.MIN_TEMPERATURE
    
    print("  ✓ All bounds are reasonable")

def test_bounds_enforcer_plasma_density():
    """Test plasma density bounds enforcement"""
    print("Testing plasma density bounds...")
    
    enforcer = BoundsEnforcer()
    bounds = enforcer.bounds
    
    # Test normal values
    normal_density = 1e20
    result = enforcer.enforce_plasma_density(normal_density)
    assert result == normal_density
    
    # Test too low
    low_density = 1e5  # Below minimum
    result = enforcer.enforce_plasma_density(low_density)
    assert result == bounds.MIN_PLASMA_DENSITY
    
    # Test too high
    high_density = 1e35  # Above maximum
    result = enforcer.enforce_plasma_density(high_density)
    assert result == bounds.MAX_PLASMA_DENSITY
    
    # Test array input
    densities = np.array([1e5, 1e20, 1e35])
    results = enforcer.enforce_plasma_density(densities)
    expected = np.array([bounds.MIN_PLASMA_DENSITY, 1e20, bounds.MAX_PLASMA_DENSITY])
    np.testing.assert_array_equal(results, expected)
    
    print("  ✓ Plasma density bounds enforced correctly")

def test_bounds_enforcer_temperature():
    """Test temperature bounds enforcement"""
    print("Testing temperature bounds...")
    
    enforcer = BoundsEnforcer()
    bounds = enforcer.bounds
    
    # Test normal values
    normal_temp = 1e6
    result = enforcer.enforce_temperature(normal_temp)
    assert result == normal_temp
    
    # Test too low
    low_temp = 50.0  # Below minimum
    result = enforcer.enforce_temperature(low_temp)
    assert result == bounds.MIN_TEMPERATURE
    
    # Test too high
    high_temp = 1e12  # Above maximum
    result = enforcer.enforce_temperature(high_temp)
    assert result == bounds.MAX_TEMPERATURE
    
    print("  ✓ Temperature bounds enforced correctly")

def test_bounds_enforcer_cross_section():
    """Test cross-section bounds enforcement"""
    print("Testing cross-section bounds...")
    
    enforcer = BoundsEnforcer()
    bounds = enforcer.bounds
    
    # Test normal values
    normal_cs = 1e-30
    result = enforcer.enforce_cross_section(normal_cs)
    assert result == normal_cs
    
    # Test too low
    low_cs = 1e-60  # Below minimum
    result = enforcer.enforce_cross_section(low_cs)
    assert result == bounds.MIN_CROSS_SECTION
    
    # Test too high
    high_cs = 1e-10  # Above maximum
    result = enforcer.enforce_cross_section(high_cs)
    assert result == bounds.MAX_CROSS_SECTION
    
    print("  ✓ Cross-section bounds enforced correctly")

def test_bounds_enforcer_optical_depth():
    """Test optical depth bounds enforcement"""
    print("Testing optical depth bounds...")
    
    enforcer = BoundsEnforcer()
    bounds = enforcer.bounds
    
    # Test normal values
    normal_od = 5.0
    result = enforcer.enforce_optical_depth(normal_od)
    assert result == normal_od
    
    # Test negative (should be clamped to 0)
    negative_od = -1.0
    result = enforcer.enforce_optical_depth(negative_od)
    assert result == 0.0
    
    # Test too high
    high_od = 100.0  # Above maximum
    result = enforcer.enforce_optical_depth(high_od)
    assert result == bounds.MAX_OPTICAL_DEPTH
    
    print("  ✓ Optical depth bounds enforced correctly")

def test_bounds_enforcer_transmission():
    """Test transmission bounds enforcement"""
    print("Testing transmission bounds...")
    
    enforcer = BoundsEnforcer()
    bounds = enforcer.bounds
    
    # Test normal values
    normal_trans = 0.5
    result = enforcer.enforce_transmission(normal_trans)
    assert result == normal_trans
    
    # Test too low
    low_trans = 1e-30  # Below minimum
    result = enforcer.enforce_transmission(low_trans)
    assert result == bounds.MIN_TRANSMISSION
    
    # Test too high
    high_trans = 2.0  # Above maximum (1.0)
    result = enforcer.enforce_transmission(high_trans)
    assert result == 1.0
    
    print("  ✓ Transmission bounds enforced correctly")

def test_bounds_enforcer_dimming():
    """Test dimming magnitude bounds enforcement"""
    print("Testing dimming magnitude bounds...")
    
    enforcer = BoundsEnforcer()
    bounds = enforcer.bounds
    
    # Test normal values
    normal_dimming = 2.0
    result = enforcer.enforce_dimming_magnitude(normal_dimming)
    assert result == normal_dimming
    
    # Test negative (should be clamped to 0)
    negative_dimming = -1.0
    result = enforcer.enforce_dimming_magnitude(negative_dimming)
    assert result == 0.0
    
    # Test too high
    high_dimming = 20.0  # Above maximum
    result = enforcer.enforce_dimming_magnitude(high_dimming)
    assert result == bounds.MAX_DIMMING_MAG
    
    print("  ✓ Dimming magnitude bounds enforced correctly")

def test_bounds_enforcer_nan_handling():
    """Test handling of NaN/Inf values"""
    print("Testing NaN/Inf handling...")
    
    enforcer = BoundsEnforcer()
    
    # Test NaN values
    nan_density = np.nan
    result = enforcer.enforce_plasma_density(nan_density)
    assert np.isfinite(result)
    assert result > 0
    
    # Test Inf values
    inf_temperature = np.inf
    result = enforcer.enforce_temperature(inf_temperature)
    assert np.isfinite(result)
    assert result <= enforcer.bounds.MAX_TEMPERATURE
    
    # Test array with mixed NaN/Inf
    mixed_values = np.array([1e20, np.nan, np.inf, -np.inf])
    results = enforcer.enforce_plasma_density(mixed_values)
    assert np.all(np.isfinite(results))
    assert np.all(results > 0)
    
    print("  ✓ NaN/Inf values handled correctly")

def test_violation_tracking():
    """Test violation counting and reporting"""
    print("Testing violation tracking...")
    
    enforcer = BoundsEnforcer()
    
    # Generate some violations
    enforcer.enforce_plasma_density(1e5)  # Too low
    enforcer.enforce_temperature(1e12)    # Too high
    enforcer.enforce_plasma_density(1e35) # Too high
    
    # Check violation summary
    violations = enforcer.get_violation_summary()
    assert 'plasma_density' in violations
    assert 'temperature' in violations
    assert violations['plasma_density'] == 2  # Two violations
    assert violations['temperature'] == 1     # One violation
    
    # Test reset
    enforcer.reset_violation_counts()
    violations = enforcer.get_violation_summary()
    assert len(violations) == 0
    
    print("  ✓ Violation tracking works correctly")

def test_create_safe_plasma_state():
    """Test safe plasma state creation"""
    print("Testing safe plasma state creation...")
    
    # Test with reasonable values
    plasma = create_safe_plasma_state(
        radius_cm=1e12,
        electron_density_cm3=1e20,
        temperature_K=1e6,
        luminosity_erg_s=1e42
    )
    
    assert 'radius_cm' in plasma
    assert 'electron_density_cm3' in plasma
    assert 'temperature_K' in plasma
    assert 'luminosity_erg_s' in plasma
    assert 'intensity_erg_cm2_s' in plasma
    assert 'photosphere_area_cm2' in plasma
    
    # All values should be finite and positive
    for key, value in plasma.items():
        assert np.isfinite(value)
        assert value > 0
    
    # Test with extreme values
    plasma_extreme = create_safe_plasma_state(
        radius_cm=1e50,      # Too large
        electron_density_cm3=1e5,  # Too small
        temperature_K=50.0,  # Too small
        luminosity_erg_s=1e60  # Too large
    )
    
    # Should be bounded
    bounds = PhysicalBounds()
    assert plasma_extreme['radius_cm'] <= bounds.MAX_RADIUS
    assert plasma_extreme['electron_density_cm3'] >= bounds.MIN_PLASMA_DENSITY
    assert plasma_extreme['temperature_K'] >= bounds.MIN_TEMPERATURE
    assert plasma_extreme['luminosity_erg_s'] <= bounds.MAX_LUMINOSITY
    
    print("  ✓ Safe plasma state creation works correctly")

def test_create_safe_scattering_results():
    """Test safe scattering results creation"""
    print("Testing safe scattering results creation...")
    
    # Test with reasonable values
    scattering = create_safe_scattering_results(
        qvd_cross_section_cm2=1e-30,
        optical_depth=5.0,
        transmission=0.1,
        dimming_magnitudes=2.5
    )
    
    assert 'qvd_cross_section_cm2' in scattering
    assert 'optical_depth' in scattering
    assert 'transmission' in scattering
    assert 'dimming_magnitudes' in scattering
    
    # All values should be finite and non-negative
    for key, value in scattering.items():
        assert np.isfinite(value)
        assert value >= 0
    
    # Test with extreme values
    scattering_extreme = create_safe_scattering_results(
        qvd_cross_section_cm2=1e-5,   # Too large
        optical_depth=1000.0,         # Too large
        transmission=2.0,             # Too large
        dimming_magnitudes=50.0       # Too large
    )
    
    # Should be bounded
    bounds = PhysicalBounds()
    assert scattering_extreme['qvd_cross_section_cm2'] <= bounds.MAX_CROSS_SECTION
    assert scattering_extreme['optical_depth'] <= bounds.MAX_OPTICAL_DEPTH
    assert scattering_extreme['transmission'] <= 1.0
    assert scattering_extreme['dimming_magnitudes'] <= bounds.MAX_DIMMING_MAG
    
    print("  ✓ Safe scattering results creation works correctly")

def test_physical_reasonableness_checks():
    """Test physical reasonableness checking functions"""
    print("Testing physical reasonableness checks...")
    
    # Test reasonable plasma
    assert is_physically_reasonable_plasma(1e20, 1e6, 1e12) == True
    
    # Test unreasonable plasma
    assert is_physically_reasonable_plasma(1e5, 1e6, 1e12) == False  # Density too low
    assert is_physically_reasonable_plasma(1e20, 50.0, 1e12) == False  # Temperature too low
    assert is_physically_reasonable_plasma(1e20, 1e6, 1e3) == False   # Radius too small
    
    # Test reasonable scattering
    assert is_physically_reasonable_scattering(1e-30, 5.0, 0.1) == True
    
    # Test unreasonable scattering
    assert is_physically_reasonable_scattering(1e-5, 5.0, 0.1) == False   # Cross-section too large
    assert is_physically_reasonable_scattering(1e-30, 100.0, 0.1) == False # Optical depth too large
    assert is_physically_reasonable_scattering(1e-30, 5.0, 2.0) == False   # Transmission > 1
    
    print("  ✓ Physical reasonableness checks work correctly")

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("PHYSICAL BOUNDS ENFORCEMENT TEST SUITE")
    print("="*60)
    
    try:
        test_physical_bounds_dataclass()
        test_bounds_enforcer_plasma_density()
        test_bounds_enforcer_temperature()
        test_bounds_enforcer_cross_section()
        test_bounds_enforcer_optical_depth()
        test_bounds_enforcer_transmission()
        test_bounds_enforcer_dimming()
        test_bounds_enforcer_nan_handling()
        test_violation_tracking()
        test_create_safe_plasma_state()
        test_create_safe_scattering_results()
        test_physical_reasonableness_checks()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("Physical bounds enforcement system is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()