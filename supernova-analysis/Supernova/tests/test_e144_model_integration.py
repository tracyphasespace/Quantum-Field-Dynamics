#!/usr/bin/env python3
"""
Integration Test for Fixed E144 Model
====================================

Comprehensive test to verify that the entire E144 supernova QVD model
works without producing NaN values after all numerical fixes.
"""

import numpy as np
import sys
import os

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from supernova_qvd_scattering import E144ExperimentalData, SupernovaParameters, E144ScaledQVDModel

def test_full_model_basic():
    """Test the full model with basic parameters"""
    print("Testing full E144 model with basic parameters...")
    
    # Create model
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Test luminance curve generation
    distance_Mpc = 100.0
    wavelength_nm = 500.0
    
    curve = model.generate_luminance_curve(
        distance_Mpc=distance_Mpc,
        wavelength_nm=wavelength_nm,
        time_range_days=(-20, 100),
        time_resolution_days=2.0
    )
    
    # Check that all arrays are finite
    for key, values in curve.items():
        if isinstance(values, np.ndarray):
            assert np.all(np.isfinite(values)), f"Non-finite values in {key}"
            if 'magnitude' in key or 'dimming' in key:
                assert np.all(values >= 0), f"Negative magnitudes in {key}"
            elif 'luminosity' in key:
                assert np.all(values >= 0), f"Negative luminosity in {key}"
            elif 'optical_depth' in key:
                assert np.all(values >= 0), f"Negative optical depth in {key}"
    
    print("  ✓ Full model basic test passed")

def test_full_model_multi_wavelength():
    """Test the full model with multiple wavelengths"""
    print("Testing full E144 model with multiple wavelengths...")
    
    # Create model
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Test multi-wavelength curves
    distance_Mpc = 100.0
    wavelengths = [400, 500, 600, 700, 800]
    
    curves = model.generate_multi_wavelength_curves(
        distance_Mpc=distance_Mpc,
        wavelengths_nm=wavelengths,
        time_range_days=(-20, 100)
    )
    
    # Check each wavelength curve
    for wavelength in wavelengths:
        key = f'{wavelength}nm'
        assert key in curves, f"Missing wavelength curve: {key}"
        
        curve = curves[key]
        for data_key, values in curve.items():
            if isinstance(values, np.ndarray):
                assert np.all(np.isfinite(values)), \
                    f"Non-finite values in {key}.{data_key}"
    
    # Check color evolution if present
    if 'color_evolution' in curves:
        color_data = curves['color_evolution']
        for key, values in color_data.items():
            if isinstance(values, np.ndarray):
                assert np.all(np.isfinite(values)), \
                    f"Non-finite values in color_evolution.{key}"
    
    print("  ✓ Multi-wavelength test passed")

def test_full_model_extreme_distances():
    """Test the full model with extreme distances"""
    print("Testing full E144 model with extreme distances...")
    
    # Create model
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Test extreme distances
    extreme_distances = [1.0, 10.0, 1000.0, 10000.0]  # Mpc
    wavelength_nm = 500.0
    
    for distance in extreme_distances:
        curve = model.generate_luminance_curve(
            distance_Mpc=distance,
            wavelength_nm=wavelength_nm,
            time_range_days=(-10, 50),
            time_resolution_days=5.0
        )
        
        # All values should be finite
        for key, values in curve.items():
            if isinstance(values, np.ndarray):
                assert np.all(np.isfinite(values)), \
                    f"Non-finite values in {key} at distance {distance} Mpc"
    
    print("  ✓ Extreme distances test passed")

def test_full_model_extreme_wavelengths():
    """Test the full model with extreme wavelengths"""
    print("Testing full E144 model with extreme wavelengths...")
    
    # Create model
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Test extreme wavelengths
    extreme_wavelengths = [200.0, 1000.0, 5000.0]  # nm
    distance_Mpc = 100.0
    
    for wavelength in extreme_wavelengths:
        curve = model.generate_luminance_curve(
            distance_Mpc=distance_Mpc,
            wavelength_nm=wavelength,
            time_range_days=(-10, 50),
            time_resolution_days=5.0
        )
        
        # All values should be finite
        for key, values in curve.items():
            if isinstance(values, np.ndarray):
                assert np.all(np.isfinite(values)), \
                    f"Non-finite values in {key} at wavelength {wavelength} nm"
    
    print("  ✓ Extreme wavelengths test passed")

def test_full_model_extreme_time_ranges():
    """Test the full model with extreme time ranges"""
    print("Testing full E144 model with extreme time ranges...")
    
    # Create model
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Test extreme time ranges
    extreme_time_ranges = [
        (-100, -50),   # Pre-explosion only
        (-10, 10),     # Around explosion
        (100, 500),    # Late times only
        (-50, 200),    # Full range
    ]
    
    distance_Mpc = 100.0
    wavelength_nm = 500.0
    
    for time_range in extreme_time_ranges:
        curve = model.generate_luminance_curve(
            distance_Mpc=distance_Mpc,
            wavelength_nm=wavelength_nm,
            time_range_days=time_range,
            time_resolution_days=5.0
        )
        
        # All values should be finite
        for key, values in curve.items():
            if isinstance(values, np.ndarray):
                assert np.all(np.isfinite(values)), \
                    f"Non-finite values in {key} for time range {time_range}"
    
    print("  ✓ Extreme time ranges test passed")

def test_full_model_extreme_parameters():
    """Test the full model with extreme supernova parameters"""
    print("Testing full E144 model with extreme parameters...")
    
    # Create model with extreme parameters
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    
    # Set extreme parameters that could cause numerical issues
    sn_params.peak_luminosity_erg_s = 1e50          # Very bright
    sn_params.initial_radius_cm = 1e6               # Very small
    sn_params.expansion_velocity_cm_s = 1e10        # Very fast
    sn_params.initial_electron_density_cm3 = 1e30   # Very dense
    sn_params.plasma_enhancement_factor = 1e10      # Very large enhancement
    
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Test basic curve generation
    curve = model.generate_luminance_curve(
        distance_Mpc=100.0,
        wavelength_nm=500.0,
        time_range_days=(-10, 50),
        time_resolution_days=5.0
    )
    
    # All values should still be finite due to bounds enforcement
    for key, values in curve.items():
        if isinstance(values, np.ndarray):
            assert np.all(np.isfinite(values)), \
                f"Non-finite values in {key} with extreme parameters"
    
    print("  ✓ Extreme parameters test passed")

def test_full_model_stress_test():
    """Stress test the full model with many calculations"""
    print("Running stress test on full E144 model...")
    
    # Create model
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Generate many curves with different parameters
    distances = [10, 50, 100, 500, 1000]  # Mpc
    wavelengths = [300, 400, 500, 600, 700, 800]  # nm
    
    finite_count = 0
    total_count = 0
    
    for distance in distances:
        for wavelength in wavelengths:
            curve = model.generate_luminance_curve(
                distance_Mpc=distance,
                wavelength_nm=wavelength,
                time_range_days=(-20, 100),
                time_resolution_days=2.0
            )
            
            # Count finite values
            for key, values in curve.items():
                if isinstance(values, np.ndarray):
                    finite_count += np.sum(np.isfinite(values))
                    total_count += len(values)
    
    # All values should be finite
    finite_fraction = finite_count / total_count
    assert finite_fraction == 1.0, f"Only {finite_fraction:.3f} of values are finite"
    
    print(f"  ✓ Stress test passed: {finite_count}/{total_count} values finite")

def test_model_comparison_with_original():
    """Compare fixed model behavior with expected patterns"""
    print("Testing model behavior patterns...")
    
    # Create model
    e144_data = E144ExperimentalData()
    sn_params = SupernovaParameters()
    model = E144ScaledQVDModel(e144_data, sn_params)
    
    # Generate reference curve
    distance_Mpc = 100.0
    wavelength_nm = 500.0
    
    curve = model.generate_luminance_curve(
        distance_Mpc=distance_Mpc,
        wavelength_nm=wavelength_nm,
        time_range_days=(-20, 100),
        time_resolution_days=1.0
    )
    
    # Check expected patterns
    time_days = curve['time_days']
    dimming = curve['dimming_magnitudes']
    optical_depth = curve['optical_depths']
    
    # All should be finite
    assert np.all(np.isfinite(time_days)), "Time array should be finite"
    assert np.all(np.isfinite(dimming)), "Dimming should be finite"
    assert np.all(np.isfinite(optical_depth)), "Optical depth should be finite"
    
    # Dimming should be non-negative
    assert np.all(dimming >= 0), "Dimming should be non-negative"
    
    # Optical depth should be non-negative
    assert np.all(optical_depth >= 0), "Optical depth should be non-negative"
    
    # For positive times, there should be some scattering effect
    positive_time_mask = time_days > 0
    if np.any(positive_time_mask):
        max_dimming = np.max(dimming[positive_time_mask])
        assert max_dimming > 0, "Should have some dimming at positive times"
    
    print("  ✓ Model behavior patterns are correct")

def run_all_tests():
    """Run all integration tests"""
    print("="*60)
    print("E144 MODEL INTEGRATION TEST SUITE")
    print("="*60)
    
    try:
        test_full_model_basic()
        test_full_model_multi_wavelength()
        test_full_model_extreme_distances()
        test_full_model_extreme_wavelengths()
        test_full_model_extreme_time_ranges()
        test_full_model_extreme_parameters()
        test_full_model_stress_test()
        test_model_comparison_with_original()
        
        print("\n" + "="*60)
        print("ALL INTEGRATION TESTS PASSED! ✓")
        print("The E144 model is now numerically stable and produces")
        print("finite results under all tested conditions.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_all_tests()