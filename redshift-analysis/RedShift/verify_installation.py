#!/usr/bin/env python3
"""
Enhanced RedShift Installation Verification Script
==================================================

Quick verification that the Enhanced RedShift QVD model is properly installed
and working correctly with all numerical stability features.
"""

import sys
import numpy as np
import time

def verify_installation():
    """Verify that the installation is working correctly"""
    
    print("Enhanced RedShift QVD Model - Installation Verification")
    print("=" * 60)
    print()
    
    # Test 1: Import all modules
    print("1. Testing imports...")
    try:
        from redshift_analyzer import EnhancedRedshiftAnalyzer
        from redshift_physics import EnhancedQVDPhysics, RedshiftBounds, RedshiftBoundsEnforcer
        from redshift_cosmology import EnhancedQVDCosmology
        from redshift_visualization import EnhancedRedshiftPlotter
        from numerical_safety import safe_power, safe_log10, validate_finite
        from error_handling import setup_qvd_logging, ErrorReporter
        print("   ✅ All imports successful")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Create enhanced analyzer instance
    print("2. Testing enhanced analyzer creation...")
    try:
        analyzer = EnhancedRedshiftAnalyzer(
            qvd_coupling=0.85,
            redshift_power=0.6,
            enable_logging=False,
            enable_bounds_checking=True
        )
        print("   ✅ Enhanced analyzer created successfully")
        print(f"      QVD coupling: {analyzer.physics.qvd_coupling:.3f}")
        print(f"      Redshift power: {analyzer.physics.redshift_power:.3f}")
        print(f"      Bounds checking: {analyzer.enable_bounds_checking}")
    except Exception as e:
        print(f"   ❌ Analyzer creation failed: {e}")
        return False
    
    # Test 3: Test numerical stability with extreme values
    print("3. Testing numerical stability...")
    try:
        extreme_redshifts = [-1.0, 0.0, 1e-10, 10.0, 100.0, np.inf, np.nan]
        all_finite = True
        
        for z in extreme_redshifts:
            dimming = analyzer.calculate_qvd_dimming(z)
            if not np.isfinite(dimming):
                all_finite = False
                print(f"      ⚠️  Non-finite result for z={z}: {dimming}")
            elif not (0.0 <= dimming <= 10.0):
                print(f"      ⚠️  Out-of-bounds result for z={z}: {dimming}")
        
        if all_finite:
            print("   ✅ All extreme values handled with finite results")
        else:
            print("   ❌ Some extreme values produced non-finite results")
            return False
            
    except Exception as e:
        print(f"   ❌ Numerical stability test failed: {e}")
        return False
    
    # Test 4: Test bounds enforcement
    print("4. Testing bounds enforcement...")
    try:
        bounds_enforcer = RedshiftBoundsEnforcer()
        
        # Test extreme parameter values
        extreme_coupling = bounds_enforcer.enforce_qvd_coupling(1000.0)  # Too high
        extreme_redshift = bounds_enforcer.enforce_redshift(-10.0)       # Too low
        
        bounds = bounds_enforcer.bounds
        coupling_ok = bounds.MIN_QVD_COUPLING <= extreme_coupling <= bounds.MAX_QVD_COUPLING
        redshift_ok = bounds.MIN_REDSHIFT <= extreme_redshift <= bounds.MAX_REDSHIFT
        
        if coupling_ok and redshift_ok:
            print("   ✅ Bounds enforcement working correctly")
            print(f"      Extreme coupling clamped to: {extreme_coupling:.6f}")
            print(f"      Extreme redshift clamped to: {extreme_redshift:.6f}")
        else:
            print("   ❌ Bounds enforcement not working properly")
            return False
            
    except Exception as e:
        print(f"   ❌ Bounds enforcement test failed: {e}")
        return False
    
    # Test 5: Test safe mathematical operations
    print("5. Testing safe mathematical operations...")
    try:
        # Test operations that could cause issues
        result1 = safe_power(0.0, 2.0)      # Zero base
        result2 = safe_log10(0.0)           # Log of zero
        result3 = safe_power(-2.0, 3.0)     # Negative base
        result4 = safe_power(2.0, 1000.0)   # Large exponent
        
        results = [result1, result2, result3, result4]
        if all(np.isfinite(r) for r in results):
            print("   ✅ Safe operations working correctly")
            print(f"      safe_power(0.0, 2.0) = {result1:.6f}")
            print(f"      safe_log10(0.0) = {result2:.6f}")
            print(f"      safe_power(-2.0, 3.0) = {result3:.6f}")
            print(f"      safe_power(2.0, 1000.0) = {result4:.6f}")
        else:
            print("   ❌ Safe operations produced non-finite results")
            return False
            
    except Exception as e:
        print(f"   ❌ Safe operations test failed: {e}")
        return False
    
    # Test 6: Test cosmological calculations
    print("6. Testing cosmological calculations...")
    try:
        cosmology = EnhancedQVDCosmology(enable_bounds_checking=True)
        
        # Test distance calculations
        test_redshifts = [0.1, 0.5, 1.0]
        for z in test_redshifts:
            d_lum = cosmology.luminosity_distance(z)
            d_ang = cosmology.angular_diameter_distance(z)
            mu = cosmology.distance_modulus(z)
            
            if not all(np.isfinite([d_lum, d_ang, mu])):
                print(f"      ❌ Non-finite cosmological result at z={z}")
                return False
        
        print("   ✅ Cosmological calculations working correctly")
        print(f"      Hubble constant: {cosmology.H0:.1f} km/s/Mpc")
        print(f"      Matter-dominated: Ωₘ = {cosmology.omega_m:.1f}")
        print(f"      No dark energy: ΩΛ = {cosmology.omega_lambda:.1f}")
        
    except Exception as e:
        print(f"   ❌ Cosmological calculations test failed: {e}")
        return False
    
    # Test 7: Test complete analysis pipeline
    print("7. Testing complete analysis pipeline...")
    try:
        start_time = time.time()
        
        # Run a small-scale analysis
        hubble_data = analyzer.generate_hubble_diagram(
            z_min=0.1, z_max=0.5, n_points=10
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check results
        success_rate = hubble_data.get('points_calculated', 0) / len(hubble_data['redshifts'])
        all_finite = np.all(np.isfinite(hubble_data['qvd_dimming']))
        
        if success_rate > 0.8 and all_finite:
            print("   ✅ Complete analysis pipeline working correctly")
            print(f"      Generated {len(hubble_data['redshifts'])} points in {duration:.2f}s")
            print(f"      Success rate: {success_rate:.1%}")
            print(f"      Max QVD dimming: {np.max(hubble_data['qvd_dimming']):.3f} mag")
        else:
            print(f"   ❌ Analysis pipeline issues: success_rate={success_rate:.1%}, finite={all_finite}")
            return False
            
    except Exception as e:
        print(f"   ❌ Complete analysis test failed: {e}")
        return False
    
    # Test 8: Test error handling system
    print("8. Testing error handling system...")
    try:
        error_reporter = ErrorReporter()
        
        # Test error reporting
        error_reporter.add_error("TestError", "This is a test error", {"test": True})
        
        if len(error_reporter.errors) == 1:
            print("   ✅ Error handling system working correctly")
            print(f"      Error reporter functional with {len(error_reporter.errors)} test error")
        else:
            print("   ❌ Error handling system not working properly")
            return False
            
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False
    
    # Summary
    print()
    print("=" * 60)
    print("🎉 INSTALLATION VERIFICATION SUCCESSFUL!")
    print("=" * 60)
    print()
    print("The Enhanced RedShift QVD model is properly installed and ready to use.")
    print()
    print("Key Features Verified:")
    print("  ✅ Enhanced numerical stability with 100% finite results")
    print("  ✅ Comprehensive bounds enforcement for all parameters")
    print("  ✅ Safe mathematical operations preventing NaN/Inf values")
    print("  ✅ Robust cosmological calculations with error handling")
    print("  ✅ Complete analysis pipeline with performance monitoring")
    print("  ✅ Professional error handling and reporting system")
    print()
    print("Next steps:")
    print("  • Run examples: python examples/basic_redshift_analysis.py")
    print("  • Run tests: python -m pytest tests/ -v")
    print("  • Run validation: python validation/validate_redshift_model.py")
    print("  • Read documentation: docs/THEORETICAL_BACKGROUND.md")
    print()
    
    return True

def test_performance_benchmark():
    """Quick performance benchmark"""
    print("Performance Benchmark:")
    print("-" * 30)
    
    try:
        from redshift_analyzer import EnhancedRedshiftAnalyzer
        
        analyzer = EnhancedRedshiftAnalyzer(enable_logging=False)
        
        # Single calculation benchmark
        start_time = time.time()
        for _ in range(100):
            dimming = analyzer.calculate_qvd_dimming(0.5)
        single_time = time.time() - start_time
        single_rate = 100 / single_time
        
        # Array calculation benchmark
        redshifts = np.linspace(0.01, 1.0, 1000)
        start_time = time.time()
        dimming_array = analyzer.calculate_qvd_dimming(redshifts)
        array_time = time.time() - start_time
        array_rate = len(redshifts) / array_time
        
        print(f"  Single calculations: {single_rate:.0f} calc/sec")
        print(f"  Array calculations: {array_rate:.0f} calc/sec")
        
        if single_rate > 1000 and array_rate > 5000:
            print("  ✅ Performance acceptable")
        else:
            print("  ⚠️  Performance below expected levels")
        
    except Exception as e:
        print(f"  ❌ Performance benchmark failed: {e}")

if __name__ == "__main__":
    success = verify_installation()
    
    if success:
        print()
        test_performance_benchmark()
        sys.exit(0)
    else:
        print("\n❌ Installation verification failed!")
        print("Please check the error messages above and ensure all dependencies are installed.")
        sys.exit(1)