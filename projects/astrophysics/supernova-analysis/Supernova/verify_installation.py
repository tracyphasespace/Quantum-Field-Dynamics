#!/usr/bin/env python3
"""
Installation Verification Script
===============================

Quick verification that the Supernova QVD model is properly installed
and working correctly.
"""

import sys
import numpy as np
import time

def verify_installation():
    """Verify that the installation is working correctly"""
    
    print("Supernova QVD Model - Installation Verification")
    print("=" * 50)
    print()
    
    # Test 1: Import all modules
    print("1. Testing imports...")
    try:
        from supernova_qvd_scattering import E144ExperimentalData, SupernovaParameters, E144ScaledQVDModel
        from numerical_safety import safe_power, safe_log10, validate_finite
        from physical_bounds import PhysicalBounds, BoundsEnforcer
        from error_handling import setup_qvd_logging
        print("   ✅ All imports successful")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Create model instance
    print("2. Testing model creation...")
    try:
        e144_data = E144ExperimentalData()
        sn_params = SupernovaParameters()
        model = E144ScaledQVDModel(e144_data, sn_params)
        print("   ✅ Model created successfully")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    # Test 3: Generate a simple curve
    print("3. Testing curve generation...")
    try:
        start_time = time.time()
        curve = model.generate_luminance_curve(
            distance_Mpc=100.0,
            wavelength_nm=500.0,
            time_range_days=(-10, 50),
            time_resolution_days=2.0
        )
        duration = time.time() - start_time
        print(f"   ✅ Curve generated in {duration:.2f}s")
    except Exception as e:
        print(f"   ❌ Curve generation failed: {e}")
        return False
    
    # Test 4: Verify numerical stability
    print("4. Testing numerical stability...")
    try:
        # Check that all values are finite
        all_finite = True
        for key, values in curve.items():
            if isinstance(values, np.ndarray):
                if not np.all(np.isfinite(values)):
                    all_finite = False
                    print(f"   ⚠️  Non-finite values in {key}")
        
        if all_finite:
            print("   ✅ All values are finite")
        else:
            print("   ❌ Some values are not finite")
            return False
            
    except Exception as e:
        print(f"   ❌ Stability check failed: {e}")
        return False
    
    # Test 5: Test safe mathematical operations
    print("5. Testing safe operations...")
    try:
        # Test operations that could cause issues
        result1 = safe_power(0.0, 2.0)  # Zero base
        result2 = safe_log10(0.0)       # Log of zero
        result3 = safe_power(2.0, 1000.0)  # Large exponent
        
        if np.all(np.isfinite([result1, result2, result3])):
            print("   ✅ Safe operations working correctly")
        else:
            print("   ❌ Safe operations produced non-finite results")
            return False
            
    except Exception as e:
        print(f"   ❌ Safe operations test failed: {e}")
        return False
    
    # Test 6: Test bounds enforcement
    print("6. Testing bounds enforcement...")
    try:
        bounds = PhysicalBounds()
        enforcer = BoundsEnforcer()
        
        # Test extreme values
        safe_density = enforcer.enforce_plasma_density(1e50)  # Too high
        safe_temp = enforcer.enforce_temperature(1.0)        # Too low
        
        if (bounds.MIN_PLASMA_DENSITY <= safe_density <= bounds.MAX_PLASMA_DENSITY and
            bounds.MIN_TEMPERATURE <= safe_temp <= bounds.MAX_TEMPERATURE):
            print("   ✅ Bounds enforcement working correctly")
        else:
            print("   ❌ Bounds enforcement not working properly")
            return False
            
    except Exception as e:
        print(f"   ❌ Bounds enforcement test failed: {e}")
        return False
    
    # Summary
    print()
    print("=" * 50)
    print("🎉 INSTALLATION VERIFICATION SUCCESSFUL!")
    print("=" * 50)
    print()
    print("The Supernova QVD model is properly installed and ready to use.")
    print()
    print("Next steps:")
    print("  • Run examples: python examples/basic_usage.py")
    print("  • Run tests: python tests/run_all_tests.py")
    print("  • Run validation: python validation/validate_e144_fixes.py")
    print("  • Read documentation: docs/README_E144_FIXES.md")
    print()
    
    return True

if __name__ == "__main__":
    success = verify_installation()
    if not success:
        print("\n❌ Installation verification failed!")
        print("Please check the error messages above and ensure all dependencies are installed.")
        sys.exit(1)
    else:
        sys.exit(0)