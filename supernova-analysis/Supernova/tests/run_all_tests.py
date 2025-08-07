#!/usr/bin/env python3
"""
Comprehensive Test Runner for Supernova QVD Model
================================================

Runs all tests and generates a comprehensive report.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_test_suite():
    """Run the complete test suite"""
    
    print("="*80)
    print("SUPERNOVA QVD MODEL - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # List of test modules to run
    test_modules = [
        "tests.test_numerical_safety",
        "tests.test_physical_bounds", 
        "tests.test_error_handling",
        "tests.test_e144_model_integration",
        "tests.test_regression_comparison"
    ]
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    for module in test_modules:
        print(f"Running {module}...")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Run the test module
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                module.replace(".", "/") + ".py", 
                "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=300)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {module} PASSED ({duration:.1f}s)")
                results[module] = {"status": "PASSED", "duration": duration}
            else:
                print(f"❌ {module} FAILED ({duration:.1f}s)")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                results[module] = {"status": "FAILED", "duration": duration, 
                                 "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {module} TIMEOUT (>300s)")
            results[module] = {"status": "TIMEOUT", "duration": 300}
            
        except Exception as e:
            print(f"💥 {module} ERROR: {e}")
            results[module] = {"status": "ERROR", "duration": 0, "error": str(e)}
        
        print()
    
    # Run validation script
    print("Running comprehensive validation...")
    print("-" * 60)
    
    validation_start = time.time()
    try:
        result = subprocess.run([
            sys.executable, "validation/validate_e144_fixes.py"
        ], capture_output=True, text=True, timeout=600)
        
        validation_duration = time.time() - validation_start
        
        if result.returncode == 0:
            print(f"✅ VALIDATION PASSED ({validation_duration:.1f}s)")
            results["validation"] = {"status": "PASSED", "duration": validation_duration}
        else:
            print(f"❌ VALIDATION FAILED ({validation_duration:.1f}s)")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            results["validation"] = {"status": "FAILED", "duration": validation_duration,
                                   "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        print("⏰ VALIDATION TIMEOUT (>600s)")
        results["validation"] = {"status": "TIMEOUT", "duration": 600}
        
    except Exception as e:
        print(f"💥 VALIDATION ERROR: {e}")
        results["validation"] = {"status": "ERROR", "duration": 0, "error": str(e)}
    
    # Generate summary report
    total_duration = time.time() - total_start_time
    
    print("="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r["status"] == "PASSED")
    failed = sum(1 for r in results.values() if r["status"] == "FAILED")
    errors = sum(1 for r in results.values() if r["status"] in ["ERROR", "TIMEOUT"])
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Errors/Timeouts: {errors} 💥")
    print(f"Success Rate: {100*passed/total:.1f}%")
    print(f"Total Duration: {total_duration:.1f}s")
    print()
    
    # Detailed results
    print("DETAILED RESULTS:")
    print("-" * 40)
    for module, result in results.items():
        status_icon = {"PASSED": "✅", "FAILED": "❌", "ERROR": "💥", "TIMEOUT": "⏰"}
        icon = status_icon.get(result["status"], "❓")
        print(f"{icon} {module}: {result['status']} ({result['duration']:.1f}s)")
        if "error" in result:
            print(f"    Error: {result['error'][:100]}...")
    
    print()
    
    # Overall result
    if passed == total:
        print("🎉 ALL TESTS PASSED! The Supernova QVD model is ready for production.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED. Please review the results above.")
        return False

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)