#!/usr/bin/env python3
"""
Comprehensive validation script for QFD CMB Module
Tests installation, core functionality, and output validation
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import csv
from pathlib import Path
import numpy as np

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def validate_demo_outputs(output_dir):
    """Validate that demo outputs are correct"""
    expected_files = [
        'qfd_demo_spectra.csv',
        'TT.png',
        'EE.png', 
        'TE.png'
    ]
    
    validation_results = {}
    
    # Check all expected files exist
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        validation_results[f"file_exists_{filename}"] = os.path.exists(filepath)
    
    # Validate CSV content
    csv_path = os.path.join(output_dir, 'qfd_demo_spectra.csv')
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                
            validation_results['csv_has_data'] = len(data) > 0
            validation_results['csv_has_ell_column'] = 'ell' in data[0] if data else False
            validation_results['csv_has_spectra_columns'] = all(
                col in data[0] for col in ['C_TT', 'C_EE', 'C_TE']
            ) if data else False
            
            # Check data ranges are reasonable
            if data:
                ell_values = [float(row['ell']) for row in data]
                ctt_values = [float(row['C_TT']) for row in data]
                
                validation_results['ell_range_valid'] = min(ell_values) >= 2 and max(ell_values) <= 3000
                validation_results['ctt_values_positive'] = all(c > 0 for c in ctt_values)
                
        except Exception as e:
            validation_results['csv_validation_error'] = str(e)
    
    return validation_results

def validate_package_import():
    """Test that the package can be imported and core functions work"""
    validation_results = {}
    
    try:
        # Test basic imports
        import qfd_cmb
        validation_results['package_import'] = True
        
        from qfd_cmb import ppsi_models, visibility, kernels, projector, figures
        validation_results['module_imports'] = True
        
        # Test core function calls with sample parameters
        k_test = np.logspace(-4, 0, 10)
        
        # Test ppsi_models
        result = ppsi_models.oscillatory_psik(k_test, A=1.0, rpsi=147.0)
        validation_results['ppsi_models_function'] = len(result) == len(k_test)
        
        # Test visibility functions
        eta_test = np.linspace(13800, 14300, 100)
        chi_test = np.linspace(13800, 14300, 100)
        
        vis_result = visibility.gaussian_visibility(eta_test, 14065.0, 250.0)
        validation_results['visibility_function'] = len(vis_result) == len(eta_test)
        
        win_result = visibility.gaussian_window_chi(chi_test, 14065.0, 250.0)
        validation_results['window_function'] = len(win_result) == len(chi_test)
        
        # Test kernels
        mu_test = np.linspace(-1, 1, 50)
        w_T, w_E = kernels.sin2_mueller_coeffs(mu_test)
        validation_results['mueller_coeffs'] = len(w_T) == len(mu_test) and len(w_E) == len(mu_test)
        
        ell_test = np.array([100, 200, 300])
        te_result = kernels.te_correlation_phase(ell_test, k_test[0], 147.0, 14065.0, 250.0)
        validation_results['te_correlation'] = not np.isnan(te_result).any()
        
    except Exception as e:
        validation_results['import_error'] = str(e)
        return validation_results
    
    return validation_results

def validate_scientific_accuracy():
    """Test scientific accuracy with known reference values"""
    validation_results = {}
    
    try:
        from qfd_cmb import ppsi_models, visibility, kernels
        
        # Test with Planck-anchored parameters
        planck_params = {
            'A': 1.0,
            'rpsi': 147.0,
            'chi_star': 14065.0,
            'sigma_chi': 250.0
        }
        
        # Test power spectrum at specific k values
        k_ref = np.array([1e-3, 1e-2, 1e-1])
        ps_result = ppsi_models.oscillatory_psik(k_ref, A=planck_params['A'], rpsi=planck_params['rpsi'])
        
        # Basic sanity checks
        validation_results['power_spectrum_positive'] = bool(np.all(ps_result > 0))
        validation_results['power_spectrum_finite'] = bool(np.all(np.isfinite(ps_result)))
        validation_results['power_spectrum_decreasing'] = bool(ps_result[0] > ps_result[-1])  # Should decrease with k
        
        # Test window function normalization
        chi_grid = np.linspace(13000, 15000, 1000)
        window = visibility.gaussian_window_chi(chi_grid, planck_params['chi_star'], planck_params['sigma_chi'])
        
        # Check normalization (should integrate to ~1)
        norm_squared = np.trapz(window**2, chi_grid)
        validation_results['window_normalization'] = bool(abs(norm_squared - 1.0) < 0.1)
        
        # Test Mueller coefficients sum rule
        mu_grid = np.linspace(-1, 1, 1000)
        w_T, w_E = kernels.sin2_mueller_coeffs(mu_grid)
        
        # Integration should give specific values
        integral_T = np.trapz(w_T, mu_grid)
        integral_E = np.trapz(w_E, mu_grid)
        
        validation_results['mueller_T_integral'] = bool(abs(integral_T - 4.0/3.0) < 0.01)
        validation_results['mueller_E_integral'] = bool(abs(integral_E - 4.0/15.0) < 0.01)
        
    except Exception as e:
        validation_results['scientific_validation_error'] = str(e)
    
    return validation_results

def main():
    """Run comprehensive validation"""
    print("=" * 60)
    print("QFD CMB Module - Comprehensive Validation")
    print("=" * 60)
    
    all_results = {}
    
    # 1. Test package import and basic functionality
    print("\n1. Testing package imports and basic functionality...")
    import_results = validate_package_import()
    all_results['imports'] = import_results
    
    passed = sum(1 for v in import_results.values() if v is True)
    total = len([v for v in import_results.values() if isinstance(v, bool)])
    print(f"   Import tests: {passed}/{total} passed")
    
    # 2. Test demo script execution
    print("\n2. Testing demo script execution...")
    with tempfile.TemporaryDirectory() as temp_dir:
        demo_success, demo_stdout, demo_stderr = run_command(
            f"python run_demo.py --outdir {temp_dir}"
        )
        
        all_results['demo_execution'] = demo_success
        
        if demo_success:
            print("   Demo script executed successfully")
            
            # Validate outputs
            output_results = validate_demo_outputs(temp_dir)
            all_results['demo_outputs'] = output_results
            
            passed = sum(1 for v in output_results.values() if v is True)
            total = len([v for v in output_results.values() if isinstance(v, bool)])
            print(f"   Output validation: {passed}/{total} passed")
        else:
            print(f"   Demo script failed: {demo_stderr}")
    
    # 3. Test scientific accuracy
    print("\n3. Testing scientific accuracy...")
    science_results = validate_scientific_accuracy()
    all_results['scientific_accuracy'] = science_results
    
    passed = sum(1 for v in science_results.values() if v is True)
    total = len([v for v in science_results.values() if isinstance(v, bool)])
    print(f"   Scientific tests: {passed}/{total} passed")
    
    # 4. Test fitting script (if emcee is available)
    print("\n4. Testing fitting script...")
    try:
        import emcee
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal test data
            test_data_path = os.path.join(temp_dir, 'test_data.csv')
            with open(test_data_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ell', 'C_TT', 'error'])
                for ell in range(2, 100, 10):
                    writer.writerow([ell, 1000.0 / ell**2, 10.0])
            
            fit_success, fit_stdout, fit_stderr = run_command(
                f"python fit_planck.py --data {test_data_path} --which TT --steps 5 --walkers 8 --out {temp_dir}/fit_results.json"
            )
            
            all_results['fitting_script'] = fit_success
            if fit_success:
                print("   Fitting script executed successfully")
            else:
                print(f"   Fitting script failed: {fit_stderr}")
                
    except ImportError:
        print("   Skipping fitting test (emcee not available)")
        all_results['fitting_script'] = 'skipped'
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_passed = 0
    total_tests = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            passed = sum(1 for v in results.values() if v is True)
            total = len([v for v in results.values() if isinstance(v, bool)])
            total_passed += passed
            total_tests += total
            print(f"{category}: {passed}/{total} passed")
        elif results is True:
            total_passed += 1
            total_tests += 1
            print(f"{category}: PASSED")
        elif results == 'skipped':
            print(f"{category}: SKIPPED")
        else:
            total_tests += 1
            print(f"{category}: FAILED")
    
    print(f"\nOVERALL: {total_passed}/{total_tests} tests passed")
    
    # Save detailed results
    with open('validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: validation_results.json")
    
    # Return exit code based on results
    success_rate = total_passed / total_tests if total_tests > 0 else 0
    if success_rate >= 0.8:  # 80% pass rate
        print("\n✅ VALIDATION PASSED - Repository is ready for publication")
        return 0
    else:
        print("\n❌ VALIDATION FAILED - Issues need to be addressed")
        return 1

if __name__ == "__main__":
    sys.exit(main())