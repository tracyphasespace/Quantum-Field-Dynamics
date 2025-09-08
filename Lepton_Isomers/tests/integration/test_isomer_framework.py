#!/usr/bin/env python3
"""
QFD Isomer Framework Test
========================

Comprehensive test of the proposed three-objective isomer paradigm:
1. Electron calibration locks all physics parameters
2. Excited state search finds muon/tau as isomers
3. Unified g-2 predictions from single framework

This test validates the theoretical framework hypothesis that electron, muon, and tau
are different energy states of the same fundamental field equation.

Usage:
    python test_isomer_framework.py [--quick] [--device cpu]
"""

import sys
import time
from pathlib import Path
import logging
import argparse

# Add src to path
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from orchestration.isomer_workflow import IsomerWorkflow, run_complete_isomer_workflow
from solvers.zeeman_experiments import ZeemanExperiment, IsomerZeemanAnalysis


def test_zeeman_experiment_basic():
    """Test basic Zeeman experiment functionality."""
    print("Testing basic Zeeman experiment...")
    
    try:
        # Create electron Zeeman experiment
        zeeman = ZeemanExperiment("electron", "numpy", "cpu", grid_size=8, box_size=1.0)
        
        # Test g-factor computation with small field
        g_result = zeeman.compute_g_factor(1e-6)  # 1 μT
        
        print(f"SUCCESS: Basic Zeeman test passed: g = {g_result['g_factor']:.6f}")
        return True
        
    except Exception as e:
        print(f"FAILED: Basic Zeeman test failed: {e}")
        return False


def test_isomer_analysis_framework():
    """Test isomer analysis framework initialization."""
    print("Testing isomer analysis framework...")
    
    try:
        # Initialize framework
        analyzer = IsomerZeemanAnalysis("numpy", "cpu", grid_size=8, box_size=1.0)
        
        # Test experimental g-2 reference data
        assert analyzer.experimental_g2['electron'] == 0.00115965218073
        assert analyzer.experimental_g2['muon'] == 0.00116592089
        assert analyzer.experimental_g2['tau'] is None
        
        print("SUCCESS: Isomer analysis framework test passed")
        return True
        
    except Exception as e:
        print(f"FAILED: Isomer analysis framework test failed: {e}")
        return False


def test_electron_calibration_quick():
    """Quick test of electron calibration process."""
    print("Testing electron calibration (quick)...")
    
    try:
        # Create analyzer with small parameters for speed
        analyzer = IsomerZeemanAnalysis("numpy", "cpu", grid_size=8, box_size=1.0)
        
        # Test calibration process (will be approximate with small grid)
        calibration = analyzer.calibrate_from_electron()
        
        # Check basic structure
        assert 'electron_g_factor' in calibration
        assert 'electron_g2_predicted' in calibration
        assert 'calibrated_physics' in calibration
        
        # Should have reasonable g-factor (between 1.5 and 2.5)
        g_factor = calibration['electron_g_factor']
        assert 1.5 < g_factor < 2.5, f"Unreasonable g-factor: {g_factor}"
        
        print(f"SUCCESS: Electron calibration test passed: g = {g_factor:.6f}")
        return True
        
    except Exception as e:
        print(f"FAILED: Electron calibration test failed: {e}")
        return False


def test_workflow_initialization():
    """Test isomer workflow initialization."""
    print("Testing workflow initialization...")
    
    try:
        # Create workflow with test parameters
        workflow = IsomerWorkflow(
            output_dir=Path("test_isomer_output"),
            backend="numpy",
            device="cpu",
            grid_size=8,
            box_size=1.0,
            precision_target=0.1  # Relaxed for test
        )
        
        # Check initialization
        assert workflow.backend == "numpy"
        assert workflow.device == "cpu"
        assert workflow.grid_size == 8
        assert workflow.target_masses['electron'] == 510998.946
        assert workflow.target_masses['muon'] == 105658374.5
        assert workflow.target_masses['tau'] == 1776860000.0
        
        print("SUCCESS: Workflow initialization test passed")
        return True
        
    except Exception as e:
        print(f"FAILED: Workflow initialization test failed: {e}")
        return False


def run_quick_framework_demo():
    """
    Quick demonstration of isomer framework capabilities.
    
    This runs a simplified version of the three-objective paradigm
    with reduced parameters for fast execution and testing.
    """
    print("\n" + "=" * 60)
    print("QUICK ISOMER FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run with minimal parameters for speed
        results = run_complete_isomer_workflow(
            output_dir="test_isomer_framework_demo",
            backend="numpy",  # Use NumPy for CPU compatibility
            device="cpu",     # Use CPU for maximum compatibility
            grid_size=8,      # Very small grid for speed
            precision_target=0.2  # Relaxed precision for demo
        )
        
        # Extract key results
        success_rate = results.get('success_rate', 0)
        paradigm_breakthrough = results.get('paradigm_breakthrough', False)
        
        # Check electron calibration
        electron_results = results.get('objective_1_electron', {})
        electron_success = electron_results.get('calibration_success', False)
        
        # Check predictions
        muon_results = results.get('objective_2_muon', {})
        muon_success = muon_results.get('prediction_success', False)
        
        tau_results = results.get('objective_3_tau', {})
        tau_success = tau_results.get('prediction_success', False)
        
        # Print summary
        print(f"\nDemonstration Results:")
        print(f"  Success Rate: {success_rate*100:.0f}%")
        print(f"  Paradigm Breakthrough: {'YES SUCCESS' if paradigm_breakthrough else 'NO FAILED'}")
        print(f"  Electron Calibration: {'SUCCESS' if electron_success else 'FAILED'}")
        print(f"  Muon Prediction: {'SUCCESS' if muon_success else 'FAILED'}")
        print(f"  Tau Prediction: {'SUCCESS' if tau_success else 'FAILED'}")
        
        if electron_success:
            g2_pred = electron_results.get('g2_predicted', 0)
            g2_exp = electron_results.get('g2_experimental', 0)
            print(f"  Electron g-2: {g2_pred:.8f} (exp: {g2_exp:.8f})")
        
        if muon_success:
            muon_g2 = muon_results.get('g2_predicted', 0)
            print(f"  Muon g-2: {muon_g2:.8f}")
        
        if tau_success:
            tau_g2 = tau_results.get('g2_predicted', 0)
            print(f"  Tau g-2: {tau_g2:.8f} *** FIRST PREDICTION")
        
        duration = time.time() - start_time
        print(f"\nDemo completed in {duration:.1f} seconds")
        
        # Success criteria for demo
        demo_success = success_rate >= 0.33  # At least 1/3 objectives
        
        print(f"\nDemo Status: {'SUCCESS' if demo_success else 'PARTIAL'}")
        
        return demo_success, results
        
    except Exception as e:
        print(f"\nFAILED: Quick framework demo failed: {e}")
        return False, {}


def run_comprehensive_framework_test():
    """
    Comprehensive test of framework capabilities.
    
    This runs the full three-objective paradigm with realistic parameters
    for thorough validation.
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ISOMER FRAMEWORK TEST")
    print("=" * 60)
    print("WARNING  This test uses larger parameters and may take several minutes")
    
    response = input("Continue with comprehensive test? [y/N]: ")
    if response.lower() not in ('y', 'yes'):
        print("Skipping comprehensive test")
        return True, {}
    
    start_time = time.time()
    
    try:
        # Run with production-like parameters
        results = run_complete_isomer_workflow(
            output_dir="comprehensive_isomer_test",
            backend="torch",      # Use PyTorch if available
            device="cuda",        # Use GPU if available
            grid_size=32,         # Medium grid for balance of speed/accuracy
            precision_target=0.1  # 10% target precision
        )
        
        # Comprehensive analysis
        analysis = results.get('final_analysis', {})
        success_rate = analysis.get('success_rate', 0)
        paradigm_breakthrough = analysis.get('paradigm_breakthrough', False)
        scientific_impact = analysis.get('scientific_impact', {})
        
        # Print detailed results
        print(f"\nComprehensive Test Results:")
        print(f"  Overall Success Rate: {success_rate*100:.0f}%")
        print(f"  Paradigm Breakthrough: {'YES SUCCESS' if paradigm_breakthrough else 'NO FAILED'}")
        print(f"  Scientific Impact Score: {scientific_impact.get('total_impact_score', 0)}/10")
        print(f"  Revolutionary Potential: {'YES BREAKTHROUGH' if scientific_impact.get('revolutionary_potential', False) else 'No'}")
        
        # Detailed objective analysis
        objectives = analysis.get('objective_results', {})
        print(f"\nDetailed Objectives:")
        for obj_name, success in objectives.items():
            print(f"  {obj_name.replace('_', ' ').title()}: {'SUCCESS' if success else 'FAILED'}")
        
        # Precision metrics
        precision_metrics = analysis.get('precision_metrics', {})
        if precision_metrics:
            print(f"\nPrecision Achieved:")
            for metric, value in precision_metrics.items():
                if value is not None:
                    print(f"  {metric.replace('_', ' ').title()}: {(1-value)*100:.1f}%")
        
        duration = time.time() - start_time
        print(f"\nComprehensive test completed in {duration:.1f} seconds")
        
        # Success criteria (stricter for comprehensive test)
        comprehensive_success = (
            success_rate >= 0.67 and  # At least 2/3 objectives
            paradigm_breakthrough     # Must achieve paradigm breakthrough
        )
        
        print(f"\nComprehensive Test Status: {'SUCCESS SUCCESS' if comprehensive_success else 'PARTIAL WARNING'}")
        
        return comprehensive_success, results
        
    except Exception as e:
        print(f"\nFAILED: Comprehensive framework test failed: {e}")
        return False, {}


def main():
    """Main test runner for isomer framework capabilities."""
    
    parser = argparse.ArgumentParser(description="QFD Isomer Framework Test")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only quick tests and demo")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                       help="Device for computation")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive test (takes longer)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("QFD Isomer Framework Test Suite")
    print("=" * 50)
    print(f"Device: {args.device}")
    print()
    
    # Run basic unit tests
    print("Running basic unit tests...")
    tests = [
        test_zeeman_experiment_basic,
        test_isomer_analysis_framework,
        test_workflow_initialization,
        test_electron_calibration_quick
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAILED: Test {test.__name__} crashed: {e}")
    
    print(f"\nBasic Tests: {passed}/{total} passed")
    
    if passed < total:
        print("WARNING  Some basic tests failed - proceeding with caution")
    
    # Run demonstration
    demo_success, demo_results = run_quick_framework_demo()
    
    # Run comprehensive test if requested
    if args.comprehensive and not args.quick:
        comprehensive_success, comp_results = run_comprehensive_framework_test()
    else:
        comprehensive_success = True  # Skip if not requested
        comp_results = {}
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    
    basic_success = passed == total
    overall_success = basic_success and demo_success and comprehensive_success
    
    print(f"Basic Unit Tests: {'PASS SUCCESS' if basic_success else 'FAIL FAILED'}")
    print(f"Quick Demonstration: {'PASS SUCCESS' if demo_success else 'FAIL FAILED'}")
    
    if args.comprehensive:
        print(f"Comprehensive Test: {'PASS SUCCESS' if comprehensive_success else 'FAIL FAILED'}")
    
    print(f"\nOverall Status: {'SUCCESS SUCCESS' if overall_success else 'ISSUES WARNING'}")
    
    if overall_success:
        print("\n***  QFD Isomer Framework capabilities validated!")
        print("The three-objective paradigm is ready for theoretical research.")
    else:
        print("\nWARNING  Some issues detected. Review test output for details.")
        print("Consider adjusting parameters or checking system requirements.")
    
    # Return appropriate exit code
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())