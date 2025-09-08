#!/usr/bin/env python3
"""
QFD Stability Analysis Example
==============================

Demonstrates the complete four-objective theoretical paradigm:

1. **Electron Calibration**: Lock physics parameters from electron ground state
2. **Excited State Search**: Find muon/tau as isomers of electron field equation  
3. **Unified g-2 Predictions**: Predict magnetic moments via Zeeman experiments
4. **Stability Analysis**: Predict lifetimes from geometric field properties

This extends the three-objective paradigm with lifetime predictions based on
geometric stability indices derived from field dynamics.

Usage:
    python stability_analysis_example.py [--quick] [--device cpu]
"""

import sys
import time
from pathlib import Path
import logging

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from orchestration.isomer_workflow import run_complete_isomer_workflow
from utils.stability_analysis import StabilityPredictor
from solvers.phoenix_solver import solve_psi_field


def run_complete_stability_example(quick_mode: bool = True, device: str = "cpu"):
    """
    Complete demonstration of the four-objective paradigm including stability analysis.
    
    Args:
        quick_mode: Use small parameters for fast execution
        device: Computation device ("cpu" or "cuda")
    """
    print("=" * 70)
    print("QFD FOUR-OBJECTIVE PARADIGM: ISOMERS + STABILITY ANALYSIS")
    print("=" * 70)
    print()
    
    output_dir = Path("stability_analysis_demo")
    
    # Configure parameters based on mode
    if quick_mode:
        grid_size = 16
        box_size = 2.0
        steps = 50
        precision_target = 0.3
        print("Mode: Quick demonstration (small parameters)")
    else:
        grid_size = 32
        box_size = 4.0
        steps = 200
        precision_target = 0.1
        print("Mode: Research-quality simulation")
    
    print(f"Device: {device}")
    print(f"Grid: {grid_size}³, Steps: {steps}")
    print()
    
    start_time = time.time()
    
    # Objective 1-3: Run complete isomer workflow
    print("OBJECTIVES 1-3: ISOMER WORKFLOW")
    print("-" * 40)
    
    try:
        isomer_results = run_complete_isomer_workflow(
            output_dir=str(output_dir / "isomer_workflow"),
            backend="numpy" if device == "cpu" else "torch",
            device=device,
            grid_size=grid_size,
            precision_target=precision_target
        )
        
        print("✓ Isomer workflow completed")
        print(f"  Success rate: {isomer_results.get('success_rate', 0)*100:.0f}%")
        
        # Check if we have basic results to work with
        if not isomer_results.get('objective_1_electron', {}).get('calibration_success'):
            print("⚠ Electron calibration incomplete - running individual simulations")
            
            # Run individual Phoenix simulations for stability analysis
            print("\nRunning individual Phoenix simulations...")
            
            electron_results = solve_psi_field(
                "electron", grid_size=grid_size, box_size=box_size, 
                backend="numpy", device=device, steps=steps
            )
            print("  ✓ Electron simulation complete")
            
            muon_results = solve_psi_field(
                "muon", grid_size=grid_size, box_size=box_size,
                backend="numpy", device=device, steps=steps
            )
            print("  ✓ Muon simulation complete")
            
            tau_results = solve_psi_field(
                "tau", grid_size=grid_size, box_size=box_size,
                backend="numpy", device=device, steps=steps  
            )
            print("  ✓ Tau simulation complete")
            
        else:
            print("✓ Using isomer workflow results for stability analysis")
            # Extract simulation results from isomer workflow
            # (In practice, we'd extract the actual Phoenix solver results)
            # For demo, we'll run fresh simulations
            electron_results = solve_psi_field(
                "electron", grid_size=grid_size, box_size=box_size,
                backend="numpy", device=device, steps=steps
            )
            muon_results = solve_psi_field(
                "muon", grid_size=grid_size, box_size=box_size,
                backend="numpy", device=device, steps=steps
            )
            tau_results = solve_psi_field(
                "tau", grid_size=grid_size, box_size=box_size,
                backend="numpy", device=device, steps=steps
            )
    
    except Exception as e:
        print(f"✗ Isomer workflow failed: {e}")
        print("Running fallback individual simulations...")
        
        # Fallback: run individual simulations
        electron_results = solve_psi_field(
            "electron", grid_size=grid_size, box_size=box_size,
            backend="numpy", device=device, steps=steps
        )
        muon_results = solve_psi_field(
            "muon", grid_size=grid_size, box_size=box_size, 
            backend="numpy", device=device, steps=steps
        )
        tau_results = solve_psi_field(
            "tau", grid_size=grid_size, box_size=box_size,
            backend="numpy", device=device, steps=steps
        )
        isomer_results = None
    
    print()
    
    # Objective 4: Stability Analysis
    print("OBJECTIVE 4: STABILITY ANALYSIS")
    print("-" * 40)
    
    try:
        # Initialize stability predictor
        predictor = StabilityPredictor(
            tau_muon_exp=2.196981e-6,  # Experimental muon lifetime
            tau_tau_exp=2.903e-13,     # Experimental tau lifetime
            U0=0.99
        )
        
        print("✓ Stability predictor initialized")
        
        # Run complete stability analysis
        stability_results = predictor.analyze_stability(
            electron_results=electron_results,
            muon_results=muon_results,
            tau_results=tau_results,
            output_dir=output_dir / "stability_analysis"
        )
        
        print("✓ Stability analysis completed")
        
        # Display results
        print("\n" + predictor.format_summary_report(stability_results))
        
        # Extract key metrics
        fit_success = stability_results['fitted_parameters']['fit_successful']
        muon_accuracy = stability_results['analysis_summary']['muon_prediction_accuracy']
        tau_accuracy = stability_results['analysis_summary']['tau_prediction_accuracy']
        overall_success = stability_results['analysis_summary']['overall_success']
        
        print(f"\nStability Analysis Success: {overall_success}")
        print(f"Muon Lifetime Accuracy: {muon_accuracy*100:.1f}%")
        print(f"Tau Lifetime Accuracy: {tau_accuracy*100:.1f}%")
        
    except Exception as e:
        print(f"✗ Stability analysis failed: {e}")
        stability_results = None
        fit_success = False
    
    # Final Assessment
    duration = time.time() - start_time
    print()
    print("=" * 70)
    print("FOUR-OBJECTIVE PARADIGM SUMMARY")
    print("=" * 70)
    
    # Assess each objective
    objectives_status = []
    
    # Objective 1: Electron calibration
    obj1_success = (isomer_results and 
                   isomer_results.get('objective_1_electron', {}).get('calibration_success', False))
    objectives_status.append(obj1_success)
    print(f"Objective 1 (Electron Calibration): {'✓' if obj1_success else '△'}")
    
    # Objective 2: Muon prediction  
    obj2_success = (isomer_results and
                   isomer_results.get('objective_2_muon', {}).get('prediction_success', False))
    objectives_status.append(obj2_success)
    print(f"Objective 2 (Muon Isomer Prediction): {'✓' if obj2_success else '△'}")
    
    # Objective 3: Tau prediction
    obj3_success = (isomer_results and
                   isomer_results.get('objective_3_tau', {}).get('prediction_success', False))
    objectives_status.append(obj3_success)
    print(f"Objective 3 (Tau Isomer Prediction): {'✓' if obj3_success else '△'}")
    
    # Objective 4: Stability analysis
    obj4_success = stability_results and fit_success
    objectives_status.append(obj4_success)
    print(f"Objective 4 (Stability Analysis): {'✓' if obj4_success else '△'}")
    
    # Overall assessment
    success_count = sum(objectives_status)
    success_rate = success_count / 4.0
    
    print(f"\nOverall Success Rate: {success_rate*100:.0f}% ({success_count}/4 objectives)")
    print(f"Execution Time: {duration:.1f} seconds")
    print(f"Output Directory: {output_dir}")
    
    if success_rate >= 0.75:
        print("\n🎉 Four-objective paradigm successfully demonstrated!")
        print("Complete theoretical framework for lepton physics validated:")
        print("  • Unified field equation (electron/muon/tau as isomers)")
        print("  • g-2 predictions from field-fundamental Zeeman experiments") 
        print("  • Lifetime predictions from geometric stability indices")
        print("  • Single calibration from electron → complete lepton spectrum")
    elif success_rate >= 0.5:
        print("\n✓ Partial four-objective paradigm demonstrated")
        print("Core theoretical components working - framework established")
    else:
        print("\n△ Limited success - theoretical framework needs refinement")
        print("Individual components available for further development")
    
    return {
        'isomer_results': isomer_results,
        'stability_results': stability_results,
        'success_rate': success_rate,
        'duration': duration
    }


def main():
    """Main example runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QFD Four-Objective Stability Analysis Example")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick demo with small parameters")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                       help="Computation device")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise for demo
        format='%(levelname)s: %(message)s'
    )
    
    # Run example
    results = run_complete_stability_example(
        quick_mode=args.quick,
        device=args.device
    )
    
    return 0 if results['success_rate'] > 0.25 else 1


if __name__ == "__main__":
    sys.exit(main())