#!/usr/bin/env python3
"""
Complete G-2 Workflow Example
============================

Demonstrates the complete QFD Phoenix workflow for lepton g-2 analysis:

1. Phoenix solver ladder targeting (energy pinning)
2. CSR parameter sweeps  
3. Bundle generation for g-2 predictors
4. G-2 prediction batch processing
5. Consolidated analysis and reporting

This example shows how to use the refactored codebase for end-to-end
lepton physics analysis, from field simulation to g-2 predictions.

Usage:
    python complete_g2_workflow_example.py [--quick] [--device cuda]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path for the example
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from orchestration.g2_workflow import G2Workflow
from orchestration.ladder_solver import LadderSolver
from orchestration.g2_predictor_batch import G2PredictorBatch


def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def quick_demonstration():
    """
    Quick demonstration using small parameters for fast execution.
    Suitable for testing and demonstration purposes.
    """
    print("=" * 60)
    print("QUICK G-2 WORKFLOW DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path("example_g2_output_quick")
    
    # Initialize workflow with CPU for compatibility
    workflow = G2Workflow(
        output_base=output_dir,
        device="cpu",  # Use CPU for maximum compatibility
        backend="numpy",  # Use NumPy backend for reliability
        q_star=414.58  # Rounded for quick demo
    )
    
    print("Step 1: Testing electron ladder solver (small grid)")
    print("-" * 50)
    
    # Quick electron test with small parameters
    electron_ladder = LadderSolver(
        particle="electron",
        target_energy=500000.0,  # Close to 511 keV but easier target
        q_star=400.0,  # Simplified Q*
        max_iterations=3,  # Just a few iterations
        tolerance=0.1,  # Loose tolerance for quick demo
        output_dir=output_dir / "electron_quick"
    )
    
    try:
        print("Running electron ladder (this may take a moment)...")
        electron_results = electron_ladder.run_ladder()
        
        if electron_ladder.converged:
            print(f"✓ Electron converged: {electron_results['H_final']:.1f} eV")
        else:
            print(f"△ Electron partial result: {electron_results['H_final']:.1f} eV")
        
        print(f"  Iterations: {len(electron_ladder.iteration_history)}")
        print(f"  Final V2: {electron_ladder.physics['V2']:.1f}")
        
    except Exception as e:
        print(f"✗ Electron test failed: {e}")
        return False
    
    print()
    print("Step 2: Testing bundle creation")
    print("-" * 50)
    
    try:
        # Create a bundle manually for testing
        bundle = workflow._create_bundle(
            "electron_demo_v1",
            electron_results,
            {"k_csr": 0.0, "demo": True}
        )
        
        print(f"✓ Bundle created: {bundle}")
        
        # Check bundle contents
        manifest_files = list(bundle.glob("*manifest*.json"))
        if manifest_files:
            print(f"  Manifest: {manifest_files[0].name}")
        
        result_files = list(bundle.glob("*results*"))
        print(f"  Result files: {len(result_files)}")
        
    except Exception as e:
        print(f"✗ Bundle creation failed: {e}")
        return False
    
    print()
    print("Step 3: Testing G-2 batch processor (without actual predictor)")
    print("-" * 50)
    
    try:
        # Initialize batch processor
        g2_batch = G2PredictorBatch(
            device="cpu",
            workdir=output_dir,
            quiet=False
        )
        
        print(f"✓ G2PredictorBatch initialized")
        print(f"  References: {g2_batch.references}")
        print(f"  Work directory: {g2_batch.workdir}")
        
        # Test JSON parsing with mock data
        mock_prediction = {
            "prediction_v3_0": {
                "value": 0.001159652
            },
            "experimental_comparison": {
                "best_match": "electron",
                "best_relative_error": "0.05%"
            }
        }
        
        g2_value, rel_error, best_match = g2_batch._parse_prediction(mock_prediction)
        print(f"✓ JSON parsing test:")
        print(f"  g-2 value: {g2_value}")
        print(f"  Relative error: {rel_error}")
        print(f"  Best match: {best_match}")
        
    except Exception as e:
        print(f"✗ G-2 batch test failed: {e}")
        return False
    
    print()
    print("=" * 60)
    print("QUICK DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    print("This demonstrates the core workflow components:")
    print("1. ✓ Phoenix ladder solver")
    print("2. ✓ Bundle generation") 
    print("3. ✓ G-2 batch processing framework")
    print()
    print("For full production runs:")
    print("- Use larger grid sizes (64³ or higher)")
    print("- Enable GPU acceleration (--device cuda)")  
    print("- Use actual g-2 predictor scripts")
    print("- Run complete CSR parameter sweeps")
    
    return True


def full_production_example(device="cuda"):
    """
    Full production example with realistic parameters.
    This will take significantly longer but produces research-quality results.
    """
    print("=" * 60)
    print("FULL PRODUCTION G-2 WORKFLOW")
    print("=" * 60)
    print()
    print("⚠️  This will take several minutes to hours depending on hardware")
    print("⚠️  Requires GPU for reasonable performance")
    print()
    
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"production_g2_workflow_{timestamp}")
    
    # Initialize production workflow
    workflow = G2Workflow(
        output_base=output_dir,
        device=device,
        backend="torch",
        q_star=414.5848693847656  # Precise Q* from calibration
    )
    
    print("Step 1: Running complete electron workflow")
    print("-" * 50)
    
    try:
        # Full electron workflow with CSR sweep
        electron_bundles = workflow.run_electron_workflow(
            target_energy=511000.0,  # Precise electron rest mass
            csr_values=[0.0, 0.002, 0.005],  # Standard CSR values
            tolerance=0.01  # 1% tolerance
        )
        
        print(f"✓ Electron workflow completed: {len(electron_bundles)} bundles")
        for name, bundle in electron_bundles.items():
            print(f"  {name}: {bundle}")
        
    except Exception as e:
        print(f"✗ Electron workflow failed: {e}")
        return False
    
    print()
    print("Step 2: Running muon workflow")
    print("-" * 50)
    
    try:
        # Muon workflow
        muon_bundles = workflow.run_muon_workflow(
            target_energy=105658000.0,  # Precise muon rest mass
            tolerance=0.01
        )
        
        print(f"✓ Muon workflow completed: {len(muon_bundles)} bundles")
        for name, bundle in muon_bundles.items():
            print(f"  {name}: {bundle}")
        
    except Exception as e:
        print(f"✗ Muon workflow failed: {e}")
        # Continue anyway - muon is more challenging
        muon_bundles = {}
    
    print()
    print("Step 3: G-2 prediction analysis")
    print("-" * 50)
    
    # Collect all bundles
    all_bundles = []
    all_bundles.extend(electron_bundles.values())
    all_bundles.extend(muon_bundles.values())
    
    try:
        # Run g-2 analysis (will try to find predictor automatically)
        analysis_report = workflow.run_g2_analysis(
            bundle_paths=all_bundles,
            predictor_path=None  # Auto-detect
        )
        
        print(f"✓ G-2 analysis completed")
        print(f"  Report: {analysis_report}")
        
    except Exception as e:
        print(f"△ G-2 analysis failed (expected without predictor): {e}")
        print("  Bundles are ready for external g-2 predictor tools")
    
    print()
    print("=" * 60)
    print("PRODUCTION WORKFLOW COMPLETED!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Generated bundles: {len(all_bundles)}")
    print()
    print("Next steps:")
    print("1. Install/configure g-2 predictor from canonical/")
    print("2. Run: qfd-g2-batch --glob 'bundles/*' --device cuda")
    print("3. Analyze results in reports/ directory")
    
    return True


def main():
    """Main example runner."""
    parser = argparse.ArgumentParser(
        description="Complete G-2 workflow example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick demonstration (fast, small parameters)"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
        help="Computation device (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print(f"QFD Phoenix G-2 Workflow Example")
    print(f"Device: {args.device}")
    print()
    
    if args.quick:
        success = quick_demonstration()
    else:
        print("Running full production workflow...")
        print("Use --quick for a fast demonstration")
        print()
        response = input("Continue with full workflow? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("Canceled. Use --quick for a fast demonstration.")
            return 0
        
        success = full_production_example(args.device)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())