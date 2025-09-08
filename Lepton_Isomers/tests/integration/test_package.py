#!/usr/bin/env python3
"""
Quick integration test for QFD Phoenix Refactored
================================================

Tests core functionality to verify the refactored package works correctly.
"""

import sys
import time
from pathlib import Path

# Add src to path for testing
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import solvers
        from solvers.backend import get_backend
        from solvers.hamiltonian import PhoenixHamiltonian
        from solvers.phoenix_solver import solve_psi_field, load_particle_constants
        from orchestration.ladder_solver import LadderSolver
        from orchestration.g2_predictor_batch import G2PredictorBatch
        from orchestration.g2_workflow import G2Workflow
        from utils.io import save_results
        from utils.analysis import analyze_results
        print("SUCCESS: All imports successful")
        return True
    except ImportError as e:
        print(f"FAILED: Import failed: {e}")
        return False

def test_backend():
    """Test backend abstraction."""
    print("Testing backend abstraction...")
    
    try:
        from solvers.backend import get_backend
        
        # Test NumPy backend
        be_np = get_backend("numpy", "cpu")
        x = be_np.randn((4, 4, 4))
        print(f"SUCCESS: NumPy backend: {be_np.name}, shape: {x.shape}")
        
        # Test PyTorch backend if available
        try:
            be_torch = get_backend("torch", "cpu")
            y = be_torch.randn((4, 4, 4))
            print(f"SUCCESS: PyTorch backend: {be_torch.name}, shape: {y.shape}")
        except:
            print("INFO: PyTorch backend not available (optional)")
        
        return True
    except Exception as e:
        print(f"FAILED: Backend test failed: {e}")
        return False

def test_constants():
    """Test particle constants loading."""
    print("Testing particle constants...")
    
    try:
        from solvers.phoenix_solver import load_particle_constants
        
        for particle in ["electron", "muon", "tau"]:
            constants = load_particle_constants(particle)
            physics = constants['physics_constants']
            print(f"SUCCESS: {particle}: V2={physics['V2']}, V4={physics['V4']}, q*={physics['q_star']}")
        
        return True
    except Exception as e:
        print(f"FAILED: Constants test failed: {e}")
        return False

def test_physics():
    """Test core physics calculations."""
    print("Testing physics calculations...")
    
    try:
        from solvers.backend import get_backend
        from solvers.hamiltonian import PhoenixHamiltonian
        import numpy as np
        
        be = get_backend("numpy", "cpu")
        hamiltonian = PhoenixHamiltonian(8, 2.0, be, 585.0, 17.915, 0.985, 0.0)
        
        # Initialize test fields
        psi_s = be.randn((8, 8, 8)) * 0.1
        psi_b = be.zeros((8, 8, 8), dtype=np.complex64)
        
        # Test energy calculation
        energy = hamiltonian.compute_energy(psi_s, psi_b)
        print(f"SUCCESS: Initial energy: {energy:.3f} eV")
        
        # Test evolution
        psi_s_new, psi_b_new = hamiltonian.evolve(psi_s, psi_b, 1e-5)
        energy_new = hamiltonian.compute_energy(psi_s_new, psi_b_new)
        print(f"SUCCESS: Energy after evolution: {energy_new:.3f} eV")
        
        return True
    except Exception as e:
        print(f"FAILED: Physics test failed: {e}")
        return False

def test_solver():
    """Test full Phoenix solver."""
    print("Testing Phoenix solver...")
    
    try:
        from solvers.phoenix_solver import solve_psi_field
        
        results = solve_psi_field(
            particle="electron",
            grid_size=8,  # Small for quick test
            box_size=4.0,
            backend="numpy",
            device="cpu",
            steps=5,
            dt_auto=False,
            dt_max=1e-4
        )
        
        print(f"SUCCESS: Solver completed: E={results['H_final']:.1f} eV, shape={results['psi_field'].shape}")
        return True
    except Exception as e:
        print(f"FAILED: Solver test failed: {e}")
        return False

def test_orchestration():
    """Test ladder solver orchestration."""
    print("Testing ladder solver...")
    
    try:
        from orchestration.ladder_solver import LadderSolver
        
        ladder = LadderSolver(
            particle="electron",
            target_energy=511000.0,
            q_star=414.5848693847656,
            max_iterations=1,  # Just test initialization
            tolerance=0.5
        )
        
        print(f"SUCCESS: LadderSolver created: target={ladder.target_energy:.0f} eV, V2={ladder.physics['V2']}")
        return True
    except Exception as e:
        print(f"FAILED: Orchestration test failed: {e}")
        return False

def test_g2_integration():
    """Test g-2 workflow integration."""
    print("Testing g-2 workflow integration...")
    
    try:
        from orchestration.g2_predictor_batch import G2PredictorBatch
        from orchestration.g2_workflow import G2Workflow
        from pathlib import Path
        
        # Test G2PredictorBatch
        batch = G2PredictorBatch(quiet=True)
        print(f"SUCCESS: G2PredictorBatch: refs={len(batch.references)}")
        
        # Test G2Workflow
        workflow = G2Workflow(Path("test_g2"), device="cpu")
        print(f"SUCCESS: G2Workflow: q*={workflow.q_star:.1f}")
        
        return True
    except Exception as e:
        print(f"FAILED: G-2 integration test failed: {e}")
        return False

def test_zeeman_experiments():
    """Test QFD Zeeman experiment capabilities."""
    print("Testing Zeeman experiments...")
    
    try:
        from solvers.zeeman_experiments import ZeemanExperiment, IsomerZeemanAnalysis
        
        # Test basic Zeeman experiment
        zeeman = ZeemanExperiment("electron", "numpy", "cpu", grid_size=8, box_size=1.0)
        print(f"SUCCESS: ZeemanExperiment created: particle=electron")
        
        # Test isomer analysis framework
        analyzer = IsomerZeemanAnalysis("numpy", "cpu", grid_size=8, box_size=1.0)
        print(f"SUCCESS: IsomerZeemanAnalysis: experimental refs loaded")
        
        return True
    except Exception as e:
        print(f"FAILED: Zeeman experiments test failed: {e}")
        return False

def test_isomer_workflow():
    """Test revolutionary isomer workflow integration."""
    print("Testing isomer workflow...")
    
    try:
        from orchestration.isomer_workflow import IsomerWorkflow, run_complete_isomer_workflow
        from pathlib import Path
        
        # Test workflow initialization
        workflow = IsomerWorkflow(
            output_dir=Path("test_isomer_workflow"),
            backend="numpy",
            device="cpu",
            grid_size=8,
            precision_target=0.1
        )
        
        print(f"SUCCESS: IsomerWorkflow created: backend={workflow.backend}")
        print(f"         Target masses: electron={workflow.target_masses['electron']:.0f} eV")
        print(f"         Experimental g-2 refs: {len([k for k, v in workflow.experimental_g2.items() if v is not None])}/3")
        
        return True
    except Exception as e:
        print(f"FAILED: Isomer workflow test failed: {e}")
        return False

def test_stability_analysis():
    """Test stability analysis integration."""
    print("Testing stability analysis...")
    
    try:
        from utils.stability_analysis import StabilityPredictor
        from solvers.phoenix_solver import solve_psi_field
        
        # Test predictor initialization
        predictor = StabilityPredictor(
            tau_muon_exp=2.196981e-6,
            tau_tau_exp=2.903e-13
        )
        
        # Test with minimal simulation results
        mock_results = {
            'H_final': 100.0,
            'energy': 100.0,
            'psi_field': None,  # Will use fallback values
            'constants': {'physics_constants': {}}
        }
        
        # Test feature extraction
        features = predictor.extract_stability_features(mock_results)
        
        # Test geometric index computation
        G = predictor.compute_geometric_index(features, features)
        
        # Test CSR handle computation
        chi = predictor.compute_csr_handle(features)
        
        print(f"SUCCESS: StabilityPredictor created and tested")
        print(f"         Features extracted: {len(features)} properties")
        print(f"         Geometric index G = {G:.3f}, CSR handle chi = {chi:.6f}")
        
        return True
    except Exception as e:
        print(f"FAILED: Stability analysis test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("QFD Phoenix Refactored - Integration Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_backend, 
        test_constants,
        test_physics,
        test_solver,
        test_orchestration,
        test_g2_integration,
        test_zeeman_experiments,
        test_isomer_workflow,
        test_stability_analysis,
    ]
    
    passed = 0
    total = len(tests)
    
    start_time = time.time()
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"FAILED: Test {test.__name__} crashed: {e}")
            print()
    
    elapsed = time.time() - start_time
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed ({elapsed:.2f}s)")
    
    if passed == total:
        print("SUCCESS: ALL TESTS PASSED! Package is working correctly.")
        print()
        print("Theoretical framework components implemented:")
        print("- Phoenix Core Hamiltonian solver")
        print("- QFD Zeeman experiments for g-2 calculations")
        print("- Isomer theory workflow (electron/muon/tau unified)")
        print("- Three-objective paradigm framework")
        print()
        print("Ready for theoretical lepton physics research!")
        return 0
    else:
        print(f"WARNING:  {total-passed} tests failed. See errors above.")
        print("Some theoretical framework components may not be fully functional.")
        return 1

if __name__ == "__main__":
    sys.exit(main())