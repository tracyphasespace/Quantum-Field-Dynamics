"""
Ladder Solver - Energy Targeting Orchestration
============================================

Python implementation of the PowerShell ladder workflow for pinning
particle energies to their rest mass targets using Q* sensitivity.

Based on run_free_electron_ladder.ps1 from canonical implementation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ..solvers.phoenix_solver import solve_psi_field, load_particle_constants
    from ..utils.io import save_results
    from ..utils.analysis import analyze_results
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    
    from solvers.phoenix_solver import solve_psi_field, load_particle_constants
    from utils.io import save_results
    from utils.analysis import analyze_results


class LadderSolver:
    """Energy ladder solver using Q* sensitivity for rapid convergence."""
    
    def __init__(
        self,
        particle: str,
        target_energy: float,
        q_star: float,
        max_iterations: int = 12,
        tolerance: float = 0.01,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize ladder solver.
        
        Args:
            particle: Particle name ('electron', 'muon', 'tau')
            target_energy: Target rest mass energy in eV
            q_star: Energy sensitivity parameter
            max_iterations: Maximum ladder steps
            tolerance: Relative energy tolerance for convergence
            output_dir: Output directory for results
        """
        self.particle = particle
        self.target_energy = target_energy
        self.q_star = q_star
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.output_dir = output_dir or Path("data/output") / particle
        
        # Load particle constants
        self.constants = load_particle_constants(particle)
        self.physics = self.constants['physics_constants'].copy()
        
        # Initialize solver parameters
        self.grid_size = 64
        self.box_size = 16.0
        self.steps = 700
        self.dt_min = 5e-5
        self.dt_max = 5e-4
        self.backend = "torch"
        self.device = "cuda"
        
        # Tracking
        self.iteration_history = []
        self.converged = False
        
        logging.info(f"Initialized {particle} ladder solver:")
        logging.info(f"  Target energy: {target_energy:.1f} eV")
        logging.info(f"  Q* sensitivity: {q_star:.6f}")
        logging.info(f"  Tolerance: {tolerance*100:.1f}%")
    
    def compute_v2_update(self, current_energy: float) -> float:
        """
        Compute V2 update using Q* sensitivity.
        
        Formula: ΔV2 = (E_target - E_current) / Q*
        
        Args:
            current_energy: Current simulation energy
            
        Returns:
            V2 update value
        """
        delta_energy = self.target_energy - current_energy
        delta_v2 = delta_energy / self.q_star
        
        # Apply safety cap that grows with iteration
        iteration = len(self.iteration_history) + 1
        max_cap = 1000 * (2 ** min(iteration - 1, 5))  # 1k, 2k, 4k, 8k, 16k, 32k
        
        # Apply 25% damping
        damped_delta = 0.25 * delta_v2
        
        # Apply cap
        if damped_delta > max_cap:
            damped_delta = max_cap
        elif damped_delta < -max_cap:
            damped_delta = -max_cap
            
        return damped_delta
    
    def adaptive_dt_max(self, iteration: int, v2: float) -> float:
        """Adapt time step based on iteration and V2 magnitude."""
        if v2 < 1e7:
            return 5e-5
        elif v2 < 3e7:
            return 2e-5
        else:
            return 1e-5
    
    def run_ladder(self, initial_v2: Optional[float] = None) -> Dict:
        """
        Run the ladder solver to pin energy to target.
        
        Args:
            initial_v2: Initial V2 value (uses constant default if None)
            
        Returns:
            Final solver results
        """
        # Initialize V2
        if initial_v2 is not None:
            self.physics['V2'] = initial_v2
        
        current_v2 = self.physics['V2']
        warm_start_path = None
        
        logging.info(f"Starting {self.particle} ladder solver...")
        logging.info(f"Initial V2: {current_v2}")
        
        for iteration in range(1, self.max_iterations + 1):
            # Adapt time step
            self.dt_max = self.adaptive_dt_max(iteration, current_v2)
            
            # Setup output directory
            run_dir = self.output_dir / f"ladder_{iteration:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"\\n=== Ladder Step {iteration:02d} ===")
            logging.info(f"V2: {current_v2:.2f}")
            logging.info(f"dt_max: {self.dt_max}")
            logging.info(f"warm_start: {warm_start_path}")
            
            try:
                # Update physics constants
                self.physics['V2'] = current_v2
                
                # Run simulation
                results = solve_psi_field(
                    particle=self.particle,
                    grid_size=self.grid_size,
                    box_size=self.box_size,
                    backend=self.backend,
                    device=self.device,
                    steps=self.steps,
                    dt_auto=True,
                    dt_min=self.dt_min,
                    dt_max=self.dt_max,
                    V2=current_v2,
                    V4=self.physics['V4'],
                    g_c=self.physics['g_c'],
                    k_csr=self.physics['k_csr'],
                    q_star=self.q_star,
                    init_from=warm_start_path
                )
                
                # Extract energy
                current_energy = results['H_final']
                
                # Save results
                save_results(results, run_dir / "results")
                
                # Log progress
                logging.info(f"Energy: {current_energy:.4f} eV")
                
                # Track iteration
                iteration_data = {
                    'iteration': iteration,
                    'V2': current_v2,
                    'energy': current_energy,
                    'dt_max': self.dt_max,
                    'converged': False
                }
                self.iteration_history.append(iteration_data)
                
                # Check convergence
                relative_error = abs(current_energy - self.target_energy) / self.target_energy
                if relative_error <= self.tolerance:
                    logging.info(f"\\n*** CONVERGED ***")
                    logging.info(f"Target reached within {self.tolerance*100:.1f}%")
                    logging.info(f"Final energy: {current_energy:.4f} eV")
                    
                    iteration_data['converged'] = True
                    self.converged = True
                    
                    # Save final state
                    final_results = {
                        'converged': True,
                        'final_energy': current_energy,
                        'final_V2': current_v2,
                        'target_energy': self.target_energy,
                        'iterations': len(self.iteration_history),
                        'tolerance_achieved': relative_error,
                        'history': self.iteration_history
                    }
                    
                    with open(self.output_dir / "ladder_summary.json", 'w') as f:
                        json.dump(final_results, f, indent=2)
                    
                    return results
                
                # Compute next V2 update
                delta_v2 = self.compute_v2_update(current_energy)
                current_v2 += delta_v2
                
                logging.info(f"Delta V2: {delta_v2:.2f}")
                logging.info(f"Next V2: {current_v2:.2f}")
                
                # Update warm start path for next iteration
                # warm_start_path = run_dir / "fields_final.pt"  # Would need field saving
                
            except Exception as e:
                logging.error(f"Simulation failed: {e}")
                logging.info("Reducing dt_max and stiffening V4, retrying...")
                
                self.dt_max = max(self.dt_min, self.dt_max / 2.0)
                self.physics['V4'] += 0.5
                iteration -= 1  # Retry same rung
                continue
        
        # Did not converge
        logging.warning(f"Failed to converge after {self.max_iterations} iterations")
        
        final_results = {
            'converged': False,
            'final_energy': current_energy if 'current_energy' in locals() else None,
            'final_V2': current_v2,
            'target_energy': self.target_energy,
            'iterations': len(self.iteration_history),
            'history': self.iteration_history
        }
        
        with open(self.output_dir / "ladder_summary.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return results if 'results' in locals() else None


def run_electron_ladder(
    output_dir: Optional[Path] = None,
    q_star: float = 414.5848693847656
) -> Dict:
    """
    Run electron energy ladder to 511 keV target.
    
    Args:
        output_dir: Output directory
        q_star: Q* sensitivity parameter
        
    Returns:
        Final simulation results
    """
    target_energy = 511000.0  # eV
    
    ladder = LadderSolver(
        particle="electron",
        target_energy=target_energy,
        q_star=q_star,
        output_dir=output_dir
    )
    
    return ladder.run_ladder()


def run_muon_ladder(
    output_dir: Optional[Path] = None,
    q_star: float = 414.5848693847656,
    initial_v2: float = 500000.0
) -> Dict:
    """
    Run muon energy ladder to 105.658 MeV target.
    
    Args:
        output_dir: Output directory
        q_star: Q* sensitivity parameter
        initial_v2: Starting V2 value for muon regime
        
    Returns:
        Final simulation results
    """
    target_energy = 105658000.0  # eV
    
    ladder = LadderSolver(
        particle="muon",
        target_energy=target_energy,
        q_star=q_star,
        max_iterations=15,  # May need more iterations for muon
        output_dir=output_dir
    )
    
    return ladder.run_ladder(initial_v2=initial_v2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import argparse
    parser = argparse.ArgumentParser(description="QFD Ladder Solver")
    parser.add_argument("--particle", choices=["electron", "muon", "tau"], 
                       default="electron", help="Particle to simulate")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--q-star", type=float, default=414.5848693847656,
                       help="Q* sensitivity parameter")
    
    args = parser.parse_args()
    
    if args.particle == "electron":
        results = run_electron_ladder(args.output_dir, args.q_star)
    elif args.particle == "muon":
        results = run_muon_ladder(args.output_dir, args.q_star)
    else:
        raise NotImplementedError(f"Tau solver not yet implemented")
    
    if results:
        print(f"\\n{args.particle.title()} ladder completed successfully!")
    else:
        print(f"\\n{args.particle.title()} ladder failed to converge.")