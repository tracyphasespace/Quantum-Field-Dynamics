#!/usr/bin/env python
"""
Ladder Solver - Energy Targeting Orchestration
============================================

Python implementation of the PowerShell ladder workflow for pinning
particle energies to their rest mass targets using Q* sensitivity.

Based on run_free_electron_ladder.ps1 from canonical implementation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from ..solvers.phoenix_solver import load_particle_constants, solve_psi_field
    from ..utils.io import save_results
except ImportError:
    # Handle direct execution
    from pathlib import Path
    import sys

    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))

    from solvers.phoenix_solver import load_particle_constants, solve_psi_field
    from utils.io import save_results


class LadderSolver:
    """Energy ladder solver using Q* sensitivity for rapid convergence."""

    EPSILON_DELTA_V2 = 1e-9
    ENERGY_MISMATCH_THRESHOLD = 1000.0
    ENERGY_COLLAPSE_THRESHOLD = 1e-10

    # V2 threshold constants to avoid magic numbers in comparisons
    V2_THRESH_LOW = 1e7
    V2_THRESH_MED = 3e7

    # Base dt values (high/medium/low) used to select dt_max depending on V2 magnitude
    DT_BASE_HIGH = 5e-5
    DT_BASE_MED = 2e-5
    DT_BASE_LOW = 1e-5

    # Iteration thresholds for progressively reducing dt_max to improve stability
    DT_REDUCE_STAGE1 = 5
    DT_REDUCE_STAGE2 = 10

    def __init__(
        self,
        particle: str,
        target_energy: float,
        q_star: float,
        max_iterations: int = 12,
        tolerance: float = 0.000001,  # 0.0001% for 99.9999% accuracy
        output_dir: Optional[Path] = None,
        num_radial_points: int = 250, # CHANGED from grid_size
        r_max: float = 10.0,          # CHANGED from box_size
        device: str = "cpu",          # Device is less relevant for NumPy solver
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
            num_radial_points: Number of radial points for the solver
            r_max: Maximum radial extent for the solver
            device: Device to run the solver on (e.g., 'cuda', 'cpu')
        """
        self.particle = particle
        self.target_energy = target_energy
        self.q_star = q_star
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.output_dir = output_dir or Path("data/output") / particle

        # Load particle constants
        self.constants = load_particle_constants(particle)
        self.physics = self.constants["physics_constants"].copy()

        # Initialize solver parameters
        self.num_radial_points = num_radial_points
        self.r_max = r_max
        self.steps = 2000 # Modified
        self.dt_min = 5e-6
        self.dt_max = 5e-5
        self.backend = "numpy" # Changed to numpy
        self.device = device

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
            current_energy: Current simulation energy (eV)

        Returns:
            V2 update value
        """
        delta_energy = self.target_energy - current_energy
        delta_v2_raw = delta_energy / self.q_star

        # Apply safety cap that grows with iteration - with 1e8 guardrail
        iteration = len(self.iteration_history) + 1
        max_cap = min(
            1000 * (2 ** min(iteration - 1, 5)), 1e8
        )  # Cap at 100M to prevent overflow

        # --- DYNAMIC PRECISION DAMPING: Reduce damping as we approach target ---
        accuracy_percent = (current_energy / self.target_energy) * 100
        
        if accuracy_percent >= 98.0:
            # PRECISION MODE: In final 2%, use progressively finer damping for 99.999% accuracy
            if accuracy_percent >= 99.9:
                damping_factor = 0.01  # Ultra-fine for 99.9%+ (final 0.1%)
                logging.info(f"  🎯 ULTRA-PRECISION MODE | Accuracy: {accuracy_percent:.3f}% | Damping: {damping_factor}")
            elif accuracy_percent >= 99.5:
                damping_factor = 0.02  # Very fine for 99.5%+ (final 0.5%)
                logging.info(f"  🎯 FINE-PRECISION MODE | Accuracy: {accuracy_percent:.3f}% | Damping: {damping_factor}")
            elif accuracy_percent >= 99.0:
                damping_factor = 0.03  # Fine for 99%+ (final 1%)
                logging.info(f"  🎯 PRECISION MODE | Accuracy: {accuracy_percent:.3f}% | Damping: {damping_factor}")
            else:
                damping_factor = 0.05  # Reduced for 98-99% (final 2%)
                logging.info(f"  🎯 APPROACHING TARGET | Accuracy: {accuracy_percent:.3f}% | Damping: {damping_factor}")
        
        # --- THE EXPONENTIAL ACCELERATOR: DYNAMIC DAMPING FOR HEAVY LEPTONS ---
        elif self.particle in ['muon', 'tau']:
            current_energy_MeV = current_energy / 1e6  # Work in MeV for clarity

            if current_energy_MeV < 1.0:
                # REGIME 1: Sub-MeV Stability (Proven safe)
                # Be moderately aggressive to quickly exit the low-energy zone.
                damping_factor = 0.65
            elif current_energy_MeV < 25.0:
                # REGIME 2: Mid-Range Acceleration (1 MeV -> 25 MeV)
                # We are in this regime now. The system is stiff and can handle large steps.
                damping_factor = 0.85
            else:
                # REGIME 3: High-Energy Push (Above 25 MeV)
                # Maximum aggression to close the final gap to 105 MeV.
                damping_factor = 0.95
                
            logging.info(f"  ⚡ ACCELERATOR ACTIVE | Energy: {current_energy_MeV:.2f} MeV | Damping: {damping_factor:.2f}")

        else:
            # Standard damping for moderate energies
            damping_factor = 0.1
        
        damped_delta = damping_factor * delta_v2_raw

        # Apply cap with guardrails
        if damped_delta > max_cap:
            damped_delta = max_cap
        elif damped_delta < -max_cap:
            damped_delta = -max_cap

        # Detailed per-iteration logging
        logging.info(f"  📊 E_target: {self.target_energy:.2e} eV")
        logging.info(f"  📊 E_current: {current_energy:.2e} eV")
        logging.info(f"  📊 delta_E: {delta_energy:.2e} eV")
        logging.info(f"  📊 q_star: {self.q_star:.6f}")
        logging.info(f"  📊 ΔV2(raw): {delta_v2_raw:.2e}")
        logging.info(f"  📊 ΔV2(damped): {damped_delta:.2e}")
        logging.info(f"  📊 Current V2: {self.physics['V2']:.2f}")
        # Unit check: Flag potential unit mismatch issues
        if abs(damped_delta) < self.EPSILON_DELTA_V2 and abs(delta_energy) > self.ENERGY_MISMATCH_THRESHOLD:
            logging.error(
                "🚨 POTENTIAL UNIT MISMATCH: |ΔV2|=%s very small but |delta_E|=%s large",
                f"{abs(damped_delta):.2e}",
                f"{abs(delta_energy):.2e}",
            )
            logging.error(
                "    This suggests possible energy unit inconsistency "
                "(simulation vs eV)"
            )
            raise ValueError(
                f"Unit mismatch detected: ΔV2={damped_delta:.2e}, delta_E={delta_energy:.2e}"
            )
        # Flag if energy is collapsing to zero
        if abs(current_energy) < self.ENERGY_COLLAPSE_THRESHOLD and self.target_energy > self.ENERGY_MISMATCH_THRESHOLD:
            logging.warning(
                "⚠️  ENERGY COLLAPSE: Current energy ~0 but target=%s eV",
                f"{self.target_energy:.0f}",
            )

        # Always return the computed (and capped/damped) ΔV2 so the signature is honored.
        return float(damped_delta)
    def adaptive_dt_max(self, iteration: int, v2: float) -> float:
        """Adapt time step based on iteration and V2 magnitude."""
        # Base timestep selected by V2 magnitude (use class constants instead of magic numbers)
        if v2 < self.V2_THRESH_LOW:
            base = self.DT_BASE_HIGH
        elif v2 < self.V2_THRESH_MED:
            base = self.DT_BASE_MED
        else:
            base = self.DT_BASE_LOW

        # Use iteration thresholds to progressively reduce dt_max on later rungs for stability.
        # Ensure we never go below dt_min.
        if iteration >= self.DT_REDUCE_STAGE2:
            return max(self.dt_min, base / 4.0)
        if iteration >= self.DT_REDUCE_STAGE1:
            return max(self.dt_min, base / 2.0)
        return max(self.dt_min, base)

    def _write_summary(self, results_data: Dict):
        """
        Write ladder summary to JSON file.

        Args:
            results_data: Dictionary containing results data.
        """
        summary_path = self.output_dir / "ladder_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_data, f, indent=2)
            # Write summary.json for fitter/zeeman consumers
            try:
                feats_path = self.output_dir / "features.json"
                cons_path  = self.output_dir / "constants.json"
                features = {}
                if feats_path.exists():
                    features = json.loads(feats_path.read_text(encoding="utf-8"))
                g_factor = None
                mag_mom  = None
                if cons_path.exists():
                    cons = json.loads(cons_path.read_text(encoding="utf-8"))
                    g_factor = cons.get("g_factor")
                    mag_mom  = cons.get("magnetic_moment")
                summary_out = {
                    "species": getattr(self, "particle", features.get("species")),
                    "energy": results_data.get("final_energy"),
                    "magnetic_moment": mag_mom,
                    "g_factor": g_factor,
                    "U_final": features.get("U0", features.get("U_final", 1.0)),
                    "R_eff_final": features.get("R_eff"),
                    "I_final": features.get("I"),
                    "Hkin_final": results_data.get("final_energy"),
                    "Q_proxy_final": features.get("q_star")
                }
                (self.output_dir / "summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
            except Exception as e:
                logging.warning(f"Failed to write summary.json: {e}")

        logging.info(f"Wrote ladder summary to {summary_path}")

    def run_ladder(self, initial_v2: Optional[float] = None) -> Optional[Dict]:
        """
        Run the ladder solver to pin energy to target.

        Args:
            initial_v2: Initial V2 value (uses constant default if None)

        Returns:
            Final solver results or None if simulation did not run
        """
        # Initialize V2
        if initial_v2 is not None:
            self.physics["V2"] = initial_v2

        current_v2 = self.physics["V2"]
        warm_start_path = None
        # Ensure variables referenced after the loop are always defined to avoid
        # "possibly unbound" warnings from static analyzers and to have a sane
        # fallback value at runtime.
        current_energy = None
        results = None
        # Track the last computed relative error (may remain None if no simulation ran)
        relative_error = None

        logging.info(f"Starting {self.particle} ladder solver...")
        logging.info(f"Initial V2: {current_v2}")

        # Initialize progress bar
        pbar = None
        if tqdm and logging.getLogger().level > logging.INFO:  # Only show tqdm if logging is suppressed
            pbar = tqdm(total=self.max_iterations, desc=f"{self.particle.title()} Ladder", 
                       unit="iter", ncols=80, leave=True)

        iteration = 1
        while iteration <= self.max_iterations:
            # Adapt time step
            self.dt_max = self.adaptive_dt_max(iteration, current_v2)

            # Setup output directory
            run_dir = self.output_dir / f"ladder_{iteration:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"\n=== Ladder Step {iteration:02d} ===")
            logging.info(f"V2: {current_v2:.2f}")
            logging.info(f"dt_max: {self.dt_max}")
            logging.info(f"warm_start: {warm_start_path}")

            try:
                # Update physics constants
                self.physics["V2"] = current_v2

                # Run simulation
                try:
                    results = solve_psi_field(
                        particle=self.particle,
                        num_radial_points=self.num_radial_points, # CHANGED
                        r_max=self.r_max,                       # CHANGED
                        # backend and device are now implicit in the NumPy-based solver
                        custom_physics={
                            "V2": current_v2,
                            "V4": self.physics["V4"],
                            "g_c": self.physics["g_c"],
                            "k_csr": self.physics["k_csr"],
                            "psi_floor": self.physics.get("psi_floor", -10.0),  # Explicitly pass it
                        },
                        init_from=warm_start_path,
                        output_dir=str(run_dir),
                        q_star=self.q_star,
                    )
                except TypeError as e:
                    import traceback
                    print(f"[ERROR] String->int conversion error in solve_psi_field:")
                    traceback.print_exc()
                    print(f"[ERROR] Error message: {e}")
                    raise

                # Extract energy
                current_energy = results["H_final"]

                # --- DYNAMIC Q* UPDATE (ROBUST IMPLEMENTATION) ---
                # If we have a previous energy, estimate the new Q*
                if len(self.iteration_history) > 1: # Need at least two previous points for a stable delta
                    prev_iter_data = self.iteration_history[-1]
                    prev_energy = prev_iter_data['energy']
                    prev_v2 = prev_iter_data['V2']
                    
                    delta_E_step = current_energy - prev_energy
                    delta_V2_step = current_v2 - prev_v2
                    
                    if abs(delta_V2_step) > 1e-6:  # Avoid division by zero
                        measured_q_star = delta_E_step / delta_V2_step
                        
                        # --- STABILITY GUARDS ---
                        is_sensible = True
                        # 1. Positivity Guard: Q* must be positive.
                        if measured_q_star <= 1e-3:
                            logging.warning(f"  🧠 Measured Q* is non-positive ({measured_q_star:.4f}). Ignoring update.")
                            is_sensible = False

                        # 2. Relative Change Guard: Don't allow updates to be excessively large or small.
                        if is_sensible and (measured_q_star > 5.0 * self.q_star or measured_q_star < 0.2 * self.q_star):
                            logging.warning(f"  🧠 Measured Q* ({measured_q_star:.4f}) is an outlier compared to current Q* ({self.q_star:.4f}). Ignoring update.")
                            is_sensible = False

                        if is_sensible:
                            # Use a weighted average to smooth the update
                            self.q_star = 0.7 * self.q_star + 0.3 * measured_q_star # Slower update rate
                            
                            # 3. Absolute Bounds Guard: Keep Q* within a reasonable physical range.
                            self.q_star = max(min(self.q_star, 1e7), 1e-2) 
                            
                            logging.info(f"  🧠 Dynamic Q* updated to: {self.q_star:.4f} (measured: {measured_q_star:.4f})")
                        else:
                            logging.info(f"  🧠 Q* update skipped. Maintaining Q* at: {self.q_star:.4f}")

                else:
                    logging.info(f"  🧠 Q* update requires more history. Maintaining Q* at: {self.q_star:.4f}")

                # Save results
                save_results(results, run_dir / "results")

                # Log progress
                logging.info(f"Energy: {current_energy:.4f} eV")

                # Track iteration
                iteration_data = {
                    "iteration": iteration,
                    "V2": current_v2,
                    "energy": current_energy,
                    "dt_max": self.dt_max,
                    "converged": False,
                }
                self.iteration_history.append(iteration_data)

                # Update progress bar
                if pbar:
                    accuracy = (current_energy / self.target_energy) * 100
                    pbar.set_postfix({"Energy": f"{current_energy:.0f} eV", "Acc": f"{accuracy:.2f}%"})
                    pbar.update(1)

                # Check convergence
                relative_error = (
                    abs(current_energy - self.target_energy) / self.target_energy
                )
                if relative_error <= self.tolerance:
                    logging.info("\n*** CONVERGED ***")
                    logging.info(f"Target reached within {self.tolerance*100:.1f}%")
                    logging.info(f"Final energy: {current_energy:.4f} eV")

                    iteration_data["converged"] = True
                    self.converged = True

                    # Save final state
                    final_results = {
                        "converged": True,
                        "final_energy": current_energy,
                        "final_V2": current_v2,
                        "target_energy": self.target_energy,
                        "iterations": len(self.iteration_history),
                        "tolerance_achieved": relative_error,
                        "history": self.iteration_history,
                        "U_final": results.get("U", 1.0),
                        "R_eff_final": results.get("R_eff", 1.0),
                        "I_final": results.get("I", 1.0),
                        "Hkin_final": results.get("K", 1.0),
                        "Q_proxy_final": self.q_star,
                    }

                    self._write_summary(final_results)

                    # Close progress bar
                    if pbar:
                        pbar.close()

                    return results

                # Compute next V2 update
                delta_v2 = self.compute_v2_update(current_energy)

                # Canonical adaptive V4/dt strategy: stiffen when step approaches cap
                iteration_cap = 1000 * (2 ** min(iteration - 1, 5))
                approaching_cap = abs(delta_v2) >= iteration_cap * 0.9

                if approaching_cap:
                    logging.info("🚨 Approaching iteration cap - stiffening V4 and reducing dt_max")
                    old_v4 = self.physics["V4"]
                    self.physics["V4"] += 0.5
                    self.dt_max = max(self.dt_min, self.dt_max / 1.5)
                    logging.info(f"🔧 V4: {old_v4:.3f} → {self.physics['V4']:.3f}")
                    logging.info(f"🔧 dt_max: {self.dt_max:.2e}")

                current_v2 += delta_v2

                logging.info(f"📈 Delta V2: {delta_v2:.2f}")
                logging.info(f"📈 Next V2: {current_v2:.2f}")

                # Update warm start path for next iteration
                potential_warm_start = run_dir / "fields_final.npz"
                if potential_warm_start.exists():
                    warm_start_path = str(potential_warm_start)
                    logging.info(f"🔄 Warm-start ready: {warm_start_path}")
                else:
                    logging.warning("⚠️ No fields_final.npz found for warm-start")

                # Move to next rung on successful simulation
                iteration += 1

            except Exception as e:
                logging.error(f"Simulation failed: {e}")
                logging.info("Reducing dt_max and stiffening V4, retrying...")

                self.dt_max = max(self.dt_min, self.dt_max / 2.0)
                self.physics["V4"] += 0.5
                # Do NOT modify iteration here; keep current value to retry same rung
                continue

        # Close progress bar
        if pbar:
            pbar.close()

        # Did not converge
        logging.warning(f"Failed to converge after {self.max_iterations} iterations")

        final_results = {
            "converged": False,
            "final_energy": current_energy if "current_energy" in locals() else None,
            "final_V2": current_v2,
            "target_energy": self.target_energy,
            "iterations": len(self.iteration_history),
            "tolerance_achieved": relative_error,
            "history": self.iteration_history,
            "U_final": results.get("U", 1.0) if results is not None else None,
            "R_eff_final": results.get("R_eff", 1.0) if results is not None else None,
            "I_final": results.get("I", 1.0) if results is not None else None,
            "Hkin_final": results.get("K", 1.0) if results is not None else None,
            "Q_proxy_final": self.q_star,
        }

        self._write_summary(final_results)

        return results if "results" in locals() else None


def run_electron_ladder(
    output_dir: Optional[Path] = None,
    q_star: float = 2.166144847869873,
    num_radial_points: int = 64,
    device: str = "cuda",
    max_iterations: int = 12,
) -> Optional[Dict]:
    """
    Run electron energy ladder to 511 keV target.

    Args:
        output_dir: Output directory
        q_star: Q* sensitivity parameter
        num_radial_points: Number of radial points for the solver
        device: Device to run the solver on (e.g., 'cuda', 'cpu')

    Returns:
        Final simulation results or None on failure
    """
    target_energy = 511000.0  # eV

    ladder = LadderSolver(
        particle="electron",
        target_energy=target_energy,
        q_star=q_star,
        output_dir=output_dir,
        num_radial_points=num_radial_points,
        device=device,
        max_iterations=max_iterations,
    )

    return ladder.run_ladder()


def run_muon_ladder(
    output_dir: Optional[Path] = None,
    q_star: float = 2.166144847869873,
    initial_v2: float = 500000.0,
    num_radial_points: int = 64,
    device: str = "cuda",
    max_iterations: int = 75,
) -> Optional[Dict]:
    """
    Run muon energy ladder to 105.658 MeV target.

    Args:
        output_dir: Output directory
        q_star: Q* sensitivity parameter
        initial_v2: Starting V2 value for muon regime
        num_radial_points: Number of radial points for the solver
        device: Device to run the solver on (e.g., 'cuda', 'cpu')

    Returns:
        Final simulation results or None on failure
    """
    target_energy = 105658000.0  # eV

    ladder = LadderSolver(
        particle="muon",
        target_energy=target_energy,
        q_star=q_star,
        max_iterations=max_iterations,
        output_dir=output_dir,
        num_radial_points=num_radial_points,
        device=device,
    )

    return ladder.run_ladder(initial_v2=initial_v2)


def run_tau_ladder(
    output_dir: Optional[Path] = None,
    q_star: float = 2.166144847869873,
    initial_v2: float = 1000000.0,
    initial_v4: Optional[float] = None,  # NEW: V4 parameter override
    num_radial_points: int = 64,
    device: str = "cuda",
    max_iterations: int = 75,
) -> Optional[Dict]:
    """
    Run tau energy ladder to 1.777 GeV target.

    Args:
        output_dir: Output directory
        q_star: Q* sensitivity parameter
        initial_v2: Starting V2 value for tau regime
        initial_v4: V4 parameter override (if None, uses tau.json default)
        num_radial_points: Number of radial points for the solver
        device: Device to run the solver on (e.g., 'cuda', 'cpu')

    Returns:
        Final simulation results or None on failure
    """
    target_energy = 1777000000.0  # eV

    ladder = LadderSolver(
        particle="tau",
        target_energy=target_energy,
        q_star=q_star,
        max_iterations=max_iterations,
        output_dir=output_dir,
        num_radial_points=num_radial_points,
        device=device,
    )
    
    # Apply V4 override if specified
    if initial_v4 is not None:
        ladder.physics["V4"] = initial_v4

    return ladder.run_ladder(initial_v2=initial_v2)


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("ladder_solver.log", mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    import argparse

    parser = argparse.ArgumentParser(description="QFD Ladder Solver")
    parser.add_argument(
        "--particle",
        choices=["electron", "muon", "tau"],
        default="electron",
        help="Particle to simulate",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--q-star",
        type=float,
        default=2.166144847869873,
        help="Q* sensitivity parameter",
    )
    parser.add_argument(
        "--num-radial-points",
        type=int,
        default=64,
        help="Number of radial points for the solver",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the solver on (e.g., 'cuda', 'cpu')",
    )

    args = parser.parse_args()

    if args.particle == "electron":
        results = run_electron_ladder(args.output_dir, args.q_star, args.num_radial_points, args.device)
    elif args.particle == "muon":
        results = run_muon_ladder(args.output_dir, args.q_star, args.num_radial_points, args.device)
    elif args.particle == "tau":
        results = run_tau_ladder(args.output_dir, args.q_star, args.num_radial_points, args.device)
    else:
        raise NotImplementedError(f"Unknown particle: {args.particle}")

    if results:
        print(f"\n{args.particle.title()} ladder completed successfully!")
    else:
        print(f"\n{args.particle.title()} ladder failed to converge.")
