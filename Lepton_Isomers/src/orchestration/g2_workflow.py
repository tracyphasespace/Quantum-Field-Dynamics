"""
G-2 Workflow Integration
=======================

End-to-end workflow for QFD Phoenix simulations and g-2 predictions.
Combines the Phoenix solver ladder targeting with g-2 prediction analysis
for complete lepton physics analysis.

This module provides:
- Complete electron/muon/tau simulation workflows
- Automatic bundle generation for g-2 predictors  
- CSR parameter sweeps and analysis
- Publication-ready reports and visualizations
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .ladder_solver import LadderSolver, run_electron_ladder, run_muon_ladder
    from .g2_predictor_batch import G2PredictorBatch
    from ..solvers.phoenix_solver import solve_psi_field
    from ..utils.io import save_results
    from ..utils.analysis import analyze_results
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    
    from orchestration.ladder_solver import LadderSolver, run_electron_ladder, run_muon_ladder
    from orchestration.g2_predictor_batch import G2PredictorBatch
    from solvers.phoenix_solver import solve_psi_field
    from utils.io import save_results
    from utils.analysis import analyze_results

logger = logging.getLogger(__name__)


class G2Workflow:
    """Complete workflow for QFD simulations and g-2 analysis."""
    
    def __init__(
        self,
        output_base: Path,
        device: str = "cuda",
        backend: str = "torch",
        q_star: float = 414.5848693847656
    ):
        """
        Initialize g-2 workflow.
        
        Args:
            output_base: Base directory for all outputs
            device: Computation device ('cuda' or 'cpu')  
            backend: Solver backend ('torch' or 'numpy')
            q_star: Q* sensitivity parameter for ladder solving
        """
        self.output_base = Path(output_base)
        self.device = device
        self.backend = backend
        self.q_star = q_star
        
        # Create output structure
        self.bundles_dir = self.output_base / "bundles"
        self.reports_dir = self.output_base / "reports"
        self.analysis_dir = self.output_base / "analysis"
        
        for dir_path in [self.bundles_dir, self.reports_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"G-2 Workflow initialized")
        logger.info(f"  Output base: {self.output_base}")
        logger.info(f"  Device: {self.device}, Backend: {self.backend}")
        logger.info(f"  Q* sensitivity: {self.q_star}")
    
    def run_electron_workflow(
        self,
        target_energy: float = 511000.0,
        csr_values: Optional[List[float]] = None,
        tolerance: float = 0.01
    ) -> Dict[str, Path]:
        """
        Complete electron workflow: simulation + bundle generation.
        
        Args:
            target_energy: Target electron energy in eV (511 keV)
            csr_values: CSR parameter values to test [0.0, 0.002, 0.005]
            tolerance: Energy convergence tolerance
            
        Returns:
            Dictionary mapping bundle names to paths
        """
        if csr_values is None:
            csr_values = [0.0, 0.002, 0.005]
        
        logger.info(f"Starting electron workflow (target: {target_energy:.0f} eV)")
        
        bundles = {}
        
        # Base electron simulation (CSR = 0)
        logger.info("Running base electron ladder solver...")
        
        base_output = self.output_base / "electron_base"
        ladder = LadderSolver(
            particle="electron",
            target_energy=target_energy,
            q_star=self.q_star,
            tolerance=tolerance,
            output_dir=base_output
        )
        
        base_results = ladder.run_ladder()
        
        if not ladder.converged:
            raise RuntimeError("Base electron ladder failed to converge")
        
        # Create base bundle
        base_bundle = self._create_bundle(
            "electron_511keV_v1",
            base_results,
            {"k_csr": 0.0}
        )
        bundles["electron_base"] = base_bundle
        
        logger.info(f"Base electron bundle: {base_bundle}")
        
        # CSR parameter sweep
        if len(csr_values) > 1:
            logger.info(f"Running CSR sweep: {csr_values}")
            
            for k_csr in csr_values:
                if k_csr == 0.0:
                    continue  # Already done
                
                logger.info(f"CSR probe: k_csr = {k_csr}")
                
                # Run with CSR enabled
                csr_results = solve_psi_field(
                    particle="electron",
                    grid_size=64,
                    box_size=16.0,
                    backend=self.backend,
                    device=self.device,
                    steps=400,
                    dt_auto=True,
                    dt_min=5e-5,
                    dt_max=5e-4,
                    V2=base_results['constants']['physics_constants']['V2'],
                    V4=base_results['constants']['physics_constants']['V4'],
                    g_c=base_results['constants']['physics_constants']['g_c'],
                    k_csr=k_csr,
                    q_star=self.q_star
                )
                
                # Re-pin energy if needed (CSR typically lowers energy)
                energy_diff = target_energy - csr_results['H_final']
                if abs(energy_diff) > tolerance * target_energy:
                    logger.info(f"Re-pinning energy (diff: {energy_diff:.1f} eV)")
                    
                    v2_correction = energy_diff / self.q_star
                    corrected_v2 = base_results['constants']['physics_constants']['V2'] + v2_correction
                    
                    csr_results = solve_psi_field(
                        particle="electron",
                        grid_size=64,
                        box_size=16.0,
                        backend=self.backend,
                        device=self.device,
                        steps=400,
                        dt_auto=True,
                        dt_min=5e-5,
                        dt_max=5e-4,
                        V2=corrected_v2,
                        V4=base_results['constants']['physics_constants']['V4'],
                        g_c=base_results['constants']['physics_constants']['g_c'],
                        k_csr=k_csr,
                        q_star=self.q_star
                    )
                
                # Create CSR bundle
                csr_tag = f"csr{int(k_csr * 1000):03d}"  # e.g., csr002
                bundle_name = f"electron_511keV_v1_{csr_tag}"
                
                csr_bundle = self._create_bundle(
                    bundle_name,
                    csr_results,
                    {"k_csr": k_csr}
                )
                bundles[f"electron_{csr_tag}"] = csr_bundle
                
                logger.info(f"CSR bundle created: {csr_bundle}")
        
        logger.info(f"Electron workflow completed: {len(bundles)} bundles")
        return bundles
    
    def run_muon_workflow(
        self,
        target_energy: float = 105658000.0,
        tolerance: float = 0.01
    ) -> Dict[str, Path]:
        """
        Complete muon workflow: simulation + bundle generation.
        
        Args:
            target_energy: Target muon energy in eV (105.658 MeV)
            tolerance: Energy convergence tolerance
            
        Returns:
            Dictionary mapping bundle names to paths  
        """
        logger.info(f"Starting muon workflow (target: {target_energy:.0f} eV)")
        
        muon_output = self.output_base / "muon_base"
        ladder = LadderSolver(
            particle="muon",
            target_energy=target_energy,
            q_star=self.q_star,
            tolerance=tolerance,
            max_iterations=15,  # Muon may need more iterations
            output_dir=muon_output
        )
        
        # Use higher initial V2 for muon regime  
        muon_results = ladder.run_ladder(initial_v2=500000.0)
        
        if not ladder.converged:
            logger.warning("Muon ladder did not fully converge - using best result")
        
        # Create muon bundle
        muon_bundle = self._create_bundle(
            "muon_105658keV_v1",
            muon_results,
            {"k_csr": 0.0}
        )
        
        logger.info(f"Muon workflow completed: {muon_bundle}")
        return {"muon_base": muon_bundle}
    
    def run_tau_workflow(
        self,
        target_energy: float = 1776840000.0,
        tolerance: float = 0.02  # Looser tolerance for tau
    ) -> Dict[str, Path]:
        """
        Complete tau workflow: simulation + bundle generation.
        
        Args:
            target_energy: Target tau energy in eV (1.777 GeV) 
            tolerance: Energy convergence tolerance
            
        Returns:
            Dictionary mapping bundle names to paths
        """
        logger.info(f"Starting tau workflow (target: {target_energy:.0f} eV)")
        logger.info("Note: Tau parameters are preliminary and need calibration")
        
        tau_output = self.output_base / "tau_base"
        ladder = LadderSolver(
            particle="tau",
            target_energy=target_energy,
            q_star=self.q_star,
            tolerance=tolerance,
            max_iterations=20,  # Tau may need many iterations
            output_dir=tau_output
        )
        
        # Use very high initial V2 for tau regime
        tau_results = ladder.run_ladder(initial_v2=5000000.0)
        
        if not ladder.converged:
            logger.warning("Tau ladder did not converge - this is expected for preliminary parameters")
        
        # Create tau bundle (even if not converged, for development)
        tau_bundle = self._create_bundle(
            "tau_1777MeV_v1",
            tau_results if tau_results else {"H_final": 0, "constants": {"physics_constants": {"k_csr": 0.0}}},
            {"k_csr": 0.0}
        )
        
        logger.info(f"Tau workflow completed: {tau_bundle}")
        return {"tau_base": tau_bundle}
    
    def _create_bundle(
        self,
        bundle_name: str,
        simulation_results: Dict,
        parameters: Dict
    ) -> Path:
        """
        Create a bundle directory for g-2 prediction.
        
        Args:
            bundle_name: Name of the bundle
            simulation_results: Results from Phoenix solver
            parameters: Additional parameters (e.g., CSR values)
            
        Returns:
            Path to created bundle directory
        """
        bundle_dir = self.bundles_dir / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # Save simulation results
        save_results(simulation_results, bundle_dir / "simulation_results")
        
        # Create manifest for g-2 predictor
        manifest = {
            "bundle_name": bundle_name,
            "particle": simulation_results.get("particle", "unknown"),
            "energy_final": simulation_results.get("H_final", 0.0),
            "physics_parameters": simulation_results.get("constants", {}).get("physics_constants", {}),
            "additional_parameters": parameters,
            "fields_file": "simulation_results.json",  # Adjust based on your format
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator": "QFD Phoenix Refactored Workflow v1.0"
        }
        
        manifest_path = bundle_dir / f"{bundle_name.split('_')[0]}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Bundle created: {bundle_dir} (manifest: {manifest_path.name})")
        return bundle_dir
    
    def run_g2_analysis(
        self,
        bundle_paths: List[Path],
        predictor_path: Optional[str] = None
    ) -> Path:
        """
        Run g-2 analysis on generated bundles.
        
        Args:
            bundle_paths: List of bundle directories
            predictor_path: Path to g-2 predictor (auto-detected if None)
            
        Returns:
            Path to analysis report
        """
        logger.info(f"Running g-2 analysis on {len(bundle_paths)} bundles")
        
        # Initialize g-2 batch processor
        processor = G2PredictorBatch(
            predictor_path=predictor_path,
            device=self.device,
            workdir=self.output_base
        )
        
        # Process all bundles
        successful = processor.process_bundles(bundle_paths)
        
        if successful == 0:
            raise RuntimeError("No bundles processed successfully")
        
        # Write consolidated reports
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.reports_dir / f"g2_analysis_{timestamp}.csv"
        md_path = self.reports_dir / f"g2_analysis_{timestamp}.md"
        
        processor.write_summary(csv_path, md_path)
        
        logger.info(f"G-2 analysis completed: {successful}/{len(bundle_paths)} bundles")
        logger.info(f"Reports: {csv_path}, {md_path}")
        
        return md_path
    
    def run_complete_workflow(
        self,
        particles: List[str] = None,
        predictor_path: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Run complete workflow for specified particles.
        
        Args:
            particles: List of particles ('electron', 'muon', 'tau')
            predictor_path: Path to g-2 predictor
            
        Returns:
            Dictionary with workflow results and report paths
        """
        if particles is None:
            particles = ["electron", "muon"]  # Skip tau by default (preliminary)
        
        logger.info(f"Starting complete g-2 workflow for: {particles}")
        
        all_bundles = []
        workflow_results = {}
        
        # Run particle simulations
        for particle in particles:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {particle.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                if particle == "electron":
                    bundles = self.run_electron_workflow()
                elif particle == "muon":
                    bundles = self.run_muon_workflow()
                elif particle == "tau":
                    bundles = self.run_tau_workflow()
                else:
                    logger.error(f"Unknown particle: {particle}")
                    continue
                
                workflow_results[particle] = bundles
                all_bundles.extend(bundles.values())
                
            except Exception as e:
                logger.error(f"Failed to process {particle}: {e}")
                workflow_results[particle] = {"error": str(e)}
        
        # Run g-2 analysis if we have bundles
        if all_bundles:
            try:
                logger.info(f"\n{'='*50}")
                logger.info("Running G-2 Analysis")  
                logger.info(f"{'='*50}")
                
                analysis_report = self.run_g2_analysis(all_bundles, predictor_path)
                workflow_results["g2_analysis"] = analysis_report
                
            except Exception as e:
                logger.error(f"G-2 analysis failed: {e}")
                workflow_results["g2_analysis"] = {"error": str(e)}
        
        # Create final summary
        summary = {
            "workflow_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "particles_processed": list(workflow_results.keys()),
            "total_bundles": len(all_bundles),
            "output_base": str(self.output_base),
            "device": self.device,
            "backend": self.backend
        }
        
        summary_path = self.reports_dir / "workflow_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        workflow_results["summary"] = summary_path
        
        logger.info(f"\n{'='*50}")
        logger.info("WORKFLOW COMPLETED")
        logger.info(f"{'='*50}")
        logger.info(f"Processed: {len(workflow_results)-2} particles")  # -2 for g2_analysis and summary
        logger.info(f"Generated: {len(all_bundles)} bundles")
        logger.info(f"Summary: {summary_path}")
        
        return workflow_results


def main():
    """Command-line interface for g-2 workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete QFD g-2 workflow")
    parser.add_argument(
        "--particles", nargs="*", 
        choices=["electron", "muon", "tau"],
        default=["electron", "muon"],
        help="Particles to process"
    )
    parser.add_argument(
        "--output", type=Path, default="g2_workflow_output",
        help="Base output directory"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
        help="Computation device"
    )
    parser.add_argument(
        "--backend", choices=["torch", "numpy"], default="torch", 
        help="Solver backend"
    )
    parser.add_argument(
        "--predictor", help="Path to g-2 predictor script"
    )
    parser.add_argument(
        "--q-star", type=float, default=414.5848693847656,
        help="Q* sensitivity parameter"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize workflow
    workflow = G2Workflow(
        output_base=args.output,
        device=args.device,
        backend=args.backend,
        q_star=args.q_star
    )
    
    # Run complete workflow
    try:
        results = workflow.run_complete_workflow(
            particles=args.particles,
            predictor_path=args.predictor
        )
        
        print(f"\nWorkflow completed successfully!")
        print(f"Results saved to: {args.output}")
        return 0
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())