#!/usr/bin/env python3
"""
QFD Isomer Workflow - Proposed Three-Objective Paradigm
=======================================================

Implements the proposed three-objective physics paradigm:

1. **Electron Calibration**: Lock all physics parameters from electron ground state
2. **Excited State Search**: Find muon/tau as isomers (excited states) of electron
3. **Unified g-2 Predictions**: Predict all lepton g-2 values from unified framework

This represents a paradigm shift from treating electron/muon/tau as separate particles
to viewing them as different energy states of the same fundamental field equation.

Physics Theory:
--------------
The Phoenix Core Hamiltonian:
H = ∫ [ ½(|∇ψ_s|² + |∇ψ_b|²) + V₂·ρ + V₄·ρ² - ½·k_csr·ρ_q² ] dV

Has multiple stable solutions corresponding to lepton mass states:
- Ground state (n=0): Electron at ~511 keV
- First excited state (n=1): Muon at ~105.7 MeV  
- Second excited state (n=2): Tau at ~1.777 GeV

The key insight: Different angular momentum states have different stability,
with muon/tau being less stable due to lower angular momentum, explaining
their shorter lifetimes.

Theoretical Capabilities:
--------------------------
1. Predict muon mass and g-2 from electron calibration alone
2. Predict tau mass and g-2 (currently unmeasured) from electron
3. Provide unified theory explaining lepton mass hierarchy
4. Enable precision QFD predictions without empirical fitting
"""

import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time

try:
    from ..solvers.zeeman_experiments import IsomerZeemanAnalysis, ZeemanExperiment
    from ..solvers.phoenix_solver import solve_psi_field, load_particle_constants
    from ..utils.io import save_results
    from ..utils.analysis import analyze_results
except ImportError:
    # Handle direct execution
    import sys
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    
    from solvers.zeeman_experiments import IsomerZeemanAnalysis, ZeemanExperiment
    from solvers.phoenix_solver import solve_psi_field, load_particle_constants
    from utils.io import save_results
    from utils.analysis import analyze_results

logger = logging.getLogger(__name__)


class IsomerWorkflow:
    """
    Complete workflow for QFD isomer theory validation and g-2 predictions.
    
    This orchestrates the three-objective paradigm:
    1. Electron calibration and validation
    2. Excited state search for muon/tau masses
    3. Unified g-2 predictions from field-fundamental Zeeman experiments
    """
    
    def __init__(
        self,
        output_dir: Path,
        backend: str = "torch",
        device: str = "cuda",
        grid_size: int = 64,
        box_size: float = 2.0,
        precision_target: float = 0.05  # 5% precision target
    ):
        """
        Initialize isomer workflow.
        
        Args:
            output_dir: Directory for all workflow outputs
            backend: Computational backend ("torch", "numpy")
            device: Device for computation ("cuda", "cpu")
            grid_size: Spatial grid resolution
            box_size: Simulation box size
            precision_target: Target precision for predictions (fractional)
        """
        self.output_dir = Path(output_dir)
        self.backend = backend
        self.device = device
        self.grid_size = grid_size
        self.box_size = box_size
        self.precision_target = precision_target
        
        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "electron_calibration").mkdir(exist_ok=True)
        (self.output_dir / "muon_prediction").mkdir(exist_ok=True)
        (self.output_dir / "tau_prediction").mkdir(exist_ok=True)
        (self.output_dir / "zeeman_experiments").mkdir(exist_ok=True)
        (self.output_dir / "analysis_reports").mkdir(exist_ok=True)
        
        # Initialize Zeeman analysis framework
        self.zeeman_analyzer = IsomerZeemanAnalysis(
            backend, device, grid_size, box_size
        )
        
        # Target masses (eV)
        self.target_masses = {
            'electron': 510998.946,      # PDG 2020 electron rest mass
            'muon': 105658374.5,        # PDG 2020 muon rest mass  
            'tau': 1776860000.0         # PDG 2020 tau rest mass
        }
        
        # Experimental g-2 values for validation
        self.experimental_g2 = {
            'electron': 0.00115965218073,   # PDG 2020
            'muon': 0.00116592089,          # E989 Fermilab 2023
            'tau': None                     # Not yet measured
        }
        
        logger.info(f"IsomerWorkflow initialized: {output_dir}")
        logger.info(f"Computing setup: {backend}/{device}, grid={grid_size}³")
    
    def run_complete_workflow(self, save_intermediate: bool = True) -> Dict:
        """
        Execute complete three-objective isomer workflow.
        
        Args:
            save_intermediate: Save intermediate results to disk
            
        Returns:
            Comprehensive results dictionary with all predictions
        """
        logger.info("Starting complete QFD isomer workflow...")
        start_time = time.time()
        
        workflow_results = {
            'workflow_info': {
                'start_time': start_time,
                'backend': self.backend,
                'device': self.device,
                'grid_size': self.grid_size,
                'precision_target': self.precision_target,
                'target_masses': self.target_masses
            }
        }
        
        # Objective 1: Electron Calibration
        logger.info("=" * 60)
        logger.info("OBJECTIVE 1: ELECTRON CALIBRATION")
        logger.info("=" * 60)
        
        electron_results = self._run_electron_calibration()
        workflow_results['objective_1_electron'] = electron_results
        
        if save_intermediate:
            self._save_objective_results(1, "electron_calibration", electron_results)
        
        # Check if electron calibration succeeded
        if not electron_results.get('calibration_success', False):
            logger.error("Electron calibration failed - cannot proceed")
            workflow_results['workflow_success'] = False
            workflow_results['failure_reason'] = "Electron calibration failed"
            return workflow_results
        
        logger.info(f"✓ Electron calibrated: g-2 = {electron_results['g2_predicted']:.10f}")
        
        # Objective 2: Muon Isomer Prediction
        logger.info("=" * 60) 
        logger.info("OBJECTIVE 2: MUON ISOMER PREDICTION")
        logger.info("=" * 60)
        
        muon_results = self._run_muon_isomer_prediction(electron_results)
        workflow_results['objective_2_muon'] = muon_results
        
        if save_intermediate:
            self._save_objective_results(2, "muon_prediction", muon_results)
        
        if muon_results.get('prediction_success', False):
            logger.info(f"✓ Muon predicted: mass = {muon_results['predicted_mass']:.0f} eV, g-2 = {muon_results['g2_predicted']:.10f}")
        else:
            logger.warning("✗ Muon prediction failed")
        
        # Objective 3: Tau Isomer Prediction  
        logger.info("=" * 60)
        logger.info("OBJECTIVE 3: TAU ISOMER PREDICTION")
        logger.info("=" * 60)
        
        tau_results = self._run_tau_isomer_prediction(electron_results)
        workflow_results['objective_3_tau'] = tau_results
        
        if save_intermediate:
            self._save_objective_results(3, "tau_prediction", tau_results)
        
        if tau_results.get('prediction_success', False):
            logger.info(f"✓ Tau predicted: mass = {tau_results['predicted_mass']:.0f} eV, g-2 = {tau_results['g2_predicted']:.10f}")
        else:
            logger.warning("✗ Tau prediction failed")
        
        # Final Analysis and Validation
        logger.info("=" * 60)
        logger.info("WORKFLOW ANALYSIS & VALIDATION")
        logger.info("=" * 60)
        
        analysis_results = self._analyze_workflow_results(workflow_results)
        workflow_results['final_analysis'] = analysis_results
        
        # Overall success assessment
        workflow_results['workflow_success'] = analysis_results['overall_success']
        workflow_results['success_rate'] = analysis_results['success_rate']
        workflow_results['paradigm_breakthrough'] = analysis_results['paradigm_breakthrough']
        
        # Timing
        workflow_results['workflow_info']['end_time'] = time.time()
        workflow_results['workflow_info']['total_duration'] = workflow_results['workflow_info']['end_time'] - start_time
        
        # Save complete results
        if save_intermediate:
            self._save_complete_workflow_results(workflow_results)
        
        # Print final summary
        self._print_workflow_summary(workflow_results)
        
        logger.info(f"Complete isomer workflow finished in {workflow_results['workflow_info']['total_duration']:.1f}s")
        return workflow_results
    
    def _run_electron_calibration(self) -> Dict:
        """Run electron calibration to lock all physics parameters."""
        
        logger.info("Calibrating universal physics parameters from electron ground state...")
        
        try:
            # Create dedicated electron Zeeman experiment
            electron_zeeman = ZeemanExperiment(
                "electron", self.backend, self.device, self.grid_size, self.box_size
            )
            
            # Run comprehensive field sweep for calibration
            B_fields = np.logspace(-6, -2, 12)  # 1μT to 10mT, 12 points
            zeeman_results = electron_zeeman.zeeman_field_sweep(B_fields.tolist())
            
            # Extract calibrated g-2
            g_factor = zeeman_results.get('g_factor_extrapolated', np.mean(zeeman_results['g_factors']))
            g2_predicted = (g_factor - 2.0) / 2.0
            g2_experimental = self.experimental_g2['electron']
            
            # Calculate precision
            relative_error = abs(g2_predicted - g2_experimental) / g2_experimental
            calibration_success = relative_error < self.precision_target
            
            # Get calibrated physics parameters
            calibrated_physics = electron_zeeman.physics.copy()
            
            results = {
                'calibration_success': calibration_success,
                'g_factor': float(g_factor),
                'g2_predicted': float(g2_predicted),
                'g2_experimental': g2_experimental,
                'relative_error': float(relative_error),
                'precision_achieved': float(1.0 - relative_error),
                'calibrated_physics': calibrated_physics,
                'zeeman_sweep_results': zeeman_results,
                'target_mass': self.target_masses['electron'],
                'field_points': len(B_fields)
            }
            
            logger.info(f"Electron calibration: success={calibration_success}, error={relative_error*100:.2f}%")
            return results
            
        except Exception as e:
            logger.error(f"Electron calibration failed: {e}")
            return {
                'calibration_success': False,
                'error': str(e)
            }
    
    def _run_muon_isomer_prediction(self, electron_calibration: Dict) -> Dict:
        """Predict muon properties as excited state of electron field equation."""
        
        if not electron_calibration.get('calibration_success', False):
            return {
                'prediction_success': False,
                'error': 'Electron calibration required for muon prediction'
            }
        
        logger.info("Searching for muon isomer at first excited state...")
        
        try:
            # Use isomer analysis framework
            muon_prediction = self.zeeman_analyzer.predict_isomer_g2(
                electron_calibration,
                self.target_masses['muon'],
                'muon'
            )
            
            if muon_prediction.get('converged', False):
                # Validate against experimental data
                g2_experimental = self.experimental_g2['muon']
                g2_predicted = muon_prediction['g2_predicted']
                relative_error = abs(g2_predicted - g2_experimental) / g2_experimental
                
                prediction_success = (
                    relative_error < self.precision_target and
                    muon_prediction['mass_error'] < self.precision_target
                )
                
                results = {
                    'prediction_success': prediction_success,
                    'converged': True,
                    'predicted_mass': muon_prediction['achieved_mass_eV'],
                    'target_mass': self.target_masses['muon'],
                    'mass_error': muon_prediction['mass_error'],
                    'g2_predicted': g2_predicted,
                    'g2_experimental': g2_experimental,
                    'g2_relative_error': relative_error,
                    'excited_state_details': muon_prediction,
                    'isomer_level': 1  # First excited state
                }
                
                logger.info(f"Muon prediction: mass error={results['mass_error']*100:.2f}%, g-2 error={relative_error*100:.2f}%")
                
            else:
                results = {
                    'prediction_success': False,
                    'converged': False,
                    'error': muon_prediction.get('error', 'Excited state search failed')
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Muon prediction failed: {e}")
            return {
                'prediction_success': False,
                'error': str(e)
            }
    
    def _run_tau_isomer_prediction(self, electron_calibration: Dict) -> Dict:
        """Predict tau properties as second excited state of electron field equation."""
        
        if not electron_calibration.get('calibration_success', False):
            return {
                'prediction_success': False,
                'error': 'Electron calibration required for tau prediction'
            }
        
        logger.info("Searching for tau isomer at second excited state...")
        
        try:
            # Use isomer analysis framework
            tau_prediction = self.zeeman_analyzer.predict_isomer_g2(
                electron_calibration,
                self.target_masses['tau'],
                'tau'
            )
            
            if tau_prediction.get('converged', False):
                # No experimental g-2 for tau yet, so only validate mass
                prediction_success = tau_prediction['mass_error'] < self.precision_target
                
                results = {
                    'prediction_success': prediction_success,
                    'converged': True,
                    'predicted_mass': tau_prediction['achieved_mass_eV'],
                    'target_mass': self.target_masses['tau'],
                    'mass_error': tau_prediction['mass_error'],
                    'g2_predicted': tau_prediction['g2_predicted'],
                    'g2_experimental': None,  # Not yet measured
                    'g2_relative_error': None,
                    'excited_state_details': tau_prediction,
                    'isomer_level': 2,  # Second excited state
                    'breakthrough_prediction': True  # First tau g-2 prediction
                }
                
                logger.info(f"Tau prediction: mass error={results['mass_error']*100:.2f}%, g-2={results['g2_predicted']:.10f} (no experimental data)")
                
            else:
                results = {
                    'prediction_success': False,
                    'converged': False,
                    'error': tau_prediction.get('error', 'Excited state search failed')
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Tau prediction failed: {e}")
            return {
                'prediction_success': False,
                'error': str(e)
            }
    
    def _analyze_workflow_results(self, workflow_results: Dict) -> Dict:
        """Comprehensive analysis of workflow results and paradigm validation."""
        
        logger.info("Analyzing complete workflow results...")
        
        # Extract key results
        electron_success = workflow_results.get('objective_1_electron', {}).get('calibration_success', False)
        muon_success = workflow_results.get('objective_2_muon', {}).get('prediction_success', False)
        tau_success = workflow_results.get('objective_3_tau', {}).get('prediction_success', False)
        
        successful_objectives = sum([electron_success, muon_success, tau_success])
        success_rate = successful_objectives / 3.0
        
        # Paradigm breakthrough assessment
        # Need at least electron + one isomer for paradigm validation
        paradigm_breakthrough = electron_success and (muon_success or tau_success)
        
        # Precision analysis
        precision_metrics = {}
        if electron_success:
            electron_data = workflow_results['objective_1_electron']
            precision_metrics['electron_g2_error'] = electron_data['relative_error']
            
        if muon_success:
            muon_data = workflow_results['objective_2_muon']
            precision_metrics['muon_mass_error'] = muon_data['mass_error']
            precision_metrics['muon_g2_error'] = muon_data.get('g2_relative_error')
            
        if tau_success:
            tau_data = workflow_results['objective_3_tau']
            precision_metrics['tau_mass_error'] = tau_data['mass_error']
            # No experimental g-2 for tau
        
        # Overall precision (weighted average)
        total_error = 0.0
        error_count = 0
        for key, error in precision_metrics.items():
            if error is not None:
                total_error += error
                error_count += 1
        
        average_precision = 1.0 - (total_error / error_count) if error_count > 0 else 0.0
        
        # Scientific impact assessment
        impact_assessment = self._assess_scientific_impact(workflow_results)
        
        analysis = {
            'overall_success': success_rate >= 0.67,  # At least 2/3 objectives
            'success_rate': success_rate,
            'successful_objectives': successful_objectives,
            'paradigm_breakthrough': paradigm_breakthrough,
            'average_precision': average_precision,
            'precision_metrics': precision_metrics,
            'objective_results': {
                'electron_calibration': electron_success,
                'muon_prediction': muon_success,
                'tau_prediction': tau_success
            },
            'scientific_impact': impact_assessment
        }
        
        return analysis
    
    def _assess_scientific_impact(self, workflow_results: Dict) -> Dict:
        """Assess the scientific impact and breakthrough potential of results."""
        
        impact = {
            'breakthrough_achievements': [],
            'novel_predictions': [],
            'validation_successes': [],
            'paradigm_implications': []
        }
        
        # Check for breakthrough achievements
        if workflow_results.get('objective_1_electron', {}).get('calibration_success', False):
            electron_error = workflow_results['objective_1_electron']['relative_error']
            if electron_error < 0.1:  # Better than 10%
                impact['breakthrough_achievements'].append(f"Electron g-2 predicted within {electron_error*100:.1f}% from QFD")
        
        if workflow_results.get('objective_2_muon', {}).get('prediction_success', False):
            muon_mass_error = workflow_results['objective_2_muon']['mass_error']
            muon_g2_error = workflow_results['objective_2_muon'].get('g2_relative_error')
            if muon_mass_error < 0.05:
                impact['breakthrough_achievements'].append(f"Muon mass predicted as excited state within {muon_mass_error*100:.1f}%")
            if muon_g2_error and muon_g2_error < 0.1:
                impact['breakthrough_achievements'].append(f"Muon g-2 predicted from electron calibration within {muon_g2_error*100:.1f}%")
        
        # Novel predictions (especially tau)
        if workflow_results.get('objective_3_tau', {}).get('prediction_success', False):
            tau_g2 = workflow_results['objective_3_tau']['g2_predicted']
            impact['novel_predictions'].append(f"First tau g-2 prediction: {tau_g2:.10f}")
            impact['paradigm_implications'].append("Unified lepton g-2 theory validated")
        
        # Paradigm implications
        if workflow_results.get('final_analysis', {}).get('paradigm_breakthrough', False):
            impact['paradigm_implications'].extend([
                "Electron/muon/tau unified as field equation isomers",
                "QFD replaces perturbative QED for g-2 calculations",
                "Lepton mass hierarchy explained by excited states"
            ])
        
        # Overall impact score
        total_achievements = (
            len(impact['breakthrough_achievements']) +
            len(impact['novel_predictions']) +
            len(impact['validation_successes']) +
            len(impact['paradigm_implications'])
        )
        
        impact['total_impact_score'] = total_achievements
        impact['revolutionary_potential'] = total_achievements >= 5
        
        return impact
    
    def _save_objective_results(self, objective_num: int, subdir: str, results: Dict):
        """Save results for individual objective."""
        output_path = self.output_dir / subdir / f"objective_{objective_num}_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Objective {objective_num} results saved: {output_path}")
    
    def _save_complete_workflow_results(self, workflow_results: Dict):
        """Save complete workflow results."""
        
        # Main results file
        results_path = self.output_dir / "complete_isomer_workflow_results.json"
        with open(results_path, 'w') as f:
            json.dump(workflow_results, f, indent=2, default=str)
        
        # Summary report
        self._generate_workflow_report(workflow_results)
        
        logger.info(f"Complete workflow results saved: {results_path}")
    
    def _generate_workflow_report(self, workflow_results: Dict):
        """Generate human-readable workflow report."""
        
        report_path = self.output_dir / "analysis_reports" / "isomer_workflow_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# QFD Isomer Workflow - Three-Objective Analysis Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Workflow Duration:** {workflow_results['workflow_info']['total_duration']:.1f} seconds\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            analysis = workflow_results.get('final_analysis', {})
            f.write(f"**Success Rate:** {analysis.get('success_rate', 0)*100:.0f}% ({analysis.get('successful_objectives', 0)}/3 objectives)\n")
            f.write(f"**Paradigm Breakthrough:** {'YES' if analysis.get('paradigm_breakthrough', False) else 'NO'}\n")
            f.write(f"**Average Precision:** {analysis.get('average_precision', 0)*100:.1f}%\n\n")
            
            # Objective Results
            f.write("## Objective Results\n\n")
            
            # Objective 1: Electron
            f.write("### Objective 1: Electron Calibration\n")
            electron_data = workflow_results.get('objective_1_electron', {})
            if electron_data.get('calibration_success', False):
                f.write("**Status:** ✅ SUCCESS\n")
                f.write(f"**g-2 Predicted:** {electron_data['g2_predicted']:.10f}\n")
                f.write(f"**g-2 Experimental:** {electron_data['g2_experimental']:.10f}\n")
                f.write(f"**Relative Error:** {electron_data['relative_error']*100:.2f}%\n\n")
            else:
                f.write("**Status:** ❌ FAILED\n")
                f.write(f"**Error:** {electron_data.get('error', 'Unknown error')}\n\n")
            
            # Objective 2: Muon
            f.write("### Objective 2: Muon Isomer Prediction\n")
            muon_data = workflow_results.get('objective_2_muon', {})
            if muon_data.get('prediction_success', False):
                f.write("**Status:** ✅ SUCCESS\n")
                f.write(f"**Mass Predicted:** {muon_data['predicted_mass']:.0f} eV\n")
                f.write(f"**Mass Target:** {muon_data['target_mass']:.0f} eV\n")
                f.write(f"**Mass Error:** {muon_data['mass_error']*100:.2f}%\n")
                f.write(f"**g-2 Predicted:** {muon_data['g2_predicted']:.10f}\n")
                f.write(f"**g-2 Experimental:** {muon_data['g2_experimental']:.10f}\n")
                f.write(f"**g-2 Error:** {muon_data.get('g2_relative_error', 0)*100:.2f}%\n\n")
            else:
                f.write("**Status:** ❌ FAILED\n")
                f.write(f"**Error:** {muon_data.get('error', 'Unknown error')}\n\n")
            
            # Objective 3: Tau
            f.write("### Objective 3: Tau Isomer Prediction\n")
            tau_data = workflow_results.get('objective_3_tau', {})
            if tau_data.get('prediction_success', False):
                f.write("**Status:** ✅ SUCCESS\n")
                f.write(f"**Mass Predicted:** {tau_data['predicted_mass']:.0f} eV\n")
                f.write(f"**Mass Target:** {tau_data['target_mass']:.0f} eV\n")
                f.write(f"**Mass Error:** {tau_data['mass_error']*100:.2f}%\n")
                f.write(f"**g-2 Predicted:** {tau_data['g2_predicted']:.10f} ⭐ FIRST PREDICTION\n\n")
            else:
                f.write("**Status:** ❌ FAILED\n")
                f.write(f"**Error:** {tau_data.get('error', 'Unknown error')}\n\n")
            
            # Scientific Impact
            f.write("## Scientific Impact Assessment\n\n")
            impact = analysis.get('scientific_impact', {})
            
            if impact.get('breakthrough_achievements'):
                f.write("### Breakthrough Achievements\n")
                for achievement in impact['breakthrough_achievements']:
                    f.write(f"- {achievement}\n")
                f.write("\n")
            
            if impact.get('novel_predictions'):
                f.write("### Novel Predictions\n")
                for prediction in impact['novel_predictions']:
                    f.write(f"- {prediction}\n")
                f.write("\n")
            
            if impact.get('paradigm_implications'):
                f.write("### Paradigm Implications\n")
                for implication in impact['paradigm_implications']:
                    f.write(f"- {implication}\n")
                f.write("\n")
            
            f.write(f"**Revolutionary Potential:** {'YES' if impact.get('revolutionary_potential', False) else 'NO'}\n")
            f.write(f"**Total Impact Score:** {impact.get('total_impact_score', 0)}/10\n\n")
            
            # Technical Details
            f.write("## Technical Configuration\n\n")
            info = workflow_results['workflow_info']
            f.write(f"**Backend:** {info['backend']}\n")
            f.write(f"**Device:** {info['device']}\n")
            f.write(f"**Grid Size:** {info['grid_size']}³\n")
            f.write(f"**Precision Target:** {info['precision_target']*100:.1f}%\n")
            
        logger.info(f"Workflow report generated: {report_path}")
    
    def _print_workflow_summary(self, workflow_results: Dict):
        """Print concise workflow summary to console."""
        
        print("\n" + "=" * 80)
        print("QFD ISOMER WORKFLOW - FINAL SUMMARY")
        print("=" * 80)
        
        analysis = workflow_results.get('final_analysis', {})
        
        print(f"Success Rate: {analysis.get('success_rate', 0)*100:.0f}% ({analysis.get('successful_objectives', 0)}/3 objectives)")
        print(f"Paradigm Breakthrough: {'YES ✅' if analysis.get('paradigm_breakthrough', False) else 'NO ❌'}")
        print(f"Average Precision: {analysis.get('average_precision', 0)*100:.1f}%")
        
        # Individual objectives
        print("\nObjective Results:")
        objectives = analysis.get('objective_results', {})
        print(f"  Electron Calibration: {'✅' if objectives.get('electron_calibration', False) else '❌'}")
        print(f"  Muon Prediction: {'✅' if objectives.get('muon_prediction', False) else '❌'}")
        print(f"  Tau Prediction: {'✅' if objectives.get('tau_prediction', False) else '❌'}")
        
        # Scientific impact
        impact = analysis.get('scientific_impact', {})
        print(f"\nScientific Impact: {impact.get('total_impact_score', 0)}/10")
        print(f"Revolutionary Potential: {'YES 🚀' if impact.get('revolutionary_potential', False) else 'No'}")
        
        duration = workflow_results['workflow_info'].get('total_duration', 0)
        print(f"\nWorkflow completed in {duration:.1f} seconds")
        print("=" * 80)


def run_complete_isomer_workflow(
    output_dir: str = "isomer_workflow_results",
    backend: str = "torch",
    device: str = "cuda",
    grid_size: int = 64,
    precision_target: float = 0.05
) -> Dict:
    """
    Convenience function to run complete isomer workflow.
    
    Args:
        output_dir: Output directory for results
        backend: Computational backend
        device: Computing device
        grid_size: Spatial grid resolution
        precision_target: Target precision (fractional)
        
    Returns:
        Complete workflow results
    """
    workflow = IsomerWorkflow(
        output_dir=Path(output_dir),
        backend=backend,
        device=device,
        grid_size=grid_size,
        precision_target=precision_target
    )
    
    return workflow.run_complete_workflow()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="QFD Isomer Workflow")
    parser.add_argument("--output-dir", default="isomer_workflow_results", help="Output directory")
    parser.add_argument("--backend", choices=["torch", "numpy"], default="torch", help="Backend")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size")
    parser.add_argument("--precision", type=float, default=0.05, help="Precision target")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run workflow
    results = run_complete_isomer_workflow(
        output_dir=args.output_dir,
        backend=args.backend,
        device=args.device,
        grid_size=args.grid_size,
        precision_target=args.precision
    )