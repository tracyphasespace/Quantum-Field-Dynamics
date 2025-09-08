#!/usr/bin/env python3
"""
QFD Stability Analysis - Integrated with Phoenix Framework
==========================================================

Post-processing lifetime prediction for QFD isomer states using geometric 
stability indices. Integrated with the refactored Phoenix framework.

Physics Model:
    tau = k₀ · G / (1 + k_csr · χ)
    
Where:
    G   = (U/U₀)^aU · (R/R_e)^aR · (I/I_e)^aI · (K_e/K)^aK  (geometric index)
    χ   = |L · Q| / I                                        (charge-spin handle)
    
This extends the three-objective paradigm with lifetime predictions:
1. Electron calibration locks physics parameters
2. Excited state search finds muon/tau isomers  
3. Unified g-2 predictions from Zeeman experiments
4. Stability prediction from geometric field properties

Usage:
    from utils.stability_analysis import StabilityPredictor
    predictor = StabilityPredictor()
    results = predictor.analyze_stability(electron_results, muon_results, tau_results)
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class StabilityPredictor:
    """
    Predicts lepton lifetimes from QFD simulation results using geometric stability indices.
    
    This extends the isomer framework with stability analysis, providing a complete
    theoretical approach to lepton physics from field dynamics.
    """
    
    def __init__(
        self,
        tau_muon_exp: float = 2.196981e-6,  # Experimental muon lifetime (s)
        tau_tau_exp: float = 2.903e-13,     # Experimental tau lifetime (s)
        U0: float = 0.99,                   # Reference velocity anchor
        exponents: Optional[Dict[str, float]] = None
    ):
        """
        Initialize stability predictor.
        
        Args:
            tau_muon_exp: Experimental muon lifetime in seconds
            tau_tau_exp: Experimental tau lifetime in seconds  
            U0: Reference velocity anchor for geometric index
            exponents: Exponents for geometric index (aU, aR, aI, aK)
        """
        self.tau_muon_exp = tau_muon_exp
        self.tau_tau_exp = tau_tau_exp
        self.U0 = U0
        
        # Default exponents for geometric index G
        default_exp = {"aU": 4.0, "aR": 1.0, "aI": 1.0, "aK": 1.0}
        self.exponents = {**default_exp, **(exponents or {})}
        
        logger.info(f"StabilityPredictor initialized: tau_mu={tau_muon_exp:.3e}s, tau_tau={tau_tau_exp:.3e}s, U0={self.U0}")
    
    def extract_stability_features(self, phoenix_results: Dict) -> Dict[str, float]:
        """
        Extract stability-relevant features from Phoenix solver results. 
        
        Args:
            phoenix_results: Results dictionary from solve_psi_field()
            
        Returns:
            Dictionary with extracted stability features
        """
        # Extract basic properties
        energy = phoenix_results.get('H_final', phoenix_results.get('energy', 0.0))
        psi_field = phoenix_results.get('psi_field')
        
        if psi_field is not None:
            # Convert to numpy if needed
            if hasattr(psi_field, 'numpy'):
                psi_field = psi_field.numpy()
            elif hasattr(psi_field, 'cpu'):
                psi_field = psi_field.cpu().numpy()
            
            # Compute field-based proxies
            field_magnitude = np.abs(psi_field)
            
            # Effective radius (RMS radius of field distribution)
            grid_size = psi_field.shape[0]
            x, y, z = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
            center = grid_size / 2
            r_sq = (x - center)**2 + (y - center)**2 + (z - center)**2
            
            total_field = np.sum(field_magnitude**2)
            if total_field > 0:
                R_eff = np.sqrt(np.sum(r_sq * field_magnitude**2) / total_field)
            else:
                R_eff = 1.0
            
            # Inertia proxy (field concentration measure)
            I_proxy = np.sum(r_sq * field_magnitude**2) / max(total_field, 1e-12)
            
            # Charge proxy (field gradient magnitude)
            grad_x = np.gradient(psi_field, axis=0)
            grad_y = np.gradient(psi_field, axis=1) 
            grad_z = np.gradient(psi_field, axis=2)
            Q_proxy = np.sqrt(np.sum(grad_x**2 + grad_y**2 + grad_z**2))
            
            # Spin proxy (field circulation/curl measure)
            # Simplified as cross-gradient correlation
            L_proxy = np.sum(np.abs(grad_x * grad_y + grad_y * grad_z + grad_z * grad_x))
            
            # Curvature proxy (second derivatives)
            d2_dx2 = np.gradient(grad_x, axis=0)
            d2_dy2 = np.gradient(grad_y, axis=1)
            d2_dz2 = np.gradient(grad_z, axis=2)
            K_proxy = np.sum(np.abs(d2_dx2 + d2_dy2 + d2_dz2))
            
        else:
            # Fallback values if field data not available
            R_eff = 1.0
            I_proxy = 1.0
            Q_proxy = 0.1
            L_proxy = 0.1
            K_proxy = 1e-6
        
        # Velocity proxy from energy and mass
        constants = phoenix_results.get('constants', {})
        physics = constants.get('physics_constants', {})
        mass_estimate = energy / (3e8)**2 if energy > 0 else 1.0  # E ≈ mc²
        U_proxy = min(0.99, np.sqrt(2 * energy / max(mass_estimate, 1e-12)) / 3e8)
        
        features = {
            'U': float(U_proxy),
            'R_eff': float(R_eff),
            'I': float(I_proxy),
            'Q': float(Q_proxy),
            'L': float(L_proxy),
            'K': float(max(K_proxy, 1e-12)),  # Floor to avoid division by zero
            'energy': float(energy),
            'grid_size': psi_field.shape[0] if psi_field is not None else 64
        }
        
        logger.debug(f"Extracted features: {features}")
        return features
    
    def compute_geometric_index(
        self, 
        features: Dict[str, float], 
        reference: Dict[str, float]
    ) -> float:
        """
        Compute geometric stability index G.
        
        G = (U/U₀)^aU · (R/R_ref)^aR · (I/I_ref)^aI · (K_ref/K)^aK
        
        Args:
            features: Features for current particle
            reference: Reference features (typically electron)
            
        Returns:
            Geometric index value
        """
        eps = 1e-12
        
        U_factor = max(features['U'], eps) / max(self.U0, eps)
        R_factor = max(features['R_eff'], eps) / max(reference['R_eff'], eps)
        I_factor = max(features['I'], eps) / max(reference['I'], eps)
        K_factor = max(reference['K'], eps) / max(features['K'], eps)  # Inverted
        
        G = (U_factor ** self.exponents['aU'] * 
             R_factor ** self.exponents['aR'] * 
             I_factor ** self.exponents['aI'] * 
             K_factor ** self.exponents['aK'])
        
        return float(G)
    
    def compute_csr_handle(self, features: Dict[str, float]) -> float:
        """
        Compute charge-spin reinforcement handle χ.
        
        χ = |L · Q| / I
        
        Args:
            features: Particle features
            
        Returns:
            CSR handle value
        """
        eps = 1e-12
        chi = abs(features['L'] * features['Q']) / max(features['I'], eps)
        return float(chi)

    def _lifetime_error(self, params, G_muon, G_tau, chi_muon, chi_tau):
        """Objective function for the optimizer."""
        k_csr, k0 = params
        
        # --- PHYSICALITY CONSTRAINT ---
        if k_csr < 0 or k0 < 0:
            return 1e9 # Return a huge error if parameters are unphysical

        # Predict lifetimes using the model
        tau_mu_pred = self.predict_lifetime(k0, k_csr, G_muon, chi_muon)
        tau_tau_pred = self.predict_lifetime(k0, k_csr, G_tau, chi_tau)
            
        # Calculate the squared log error (good for multi-scale data)
        # Add a small epsilon to avoid log(0)
        eps = 1e-30
        err_mu = (math.log(tau_mu_pred + eps) - math.log(self.tau_muon_exp + eps))**2
        err_tau = (math.log(tau_tau_pred + eps) - math.log(self.tau_tau_exp + eps))**2
            
        return err_mu + err_tau

    def fit_stability_parameters(
        self,
        G_muon: float, G_tau: float,
        chi_muon: float, chi_tau: float
    ) -> Tuple[float, float, bool]:
        """
        Solve for stability parameters k_csr and k0 using a constrained
        numerical optimizer.
        """
        initial_guess = [0.1, 1e-5] # Start with small, positive values
        bounds = [(0, None), (0, None)] # Enforce k_csr >= 0 and k0 >= 0

        result = minimize(
            self._lifetime_error,
            initial_guess,
            args=(G_muon, G_tau, chi_muon, chi_tau),
            bounds=bounds,
            method='L-BFGS-B' # A good choice for bounded optimization
        )
        
        k_csr_fit, k0_fit = result.x
        
        return float(k_csr_fit), float(k0_fit), result.success
    
    def predict_lifetime(self, k0: float, k_csr: float, G: float, chi: float) -> float:
        """Predict lifetime from stability model."""
        eps = 1e-12
        tau = k0 * G / (1.0 + k_csr * chi + eps)
        return float(tau)
    
    def analyze_stability(
        self,
        electron_results: Dict,
        muon_results: Dict, 
        tau_results: Dict,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Complete stability analysis for electron/muon/tau triplet.
        
        Args:
            electron_results: Phoenix solver results for electron
            muon_results: Phoenix solver results for muon
            tau_results: Phoenix solver results for tau
            output_dir: Optional directory to save results
            
        Returns:
            Complete stability analysis results
        """
        logger.info("Starting complete stability analysis...")
        
        # Extract features
        electron_features = self.extract_stability_features(electron_results)
        muon_features = self.extract_stability_features(muon_results) 
        tau_features = self.extract_stability_features(tau_results)
        
        # Compute geometric indices (electron as reference)
        G_electron = self.compute_geometric_index(electron_features, electron_features)
        G_muon = self.compute_geometric_index(muon_features, electron_features)
        G_tau = self.compute_geometric_index(tau_features, electron_features)
        
        # Compute CSR handles
        chi_electron = self.compute_csr_handle(electron_features)
        chi_muon = self.compute_csr_handle(muon_features)
        chi_tau = self.compute_csr_handle(tau_features)
        
        # Fit stability parameters
        k_csr, k0, fit_success = self.fit_stability_parameters(
            G_muon, G_tau, chi_muon, chi_tau
        )
        
        # Predict lifetimes
        tau_electron_pred = self.predict_lifetime(k0, k_csr, G_electron, chi_electron)
        tau_muon_pred = self.predict_lifetime(k0, k_csr, G_muon, chi_muon)
        tau_tau_pred = self.predict_lifetime(k0, k_csr, G_tau, chi_tau)
        
        # Compute errors
        muon_error = tau_muon_pred - self.tau_muon_exp
        tau_error = tau_tau_pred - self.tau_tau_exp
        
        # Assemble results
        analysis_results = {
            'model_info': {
                'description': 'QFD Stability Analysis via Constrained Numerical Optimization',
                'tau_muon_experimental': self.tau_muon_exp,
                'tau_tau_experimental': self.tau_tau_exp,
                'U0_reference': self.U0,
                'exponents': self.exponents.copy()
            },
            'features': {
                'electron': electron_features,
                'muon': muon_features,
                'tau': tau_features
            },
            'geometric_indices': {
                'G_electron': G_electron,
                'G_muon': G_muon,
                'G_tau': G_tau
            },
            'csr_handles': {
                'chi_electron': chi_electron,
                'chi_muon': chi_muon,
                'chi_tau': chi_tau
            },
            'fitted_parameters': {
                'k_csr': k_csr,
                'k0': k0,
                'fit_successful': fit_success
            },
            'predicted_lifetimes': {
                'electron_model_s': tau_electron_pred,
                'muon_s': tau_muon_pred,
                'tau_s': tau_tau_pred
            },
            'prediction_errors': {
                'muon_error_s': muon_error,
                'tau_error_s': tau_error,
                'muon_relative_error': abs(muon_error) / self.tau_muon_exp if self.tau_muon_exp else 0,
                'tau_relative_error': abs(tau_error) / self.tau_tau_exp if self.tau_tau_exp else 0
            },
            'analysis_summary': {
                'model_valid': fit_success,
                'muon_prediction_accuracy': 1.0 - (abs(muon_error) / self.tau_muon_exp if self.tau_muon_exp else 1),
                'tau_prediction_accuracy': 1.0 - (abs(tau_error) / self.tau_tau_exp if self.tau_tau_exp else 1),
                'overall_success': fit_success
            }
        }
        
        # Save results if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = output_dir / "stability_analysis_constrained.json"
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Constrained stability analysis results saved: {results_file}")
        
        # Log summary
        logger.info("Constrained stability analysis completed:")
        logger.info(f"  k_csr = {k_csr:.6g}, k0 = {k0:.6g}")
        logger.info(f"  Muon: tau_pred = {tau_muon_pred:.3e}s (error: {muon_error/self.tau_muon_exp*100:.1f}%)")
        logger.info(f"  Tau:  tau_pred = {tau_tau_pred:.3e}s (error: {tau_error/self.tau_tau_exp*100:.1f}%)")
        
        return analysis_results
    
    def format_summary_report(self, analysis_results: Dict) -> str:
        """Generate human-readable summary report."""
        
        def _fmt(x: float) -> str:
            """Format numbers with appropriate precision."""
            if x == 0.0:
                return "0"
            mag = abs(x)
            if 1e-3 <= mag <= 1e4:
                return f"{x:.6g}"
            return f"{x:.3e}"
        
        report = []
        report.append("QFD Constrained Stability Analysis Summary")
        report.append("=" * 50)
        report.append("")
        
        # Model parameters
        params = analysis_results['fitted_parameters']
        report.append("Fitted Parameters (Constrained):")
        report.append(f"  k_csr = {_fmt(params['k_csr'])}")
        report.append(f"  k0    = {_fmt(params['k0'])}")
        report.append(f"  Fit successful: {params['fit_successful']}")
        report.append("")
        
        # Predictions vs experiment
        pred = analysis_results['predicted_lifetimes']
        errors = analysis_results['prediction_errors']
        exp_mu = analysis_results['model_info']['tau_muon_experimental']
        exp_tau = analysis_results['model_info']['tau_tau_experimental']
        
        report.append("Lifetime Predictions:")
        report.append(f"  Muon:     tau_pred = {_fmt(pred['muon_s'])}s, tau_exp = {_fmt(exp_mu)}s")
        report.append(f"            Error = {_fmt(errors['muon_error_s'])}s ({errors['muon_relative_error']*100:.1f}%)")
        report.append(f"  Tau:      tau_pred = {_fmt(pred['tau_s'])}s, tau_exp = {_fmt(exp_tau)}s") 
        report.append(f"            Error = {_fmt(errors['tau_error_s'])}s ({errors['tau_relative_error']*100:.1f}%)")
        report.append(f"  Electron: tau_model = {_fmt(pred['electron_model_s'])}s (stable)")
        report.append("")
        
        # Summary assessment
        summary = analysis_results['analysis_summary']
        report.append("Analysis Assessment:")
        report.append(f"  Model validity: {summary['model_valid']}")
        report.append(f"  Muon accuracy:  {summary['muon_prediction_accuracy']*100:.1f}%")
        report.append(f"  Tau accuracy:   {summary['tau_prediction_accuracy']*100:.1f}%")
        report.append(f"  Overall success: {summary['overall_success']}")
        
        return "\n".join(report)

def run_stability_analysis_cli():
    """Command-line interface for stability analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QFD Stability Analysis")
    parser.add_argument("--electron-results", required=True, 
                       help="Path to electron results JSON file")
    parser.add_argument("--muon-results", required=True,
                       help="Path to muon results JSON file") 
    parser.add_argument("--tau-results", required=True,
                       help="Path to tau results JSON file")
    parser.add_argument("--output-dir", default="stability_analysis_output",
                       help="Output directory for analysis results")
    parser.add_argument("--tau-muon", type=float, default=2.196981e-6,
                       help="Experimental muon lifetime (s)")
    parser.add_argument("--tau-tau", type=float, default=2.903e-13,
                       help="Experimental tau lifetime (s)")
    
    args = parser.parse_args()
    
    # Load results
    with open(args.electron_results) as f:
        electron_results = json.load(f)
    with open(args.muon_results) as f:
        muon_results = json.load(f)
    with open(args.tau_results) as f:
        tau_results = json.load(f)
    
    # Run analysis
    predictor = StabilityPredictor(
        tau_muon_exp=args.tau_muon,
        tau_tau_exp=args.tau_tau
    )
    
    results = predictor.analyze_stability(
        electron_results, muon_results, tau_results,
        output_dir=Path(args.output_dir)
    )
    
    # Print summary
    print(predictor.format_summary_report(results))


if __name__ == "__main__":
    run_stability_analysis_cli()
