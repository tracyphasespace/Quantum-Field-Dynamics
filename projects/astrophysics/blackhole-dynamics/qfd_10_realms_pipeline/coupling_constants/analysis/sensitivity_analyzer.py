"""
Sensitivity analyzer for coupling constants.

This module provides tools for analyzing parameter sensitivity through
numerical derivatives, Monte Carlo sampling, and statistical analysis.
"""

import numpy as np
import time
from typing import Dict, List, Callable, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

from ..registry.parameter_registry import ParameterRegistry


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis for a single observable."""
    observable_name: str
    parameter_sensitivities: Dict[str, float]  # parameter -> sensitivity value
    baseline_value: float
    analysis_method: str  # 'numerical_derivative', 'monte_carlo', 'finite_difference'
    perturbation_size: float
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo sensitivity analysis."""
    observable_name: str
    parameter_statistics: Dict[str, Dict[str, float]]  # parameter -> {mean, std, min, max, percentiles}
    correlation_matrix: np.ndarray
    parameter_names: List[str]
    n_samples: int
    baseline_value: float
    execution_time_ms: float
    convergence_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterRanking:
    """Ranking of parameters by their impact on observables."""
    observable_name: str
    ranked_parameters: List[Tuple[str, float]]  # (parameter_name, impact_score)
    ranking_method: str
    total_variance_explained: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type alias for observable functions
# Takes ParameterRegistry, returns observable value
ObservableFunction = Callable[[ParameterRegistry], float]


class SensitivityAnalyzer:
    """
    Analyzes parameter sensitivity for coupling constants.
    
    Provides numerical derivative computation, Monte Carlo analysis,
    and parameter ranking capabilities.
    """
    
    def __init__(self, registry: ParameterRegistry):
        """
        Initialize sensitivity analyzer.
        
        Args:
            registry: ParameterRegistry instance to analyze
        """
        self.registry = registry
        self.observables: Dict[str, ObservableFunction] = {}
        self.sensitivity_results: List[SensitivityResult] = []
        self.monte_carlo_results: List[MonteCarloResult] = []
        self.parameter_rankings: List[ParameterRanking] = []
        
    def register_observable(self, name: str, observable_func: ObservableFunction) -> None:
        """
        Register an observable function for sensitivity analysis.
        
        Args:
            name: Name of the observable
            observable_func: Function that computes observable from registry
        """
        self.observables[name] = observable_func
        
    def compute_parameter_sensitivity(self, observable_name: str, 
                                    perturbation_size: float = 1e-6,
                                    method: str = 'central_difference') -> SensitivityResult:
        """
        Compute sensitivity of an observable to all parameters using numerical derivatives.
        
        Args:
            observable_name: Name of registered observable
            perturbation_size: Size of parameter perturbation for numerical derivative
            method: Method for numerical derivative ('forward', 'backward', 'central_difference')
            
        Returns:
            SensitivityResult with parameter sensitivities
        """
        if observable_name not in self.observables:
            raise ValueError(f"Observable '{observable_name}' not registered")
        
        start_time = time.time()
        observable_func = self.observables[observable_name]
        
        # Get baseline value
        baseline_value = observable_func(self.registry)
        
        # Get all parameters with values
        all_params = self.registry.get_all_parameters()
        parameters_to_analyze = {
            name: param for name, param in all_params.items() 
            if param.value is not None
        }
        
        sensitivities = {}
        
        for param_name, param_state in parameters_to_analyze.items():
            original_value = param_state.value
            
            # Validate method first
            if method not in ['forward', 'backward', 'central_difference']:
                raise ValueError(f"Unknown method: {method}")
            
            try:
                if method == 'forward':
                    # Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h
                    perturbed_value = original_value + perturbation_size
                    self._set_parameter_value(param_name, perturbed_value)
                    perturbed_observable = observable_func(self.registry)
                    
                    sensitivity = (perturbed_observable - baseline_value) / perturbation_size
                    
                elif method == 'backward':
                    # Backward difference: f'(x) ≈ (f(x) - f(x-h)) / h
                    perturbed_value = original_value - perturbation_size
                    self._set_parameter_value(param_name, perturbed_value)
                    perturbed_observable = observable_func(self.registry)
                    
                    sensitivity = (baseline_value - perturbed_observable) / perturbation_size
                    
                elif method == 'central_difference':
                    # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
                    # Forward perturbation
                    self._set_parameter_value(param_name, original_value + perturbation_size)
                    forward_value = observable_func(self.registry)
                    
                    # Backward perturbation
                    self._set_parameter_value(param_name, original_value - perturbation_size)
                    backward_value = observable_func(self.registry)
                    
                    sensitivity = (forward_value - backward_value) / (2 * perturbation_size)
                
                sensitivities[param_name] = sensitivity
                
            except Exception as e:
                # If perturbation causes error, set sensitivity to 0
                sensitivities[param_name] = 0.0
                
            finally:
                # Restore original value
                self._set_parameter_value(param_name, original_value)
        
        execution_time = (time.time() - start_time) * 1000
        
        result = SensitivityResult(
            observable_name=observable_name,
            parameter_sensitivities=sensitivities,
            baseline_value=baseline_value,
            analysis_method=f"numerical_derivative_{method}",
            perturbation_size=perturbation_size,
            execution_time_ms=execution_time,
            metadata={
                'parameters_analyzed': len(sensitivities),
                'method': method
            }
        )
        
        self.sensitivity_results.append(result)
        return result
    
    def perform_monte_carlo_analysis(self, observable_name: str, 
                                   n_samples: int = 1000,
                                   parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> MonteCarloResult:
        """
        Perform Monte Carlo sensitivity analysis.
        
        Args:
            observable_name: Name of registered observable
            n_samples: Number of Monte Carlo samples
            parameter_ranges: Optional dict of parameter -> (min, max) ranges
            
        Returns:
            MonteCarloResult with statistical analysis
        """
        if observable_name not in self.observables:
            raise ValueError(f"Observable '{observable_name}' not registered")
        
        start_time = time.time()
        observable_func = self.observables[observable_name]
        
        # Get baseline value
        baseline_value = observable_func(self.registry)
        
        # Get parameters to vary
        all_params = self.registry.get_all_parameters()
        parameters_to_vary = {}
        
        for param_name, param_state in all_params.items():
            if param_state.value is not None:
                # Determine parameter range
                if parameter_ranges and param_name in parameter_ranges:
                    param_min, param_max = parameter_ranges[param_name]
                else:
                    # Use constraint bounds if available
                    bounds = self._get_parameter_bounds(param_state)
                    if bounds:
                        param_min, param_max = bounds
                    else:
                        # Default: ±20% of current value
                        current_val = param_state.value
                        param_min = current_val * 0.8
                        param_max = current_val * 1.2
                
                parameters_to_vary[param_name] = (param_min, param_max, param_state.value)
        
        if not parameters_to_vary:
            raise ValueError("No parameters available for Monte Carlo analysis")
        
        # Store original values
        original_values = {name: info[2] for name, info in parameters_to_vary.items()}
        
        # Generate samples
        param_names = list(parameters_to_vary.keys())
        n_params = len(param_names)
        
        # Latin Hypercube Sampling for better coverage
        samples = self._latin_hypercube_sample(n_samples, n_params)
        
        # Scale samples to parameter ranges
        scaled_samples = np.zeros_like(samples)
        for i, param_name in enumerate(param_names):
            param_min, param_max, _ = parameters_to_vary[param_name]
            scaled_samples[:, i] = param_min + samples[:, i] * (param_max - param_min)
        
        # Evaluate observable for each sample
        observable_values = []
        parameter_samples = {name: [] for name in param_names}
        
        for sample_idx in range(n_samples):
            try:
                # Set parameter values for this sample
                for i, param_name in enumerate(param_names):
                    sample_value = scaled_samples[sample_idx, i]
                    self._set_parameter_value(param_name, sample_value)
                    parameter_samples[param_name].append(sample_value)
                
                # Evaluate observable
                obs_value = observable_func(self.registry)
                observable_values.append(obs_value)
                
            except Exception:
                # If evaluation fails, use baseline value
                observable_values.append(baseline_value)
                
        # Restore original values
        for param_name, original_value in original_values.items():
            self._set_parameter_value(param_name, original_value)
        
        # Compute statistics
        parameter_statistics = {}
        for param_name in param_names:
            param_values = np.array(parameter_samples[param_name])
            parameter_statistics[param_name] = {
                'mean': float(np.mean(param_values)),
                'std': float(np.std(param_values)),
                'min': float(np.min(param_values)),
                'max': float(np.max(param_values)),
                'percentile_25': float(np.percentile(param_values, 25)),
                'percentile_50': float(np.percentile(param_values, 50)),
                'percentile_75': float(np.percentile(param_values, 75))
            }
        
        # Compute correlation matrix
        sample_matrix = np.column_stack([parameter_samples[name] for name in param_names])
        correlation_matrix = np.corrcoef(sample_matrix.T)
        
        # Convergence analysis
        convergence_info = self._analyze_convergence(observable_values)
        
        execution_time = (time.time() - start_time) * 1000
        
        result = MonteCarloResult(
            observable_name=observable_name,
            parameter_statistics=parameter_statistics,
            correlation_matrix=correlation_matrix,
            parameter_names=param_names,
            n_samples=n_samples,
            baseline_value=baseline_value,
            execution_time_ms=execution_time,
            convergence_info=convergence_info
        )
        
        self.monte_carlo_results.append(result)
        return result
    
    def rank_parameters_by_impact(self, observables: List[str], 
                                method: str = 'variance_based') -> List[ParameterRanking]:
        """
        Rank parameters by their impact on multiple observables.
        
        Args:
            observables: List of observable names to analyze
            method: Ranking method ('variance_based', 'sensitivity_based')
            
        Returns:
            List of ParameterRanking objects
        """
        rankings = []
        
        for observable_name in observables:
            if method == 'variance_based':
                ranking = self._rank_by_variance(observable_name)
            elif method == 'sensitivity_based':
                ranking = self._rank_by_sensitivity(observable_name)
            else:
                raise ValueError(f"Unknown ranking method: {method}")
            
            rankings.append(ranking)
        
        self.parameter_rankings.extend(rankings)
        return rankings
    
    def _rank_by_sensitivity(self, observable_name: str) -> ParameterRanking:
        """Rank parameters by absolute sensitivity values."""
        # Find sensitivity result for this observable
        sensitivity_result = None
        for result in self.sensitivity_results:
            if result.observable_name == observable_name:
                sensitivity_result = result
                break
        
        if sensitivity_result is None:
            # Compute sensitivity if not available
            sensitivity_result = self.compute_parameter_sensitivity(observable_name)
        
        # Rank by absolute sensitivity
        sensitivities = sensitivity_result.parameter_sensitivities
        ranked_params = sorted(
            sensitivities.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Calculate total variance explained (approximation)
        total_abs_sensitivity = sum(abs(sens) for _, sens in ranked_params)
        total_variance_explained = 1.0 if total_abs_sensitivity == 0 else total_abs_sensitivity
        
        return ParameterRanking(
            observable_name=observable_name,
            ranked_parameters=ranked_params,
            ranking_method='sensitivity_based',
            total_variance_explained=total_variance_explained,
            metadata={
                'analysis_method': sensitivity_result.analysis_method,
                'perturbation_size': sensitivity_result.perturbation_size
            }
        )
    
    def _rank_by_variance(self, observable_name: str) -> ParameterRanking:
        """Rank parameters by their contribution to output variance."""
        # Find Monte Carlo result for this observable
        mc_result = None
        for result in self.monte_carlo_results:
            if result.observable_name == observable_name:
                mc_result = result
                break
        
        if mc_result is None:
            # Perform Monte Carlo analysis if not available
            mc_result = self.perform_monte_carlo_analysis(observable_name)
        
        # Compute variance contribution (simplified Sobol indices approximation)
        param_variances = {}
        for param_name in mc_result.parameter_names:
            param_stats = mc_result.parameter_statistics[param_name]
            param_variances[param_name] = param_stats['std'] ** 2
        
        # Rank by variance contribution
        ranked_params = sorted(
            param_variances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        total_variance = sum(param_variances.values())
        
        return ParameterRanking(
            observable_name=observable_name,
            ranked_parameters=ranked_params,
            ranking_method='variance_based',
            total_variance_explained=total_variance,
            metadata={
                'n_samples': mc_result.n_samples,
                'baseline_value': mc_result.baseline_value
            }
        )
    
    def _set_parameter_value(self, param_name: str, value: float) -> None:
        """Safely set parameter value."""
        param = self.registry.get_parameter(param_name)
        if param and param.history:
            # Use the same realm as the last update
            last_realm = param.history[-1].realm
            self.registry.update_parameter(param_name, value, last_realm, "Sensitivity analysis")
        else:
            # Use a generic realm for sensitivity analysis
            self.registry.update_parameter(param_name, value, "sensitivity_analysis", "Parameter perturbation")
    
    def _get_parameter_bounds(self, param_state) -> Optional[Tuple[float, float]]:
        """Get parameter bounds from constraints."""
        min_val = None
        max_val = None
        
        for constraint in param_state.get_active_constraints():
            if constraint.min_value is not None:
                min_val = constraint.min_value if min_val is None else max(min_val, constraint.min_value)
            if constraint.max_value is not None:
                max_val = constraint.max_value if max_val is None else min(max_val, constraint.max_value)
        
        if min_val is not None and max_val is not None:
            return (min_val, max_val)
        return None
    
    def _latin_hypercube_sample(self, n_samples: int, n_dims: int) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        # Simple Latin Hypercube implementation
        samples = np.zeros((n_samples, n_dims))
        
        for dim in range(n_dims):
            # Create stratified samples
            intervals = np.linspace(0, 1, n_samples + 1)
            samples[:, dim] = np.random.uniform(intervals[:-1], intervals[1:])
            
            # Shuffle to break correlation between dimensions
            np.random.shuffle(samples[:, dim])
        
        return samples
    
    def _analyze_convergence(self, values: List[float]) -> Dict[str, Any]:
        """Analyze convergence of Monte Carlo samples."""
        values_array = np.array(values)
        n_samples = len(values_array)
        
        # Running mean and std
        running_means = np.cumsum(values_array) / np.arange(1, n_samples + 1)
        running_stds = np.array([np.std(values_array[:i+1]) for i in range(n_samples)])
        
        # Convergence metrics
        final_mean = running_means[-1]
        final_std = running_stds[-1]
        
        # Check if last 10% of samples have stabilized
        if n_samples >= 100:
            last_10_percent = int(0.1 * n_samples)
            recent_means = running_means[-last_10_percent:]
            mean_stability = np.std(recent_means) / abs(final_mean) if final_mean != 0 else np.std(recent_means)
            converged = mean_stability < 0.01  # 1% relative stability
        else:
            converged = False
            mean_stability = float('inf')
        
        return {
            'converged': converged,
            'final_mean': float(final_mean),
            'final_std': float(final_std),
            'mean_stability': float(mean_stability),
            'running_means': running_means[-10:].tolist(),  # Last 10 values
            'running_stds': running_stds[-10:].tolist()
        }
    
    def generate_sensitivity_plots(self, output_dir: str) -> None:
        """
        Generate sensitivity analysis plots.
        
        Args:
            output_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import os
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot sensitivity results
            for result in self.sensitivity_results:
                self._plot_sensitivity_result(result, output_dir)
            
            # Plot Monte Carlo results
            for result in self.monte_carlo_results:
                self._plot_monte_carlo_result(result, output_dir)
            
            # Plot parameter rankings
            for ranking in self.parameter_rankings:
                self._plot_parameter_ranking(ranking, output_dir)
                
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def _plot_sensitivity_result(self, result: SensitivityResult, output_dir: str) -> None:
        """Plot sensitivity analysis result."""
        import matplotlib.pyplot as plt
        
        params = list(result.parameter_sensitivities.keys())
        sensitivities = list(result.parameter_sensitivities.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(params)), sensitivities)
        plt.xlabel('Parameters')
        plt.ylabel('Sensitivity')
        plt.title(f'Parameter Sensitivity for {result.observable_name}')
        plt.xticks(range(len(params)), params, rotation=45, ha='right')
        
        # Color bars by magnitude
        max_sens = max(abs(s) for s in sensitivities) if sensitivities else 1
        for bar, sens in zip(bars, sensitivities):
            color_intensity = abs(sens) / max_sens
            bar.set_color(plt.cm.RdYlBu_r(color_intensity))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sensitivity_{result.observable_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_monte_carlo_result(self, result: MonteCarloResult, output_dir: str) -> None:
        """Plot Monte Carlo analysis result."""
        import matplotlib.pyplot as plt
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(result.correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title(f'Parameter Correlation Matrix - {result.observable_name}')
        plt.xticks(range(len(result.parameter_names)), result.parameter_names, rotation=45, ha='right')
        plt.yticks(range(len(result.parameter_names)), result.parameter_names)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_{result.observable_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_ranking(self, ranking: ParameterRanking, output_dir: str) -> None:
        """Plot parameter ranking."""
        import matplotlib.pyplot as plt
        
        params = [p[0] for p in ranking.ranked_parameters[:10]]  # Top 10
        impacts = [p[1] for p in ranking.ranked_parameters[:10]]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(params)), impacts)
        plt.xlabel('Parameters')
        plt.ylabel('Impact Score')
        plt.title(f'Parameter Impact Ranking - {ranking.observable_name} ({ranking.ranking_method})')
        plt.xticks(range(len(params)), params, rotation=45, ha='right')
        
        # Color bars by rank
        for i, bar in enumerate(bars):
            color_intensity = 1 - (i / len(bars))
            bar.set_color(plt.cm.viridis(color_intensity))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ranking_{ranking.observable_name}_{ranking.ranking_method}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self, filename: str) -> None:
        """Export all sensitivity analysis results to JSON."""
        export_data = {
            'sensitivity_results': [
                {
                    'observable_name': result.observable_name,
                    'parameter_sensitivities': result.parameter_sensitivities,
                    'baseline_value': result.baseline_value,
                    'analysis_method': result.analysis_method,
                    'perturbation_size': result.perturbation_size,
                    'execution_time_ms': result.execution_time_ms,
                    'metadata': result.metadata
                }
                for result in self.sensitivity_results
            ],
            'monte_carlo_results': [
                {
                    'observable_name': result.observable_name,
                    'parameter_statistics': result.parameter_statistics,
                    'parameter_names': result.parameter_names,
                    'n_samples': result.n_samples,
                    'baseline_value': result.baseline_value,
                    'execution_time_ms': result.execution_time_ms,
                    'convergence_info': result.convergence_info
                }
                for result in self.monte_carlo_results
            ],
            'parameter_rankings': [
                {
                    'observable_name': ranking.observable_name,
                    'ranked_parameters': ranking.ranked_parameters,
                    'ranking_method': ranking.ranking_method,
                    'total_variance_explained': ranking.total_variance_explained,
                    'metadata': ranking.metadata
                }
                for ranking in self.parameter_rankings
            ]
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        export_data = convert_numpy_types(export_data)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)


# Common observable functions for QFD analysis

def create_ppn_gamma_observable() -> ObservableFunction:
    """Create observable function for PPN gamma parameter."""
    def ppn_gamma_observable(registry: ParameterRegistry) -> float:
        ppn_gamma = registry.get_parameter("PPN_gamma")
        return ppn_gamma.value if ppn_gamma and ppn_gamma.value is not None else 1.0
    return ppn_gamma_observable


def create_ppn_beta_observable() -> ObservableFunction:
    """Create observable function for PPN beta parameter."""
    def ppn_beta_observable(registry: ParameterRegistry) -> float:
        ppn_beta = registry.get_parameter("PPN_beta")
        return ppn_beta.value if ppn_beta and ppn_beta.value is not None else 1.0
    return ppn_beta_observable


def create_cmb_temperature_observable() -> ObservableFunction:
    """Create observable function for CMB temperature."""
    def cmb_temperature_observable(registry: ParameterRegistry) -> float:
        t_cmb = registry.get_parameter("T_CMB_K")
        return t_cmb.value if t_cmb and t_cmb.value is not None else 2.725
    return cmb_temperature_observable


def create_vacuum_refractive_index_observable() -> ObservableFunction:
    """Create observable function for vacuum refractive index."""
    def vacuum_index_observable(registry: ParameterRegistry) -> float:
        # In a full implementation, this would compute n_vacuum from coupling constants
        # For now, return 1.0 + small perturbation based on k_J
        k_j = registry.get_parameter("k_J")
        if k_j and k_j.value is not None:
            return 1.0 + k_j.value * 1e-6  # Small perturbation
        return 1.0
    return vacuum_index_observable