#!/usr/bin/env python3
"""
Sensitivity analysis demonstration.

This script demonstrates the sensitivity analysis capabilities
for coupling constants in the QFD framework, including numerical
derivatives, Monte Carlo analysis, and parameter ranking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from coupling_constants.registry.parameter_registry import ParameterRegistry
from coupling_constants.config.yaml_loader import load_parameters_from_yaml
from coupling_constants.analysis.sensitivity_analyzer import (
    SensitivityAnalyzer, create_ppn_gamma_observable, create_ppn_beta_observable,
    create_cmb_temperature_observable, create_vacuum_refractive_index_observable
)
import numpy as np


def create_composite_observable():
    """Create a composite observable that depends on multiple parameters."""
    def composite_observable(registry):
        # Composite function: combines PPN parameters, CMB temperature, and vacuum effects
        ppn_gamma = registry.get_parameter("PPN_gamma")
        ppn_beta = registry.get_parameter("PPN_beta")
        t_cmb = registry.get_parameter("T_CMB_K")
        k_j = registry.get_parameter("k_J")
        xi = registry.get_parameter("xi")
        
        gamma_val = ppn_gamma.value if ppn_gamma and ppn_gamma.value is not None else 1.0
        beta_val = ppn_beta.value if ppn_beta and ppn_beta.value is not None else 1.0
        temp_val = t_cmb.value if t_cmb and t_cmb.value is not None else 2.725
        k_j_val = k_j.value if k_j and k_j.value is not None else 0.0
        xi_val = xi.value if xi and xi.value is not None else 1.0
        
        # Composite metric: deviation from GR + CMB consistency + vacuum effects
        gr_deviation = (gamma_val - 1.0)**2 + (beta_val - 1.0)**2
        cmb_deviation = (temp_val - 2.725)**2
        vacuum_effect = k_j_val**2 + 0.1 * (xi_val - 1.0)**2
        
        return gr_deviation + 1000 * cmb_deviation + 1e6 * vacuum_effect
    
    return composite_observable


def main():
    print("=== QFD Sensitivity Analysis Demo ===\n")
    
    # 1. Load configuration and set up registry
    print("1. Loading QFD configuration and setting up parameter registry...")
    registry = ParameterRegistry()
    load_parameters_from_yaml("qfd_params/defaults.yaml", registry)
    
    # Set realistic parameter values
    registry.update_parameter("PPN_gamma", 1.000001, "realm3_scales", "Slightly off GR")
    registry.update_parameter("PPN_beta", 0.99999, "realm3_scales", "Close to GR")
    registry.update_parameter("T_CMB_K", 2.725, "cmb_config", "CMB temperature")
    registry.update_parameter("k_J", 1e-12, "realm0_cmb", "Vacuum drag")
    registry.update_parameter("xi", 2.0, "realm4_em", "EM response")
    registry.update_parameter("psi_s0", -1.5, "realm0_cmb", "Thermalization")
    registry.update_parameter("E0", 1e3, "realm3_scales", "Energy scale")
    registry.update_parameter("L0", 1e-10, "realm3_scales", "Length scale")
    
    print(f"   ✓ Loaded {len(registry.get_all_parameters())} parameters")
    print(f"   ✓ Set values for 8 key parameters")
    
    # 2. Create sensitivity analyzer and register observables
    print("\n2. Setting up sensitivity analyzer with physics observables...")
    analyzer = SensitivityAnalyzer(registry)
    
    # Register physics observables
    analyzer.register_observable("ppn_gamma", create_ppn_gamma_observable())
    analyzer.register_observable("ppn_beta", create_ppn_beta_observable())
    analyzer.register_observable("cmb_temperature", create_cmb_temperature_observable())
    analyzer.register_observable("vacuum_index", create_vacuum_refractive_index_observable())
    analyzer.register_observable("composite_metric", create_composite_observable())
    
    print(f"   ✓ Registered {len(analyzer.observables)} observables:")
    for obs_name in analyzer.observables.keys():
        print(f"     • {obs_name}")
    
    # 3. Numerical derivative sensitivity analysis
    print("\n3. Performing numerical derivative sensitivity analysis...")
    
    observables_to_analyze = ["ppn_gamma", "ppn_beta", "composite_metric"]
    sensitivity_results = {}
    
    for obs_name in observables_to_analyze:
        print(f"\n   Analyzing {obs_name}:")
        
        # Compare different numerical methods
        methods = ["forward", "backward", "central_difference"]
        method_results = {}
        
        for method in methods:
            result = analyzer.compute_parameter_sensitivity(
                obs_name, perturbation_size=1e-8, method=method
            )
            method_results[method] = result
            print(f"     {method}: {result.execution_time_ms:.2f}ms")
        
        sensitivity_results[obs_name] = method_results
        
        # Show top sensitive parameters (using central difference)
        central_result = method_results["central_difference"]
        sorted_sensitivities = sorted(
            central_result.parameter_sensitivities.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        print(f"     Top 5 sensitive parameters:")
        for i, (param, sens) in enumerate(sorted_sensitivities[:5]):
            print(f"       {i+1}. {param}: {sens:.2e}")
    
    # 4. Monte Carlo sensitivity analysis
    print("\n4. Performing Monte Carlo sensitivity analysis...")
    
    mc_results = {}
    for obs_name in ["ppn_gamma", "composite_metric"]:
        print(f"\n   Monte Carlo analysis for {obs_name}:")
        
        # Define parameter ranges for Monte Carlo
        parameter_ranges = {
            "PPN_gamma": (0.999995, 1.000005),  # ±5e-6 around 1.0
            "PPN_beta": (0.9999, 1.0001),       # ±1e-4 around 1.0
            "k_J": (1e-15, 1e-9),               # Wide range for vacuum drag
            "xi": (1.0, 3.0),                   # EM response range
            "psi_s0": (-2.0, -1.0),             # Thermalization range
        }
        
        result = analyzer.perform_monte_carlo_analysis(
            obs_name, n_samples=500, parameter_ranges=parameter_ranges
        )
        mc_results[obs_name] = result
        
        print(f"     Samples: {result.n_samples}")
        print(f"     Execution time: {result.execution_time_ms:.2f}ms")
        print(f"     Baseline value: {result.baseline_value:.6e}")
        print(f"     Converged: {result.convergence_info['converged']}")
        
        # Show parameter statistics
        print(f"     Parameter statistics:")
        for param_name in result.parameter_names[:5]:  # Show first 5
            stats = result.parameter_statistics[param_name]
            print(f"       {param_name}: mean={stats['mean']:.6e}, std={stats['std']:.6e}")
    
    # 5. Parameter ranking analysis
    print("\n5. Parameter ranking analysis...")
    
    # Rank by sensitivity
    print("\n   Ranking by sensitivity:")
    sensitivity_rankings = analyzer.rank_parameters_by_impact(
        ["ppn_gamma", "composite_metric"], method="sensitivity_based"
    )
    
    for ranking in sensitivity_rankings:
        print(f"     {ranking.observable_name}:")
        for i, (param, impact) in enumerate(ranking.ranked_parameters[:5]):
            print(f"       {i+1}. {param}: {impact:.2e}")
    
    # Rank by variance contribution
    print("\n   Ranking by variance contribution:")
    variance_rankings = analyzer.rank_parameters_by_impact(
        ["ppn_gamma", "composite_metric"], method="variance_based"
    )
    
    for ranking in variance_rankings:
        print(f"     {ranking.observable_name}:")
        for i, (param, impact) in enumerate(ranking.ranked_parameters[:5]):
            print(f"       {i+1}. {param}: {impact:.2e}")
    
    # 6. Cross-observable sensitivity comparison
    print("\n6. Cross-observable sensitivity comparison...")
    
    # Find parameters that are sensitive across multiple observables
    all_sensitivities = {}
    for obs_name in ["ppn_gamma", "ppn_beta", "composite_metric"]:
        if obs_name in sensitivity_results:
            central_result = sensitivity_results[obs_name]["central_difference"]
            all_sensitivities[obs_name] = central_result.parameter_sensitivities
    
    # Find parameters with high sensitivity across observables
    param_cross_sensitivity = {}
    for obs_name, sensitivities in all_sensitivities.items():
        for param, sens in sensitivities.items():
            if param not in param_cross_sensitivity:
                param_cross_sensitivity[param] = []
            param_cross_sensitivity[param].append((obs_name, abs(sens)))
    
    # Calculate average sensitivity across observables
    param_avg_sensitivity = {}
    for param, obs_sens_list in param_cross_sensitivity.items():
        avg_sens = np.mean([sens for _, sens in obs_sens_list])
        param_avg_sensitivity[param] = avg_sens
    
    # Sort by average sensitivity
    sorted_cross_sens = sorted(
        param_avg_sensitivity.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("   Parameters with highest cross-observable sensitivity:")
    for i, (param, avg_sens) in enumerate(sorted_cross_sens[:8]):
        print(f"     {i+1}. {param}: {avg_sens:.2e} (avg across observables)")
    
    # 7. Uncertainty propagation analysis
    print("\n7. Uncertainty propagation analysis...")
    
    if "composite_metric" in mc_results:
        mc_result = mc_results["composite_metric"]
        
        # Analyze correlation structure
        corr_matrix = mc_result.correlation_matrix
        param_names = mc_result.parameter_names
        
        print("   Strong parameter correlations (|r| > 0.3):")
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names):
                if i < j:  # Avoid duplicates
                    correlation = corr_matrix[i, j]
                    if abs(correlation) > 0.3:
                        print(f"     {param1} ↔ {param2}: r = {correlation:.3f}")
        
        # Convergence analysis
        conv_info = mc_result.convergence_info
        print(f"   Monte Carlo convergence:")
        print(f"     Final mean: {conv_info['final_mean']:.6e}")
        print(f"     Final std: {conv_info['final_std']:.6e}")
        print(f"     Mean stability: {conv_info['mean_stability']:.6e}")
        print(f"     Converged: {conv_info['converged']}")
    
    # 8. Export results and generate plots
    print("\n8. Exporting results and generating visualizations...")
    
    # Export sensitivity analysis results
    analyzer.export_results("sensitivity_analysis_results.json")
    print("   ✓ Exported results to 'sensitivity_analysis_results.json'")
    
    # Generate plots if matplotlib is available
    try:
        analyzer.generate_sensitivity_plots("sensitivity_plots")
        print("   ✓ Generated sensitivity plots in 'sensitivity_plots/' directory")
    except ImportError:
        print("   ⚠ Matplotlib not available - skipping plot generation")
    except Exception as e:
        print(f"   ⚠ Plot generation failed: {e}")
    
    # 9. Summary and recommendations
    print("\n9. Sensitivity analysis summary:")
    print("   " + "="*50)
    
    # Most sensitive parameters overall
    if sorted_cross_sens:
        most_sensitive = sorted_cross_sens[0]
        print(f"   Most sensitive parameter: {most_sensitive[0]} (avg sensitivity: {most_sensitive[1]:.2e})")
    
    # Method comparison
    if "composite_metric" in sensitivity_results:
        methods_data = sensitivity_results["composite_metric"]
        print("   Numerical method comparison (execution time):")
        for method, result in methods_data.items():
            print(f"     {method}: {result.execution_time_ms:.2f}ms")
    
    # Monte Carlo insights
    if mc_results:
        total_mc_time = sum(r.execution_time_ms for r in mc_results.values())
        total_mc_samples = sum(r.n_samples for r in mc_results.values())
        print(f"   Monte Carlo analysis: {total_mc_samples} total samples in {total_mc_time:.2f}ms")
    
    print("\n=== Sensitivity Analysis Complete ===")
    print("\nKey insights:")
    print("• Numerical derivatives provide precise local sensitivity information")
    print("• Monte Carlo analysis reveals parameter interactions and uncertainty propagation")
    print("• Parameter ranking identifies the most critical parameters for each observable")
    print("• Cross-observable analysis reveals parameters affecting multiple physics quantities")
    print("\nThis analysis enables targeted parameter optimization and uncertainty quantification")
    print("for the QFD coupling constants across all physical realms.")


if __name__ == "__main__":
    main()