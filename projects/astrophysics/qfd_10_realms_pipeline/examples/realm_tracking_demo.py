#!/usr/bin/env python3
"""
Realm tracking and execution demonstration.

This script demonstrates the realm tracking capabilities for managing
QFD realm execution, parameter convergence, and validation integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from coupling_constants.registry.parameter_registry import ParameterRegistry
from coupling_constants.config.yaml_loader import load_parameters_from_yaml
from coupling_constants.analysis.realm_tracker import RealmTracker, RealmStatus
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.basic_validators import (
    BoundsValidator, FixedValueValidator, TargetValueValidator, ConflictValidator
)
from coupling_constants.validation.ppn_validator import PPNValidator
from coupling_constants.validation.cmb_validator import CMBValidator
import yaml


def create_mock_qfd_realms(tracker, registry):
    """Create mock QFD realm functions for demonstration."""
    
    def realm0_cmb(registry):
        """CMB realm: Set vacuum constraints and thermalization."""
        # Work with parameters that aren't pre-fixed
        
        # Set vacuum drag to near zero
        k_j = registry.get_parameter("k_J")
        if k_j:
            current_val = k_j.value if k_j.value else 0.0
            new_val = current_val * 0.1  # Move towards zero
            registry.update_parameter("k_J", new_val, "realm0_cmb", "Vacuum drag constraint")
        
        # Set thermalization zeropoint
        registry.update_parameter("psi_s0", -1.5, "realm0_cmb", "Thermalization zeropoint")
        
        # Set EM energy response
        registry.update_parameter("xi", 1.8, "realm0_cmb", "Initial EM response")
        
        return {
            "status": "ok",
            "fixed": {"psi_s0": -1.5, "xi": 1.8},
            "narrowed": {"k_J": "Near zero for vacuum"},
            "notes": ["Vacuum constraints applied", "Thermalization set"]
        }
    
    def realm3_scales(registry):
        """Scales realm: Set PPN parameters and fundamental scales."""
        # Set PPN parameters close to GR values
        ppn_gamma = registry.get_parameter("PPN_gamma")
        ppn_beta = registry.get_parameter("PPN_beta")
        
        if ppn_gamma:
            current_gamma = ppn_gamma.value if ppn_gamma.value else 1.0
            # Move towards GR value with small perturbation
            target_gamma = 1.0 + 1e-6
            new_gamma = current_gamma + 0.1 * (target_gamma - current_gamma)
            registry.update_parameter("PPN_gamma", new_gamma, "realm3_scales", "PPN gamma parameter")
        
        if ppn_beta:
            current_beta = ppn_beta.value if ppn_beta.value else 1.0
            # Move towards GR value with small perturbation
            target_beta = 1.0 - 1e-5
            new_beta = current_beta + 0.1 * (target_beta - current_beta)
            registry.update_parameter("PPN_beta", new_beta, "realm3_scales", "PPN beta parameter")
        
        # Set fundamental scales
        registry.update_parameter("E0", 1e3, "realm3_scales", "Energy scale")
        registry.update_parameter("L0", 1e-10, "realm3_scales", "Length scale")
        
        return {
            "status": "ok",
            "fixed": {"E0": 1e3, "L0": 1e-10},
            "narrowed": {"PPN_gamma": "Close to GR", "PPN_beta": "Close to GR"},
            "notes": ["PPN parameters set", "Fundamental scales established"]
        }
    
    def realm4_em(registry):
        """EM realm: Set electromagnetic coupling parameters."""
        # Set EM response parameter
        xi = registry.get_parameter("xi")
        if xi:
            current_xi = xi.value if xi.value else 1.0
            # Move towards target value
            target_xi = 2.0
            new_xi = current_xi + 0.2 * (target_xi - current_xi)
            registry.update_parameter("xi", new_xi, "realm4_em", "EM energy response")
        
        # Set EM coupling constants
        registry.update_parameter("k_c2", 0.5, "realm4_em", "EM coupling constant")
        registry.update_parameter("k_EM", 1.5, "realm4_em", "EM field coupling")
        
        return {
            "status": "ok",
            "fixed": {"k_c2": 0.5, "k_EM": 1.5},
            "narrowed": {"xi": "EM response parameter"},
            "notes": ["EM coupling parameters set"]
        }
    
    def realm5_electron(registry):
        """Electron realm: Set electron-related parameters."""
        # Set potential terms
        registry.update_parameter("V2", -2.0, "realm5_electron", "Quadratic potential")
        registry.update_parameter("V4", 1.0, "realm5_electron", "Quartic potential")
        
        # Adjust parameters based on other realms
        xi = registry.get_parameter("xi")
        if xi and xi.value:
            # Set kappa based on xi
            kappa_val = 0.1 * xi.value
            registry.update_parameter("kappa", kappa_val, "realm5_electron", "Based on xi")
        
        return {
            "status": "ok",
            "fixed": {"V2": -2.0, "V4": 1.0},
            "derived": {"kappa": "Based on xi"},
            "notes": ["Electron parameters set"]
        }
    
    def unstable_realm(registry):
        """A realm that causes parameter oscillation for testing."""
        # This realm will cause oscillation in lambda_t
        lambda_t = registry.get_parameter("lambda_t")
        if lambda_t:
            current_val = lambda_t.value if lambda_t.value else 1.0
            # Oscillate around 2.0
            import math
            oscillation = 0.5 * math.sin(len(registry.get_parameter("lambda_t").history) * 0.5)
            new_val = 2.0 + oscillation
            registry.update_parameter("lambda_t", new_val, "unstable_realm", "Oscillating parameter")
        
        return {
            "status": "ok",
            "notes": ["Caused parameter oscillation"]
        }
    
    # Register realms with dependencies
    tracker.register_realm("realm0_cmb", realm0_cmb)
    tracker.register_realm("realm3_scales", realm3_scales, dependencies=["realm0_cmb"])
    tracker.register_realm("realm4_em", realm4_em, dependencies=["realm3_scales"])
    tracker.register_realm("realm5_electron", realm5_electron, dependencies=["realm4_em"])
    tracker.register_realm("unstable_realm", unstable_realm)


def main():
    print("=== QFD Realm Tracking and Execution Demo ===\n")
    
    # 1. Load configuration and set up registry
    print("1. Loading QFD configuration and setting up parameter registry...")
    registry = ParameterRegistry()
    load_parameters_from_yaml("qfd_params/defaults.yaml", registry)
    
    # Load config for validators
    with open("qfd_params/defaults.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"   ✓ Loaded {len(registry.get_all_parameters())} parameters")
    
    # 2. Set up comprehensive validation suite
    print("\n2. Setting up validation suite for realm tracking...")
    validator = CompositeValidator("QFD Realm Validation Suite")
    
    # Add all validators
    validator.add_validator(BoundsValidator())
    validator.add_validator(FixedValueValidator())
    validator.add_validator(TargetValueValidator())
    validator.add_validator(ConflictValidator())
    validator.add_validator(PPNValidator())
    validator.add_validator(CMBValidator.from_config(config))
    
    print(f"   ✓ Registered {len(validator.get_validator_names())} validators")
    
    # 3. Create realm tracker and register realms
    print("\n3. Creating realm tracker and registering QFD realms...")
    tracker = RealmTracker(registry, validator)
    
    # Set convergence thresholds for key parameters
    tracker.set_convergence_threshold("PPN_gamma", 1e-7)
    tracker.set_convergence_threshold("PPN_beta", 1e-6)
    tracker.set_convergence_threshold("T_CMB_K", 1e-8)
    tracker.set_convergence_threshold("k_J", 1e-12)
    tracker.set_convergence_threshold("xi", 1e-3)
    
    # Register mock QFD realms
    create_mock_qfd_realms(tracker, registry)
    
    print(f"   ✓ Registered {len(tracker.realm_functions)} realms:")
    for realm_name in tracker.realm_functions.keys():
        deps = tracker.realm_dependencies.get(realm_name, [])
        dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
        print(f"     • {realm_name}{dep_str}")
    
    # 4. Execute individual realms
    print("\n4. Executing individual realms...")
    
    # Execute CMB realm
    print("\n   Executing realm0_cmb:")
    result = tracker.execute_realm("realm0_cmb", validate_after=True)
    print(f"     Status: {result.status.value}")
    print(f"     Execution time: {result.execution_time_ms:.2f}ms")
    print(f"     Parameters modified: {', '.join(result.parameters_modified)}")
    print(f"     Constraints added: {result.constraints_added}")
    if result.validation_report:
        print(f"     Validation status: {result.validation_report.overall_status.value}")
        print(f"     Validation violations: {len(result.validation_report.get_all_violations())}")
    
    # Try to execute a realm with missing dependencies
    print("\n   Attempting to execute realm4_em (missing dependencies):")
    result = tracker.execute_realm("realm4_em", validate_after=False)
    print(f"     Status: {result.status.value}")
    print(f"     Error: {result.error_message}")
    
    # Execute realms in proper order
    print("\n   Executing realms in dependency order:")
    for realm_name in ["realm3_scales", "realm4_em", "realm5_electron"]:
        result = tracker.execute_realm(realm_name, validate_after=False)
        print(f"     {realm_name}: {result.status.value} ({result.execution_time_ms:.2f}ms)")
        if result.parameters_modified:
            print(f"       Modified: {', '.join(result.parameters_modified[:3])}")
    
    # 5. Execution summary
    print("\n5. Individual realm execution summary:")
    summary = tracker.get_execution_summary()
    print(f"   Total executions: {summary['total_executions']}")
    print(f"   Successful: {summary['successful_executions']}")
    print(f"   Failed: {summary['failed_executions']}")
    print(f"   Total execution time: {summary['total_execution_time_ms']:.2f}ms")
    print(f"   Average execution time: {summary['average_execution_time_ms']:.2f}ms")
    print(f"   Parameters modified: {summary['total_parameters_modified']}")
    print(f"   Constraints added: {summary['total_constraints_added']}")
    
    # 6. Reset and demonstrate realm sequence execution
    print("\n6. Demonstrating realm sequence execution with convergence...")
    tracker.reset_execution_history()
    
    # Execute a convergent sequence
    print("\n   Executing convergent realm sequence:")
    sequence_result = tracker.execute_realm_sequence(
        realm_order=["realm0_cmb", "realm3_scales", "realm4_em", "realm5_electron"],
        max_iterations=3,
        convergence_check=True,
        validate_each_realm=False  # Skip individual validation for speed
    )
    
    print(f"\n   Sequence execution results:")
    print(f"     Total time: {sequence_result.total_execution_time_ms:.2f}ms")
    print(f"     Iterations completed: {sequence_result.iterations_completed}/{sequence_result.max_iterations}")
    print(f"     Realms executed: {len(sequence_result.realms_executed)}")
    print(f"     Convergence achieved: {sequence_result.convergence_achieved}")
    print(f"     Final validation: {sequence_result.final_validation_report.overall_status.value}")
    
    if sequence_result.convergence_metrics:
        converged_params = [m for m in sequence_result.convergence_metrics if m.is_converged]
        print(f"     Converged parameters: {len(converged_params)}/{len(sequence_result.convergence_metrics)}")
        
        print(f"     Top 5 parameter convergence status:")
        for i, metric in enumerate(sequence_result.convergence_metrics[:5]):
            status = "✓" if metric.is_converged else "○"
            print(f"       {status} {metric.parameter_name}: change={metric.change_magnitude:.2e}, threshold={metric.convergence_threshold:.2e}")
    
    # 7. Demonstrate oscillation detection
    print("\n7. Demonstrating oscillation and divergence detection...")
    tracker.reset_execution_history()
    
    print("\n   Executing sequence with unstable realm:")
    unstable_result = tracker.execute_realm_sequence(
        realm_order=["realm0_cmb", "unstable_realm"],
        max_iterations=5,
        convergence_check=True,
        validate_each_realm=False
    )
    
    print(f"     Iterations: {unstable_result.iterations_completed}/{unstable_result.max_iterations}")
    print(f"     Early termination: {unstable_result.early_termination_reason or 'None'}")
    
    if unstable_result.convergence_metrics:
        oscillating = [m for m in unstable_result.convergence_metrics if m.oscillation_detected]
        diverging = [m for m in unstable_result.convergence_metrics if m.divergence_detected]
        
        if oscillating:
            print(f"     Oscillating parameters: {[m.parameter_name for m in oscillating]}")
        if diverging:
            print(f"     Diverging parameters: {[m.parameter_name for m in diverging]}")
    
    # 8. Parameter state analysis
    print("\n8. Final parameter state analysis:")
    
    # Show key physics parameters
    key_params = ["T_CMB_K", "PPN_gamma", "PPN_beta", "k_J", "xi"]
    print("   Key physics parameters:")
    for param_name in key_params:
        param = registry.get_parameter(param_name)
        if param and param.value is not None:
            last_realm = param.history[-1].realm if param.history else "unknown"
            print(f"     {param_name}: {param.value:.6e} (set by {last_realm})")
        else:
            print(f"     {param_name}: not set")
    
    # Show parameter change history
    print("\n   Parameter change history (top 5 most modified):")
    param_change_counts = {}
    for param_name, param in registry.get_all_parameters().items():
        param_change_counts[param_name] = len(param.history)
    
    most_changed = sorted(param_change_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for param_name, change_count in most_changed:
        print(f"     {param_name}: {change_count} changes")
    
    # 9. Export execution logs
    print("\n9. Exporting execution logs and analysis...")
    
    # Export execution log
    tracker.export_execution_log("realm_execution_log.json")
    print("   ✓ Exported execution log to 'realm_execution_log.json'")
    
    # Export final parameter state
    final_state = registry.export_state("json")
    with open("final_parameter_state.json", "w") as f:
        f.write(final_state)
    print("   ✓ Exported final parameter state to 'final_parameter_state.json'")
    
    # 10. Summary and recommendations
    print("\n10. Realm tracking summary and recommendations:")
    print("    " + "="*50)
    
    total_executions = len(tracker.execution_history)
    successful_executions = len([r for r in tracker.execution_history if r.status == RealmStatus.COMPLETED])
    
    print(f"    Realm executions: {successful_executions}/{total_executions} successful")
    print(f"    Total execution time: {sum(r.execution_time_ms for r in tracker.execution_history):.2f}ms")
    
    if sequence_result.convergence_achieved:
        print("    ✓ Parameter convergence achieved")
    else:
        print("    ○ Parameter convergence not fully achieved")
    
    # Validation summary
    if sequence_result.final_validation_report:
        violations = len(sequence_result.final_validation_report.get_all_violations())
        warnings = sequence_result.final_validation_report.total_warnings
        print(f"    Final validation: {violations} violations, {warnings} warnings")
    
    print("\n=== Realm Tracking Demo Complete ===")
    print("\nKey capabilities demonstrated:")
    print("• Realm registration with dependency management")
    print("• Individual realm execution with validation")
    print("• Realm sequence execution with convergence checking")
    print("• Parameter change tracking and history")
    print("• Oscillation and divergence detection")
    print("• Comprehensive execution logging and export")
    print("\nThe realm tracking system provides robust coordination of QFD realm")
    print("execution with parameter convergence monitoring and validation integration.")


if __name__ == "__main__":
    main()