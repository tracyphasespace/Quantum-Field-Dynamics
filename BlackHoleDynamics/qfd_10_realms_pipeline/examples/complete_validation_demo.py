#!/usr/bin/env python3
"""
Complete validation framework demonstration.

This script demonstrates the full validation system including:
- Basic constraint validation (bounds, fixed, targets, conflicts)
- Physics-specific validation (PPN, CMB, vacuum)
- Derived constraint evaluation
- Comprehensive reporting and analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, Constraint, ConstraintType
)
from coupling_constants.config.yaml_loader import load_parameters_from_yaml
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.basic_validators import (
    BoundsValidator, FixedValueValidator, TargetValueValidator, ConflictValidator
)
from coupling_constants.validation.ppn_validator import PPNValidator
from coupling_constants.validation.cmb_validator import CMBValidator
from coupling_constants.validation.derived_validator import (
    DerivedConstraintValidator, create_vacuum_refractive_index_evaluator
)
import yaml


def main():
    print("=== Complete QFD Validation Framework Demo ===\n")
    
    # 1. Load configuration and set up registry
    print("1. Loading QFD configuration and setting up parameter registry...")
    registry = ParameterRegistry()
    load_parameters_from_yaml("qfd_params/defaults.yaml", registry)
    
    # Load config for CMB validator
    with open("qfd_params/defaults.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"   ✓ Loaded {len(registry.get_all_parameters())} parameters")
    print(f"   ✓ Total constraints: {sum(len(p.constraints) for p in registry.get_all_parameters().values())}")
    print()
    
    # 2. Simulate realm execution with realistic parameter values
    print("2. Simulating realm execution with parameter updates...")
    
    # Realm 0 (CMB) - thermalization and vacuum constraints
    # T_CMB_K is already fixed by config, so we'll update it using the same realm
    registry.update_parameter("T_CMB_K", 2.725, "cmb_config", "Confirmed by CMB observations")
    registry.update_parameter("k_J", 1e-12, "realm0_cmb", "Near-zero for vacuum (no spectral distortions)")
    registry.update_parameter("psi_s0", -1.5, "realm0_cmb", "Thermalization zeropoint")
    
    # Realm 3 (PPN) - post-Newtonian parameters
    registry.update_parameter("PPN_gamma", 1.000001, "realm3_scales", "Slightly off GR (within solar system bounds)")
    registry.update_parameter("PPN_beta", 0.99999, "realm3_scales", "Close to GR (within LLR bounds)")
    
    # Other realms - various coupling constants
    registry.update_parameter("xi", 2.0, "realm1_cosmic", "EM energy response strength")
    registry.update_parameter("E0", 1e3, "realm3_scales", "Energy scale")
    registry.update_parameter("L0", 1e-10, "realm3_scales", "Length scale (within bounds)")
    registry.update_parameter("k_c2", 0.5, "realm4_em", "EM coupling")
    registry.update_parameter("V2", -5.0, "realm5_electron", "Potential term")
    
    print("   ✓ Set 10 parameter values across multiple realms")
    print()
    
    # 3. Set up comprehensive validation suite
    print("3. Setting up comprehensive validation suite...")
    
    validator = CompositeValidator("Complete QFD Validation Suite")
    
    # Basic constraint validators
    validator.add_validator(BoundsValidator())
    validator.add_validator(FixedValueValidator())
    validator.add_validator(TargetValueValidator())
    validator.add_validator(ConflictValidator())
    
    # Physics-specific validators
    validator.add_validator(PPNValidator(
        gamma_target=1.0,
        beta_target=1.0,
        gamma_tolerance=1e-5,  # Solar system bounds
        beta_tolerance=1e-4    # LLR bounds
    ))
    
    validator.add_validator(CMBValidator.from_config(config))
    
    # Derived physics constraints
    derived_validator = DerivedConstraintValidator("Physics Relations")
    derived_validator.add_evaluator("n_vacuum", create_vacuum_refractive_index_evaluator())
    validator.add_validator(derived_validator)
    
    print(f"   ✓ Registered {len(validator.get_validator_names())} validators:")
    for i, name in enumerate(validator.get_validator_names(), 1):
        print(f"     {i}. {name}")
    print()
    
    # 4. Run comprehensive validation
    print("4. Running comprehensive validation analysis...")
    report = validator.validate_all(registry)
    
    print(f"   Overall Status: {report.overall_status.value.upper()}")
    print(f"   Execution Time: {report.execution_time_ms:.2f} ms")
    print(f"   Parameters Analyzed: {report.total_parameters}")
    print(f"   Constraints Checked: {report.total_constraints}")
    print(f"   Total Violations: {report.total_violations}")
    print(f"   Total Warnings: {report.total_warnings}")
    print()
    
    # 5. Detailed validator results
    print("5. Detailed validation results by validator:")
    print("   " + "="*60)
    
    for result in report.validator_results:
        status_symbol = {"valid": "✓", "warning": "⚠", "invalid": "✗", "skipped": "○"}
        symbol = status_symbol.get(result.status.value, "?")
        
        print(f"   {symbol} {result.validator_name}")
        print(f"     Status: {result.status.value}")
        print(f"     Performance: {result.execution_time_ms:.2f}ms, {result.parameters_checked} params, {result.constraints_checked} constraints")
        
        if "violation_types" in result.metadata and result.metadata["violation_types"]:
            print(f"     Violation types: {result.metadata['violation_types']}")
        
        if result.violations:
            print(f"     Violations ({len(result.violations)}):")
            for violation in result.violations[:3]:  # Show first 3
                print(f"       • {violation.parameter_name}: {violation.violation_type}")
                print(f"         {violation.message}")
            if len(result.violations) > 3:
                print(f"       ... and {len(result.violations) - 3} more")
        
        if result.warnings:
            print(f"     Warnings ({len(result.warnings)}):")
            for warning in result.warnings[:2]:  # Show first 2
                print(f"       • {warning}")
            if len(result.warnings) > 2:
                print(f"       ... and {len(result.warnings) - 2} more")
        
        if result.info_messages:
            print(f"     Info ({len(result.info_messages)}):")
            for info in result.info_messages[:2]:  # Show first 2
                print(f"       • {info}")
            if len(result.info_messages) > 2:
                print(f"       ... and {len(result.info_messages) - 2} more")
        
        print()
    
    # 6. Parameter-centric analysis
    print("6. Parameter-centric violation analysis:")
    print("   " + "="*50)
    
    violations_by_param = report.get_violations_by_parameter()
    if violations_by_param:
        for param_name, violations in violations_by_param.items():
            print(f"   Parameter: {param_name}")
            param = registry.get_parameter(param_name)
            if param and param.value is not None:
                print(f"     Current value: {param.value}")
            
            for violation in violations:
                print(f"     • {violation.violation_type} ({violation.constraint_realm})")
                print(f"       {violation.message}")
                if violation.expected_value is not None:
                    print(f"       Expected: {violation.expected_value}")
                if violation.expected_range is not None:
                    print(f"       Range: {violation.expected_range}")
            print()
    else:
        print("   ✓ No parameter violations found!")
        print()
    
    # 7. Physics consistency summary
    print("7. Physics consistency summary:")
    print("   " + "="*40)
    
    # PPN consistency
    ppn_gamma = registry.get_parameter("PPN_gamma")
    ppn_beta = registry.get_parameter("PPN_beta")
    if ppn_gamma and ppn_beta and ppn_gamma.value and ppn_beta.value:
        gamma_dev = abs(ppn_gamma.value - 1.0)
        beta_dev = abs(ppn_beta.value - 1.0)
        print(f"   PPN Parameters:")
        print(f"     γ = {ppn_gamma.value:.6f} (deviation from GR: {gamma_dev:.2e})")
        print(f"     β = {ppn_beta.value:.6f} (deviation from GR: {beta_dev:.2e})")
        
        if gamma_dev <= 1e-5 and beta_dev <= 1e-4:
            print("     ✓ Within observational bounds")
        else:
            print("     ✗ Outside observational bounds")
    
    # CMB consistency
    t_cmb = registry.get_parameter("T_CMB_K")
    k_j = registry.get_parameter("k_J")
    if t_cmb and t_cmb.value:
        temp_dev = abs(t_cmb.value - 2.725)
        print(f"   CMB Temperature: {t_cmb.value:.6f} K (deviation: {temp_dev:.2e} K)")
        if temp_dev <= 1e-6:
            print("     ✓ Consistent with FIRAS/Planck")
        else:
            print("     ✗ Inconsistent with observations")
    
    if k_j and k_j.value:
        print(f"   Vacuum Drag: |k_J| = {abs(k_j.value):.2e}")
        if abs(k_j.value) <= 1e-10:
            print("     ✓ No CMB spectral distortions")
        else:
            print("     ⚠ May cause spectral distortions")
    
    print()
    
    # 8. Recommendations
    print("8. Validation recommendations:")
    print("   " + "="*35)
    
    if report.total_violations == 0:
        print("   ✓ All constraints satisfied - parameter space is physically viable")
    else:
        print("   Recommendations for resolving violations:")
        
        # Group recommendations by type
        bounds_violations = [v for v in report.get_all_violations() if v.violation_type == "bounds_violation"]
        fixed_violations = [v for v in report.get_all_violations() if v.violation_type.startswith("missing_fixed")]
        
        if bounds_violations:
            print(f"   • {len(bounds_violations)} parameters outside allowed bounds")
            print("     → Adjust parameter values or relax constraints")
        
        if fixed_violations:
            print(f"   • {len(fixed_violations)} required parameters not set")
            print("     → Set values for fixed parameters")
        
        conflict_violations = [v for v in report.get_all_violations() if "conflict" in v.violation_type]
        if conflict_violations:
            print(f"   • {len(conflict_violations)} constraint conflicts detected")
            print("     → Resolve conflicting requirements between realms")
    
    if report.total_warnings > 0:
        print(f"   ⚠ {report.total_warnings} warnings - review for potential issues")
    
    print()
    
    # 9. Export summary
    print("9. Validation report summary:")
    print(report.get_summary())
    
    print("=== Complete Validation Demo Finished ===")
    print(f"\nSystem Status: {report.overall_status.value.upper()}")
    print("The QFD coupling constants validation framework is fully operational!")


if __name__ == "__main__":
    main()