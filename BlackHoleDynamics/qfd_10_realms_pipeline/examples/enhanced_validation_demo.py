#!/usr/bin/env python3
"""
Enhanced validation framework demonstration.

This script demonstrates the surgical improvements to the validation system:
- Enhanced conflict detection (FIXED vs BOUNDED, FIXED vs TARGET)
- Derived constraint validation
- PPN parameter validation
- YAML schema validation
- Validation metadata and reporting
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
from coupling_constants.validation.derived_validator import (
    DerivedConstraintValidator, create_vacuum_refractive_index_evaluator
)


def main():
    print("=== Enhanced Validation Framework Demo ===\n")
    
    # 1. Load configuration with schema validation
    print("1. Loading configuration with enhanced schema validation...")
    registry = ParameterRegistry()
    try:
        load_parameters_from_yaml("qfd_params/defaults.yaml", registry)
        print("   ✓ YAML schema validation passed")
        print(f"   ✓ Loaded {len(registry.get_all_parameters())} parameters")
    except ValueError as e:
        print(f"   ✗ YAML schema validation failed: {e}")
        return
    print()
    
    # 2. Set up some parameter values and create conflicts
    print("2. Setting up parameters and creating conflicts for demonstration...")
    
    # Create a FIXED vs BOUNDED conflict
    fixed_constraint = Constraint(
        realm="demo_realm",
        constraint_type=ConstraintType.FIXED,
        target_value=15.0,  # Outside the config bounds [-10, 10]
        tolerance=1e-6,
        notes="Demo fixed value outside bounds"
    )
    registry.add_constraint("V2", fixed_constraint)
    
    # Create a FIXED vs TARGET conflict
    target_constraint = Constraint(
        realm="another_realm",
        constraint_type=ConstraintType.TARGET,
        target_value=0.95,  # Different from PPN_gamma target of 1.0
        tolerance=1e-5,
        notes="Demo conflicting target"
    )
    registry.add_constraint("PPN_gamma", target_constraint)
    
    # Set some parameter values
    registry.update_parameter("k_J", 1e-8, "realm0_cmb", "Near-zero for vacuum")
    registry.update_parameter("PPN_gamma", 1.000001, "realm3", "Slightly off GR")
    registry.update_parameter("PPN_beta", 0.99999, "realm3", "Close to GR")
    
    print("   Added conflicting constraints and parameter values")
    print()
    
    # 3. Enhanced conflict detection
    print("3. Running enhanced conflict detection...")
    conflicts = registry.get_conflicting_constraints()
    
    print(f"   Found {len(conflicts)} conflicts:")
    for i, conflict in enumerate(conflicts, 1):
        print(f"   {i}. {conflict['type']} on parameter '{conflict['parameter']}'")
        if conflict['type'] == 'fixed_outside_bounds':
            print(f"      Fixed value {conflict['fixed_value']} from {conflict['fixed_realm']}")
            print(f"      Outside bounds {conflict['bounds']} from {conflict['bounds_realm']}")
        elif conflict['type'] == 'fixed_vs_target_mismatch':
            print(f"      Fixed: {conflict['fixed_value']} ({conflict['fixed_realm']})")
            print(f"      Target: {conflict['target_value']} ± {conflict['tolerance']} ({conflict['target_realm']})")
            print(f"      Difference: {conflict['actual_difference']:.6f}")
    print()
    
    # 4. Set up enhanced validation suite
    print("4. Setting up enhanced validation suite...")
    validator = CompositeValidator("Enhanced QFD Validator")
    
    # Basic validators
    validator.add_validator(BoundsValidator())
    validator.add_validator(FixedValueValidator())
    validator.add_validator(TargetValueValidator())
    validator.add_validator(ConflictValidator())
    
    # Physics-specific validators
    validator.add_validator(PPNValidator())
    
    # Derived constraint validator with vacuum check
    derived_validator = DerivedConstraintValidator("Physics Derived Constraints")
    derived_validator.add_evaluator("n_vacuum", create_vacuum_refractive_index_evaluator())
    validator.add_validator(derived_validator)
    
    print(f"   Registered {len(validator.get_validator_names())} validators:")
    for name in validator.get_validator_names():
        print(f"     - {name}")
    print()
    
    # 5. Run comprehensive validation
    print("5. Running comprehensive validation...")
    report = validator.validate_all(registry)
    
    print(f"   Overall Status: {report.overall_status.value}")
    print(f"   Execution Time: {report.execution_time_ms:.2f}ms")
    print(f"   Parameters: {report.total_parameters}")
    print(f"   Constraints: {report.total_constraints}")
    print(f"   Violations: {report.total_violations}")
    print(f"   Warnings: {report.total_warnings}")
    print()
    
    # 6. Detailed results with metadata
    print("6. Detailed validation results with metadata:")
    for result in report.validator_results:
        print(f"   {result.validator_name}:")
        print(f"     Status: {result.status.value}")
        print(f"     Time: {result.execution_time_ms:.2f}ms")
        print(f"     Checked: {result.parameters_checked} params, {result.constraints_checked} constraints")
        
        # Show violation type metadata
        if "violation_types" in result.metadata:
            print(f"     Violation types: {result.metadata['violation_types']}")
        
        if result.violations:
            print(f"     Violations ({len(result.violations)}):")
            for violation in result.violations:
                print(f"       - {violation.parameter_name}: {violation.violation_type}")
                print(f"         {violation.message}")
        
        if result.warnings:
            print(f"     Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"       - {warning}")
        
        if result.info_messages:
            print(f"     Info ({len(result.info_messages)}):")
            for info in result.info_messages:
                print(f"       - {info}")
        print()
    
    # 7. Violations grouped by parameter
    print("7. Violations grouped by parameter:")
    violations_by_param = report.get_violations_by_parameter()
    for param_name, violations in violations_by_param.items():
        print(f"   {param_name}:")
        for violation in violations:
            print(f"     - {violation.violation_type}: {violation.message}")
            if violation.expected_value is not None:
                print(f"       Expected: {violation.expected_value}")
            if violation.actual_value is not None:
                print(f"       Actual: {violation.actual_value}")
            if violation.expected_range is not None:
                print(f"       Range: {violation.expected_range}")
    print()
    
    # 8. Summary
    print("8. Enhanced Validation Summary:")
    print(report.get_summary())
    
    print("=== Enhanced Demo Complete ===")
    print("\nKey enhancements demonstrated:")
    print("✓ Enhanced conflict detection (FIXED vs BOUNDED/TARGET)")
    print("✓ YAML schema validation with numeric type checking")
    print("✓ Derived constraint validation framework")
    print("✓ PPN parameter validation")
    print("✓ Validation metadata with violation type counts")
    print("✓ Comprehensive error reporting and categorization")


if __name__ == "__main__":
    main()