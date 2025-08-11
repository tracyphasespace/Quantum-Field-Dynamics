#!/usr/bin/env python3
"""
Validation framework demonstration.

This script shows how to use the validation framework to check
coupling constants against various types of constraints.
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


def main():
    print("=== Validation Framework Demo ===\n")
    
    # 1. Set up registry with configuration
    print("1. Loading parameters and setting up validation...")
    registry = ParameterRegistry()
    load_parameters_from_yaml("qfd_params/defaults.yaml", registry)
    
    # 2. Set some parameter values (simulating realm execution)
    print("2. Setting parameter values (simulating realm execution)...")
    registry.update_parameter("k_J", 1e-8, "realm0_cmb", "Near-zero for vacuum")
    registry.update_parameter("psi_s0", -1.5, "realm0_cmb", "Thermalization zeropoint")
    registry.update_parameter("xi", 2.0, "realm1", "EM energy response")
    registry.update_parameter("PPN_gamma", 1.000001, "realm3", "Slightly off GR value")
    registry.update_parameter("PPN_beta", 0.99999, "realm3", "Close to GR value")
    
    # Add a problematic value
    registry.update_parameter("V2", -15.0, "realm4", "Outside bounds!")
    
    print("   Set 6 parameter values\n")
    
    # 3. Set up composite validator
    print("3. Setting up validation framework...")
    validator = CompositeValidator("QFD Coupling Constants Validator")
    validator.add_validator(BoundsValidator())
    validator.add_validator(FixedValueValidator())
    validator.add_validator(TargetValueValidator())
    validator.add_validator(ConflictValidator())
    
    print(f"   Registered {len(validator.get_validator_names())} validators:")
    for name in validator.get_validator_names():
        print(f"     - {name}")
    print()
    
    # 4. Run validation
    print("4. Running comprehensive validation...")
    report = validator.validate_all(registry)
    
    print(f"   Overall Status: {report.overall_status.value}")
    print(f"   Execution Time: {report.execution_time_ms:.2f}ms")
    print(f"   Parameters Checked: {report.total_parameters}")
    print(f"   Constraints Checked: {report.total_constraints}")
    print(f"   Total Violations: {report.total_violations}")
    print(f"   Total Warnings: {report.total_warnings}")
    print()
    
    # 5. Show detailed results
    print("5. Detailed validation results:")
    for result in report.validator_results:
        print(f"   {result.validator_name}: {result.status.value}")
        print(f"     Parameters: {result.parameters_checked}, Constraints: {result.constraints_checked}")
        print(f"     Time: {result.execution_time_ms:.2f}ms")
        
        if result.violations:
            print(f"     Violations ({len(result.violations)}):")
            for violation in result.violations:
                print(f"       - {violation.parameter_name}: {violation.message}")
        
        if result.warnings:
            print(f"     Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"       - {warning}")
        
        if result.info_messages:
            print(f"     Info ({len(result.info_messages)}):")
            for info in result.info_messages:
                print(f"       - {info}")
        print()
    
    # 6. Show violations by parameter
    if report.total_violations > 0:
        print("6. Violations grouped by parameter:")
        violations_by_param = report.get_violations_by_parameter()
        for param_name, violations in violations_by_param.items():
            print(f"   {param_name}:")
            for violation in violations:
                print(f"     - {violation.violation_type}: {violation.message}")
        print()
    
    # 7. Show summary
    print("7. Validation Summary:")
    print(report.get_summary())
    
    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()