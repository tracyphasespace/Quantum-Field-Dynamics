#!/usr/bin/env python3
"""
Basic usage example of the coupling constants mapping system.

This script demonstrates how to:
1. Load parameters from YAML configuration
2. Add constraints from different realms
3. Update parameter values
4. Validate constraints
5. Export the registry state
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, Constraint, ConstraintType
)
from coupling_constants.config.yaml_loader import load_parameters_from_yaml, get_parameter_summary


def main():
    print("=== Coupling Constants Mapping System Demo ===\n")
    
    # 1. Create registry and load configuration
    print("1. Loading parameters from YAML configuration...")
    registry = ParameterRegistry()
    load_parameters_from_yaml("qfd_params/defaults.yaml", registry)
    
    summary = get_parameter_summary(registry)
    print(f"   Loaded {summary['total_parameters']} parameters")
    print(f"   Total constraints: {summary['total_constraints']}")
    print(f"   Constraint types: {summary['constraint_types']}\n")
    
    # 2. Simulate realm execution - CMB realm confirms T_CMB
    print("2. Simulating Realm 0 (CMB) execution...")
    # T_CMB_K is already fixed by config, so let's work with a different parameter
    registry.update_parameter("psi_s0", -1.5, "realm0_cmb", "Set by thermalization zeropoint")
    
    # Add vacuum constraint from CMB realm
    vacuum_constraint = Constraint(
        realm="realm0_cmb",
        constraint_type=ConstraintType.BOUNDED,
        min_value=-1e-6,
        max_value=1e-6,
        notes="Vacuum drag must be ~0 to avoid CMB spectral distortions"
    )
    registry.add_constraint("k_J", vacuum_constraint)
    print("   Set psi_s0 = -1.5 (thermalization zeropoint)")
    print("   Added vacuum drag constraint for k_J\n")
    
    # 3. Simulate another realm trying to modify fixed parameter
    print("3. Testing parameter protection...")
    try:
        registry.update_parameter("T_CMB_K", 2.8, "realm1", "Attempt to modify")
        print("   ERROR: Should not have allowed modification!")
    except ValueError as e:
        print(f"   ✓ Protected parameter: {str(e)[:80]}...\n")
    
    # 4. Update a non-fixed parameter
    print("4. Updating non-fixed parameter...")
    registry.update_parameter("k_J", 1e-8, "realm0_cmb", "Set to near-zero for vacuum")
    k_j_param = registry.get_parameter("k_J")
    print(f"   k_J updated to {k_j_param.value}")
    print(f"   Change history: {len(k_j_param.history)} entries\n")
    
    # 5. Validate all constraints
    print("5. Validating all constraints...")
    violations = registry.validate_all_parameters()
    if violations:
        print("   Constraint violations found:")
        for param, viols in violations.items():
            print(f"     {param}: {viols}")
    else:
        print("   ✓ All constraints satisfied\n")
    
    # 6. Check for conflicts
    print("6. Checking for constraint conflicts...")
    conflicts = registry.get_conflicting_constraints()
    if conflicts:
        print("   Conflicts found:")
        for conflict in conflicts:
            print(f"     {conflict}")
    else:
        print("   ✓ No conflicts detected\n")
    
    # 7. Export registry state
    print("7. Exporting registry state...")
    export_data = registry.export_state("json")
    print(f"   Exported {len(export_data)} characters of JSON data")
    
    # Show a sample of the export
    import json
    data = json.loads(export_data)
    sample_param = list(data.keys())[0]
    print(f"   Sample parameter '{sample_param}':")
    print(f"     Value: {data[sample_param]['value']}")
    print(f"     Constraints: {len(data[sample_param]['constraints'])}")
    print(f"     Changes: {data[sample_param]['change_count']}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()