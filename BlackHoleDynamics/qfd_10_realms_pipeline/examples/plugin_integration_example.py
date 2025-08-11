#!/usr/bin/env python3
"""
Example demonstrating plugin system integration with the validation framework.

This example shows how to:
1. Register custom constraint plugins
2. Integrate plugins with the main validation pipeline
3. Handle plugin conflicts and resolution
4. Generate comprehensive validation reports
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, Constraint, ConstraintType
)
from coupling_constants.plugins.plugin_manager import PluginManager
from coupling_constants.plugins.constraint_plugins.example_plugins import (
    PhotonMassConstraintPlugin, VacuumStabilityPlugin, CosmologicalConstantPlugin
)
from coupling_constants.validation.plugin_validator import PluginValidator, PluginValidatorFactory
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.ppn_validator import PPNValidator
from coupling_constants.validation.cmb_validator import CMBValidator


def main():
    """Demonstrate plugin system integration."""
    
    print("=== QFD Plugin System Integration Example ===\n")
    
    # 1. Create parameter registry and add some test parameters
    print("1. Setting up parameter registry...")
    registry = ParameterRegistry()
    
    # Add physics parameters
    registry.update_parameter("m_gamma", 1e-20, "experimental", "Photon mass (eV)")
    registry.update_parameter("n_vac", 1.0, "vacuum", "Vacuum refractive index")
    registry.update_parameter("k_J", 1e-15, "qfd", "Incoherent photon drag parameter")
    registry.update_parameter("xi", 2.0, "qfd", "QFD coupling parameter")
    registry.update_parameter("Lambda", 1.1e-52, "cosmology", "Cosmological constant")
    registry.update_parameter("H0", 70.0, "cosmology", "Hubble constant (km/s/Mpc)")
    registry.update_parameter("psi_s0", -1.5, "qfd", "QFD scalar field parameter")
    
    # Add some constraints
    constraint1 = Constraint(
        realm="experimental",
        constraint_type=ConstraintType.BOUNDED,
        min_value=0.0,
        max_value=1e-18,
        notes="Experimental photon mass limit"
    )
    registry.add_constraint("m_gamma", constraint1)
    
    constraint2 = Constraint(
        realm="vacuum",
        constraint_type=ConstraintType.FIXED,
        target_value=1.0,
        tolerance=1e-10,
        notes="Vacuum refractive index must be exactly 1"
    )
    registry.add_constraint("n_vac", constraint2)
    
    print(f"   Added {len(registry.get_all_parameters())} parameters")
    print(f"   Added {sum(len(p.constraints) for p in registry.get_all_parameters().values())} constraints")
    
    # 2. Create plugin manager and register plugins
    print("\n2. Setting up plugin system...")
    plugin_manager = PluginManager()
    
    # Register example plugins
    photon_plugin = PhotonMassConstraintPlugin()
    vacuum_plugin = VacuumStabilityPlugin()
    cosmo_plugin = CosmologicalConstantPlugin()
    
    success_count = 0
    success_count += plugin_manager.register_plugin(photon_plugin)
    success_count += plugin_manager.register_plugin(vacuum_plugin)
    success_count += plugin_manager.register_plugin(cosmo_plugin)
    
    print(f"   Successfully registered {success_count}/3 plugins")
    
    # Show plugin information
    plugins_info = plugin_manager.get_registered_plugins()
    for name, info in plugins_info.items():
        print(f"   - {name}: {info.description} (Priority: {info.priority.name})")
    
    # 3. Create plugin validator and integrate with validation framework
    print("\n3. Creating integrated validation framework...")
    
    # Create plugin validator
    plugin_validator = PluginValidatorFactory.create_default_plugin_validator(plugin_manager)
    
    # Create composite validator with built-in and plugin validators
    composite_validator = CompositeValidator("QFD Comprehensive Validator")
    
    # Add built-in validators (if available)
    try:
        ppn_validator = PPNValidator()
        composite_validator.add_validator(ppn_validator)
        print("   Added PPN validator")
    except:
        print("   PPN validator not available")
    
    try:
        cmb_validator = CMBValidator()
        composite_validator.add_validator(cmb_validator)
        print("   Added CMB validator")
    except:
        print("   CMB validator not available")
    
    # Add plugin validator
    composite_validator.add_validator(plugin_validator)
    print("   Added plugin validator")
    
    print(f"   Total validators: {len(composite_validator.validators)}")
    
    # 4. Run comprehensive validation
    print("\n4. Running comprehensive validation...")
    
    validation_report = composite_validator.validate_all(registry)
    
    print(f"   Overall status: {validation_report.overall_status.value}")
    print(f"   Total violations: {validation_report.total_violations}")
    print(f"   Total warnings: {validation_report.total_warnings}")
    print(f"   Execution time: {validation_report.execution_time_ms:.2f} ms")
    
    # Show detailed results
    print("\n   Validator Results:")
    for result in validation_report.validator_results:
        status_icon = "✓" if result.is_valid() else "✗"
        print(f"   {status_icon} {result.validator_name}: {result.status.value}")
        
        if result.violations:
            for violation in result.violations[:3]:  # Show first 3 violations
                print(f"      - {violation.violation_type}: {violation.message}")
            if len(result.violations) > 3:
                print(f"      - ... and {len(result.violations) - 3} more violations")
        
        if result.warnings:
            for warning in result.warnings[:2]:  # Show first 2 warnings
                print(f"      ⚠ {warning}")
            if len(result.warnings) > 2:
                print(f"      ⚠ ... and {len(result.warnings) - 2} more warnings")
    
    # 5. Test plugin conflict detection and resolution
    print("\n5. Testing plugin conflict detection...")
    
    # Create a conflicting parameter value to trigger conflicts
    registry.update_parameter("m_gamma", 1e-15, "test", "Conflicting photon mass")
    
    conflicts = plugin_manager.get_plugin_conflicts(registry)
    print(f"   Found {len(conflicts)} plugin conflicts")
    
    if conflicts:
        for conflict in conflicts:
            print(f"   - Parameter: {conflict['parameter']}")
            print(f"     Conflict type: {conflict['conflict_type']}")
            print(f"     Valid plugins: {conflict.get('valid_plugins', [])}")
            print(f"     Invalid plugins: {conflict.get('invalid_plugins', [])}")
        
        # Resolve conflicts
        resolution = plugin_manager.resolve_plugin_conflicts(conflicts, "priority")
        print(f"   Resolution actions:")
        print(f"   - Disabled plugins: {resolution.get('disabled_plugins', [])}")
        print(f"   - Priority overrides: {len(resolution.get('priority_overrides', []))}")
        print(f"   - Warnings: {len(resolution.get('warnings', []))}")
    
    # 6. Generate plugin summary report
    print("\n6. Plugin system summary:")
    
    plugin_summary = plugin_validator.get_plugin_summary()
    print(f"   Total plugins: {plugin_summary['total_plugins']}")
    print(f"   Active plugins: {plugin_summary['active_plugins']}")
    print(f"   Inactive plugins: {plugin_summary['inactive_plugins']}")
    
    print("   Plugins by priority:")
    for priority, plugins in plugin_summary['plugins_by_priority'].items():
        print(f"   - {priority}: {', '.join(plugins)}")
    
    # 7. Export plugin information
    print("\n7. Exporting plugin information...")
    
    plugin_manager.export_plugin_info("plugin_info_export.json")
    print("   Plugin information exported to plugin_info_export.json")
    
    # 8. Demonstrate different conflict resolution strategies
    print("\n8. Testing different conflict resolution strategies...")
    
    # Reset parameter to conflicting value
    registry.update_parameter("m_gamma", 1e-15, "test", "Conflicting photon mass")
    
    strategies = ["priority", "disable_lower", "user_choice"]
    for strategy in strategies:
        print(f"\n   Testing {strategy} strategy:")
        
        # Create fresh plugin validator with different strategy
        test_validator = PluginValidator(plugin_manager, f"Test {strategy} Validator")
        test_validator.set_conflict_resolution_strategy(strategy)
        
        # Run validation
        result = test_validator.validate(registry)
        print(f"   - Status: {result.status.value}")
        print(f"   - Violations: {len(result.violations)}")
        print(f"   - Warnings: {len(result.warnings)}")
        
        if result.metadata.get('plugin_conflicts', 0) > 0:
            print(f"   - Conflicts detected: {result.metadata['plugin_conflicts']}")
    
    print("\n=== Plugin Integration Example Complete ===")


if __name__ == "__main__":
    main()