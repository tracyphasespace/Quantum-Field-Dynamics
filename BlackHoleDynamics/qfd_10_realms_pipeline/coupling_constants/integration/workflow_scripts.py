"""
Workflow scripts for integrated realm execution with coupling constants analysis.

This module provides high-level scripts that integrate the coupling constants
framework with the existing QFD realm workflow.
"""

import os
import sys
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .realm_integration import RealmIntegrationManager, RealmExecutionHook
from ..registry.parameter_registry import Constraint, ConstraintType


def setup_logging(verbose: bool = False) -> None:
    """Set up logging for workflow scripts."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_default_realm_sequence() -> List[Tuple[str, str]]:
    """Get the default QFD realm execution sequence."""
    return [
        ('realm0_cmb', 'realms.realm0_cmb'),
        ('realm1_cosmic_baseline', 'realms.realm1_cosmic_baseline'),
        ('realm2_dark_energy', 'realms.realm2_dark_energy'),
        ('realm3_scales', 'realms.realm3_scales'),
        ('realm4_em_charge', 'realms.realm4_em_charge'),
        ('realm5_electron', 'realms.realm5_electron'),
        ('realm6_leptons_isomer', 'realms.realm6_leptons_isomer'),
        ('realm7_proton', 'realms.realm7_proton'),
        ('realm8_neutron_beta', 'realms.realm8_neutron_beta'),
        ('realm9_deuteron', 'realms.realm9_deuteron'),
        ('realm10_isotopes', 'realms.realm10_isotopes')
    ]


def setup_default_realm_hooks(manager: RealmIntegrationManager) -> None:
    """Set up default hooks for QFD realms."""
    
    # CMB realm hook
    def cmb_pre_hook(registry, params):
        """Pre-execution hook for CMB realm."""
        # Set CMB temperature constraint
        constraint = Constraint(
            realm="realm0_cmb",
            constraint_type=ConstraintType.FIXED,
            target_value=2.725,
            tolerance=1e-6,
            notes="CMB temperature constraint"
        )
        registry.add_constraint("T_CMB_K", constraint)
    
    def cmb_post_hook(registry, result):
        """Post-execution hook for CMB realm."""
        # Add vacuum constraints based on CMB results
        if 'fixed' in result and 'T_CMB_K' in result['fixed']:
            registry.update_parameter("T_CMB_K", result['fixed']['T_CMB_K'], "realm0_cmb", "CMB thermalization")
    
    cmb_hook = RealmExecutionHook(
        realm_name="realm0_cmb",
        pre_execution_hook=cmb_pre_hook,
        post_execution_hook=cmb_post_hook,
        validation_required=True
    )
    manager.register_realm_hook(cmb_hook)
    
    # Scales realm hook
    def scales_post_hook(registry, result):
        """Post-execution hook for scales realm."""
        # Update scale-dependent parameters
        if 'narrowed' in result:
            for param_name, constraint_info in result['narrowed'].items():
                if isinstance(constraint_info, str) and 'scale' in constraint_info.lower():
                    # Add scale-dependent constraint
                    constraint = Constraint(
                        realm="realm3_scales",
                        constraint_type=ConstraintType.BOUNDED,
                        min_value=0.0,
                        max_value=1.0,
                        notes=f"Scale constraint: {constraint_info}"
                    )
                    registry.add_constraint(param_name, constraint)
    
    scales_hook = RealmExecutionHook(
        realm_name="realm3_scales",
        post_execution_hook=scales_post_hook,
        validation_required=True
    )
    manager.register_realm_hook(scales_hook)
    
    # EM charge realm hook
    def em_post_hook(registry, result):
        """Post-execution hook for EM charge realm."""
        # Update electromagnetic coupling parameters
        if 'fixed' in result:
            for param_name, value in result['fixed'].items():
                if 'alpha' in param_name.lower() or 'charge' in param_name.lower():
                    if isinstance(value, (int, float)):
                        registry.update_parameter(param_name, float(value), "realm4_em_charge", "EM coupling fixed")
    
    em_hook = RealmExecutionHook(
        realm_name="realm4_em_charge",
        post_execution_hook=em_post_hook,
        validation_required=True
    )
    manager.register_realm_hook(em_hook)


def run_integrated_analysis(config_path: str, output_dir: str = "integrated_analysis",
                          realm_sequence: Optional[List[Tuple[str, str]]] = None,
                          enable_plugins: Optional[List[str]] = None,
                          generate_visualizations: bool = True,
                          verbose: bool = False) -> str:
    """
    Run integrated coupling constants analysis with realm execution.
    
    Args:
        config_path: Path to QFD configuration file
        output_dir: Directory for analysis outputs
        realm_sequence: Custom realm sequence (uses default if None)
        enable_plugins: List of plugins to enable
        generate_visualizations: Whether to generate visualization plots
        verbose: Enable verbose logging
        
    Returns:
        Path to the generated analysis report
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting integrated coupling constants analysis")
    
    # Initialize integration manager
    manager = RealmIntegrationManager(config_path, output_dir)
    
    # Set up default realm hooks
    setup_default_realm_hooks(manager)
    
    # Register plugins if requested
    if enable_plugins:
        for plugin_name in enable_plugins:
            if manager.register_plugin(plugin_name):
                logger.info(f"Registered plugin: {plugin_name}")
            else:
                logger.warning(f"Failed to register plugin: {plugin_name}")
    
    # Use default realm sequence if none provided
    if realm_sequence is None:
        realm_sequence = get_default_realm_sequence()
    
    # Execute realm sequence
    logger.info(f"Executing realm sequence: {[r[0] for r in realm_sequence]}")
    results = manager.execute_realm_sequence(realm_sequence)
    
    # Generate analysis report
    report_path = manager.generate_analysis_report(include_visualizations=generate_visualizations)
    
    # Print execution summary
    summary = manager.get_execution_summary()
    logger.info("Execution Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"Integrated analysis complete. Report: {report_path}")
    return report_path


def run_realm_sequence_with_analysis(realm_names: List[str], config_path: str,
                                    output_dir: str = "realm_analysis",
                                    realm_params: Optional[Dict[str, Dict[str, Any]]] = None,
                                    enable_plugins: Optional[List[str]] = None,
                                    verbose: bool = False) -> str:
    """
    Run a specific sequence of realms with coupling constants analysis.
    
    Args:
        realm_names: List of realm names to execute
        config_path: Path to QFD configuration file
        output_dir: Directory for analysis outputs
        realm_params: Parameters for each realm
        enable_plugins: List of plugins to enable
        verbose: Enable verbose logging
        
    Returns:
        Path to the generated analysis report
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running realm sequence with analysis: {realm_names}")
    
    # Build realm sequence from names
    realm_sequence = []
    for realm_name in realm_names:
        module_path = f"realms.{realm_name}"
        realm_sequence.append((realm_name, module_path))
    
    # Initialize integration manager
    manager = RealmIntegrationManager(config_path, output_dir)
    
    # Set up default realm hooks
    setup_default_realm_hooks(manager)
    
    # Register plugins if requested
    if enable_plugins:
        for plugin_name in enable_plugins:
            manager.register_plugin(plugin_name)
    
    # Execute realm sequence
    results = manager.execute_realm_sequence(realm_sequence, realm_params)
    
    # Generate analysis report
    report_path = manager.generate_analysis_report(include_visualizations=True)
    
    # Print results
    logger.info("Realm Execution Results:")
    for result in results:
        status_symbol = "✓" if result.status.value == "completed" else "✗"
        logger.info(f"  {status_symbol} {result.realm_name}: {result.status.value} ({result.execution_time_ms:.2f}ms)")
        if result.parameters_modified:
            logger.info(f"    Modified parameters: {', '.join(result.parameters_modified)}")
    
    return report_path


def create_realm_integration_script(realm_name: str, output_path: str) -> None:
    """
    Create a standalone integration script for a specific realm.
    
    Args:
        realm_name: Name of the realm
        output_path: Path to save the integration script
    """
    script_content = f'''#!/usr/bin/env python3
"""
Integration script for {realm_name} with coupling constants analysis.

This script runs {realm_name} with full coupling constants integration,
including parameter tracking, constraint validation, and analysis reporting.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from coupling_constants.integration.workflow_scripts import run_realm_sequence_with_analysis


def main():
    """Main entry point for {realm_name} integration."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config_path = "qfd_params/defaults.yaml"
    output_dir = f"{realm_name}_analysis"
    
    # Run realm with analysis
    try:
        report_path = run_realm_sequence_with_analysis(
            realm_names=["{realm_name}"],
            config_path=config_path,
            output_dir=output_dir,
            enable_plugins=["vacuum_stability", "photon_mass"],
            verbose=True
        )
        
        print(f"\\n{realm_name} analysis complete!")
        print(f"Report generated: {{report_path}}")
        
    except Exception as e:
        logging.error(f"Failed to run {realm_name} analysis: {{e}}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    # Write script to file
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    try:
        os.chmod(output_path, 0o755)
    except:
        pass  # Windows doesn't support chmod
    
    print(f"Created integration script: {output_path}")


def validate_realm_integration(config_path: str, realm_name: str, 
                             enable_plugins: Optional[List[str]] = None) -> bool:
    """
    Validate that a realm can be integrated with the coupling constants framework.
    
    Args:
        config_path: Path to QFD configuration file
        realm_name: Name of the realm to validate
        enable_plugins: List of plugins to enable for validation
        
    Returns:
        True if integration is successful, False otherwise
    """
    try:
        # Initialize integration manager
        manager = RealmIntegrationManager(config_path, "validation_test")
        
        # Register plugins if requested
        if enable_plugins:
            for plugin_name in enable_plugins:
                manager.register_plugin(plugin_name)
        
        # Try to execute the realm
        module_path = f"realms.{realm_name}"
        result = manager.execute_realm_with_integration(realm_name, module_path)
        
        # Check if execution was successful
        success = result.status.value == "completed"
        
        if success:
            print(f"✓ {realm_name} integration validation successful")
            print(f"  Execution time: {result.execution_time_ms:.2f}ms")
            print(f"  Parameters modified: {len(result.parameters_modified)}")
            print(f"  Constraints added: {result.constraints_added}")
        else:
            print(f"✗ {realm_name} integration validation failed")
            if 'error' in result.metadata:
                print(f"  Error: {result.metadata['error']}")
        
        return success
        
    except Exception as e:
        print(f"✗ {realm_name} integration validation failed: {e}")
        return False