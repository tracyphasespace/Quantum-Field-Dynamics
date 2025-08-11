#!/usr/bin/env python3
"""
Main command-line interface for QFD coupling constants analysis.

This CLI provides access to all major functionality of the coupling constants
framework including parameter analysis, constraint validation, dependency mapping,
sensitivity analysis, and report generation.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..registry.parameter_registry import ParameterRegistry
from ..config.yaml_loader import load_parameters_from_yaml
from ..analysis.dependency_mapper import DependencyMapper
from ..analysis.sensitivity_analyzer import SensitivityAnalyzer
from ..analysis.realm_tracker import RealmTracker
from ..validation.base_validator import CompositeValidator
from ..validation.ppn_validator import PPNValidator
from ..validation.cmb_validator import CMBValidator
from ..validation.basic_validators import BoundsValidator, FixedValueValidator, TargetValueValidator
from ..visualization.coupling_visualizer import CouplingVisualizer
from ..visualization.export_manager import ExportManager
from ..plugins.plugin_manager import PluginManager
from ..plugins.constraint_plugins import (
    PhotonMassConstraintPlugin,
    VacuumStabilityPlugin,
    CosmologicalConstantPlugin
)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Set up logging configuration."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_registry(config_path: str) -> ParameterRegistry:
    """Load parameter registry from configuration file."""
    registry = ParameterRegistry()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    load_parameters_from_yaml(config_path, registry)
    logging.info(f"Loaded {len(registry.get_all_parameters())} parameters from {config_path}")
    
    return registry


def setup_validators(registry: ParameterRegistry) -> CompositeValidator:
    """Set up constraint validators."""
    composite = CompositeValidator("QFD Constraint Validator")
    
    # Add core validators
    composite.add_validator(PPNValidator())
    composite.add_validator(CMBValidator())
    composite.add_validator(BoundsValidator())
    composite.add_validator(FixedValueValidator())
    composite.add_validator(TargetValueValidator())
    
    logging.info(f"Set up {len(composite.validators)} core validators")
    return composite


def setup_plugins(plugin_manager: PluginManager, enable_plugins: List[str]) -> None:
    """Set up constraint plugins."""
    available_plugins = {
        'photon_mass': PhotonMassConstraintPlugin,
        'vacuum_stability': VacuumStabilityPlugin,
        'cosmological_constant': CosmologicalConstantPlugin
    }
    
    for plugin_name in enable_plugins:
        if plugin_name in available_plugins:
            plugin = available_plugins[plugin_name]()
            if plugin_manager.register_plugin(plugin):
                logging.info(f"Registered plugin: {plugin_name}")
            else:
                logging.warning(f"Failed to register plugin: {plugin_name}")
        else:
            logging.warning(f"Unknown plugin: {plugin_name}")


def cmd_validate(args) -> int:
    """Validate coupling constants constraints."""
    try:
        registry = load_registry(args.config)
        
        # Set up validators
        composite = setup_validators(registry)
        
        # Set up plugins if requested
        plugin_manager = PluginManager()
        if args.plugins:
            setup_plugins(plugin_manager, args.plugins)
        
        # Run validation
        logging.info("Running constraint validation...")
        report = composite.validate_all(registry)
        
        # Run plugin validation if plugins are enabled
        plugin_results = []
        if args.plugins:
            plugin_results = plugin_manager.validate_all_plugin_constraints(registry)
        
        # Output results
        if args.output_format == 'json':
            output_data = {
                'validation_report': {
                    'timestamp': report.timestamp.isoformat(),
                    'overall_status': report.overall_status.value,
                    'total_violations': report.total_violations,
                    'total_warnings': report.total_warnings,
                    'execution_time_ms': report.execution_time_ms,
                    'validator_results': [
                        {
                            'validator_name': result.validator_name,
                            'status': result.status.value,
                            'violations': len(result.violations),
                            'warnings': len(result.warnings)
                        }
                        for result in report.validator_results
                    ]
                },
                'plugin_results': [
                    {
                        'plugin_name': result.validator_name,
                        'status': result.status.value,
                        'violations': len(result.violations),
                        'warnings': len(result.warnings)
                    }
                    for result in plugin_results
                ]
            }
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                logging.info(f"Validation results saved to {args.output}")
            else:
                print(json.dumps(output_data, indent=2))
        
        else:  # text format
            output_lines = []
            output_lines.append("=== QFD Coupling Constants Validation Report ===")
            output_lines.append(f"Overall Status: {report.overall_status.value.upper()}")
            output_lines.append(f"Total Violations: {report.total_violations}")
            output_lines.append(f"Total Warnings: {report.total_warnings}")
            output_lines.append(f"Execution Time: {report.execution_time_ms:.2f}ms")
            output_lines.append("")
            
            # Core validator results
            output_lines.append("Core Validator Results:")
            for result in report.validator_results:
                status_symbol = "✓" if result.is_valid() else "✗"
                output_lines.append(f"  {status_symbol} {result.validator_name}: {result.status.value}")
                if result.violations:
                    for violation in result.violations[:3]:  # Show first 3 violations
                        output_lines.append(f"    - {violation.message}")
                    if len(result.violations) > 3:
                        output_lines.append(f"    ... and {len(result.violations) - 3} more violations")
            
            # Plugin results
            if plugin_results:
                output_lines.append("")
                output_lines.append("Plugin Validator Results:")
                for result in plugin_results:
                    status_symbol = "✓" if result.is_valid() else "✗"
                    plugin_name = result.validator_name.replace("plugin_", "")
                    output_lines.append(f"  {status_symbol} {plugin_name}: {result.status.value}")
                    if result.violations:
                        for violation in result.violations[:2]:  # Show first 2 violations
                            output_lines.append(f"    - {violation.message}")
            
            output_text = "\n".join(output_lines)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_text)
                logging.info(f"Validation report saved to {args.output}")
            else:
                print(output_text)
        
        # Return appropriate exit code
        return 0 if report.overall_status.value in ['valid', 'warning'] else 1
        
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return 1


def cmd_analyze(args) -> int:
    """Perform dependency and sensitivity analysis."""
    try:
        registry = load_registry(args.config)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dependency analysis
        logging.info("Building dependency graph...")
        dependency_mapper = DependencyMapper(registry)
        dependency_mapper.build_dependency_graph()
        
        # Sensitivity analysis
        sensitivity_analyzer = None
        if args.sensitivity:
            logging.info("Performing sensitivity analysis...")
            sensitivity_analyzer = SensitivityAnalyzer(registry)
            
            # Run sensitivity analysis for key observables
            observables = args.observables or ['PPN_gamma', 'PPN_beta', 'T_CMB']
            for observable in observables:
                try:
                    sensitivity_analyzer.compute_parameter_sensitivity(observable)
                    logging.info(f"Computed sensitivity for {observable}")
                except Exception as e:
                    logging.warning(f"Failed to compute sensitivity for {observable}: {e}")
            
            # Monte Carlo analysis if requested
            if args.monte_carlo:
                logging.info(f"Running Monte Carlo analysis with {args.monte_carlo} samples...")
                sensitivity_analyzer.perform_monte_carlo_analysis(args.monte_carlo)
        
        # Generate reports
        export_manager = ExportManager(registry)
        
        if args.format == 'comprehensive':
            export_manager.create_comprehensive_report(
                str(output_dir),
                dependency_mapper=dependency_mapper,
                sensitivity_analyzer=sensitivity_analyzer
            )
        elif args.format == 'publication':
            key_params = args.key_parameters or ['k_J', 'xi', 'psi_s0', 'PPN_gamma', 'PPN_beta']
            export_manager.export_for_publication(
                str(output_dir),
                key_parameters=key_params,
                paper_title=args.title or "QFD Coupling Constants Analysis"
            )
        else:
            # Individual format exports
            if 'json' in args.format:
                export_manager.export_parameters_json(str(output_dir / "parameters.json"))
            if 'yaml' in args.format:
                export_manager.export_parameters_yaml(str(output_dir / "parameters.yaml"))
            if 'csv' in args.format:
                export_manager.export_parameters_csv(str(output_dir / "parameters.csv"))
            if 'latex' in args.format:
                export_manager.export_latex_table(str(output_dir / "parameters.tex"))
        
        # Generate visualizations if requested
        if args.visualize:
            logging.info("Generating visualizations...")
            visualizer = CouplingVisualizer(registry)
            
            # Dependency graph
            visualizer.plot_dependency_graph(
                dependency_mapper,
                str(output_dir / "dependency_graph.png")
            )
            
            # Parameter constraints
            visualizer.plot_parameter_constraints(
                str(output_dir / "parameter_constraints.png")
            )
            
            # Create dashboard
            visualizer.create_dashboard(
                dependency_mapper,
                sensitivity_analyzer=sensitivity_analyzer,
                output_dir=str(output_dir / "dashboard")
            )
        
        logging.info(f"Analysis complete. Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return 1


def cmd_export(args) -> int:
    """Export coupling constants data."""
    try:
        registry = load_registry(args.config)
        export_manager = ExportManager(registry)
        
        # Create output directory
        output_path = Path(args.output)
        if args.format in ['comprehensive', 'publication']:
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            if output_path.parent != Path('.'):
                output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'json':
            export_manager.export_parameters_json(str(output_path))
        elif args.format == 'yaml':
            export_manager.export_parameters_yaml(str(output_path))
        elif args.format == 'csv':
            export_manager.export_parameters_csv(str(output_path))
        elif args.format == 'latex':
            export_manager.export_latex_table(
                str(output_path),
                parameters=args.parameters,
                table_caption=args.caption or "QFD Coupling Constants"
            )
        elif args.format == 'comprehensive':
            export_manager.create_comprehensive_report(str(output_path))
        elif args.format == 'publication':
            export_manager.export_for_publication(
                str(output_path),
                key_parameters=args.parameters,
                paper_title=args.title or "QFD Coupling Constants Analysis"
            )
        
        logging.info(f"Export complete: {output_path}")
        return 0
        
    except Exception as e:
        logging.error(f"Export failed: {e}")
        return 1


def cmd_visualize(args) -> int:
    """Generate visualizations."""
    try:
        registry = load_registry(args.config)
        visualizer = CouplingVisualizer(registry)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build dependency mapper for graph visualizations
        dependency_mapper = DependencyMapper(registry)
        dependency_mapper.build_dependency_graph()
        
        if args.type == 'dependency' or args.type == 'all':
            visualizer.plot_dependency_graph(
                dependency_mapper,
                str(output_dir / "dependency_graph.png"),
                layout=args.layout,
                show_edge_labels=args.show_labels,
                highlight_critical_path=args.highlight_critical
            )
        
        if args.type == 'constraints' or args.type == 'all':
            visualizer.plot_parameter_constraints(
                str(output_dir / "parameter_constraints.png"),
                parameters=args.parameters
            )
        
        if args.type == 'evolution' or args.type == 'all':
            if args.parameters:
                visualizer.plot_parameter_evolution(
                    args.parameters,
                    str(output_dir / "parameter_evolution.png")
                )
        
        if args.type == 'dashboard' or args.type == 'all':
            # Set up sensitivity analyzer if needed
            sensitivity_analyzer = None
            if args.sensitivity:
                sensitivity_analyzer = SensitivityAnalyzer(registry)
                # Quick sensitivity analysis for dashboard
                try:
                    sensitivity_analyzer.compute_parameter_sensitivity('PPN_gamma')
                except:
                    pass  # Continue without sensitivity data
            
            visualizer.create_dashboard(
                dependency_mapper,
                sensitivity_analyzer=sensitivity_analyzer,
                output_dir=str(output_dir / "dashboard")
            )
        
        logging.info(f"Visualizations saved to {output_dir}")
        return 0
        
    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        return 1


def cmd_plugins(args) -> int:
    """Manage constraint plugins."""
    try:
        plugin_manager = PluginManager()
        
        if args.action == 'list':
            # List available plugins
            available_plugins = {
                'photon_mass': 'Photon mass constraint from experimental limits',
                'vacuum_stability': 'Vacuum stability constraint (n_vac = 1)',
                'cosmological_constant': 'Cosmological constant consistency'
            }
            
            print("Available Plugins:")
            for name, description in available_plugins.items():
                print(f"  {name}: {description}")
            
            # List registered plugins
            registered = plugin_manager.get_registered_plugins()
            if registered:
                print("\nRegistered Plugins:")
                for name, info in registered.items():
                    status = "active" if info.active else "inactive"
                    print(f"  {name} (v{info.version}): {info.description} [{status}]")
            else:
                print("\nNo plugins currently registered.")
        
        elif args.action == 'register':
            if not args.plugin:
                logging.error("Plugin name required for registration")
                return 1
            
            setup_plugins(plugin_manager, [args.plugin])
            
            if args.output:
                plugin_manager.export_plugin_info(args.output)
                logging.info(f"Plugin info exported to {args.output}")
        
        elif args.action == 'validate':
            if not args.config:
                logging.error("Configuration file required for plugin validation")
                return 1
            
            registry = load_registry(args.config)
            
            # Register requested plugins
            if args.plugin:
                setup_plugins(plugin_manager, [args.plugin])
            else:
                # Register all available plugins
                setup_plugins(plugin_manager, ['photon_mass', 'vacuum_stability', 'cosmological_constant'])
            
            # Run plugin validation
            results = plugin_manager.validate_all_plugin_constraints(registry)
            
            print("Plugin Validation Results:")
            for result in results:
                status_symbol = "✓" if result.is_valid() else "✗"
                plugin_name = result.validator_name.replace("plugin_", "")
                print(f"  {status_symbol} {plugin_name}: {result.status.value}")
                
                if result.violations:
                    for violation in result.violations:
                        print(f"    - {violation.message}")
                
                if result.warnings:
                    for warning in result.warnings:
                        print(f"    ! {warning}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Plugin operation failed: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="QFD Coupling Constants Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate constraints
  qfd-coupling validate --config qfd_params/defaults.yaml
  
  # Full analysis with visualizations
  qfd-coupling analyze --config qfd_params/defaults.yaml --output-dir results --visualize --sensitivity
  
  # Export for publication
  qfd-coupling export --config qfd_params/defaults.yaml --format publication --output pub_data
  
  # Generate dependency graph
  qfd-coupling visualize --config qfd_params/defaults.yaml --type dependency --output-dir plots
  
  # List available plugins
  qfd-coupling plugins list
        """
    )
    
    # Global options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress all output except errors')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate coupling constants constraints')
    validate_parser.add_argument('--config', required=True,
                                help='Path to QFD configuration file')
    validate_parser.add_argument('--plugins', nargs='*', 
                                choices=['photon_mass', 'vacuum_stability', 'cosmological_constant'],
                                help='Enable constraint plugins')
    validate_parser.add_argument('--output', help='Output file for results')
    validate_parser.add_argument('--output-format', choices=['text', 'json'], default='text',
                                help='Output format')
    validate_parser.set_defaults(func=cmd_validate)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Perform comprehensive analysis')
    analyze_parser.add_argument('--config', required=True,
                               help='Path to QFD configuration file')
    analyze_parser.add_argument('--output-dir', required=True,
                               help='Output directory for analysis results')
    analyze_parser.add_argument('--format', choices=['comprehensive', 'publication', 'json', 'yaml', 'csv', 'latex'],
                               default='comprehensive', help='Output format')
    analyze_parser.add_argument('--sensitivity', action='store_true',
                               help='Perform sensitivity analysis')
    analyze_parser.add_argument('--monte-carlo', type=int, metavar='N',
                               help='Number of Monte Carlo samples for uncertainty analysis')
    analyze_parser.add_argument('--observables', nargs='*',
                               help='Observables for sensitivity analysis')
    analyze_parser.add_argument('--key-parameters', nargs='*',
                               help='Key parameters for publication format')
    analyze_parser.add_argument('--title', help='Title for publication format')
    analyze_parser.add_argument('--visualize', action='store_true',
                               help='Generate visualizations')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export coupling constants data')
    export_parser.add_argument('--config', required=True,
                              help='Path to QFD configuration file')
    export_parser.add_argument('--format', required=True,
                              choices=['json', 'yaml', 'csv', 'latex', 'comprehensive', 'publication'],
                              help='Export format')
    export_parser.add_argument('--output', required=True,
                              help='Output file or directory')
    export_parser.add_argument('--parameters', nargs='*',
                              help='Specific parameters to export')
    export_parser.add_argument('--caption', help='Table caption for LaTeX format')
    export_parser.add_argument('--title', help='Title for publication format')
    export_parser.set_defaults(func=cmd_export)
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    visualize_parser.add_argument('--config', required=True,
                                 help='Path to QFD configuration file')
    visualize_parser.add_argument('--output-dir', required=True,
                                 help='Output directory for visualizations')
    visualize_parser.add_argument('--type', choices=['dependency', 'constraints', 'evolution', 'dashboard', 'all'],
                                 default='all', help='Type of visualization')
    visualize_parser.add_argument('--parameters', nargs='*',
                                 help='Specific parameters to visualize')
    visualize_parser.add_argument('--layout', choices=['spring', 'circular', 'hierarchical'],
                                 default='spring', help='Graph layout algorithm')
    visualize_parser.add_argument('--show-labels', action='store_true',
                                 help='Show edge labels in dependency graph')
    visualize_parser.add_argument('--highlight-critical', action='store_true',
                                 help='Highlight critical path in dependency graph')
    visualize_parser.add_argument('--sensitivity', action='store_true',
                                 help='Include sensitivity analysis in dashboard')
    visualize_parser.set_defaults(func=cmd_visualize)
    
    # Plugins command
    plugins_parser = subparsers.add_parser('plugins', help='Manage constraint plugins')
    plugins_parser.add_argument('action', choices=['list', 'register', 'validate'],
                               help='Plugin action')
    plugins_parser.add_argument('--plugin', help='Plugin name')
    plugins_parser.add_argument('--config', help='Path to QFD configuration file (for validate)')
    plugins_parser.add_argument('--output', help='Output file for plugin info')
    plugins_parser.set_defaults(func=cmd_plugins)
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose, args.quiet)
    
    # Check if command was provided
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())