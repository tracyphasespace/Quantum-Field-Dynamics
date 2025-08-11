"""
Integration manager for coupling constants analysis with existing realm workflow.

This module provides hooks and managers to integrate the coupling constants
framework with the existing QFD realm execution system.
"""

import os
import sys
import json
import yaml
import logging
import importlib
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..registry.parameter_registry import ParameterRegistry, Constraint, ConstraintType
from ..config.yaml_loader import load_parameters_from_yaml
from ..analysis.realm_tracker import RealmTracker, RealmExecutionResult, RealmStatus
from ..validation.base_validator import CompositeValidator
from ..validation.ppn_validator import PPNValidator
from ..validation.cmb_validator import CMBValidator
from ..validation.basic_validators import BoundsValidator, FixedValueValidator, TargetValueValidator
from ..plugins.plugin_manager import PluginManager
from ..visualization.export_manager import ExportManager
from ..visualization.coupling_visualizer import CouplingVisualizer
from ..analysis.dependency_mapper import DependencyMapper


@dataclass
class RealmExecutionHook:
    """Configuration for realm execution hooks."""
    realm_name: str
    pre_execution_hook: Optional[Callable] = None
    post_execution_hook: Optional[Callable] = None
    parameter_mapping: Optional[Dict[str, str]] = None
    constraint_updates: Optional[List[Constraint]] = None
    validation_required: bool = True


class RealmIntegrationManager:
    """
    Manager for integrating coupling constants analysis with realm execution.
    
    This class provides hooks into the realm execution workflow to:
    - Track parameter changes during realm execution
    - Validate constraints after each realm
    - Generate analysis reports
    - Coordinate between realms based on parameter dependencies
    """
    
    def __init__(self, config_path: str, output_dir: str = "coupling_analysis"):
        """
        Initialize the integration manager.
        
        Args:
            config_path: Path to QFD configuration file
            output_dir: Directory for analysis outputs
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.registry = ParameterRegistry()
        self.realm_tracker = RealmTracker(self.registry)
        self.plugin_manager = PluginManager()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_configuration()
        
        # Set up validators
        self.validator = CompositeValidator("Integrated QFD Validator")
        self._setup_validators()
        
        # Execution hooks
        self.realm_hooks: Dict[str, RealmExecutionHook] = {}
        self.execution_history: List[RealmExecutionResult] = []
        
        # Analysis components
        self.dependency_mapper = DependencyMapper(self.registry)
        self.export_manager = ExportManager(self.registry)
        self.visualizer = CouplingVisualizer(self.registry)
    
    def _load_configuration(self) -> None:
        """Load QFD configuration and populate parameter registry."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        load_parameters_from_yaml(self.config_path, self.registry)
        self.logger.info(f"Loaded {len(self.registry.get_all_parameters())} parameters from {self.config_path}")
    
    def _setup_validators(self) -> None:
        """Set up constraint validators."""
        self.validator.add_validator(PPNValidator())
        self.validator.add_validator(CMBValidator())
        self.validator.add_validator(BoundsValidator())
        self.validator.add_validator(FixedValueValidator())
        self.validator.add_validator(TargetValueValidator())
        
        self.logger.info(f"Set up {len(self.validator.validators)} validators")
    
    def register_realm_hook(self, hook: RealmExecutionHook) -> None:
        """Register a realm execution hook."""
        self.realm_hooks[hook.realm_name] = hook
        self.logger.info(f"Registered hook for realm: {hook.realm_name}")
    
    def register_plugin(self, plugin_name: str) -> bool:
        """Register a constraint plugin."""
        from ..plugins.constraint_plugins import (
            PhotonMassConstraintPlugin,
            VacuumStabilityPlugin,
            CosmologicalConstantPlugin
        )
        
        available_plugins = {
            'photon_mass': PhotonMassConstraintPlugin,
            'vacuum_stability': VacuumStabilityPlugin,
            'cosmological_constant': CosmologicalConstantPlugin
        }
        
        if plugin_name in available_plugins:
            plugin = available_plugins[plugin_name]()
            return self.plugin_manager.register_plugin(plugin)
        else:
            self.logger.warning(f"Unknown plugin: {plugin_name}")
            return False   
 
    def execute_realm_with_integration(self, realm_name: str, realm_module_path: str, 
                                     realm_params: Dict[str, Any] = None) -> RealmExecutionResult:
        """
        Execute a realm with full coupling constants integration.
        
        Args:
            realm_name: Name of the realm (e.g., 'realm0_cmb')
            realm_module_path: Python module path (e.g., 'realms.realm0_cmb')
            realm_params: Parameters to pass to the realm
            
        Returns:
            RealmExecutionResult with execution details
        """
        start_time = datetime.now()
        self.logger.info(f"Starting integrated execution of {realm_name}")
        
        try:
            # Pre-execution hook
            if realm_name in self.realm_hooks:
                hook = self.realm_hooks[realm_name]
                if hook.pre_execution_hook:
                    hook.pre_execution_hook(self.registry, realm_params or {})
            
            # Import and execute realm
            realm_module = importlib.import_module(realm_module_path)
            
            # Get current parameter state
            pre_execution_params = {
                name: param.value for name, param in self.registry.get_all_parameters().items()
                if param.value is not None
            }
            
            # Execute realm
            if hasattr(realm_module, 'run'):
                realm_result = realm_module.run(realm_params or {}, getattr(realm_module, 'cfg', None))
            else:
                raise AttributeError(f"Realm module {realm_module_path} has no 'run' function")
            
            # Update parameter registry with realm results
            parameters_modified = self._update_registry_from_realm_result(
                realm_name, realm_result, pre_execution_params
            )
            
            # Post-execution hook
            if realm_name in self.realm_hooks:
                hook = self.realm_hooks[realm_name]
                if hook.post_execution_hook:
                    hook.post_execution_hook(self.registry, realm_result)
                
                # Apply constraint updates
                if hook.constraint_updates:
                    for constraint in hook.constraint_updates:
                        param_name = constraint.realm  # Assuming constraint.realm contains parameter name
                        self.registry.add_constraint(param_name, constraint)
            
            # Validate constraints if required
            validation_result = None
            plugin_results = []
            if realm_name not in self.realm_hooks or self.realm_hooks[realm_name].validation_required:
                validation_result = self.validator.validate_all(self.registry)
                
                # Run plugin validation
                plugin_results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
                
                if validation_result.overall_status.value == 'invalid':
                    self.logger.warning(f"Validation failed after {realm_name}: {validation_result.total_violations} violations")
                else:
                    self.logger.info(f"Validation passed after {realm_name}")
            
            # Create execution result
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            result = RealmExecutionResult(
                realm_name=realm_name,
                status=RealmStatus.COMPLETED,
                execution_time_ms=execution_time,
                parameters_modified=parameters_modified,
                constraints_added=len(self.realm_hooks.get(realm_name, RealmExecutionHook("")).constraint_updates or []),
                validation_report=validation_result,
                metadata={
                    'realm_result': realm_result,
                    'pre_execution_params': pre_execution_params,
                    'plugin_validation_results': len(plugin_results)
                }
            )
            
            # Track execution
            self.execution_history.append(result)
            
            self.logger.info(f"Completed integrated execution of {realm_name} in {execution_time:.2f}ms")
            return result
            
        except Exception as e:
            # Create error result
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            result = RealmExecutionResult(
                realm_name=realm_name,
                status=RealmStatus.FAILED,
                execution_time_ms=execution_time,
                parameters_modified=[],
                constraints_added=0,
                validation_report=None,
                metadata={'error': str(e)}
            )
            
            self.execution_history.append(result)
            
            self.logger.error(f"Failed to execute {realm_name}: {e}")
            return result  
  
    def _update_registry_from_realm_result(self, realm_name: str, realm_result: Dict[str, Any], 
                                         pre_execution_params: Dict[str, Any]) -> List[str]:
        """
        Update parameter registry based on realm execution results.
        
        Args:
            realm_name: Name of the executed realm
            realm_result: Result dictionary from realm execution
            pre_execution_params: Parameter values before execution
            
        Returns:
            List of parameter names that were modified
        """
        parameters_modified = []
        
        # Extract parameter updates from realm result
        if 'fixed' in realm_result:
            for param_name, value in realm_result['fixed'].items():
                if isinstance(value, (int, float)):
                    self.registry.update_parameter(param_name, float(value), realm_name, "Fixed by realm")
                    parameters_modified.append(param_name)
                    
                    # Add fixed constraint
                    constraint = Constraint(
                        realm=realm_name,
                        constraint_type=ConstraintType.FIXED,
                        target_value=float(value),
                        tolerance=1e-10,
                        notes=f"Fixed by {realm_name}"
                    )
                    self.registry.add_constraint(param_name, constraint)
        
        if 'narrowed' in realm_result:
            for param_name, constraint_info in realm_result['narrowed'].items():
                if isinstance(constraint_info, dict):
                    # Extract bounds from constraint info
                    min_val = constraint_info.get('min')
                    max_val = constraint_info.get('max')
                    target_val = constraint_info.get('target')
                    
                    if min_val is not None or max_val is not None:
                        constraint = Constraint(
                            realm=realm_name,
                            constraint_type=ConstraintType.BOUNDED,
                            min_value=min_val,
                            max_value=max_val,
                            notes=f"Narrowed by {realm_name}"
                        )
                        self.registry.add_constraint(param_name, constraint)
                    
                    if target_val is not None:
                        constraint = Constraint(
                            realm=realm_name,
                            constraint_type=ConstraintType.TARGET,
                            target_value=target_val,
                            tolerance=constraint_info.get('tolerance', 0.01),
                            notes=f"Target set by {realm_name}"
                        )
                        self.registry.add_constraint(param_name, constraint)
        
        return parameters_modified
    
    def execute_realm_sequence(self, realm_sequence: List[Tuple[str, str]], 
                             realm_params: Dict[str, Dict[str, Any]] = None) -> List[RealmExecutionResult]:
        """
        Execute a sequence of realms with integrated analysis.
        
        Args:
            realm_sequence: List of (realm_name, module_path) tuples
            realm_params: Dictionary of parameters for each realm
            
        Returns:
            List of RealmExecutionResult objects
        """
        results = []
        realm_params = realm_params or {}
        
        self.logger.info(f"Starting realm sequence execution: {[r[0] for r in realm_sequence]}")
        
        for realm_name, module_path in realm_sequence:
            params = realm_params.get(realm_name, {})
            result = self.execute_realm_with_integration(realm_name, module_path, params)
            results.append(result)
            
            # Stop execution if a realm fails
            if result.status == RealmStatus.FAILED:
                self.logger.error(f"Realm sequence stopped due to failure in {realm_name}")
                break
        
        self.logger.info(f"Completed realm sequence execution")
        return results
    
    def generate_analysis_report(self, include_visualizations: bool = True) -> str:
        """
        Generate comprehensive analysis report after realm execution.
        
        Args:
            include_visualizations: Whether to generate visualization plots
            
        Returns:
            Path to the generated report directory
        """
        report_dir = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating analysis report in {report_dir}")
        
        # Build dependency graph
        self.dependency_mapper.build_dependency_graph()
        
        # Generate comprehensive report
        self.export_manager.create_comprehensive_report(
            str(report_dir),
            dependency_mapper=self.dependency_mapper,
            realm_tracker=self.realm_tracker
        )
        
        # Generate visualizations if requested
        if include_visualizations:
            viz_dir = report_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Dependency graph
            self.visualizer.plot_dependency_graph(
                self.dependency_mapper,
                str(viz_dir / "dependency_graph.png")
            )
            
            # Parameter constraints
            self.visualizer.plot_parameter_constraints(
                str(viz_dir / "parameter_constraints.png")
            )
            
            # Realm execution flow
            if self.execution_history:
                self.visualizer.plot_realm_execution_flow(
                    self.execution_history,
                    str(viz_dir / "realm_execution_flow.png")
                )
            
            # Create dashboard
            self.visualizer.create_dashboard(
                self.dependency_mapper,
                execution_history=self.execution_history,
                output_dir=str(viz_dir / "dashboard")
            )
        
        # Export execution log
        self.realm_tracker.export_execution_log(str(report_dir / "realm_execution_log.json"))
        
        self.logger.info(f"Analysis report generated: {report_dir}")
        return str(report_dir)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of realm execution and analysis."""
        return {
            'total_realms_executed': len(self.execution_history),
            'successful_executions': len([r for r in self.execution_history if r.status == RealmStatus.COMPLETED]),
            'failed_executions': len([r for r in self.execution_history if r.status == RealmStatus.FAILED]),
            'total_execution_time_ms': sum(r.execution_time_ms for r in self.execution_history),
            'total_parameters': len(self.registry.get_all_parameters()),
            'parameters_with_values': len([p for p in self.registry.get_all_parameters().values() if p.value is not None]),
            'total_constraints': sum(len(p.constraints) for p in self.registry.get_all_parameters().values()),
            'registered_plugins': len(self.plugin_manager.get_registered_plugins()),
            'dependency_graph_nodes': len(self.dependency_mapper.dependency_graph.nodes()) if hasattr(self.dependency_mapper, 'dependency_graph') else 0
        }