"""
Realm execution tracker for coupling constants.

This module provides tools for tracking realm execution state,
monitoring parameter convergence, and managing the realm sequence.
"""

import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from ..registry.parameter_registry import ParameterRegistry
from ..validation.base_validator import CompositeValidator, ValidationReport


class RealmStatus(Enum):
    """Status of realm execution."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RealmExecutionResult:
    """Result from executing a single realm."""
    realm_name: str
    status: RealmStatus
    execution_time_ms: float
    parameters_modified: List[str]
    constraints_added: int
    validation_report: Optional[ValidationReport] = None
    notes: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvergenceMetrics:
    """Metrics for parameter convergence analysis."""
    parameter_name: str
    current_value: float
    previous_value: Optional[float]
    change_magnitude: float
    relative_change: float
    convergence_threshold: float
    is_converged: bool
    oscillation_detected: bool
    divergence_detected: bool
    history_length: int


@dataclass
class RealmSequenceResult:
    """Result from executing a complete realm sequence."""
    sequence_name: str
    total_execution_time_ms: float
    realms_executed: List[RealmExecutionResult]
    final_validation_report: ValidationReport
    convergence_achieved: bool
    convergence_metrics: List[ConvergenceMetrics]
    iterations_completed: int
    max_iterations: int
    early_termination_reason: Optional[str] = None


# Type alias for realm execution functions
# Takes ParameterRegistry, returns Dict with status, fixed params, notes
RealmFunction = Callable[[ParameterRegistry], Dict[str, Any]]


class RealmTracker:
    """
    Tracks realm execution state and manages the realm sequence.
    
    Provides monitoring of parameter convergence, validation after
    each realm, and coordination of the complete realm execution sequence.
    """
    
    def __init__(self, registry: ParameterRegistry, validator: Optional[CompositeValidator] = None):
        """
        Initialize realm tracker.
        
        Args:
            registry: ParameterRegistry instance to track
            validator: Optional validator for post-realm validation
        """
        self.registry = registry
        self.validator = validator
        self.realm_functions: Dict[str, RealmFunction] = {}
        self.realm_dependencies: Dict[str, List[str]] = {}
        self.execution_history: List[RealmExecutionResult] = []
        self.convergence_thresholds: Dict[str, float] = {}
        self.default_convergence_threshold = 1e-6
        
        # Execution state
        self.current_realm: Optional[str] = None
        self.sequence_running = False
        
        # Default realm execution order for QFD
        self.default_realm_order = [
            "realm0_cmb",
            "realm1_cosmic_baseline", 
            "realm2_dark_energy",
            "realm3_scales",
            "realm4_em_charge",
            "realm5_electron",
            "realm6_leptons_isomer",
            "realm7_proton",
            "realm8_neutron_beta",
            "realm9_deuteron",
            "realm10_isotopes"
        ]
    
    def register_realm(self, realm_name: str, realm_func: RealmFunction, 
                      dependencies: Optional[List[str]] = None) -> None:
        """
        Register a realm function for execution.
        
        Args:
            realm_name: Name of the realm
            realm_func: Function that executes the realm
            dependencies: List of realm names this realm depends on
        """
        self.realm_functions[realm_name] = realm_func
        self.realm_dependencies[realm_name] = dependencies or []
    
    def set_convergence_threshold(self, parameter_name: str, threshold: float) -> None:
        """Set convergence threshold for a specific parameter."""
        self.convergence_thresholds[parameter_name] = threshold
    
    def execute_realm(self, realm_name: str, validate_after: bool = True) -> RealmExecutionResult:
        """
        Execute a single realm and track the results.
        
        Args:
            realm_name: Name of the realm to execute
            validate_after: Whether to run validation after realm execution
            
        Returns:
            RealmExecutionResult with execution details
        """
        if realm_name not in self.realm_functions:
            return RealmExecutionResult(
                realm_name=realm_name,
                status=RealmStatus.FAILED,
                execution_time_ms=0.0,
                parameters_modified=[],
                constraints_added=0,
                error_message=f"Realm '{realm_name}' not registered"
            )
        
        # Check dependencies
        missing_deps = self._check_dependencies(realm_name)
        if missing_deps:
            return RealmExecutionResult(
                realm_name=realm_name,
                status=RealmStatus.FAILED,
                execution_time_ms=0.0,
                parameters_modified=[],
                constraints_added=0,
                error_message=f"Missing dependencies: {missing_deps}"
            )
        
        # Record state before execution
        params_before = {name: param.value for name, param in self.registry.get_all_parameters().items()}
        constraints_before = sum(len(param.constraints) for param in self.registry.get_all_parameters().values())
        
        # Execute realm
        start_time = time.time()
        self.current_realm = realm_name
        
        try:
            realm_func = self.realm_functions[realm_name]
            result_dict = realm_func(self.registry)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Analyze changes
            params_after = {name: param.value for name, param in self.registry.get_all_parameters().items()}
            constraints_after = sum(len(param.constraints) for param in self.registry.get_all_parameters().values())
            
            modified_params = []
            for param_name in params_before:
                if params_before[param_name] != params_after.get(param_name):
                    modified_params.append(param_name)
            
            constraints_added = constraints_after - constraints_before
            
            # Create result
            result = RealmExecutionResult(
                realm_name=realm_name,
                status=RealmStatus.COMPLETED,
                execution_time_ms=execution_time,
                parameters_modified=modified_params,
                constraints_added=constraints_added,
                notes=result_dict.get('notes', []),
                metadata={
                    'realm_result': result_dict,
                    'parameters_before': len([p for p in params_before.values() if p is not None]),
                    'parameters_after': len([p for p in params_after.values() if p is not None])
                }
            )
            
            # Run validation if requested
            if validate_after and self.validator:
                validation_report = self.validator.validate_all(self.registry)
                result.validation_report = validation_report
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            result = RealmExecutionResult(
                realm_name=realm_name,
                status=RealmStatus.FAILED,
                execution_time_ms=execution_time,
                parameters_modified=[],
                constraints_added=0,
                error_message=str(e)
            )
            
            self.execution_history.append(result)
            return result
            
        finally:
            self.current_realm = None
    
    def execute_realm_sequence(self, realm_order: Optional[List[str]] = None,
                             max_iterations: int = 10,
                             convergence_check: bool = True,
                             validate_each_realm: bool = True) -> RealmSequenceResult:
        """
        Execute a complete sequence of realms with convergence checking.
        
        Args:
            realm_order: Order of realms to execute (uses default if None)
            max_iterations: Maximum number of iterations for convergence
            convergence_check: Whether to check for parameter convergence
            validate_each_realm: Whether to validate after each realm
            
        Returns:
            RealmSequenceResult with complete execution details
        """
        if self.sequence_running:
            raise RuntimeError("Realm sequence already running")
        
        self.sequence_running = True
        start_time = time.time()
        
        try:
            realm_order = realm_order or self.default_realm_order
            sequence_results = []
            iteration = 0
            converged = False
            early_termination_reason = None
            
            # Filter realm order to only include registered realms
            available_realms = [realm for realm in realm_order if realm in self.realm_functions]
            
            if not available_realms:
                raise ValueError("No registered realms found in execution order")
            
            while iteration < max_iterations and not converged:
                iteration += 1
                iteration_start_time = time.time()
                
                print(f"Starting realm sequence iteration {iteration}/{max_iterations}")
                
                # Store parameter values before iteration for convergence check
                params_before_iteration = self._get_parameter_snapshot()
                
                # Execute each realm in order
                iteration_results = []
                for realm_name in available_realms:
                    print(f"  Executing {realm_name}...")
                    
                    result = self.execute_realm(realm_name, validate_after=validate_each_realm)
                    iteration_results.append(result)
                    
                    if result.status == RealmStatus.FAILED:
                        print(f"  ✗ {realm_name} failed: {result.error_message}")
                        early_termination_reason = f"Realm {realm_name} failed: {result.error_message}"
                        break
                    else:
                        print(f"  ✓ {realm_name} completed ({result.execution_time_ms:.2f}ms)")
                        if result.parameters_modified:
                            print(f"    Modified parameters: {', '.join(result.parameters_modified[:5])}")
                            if len(result.parameters_modified) > 5:
                                print(f"    ... and {len(result.parameters_modified) - 5} more")
                
                sequence_results.extend(iteration_results)
                
                # Check for early termination
                if early_termination_reason:
                    break
                
                # Check convergence if enabled
                if convergence_check:
                    params_after_iteration = self._get_parameter_snapshot()
                    convergence_metrics = self._analyze_convergence(params_before_iteration, params_after_iteration)
                    
                    # Check if all parameters have converged
                    converged_params = [m for m in convergence_metrics if m.is_converged]
                    total_params = len([m for m in convergence_metrics if m.current_value is not None])
                    
                    if total_params > 0:
                        convergence_ratio = len(converged_params) / total_params
                        print(f"  Convergence: {len(converged_params)}/{total_params} parameters ({convergence_ratio:.1%})")
                        
                        # Consider converged if 95% of parameters have converged
                        if convergence_ratio >= 0.95:
                            converged = True
                            print("  ✓ Sequence converged!")
                        
                        # Check for oscillation or divergence
                        oscillating = [m for m in convergence_metrics if m.oscillation_detected]
                        diverging = [m for m in convergence_metrics if m.divergence_detected]
                        
                        if oscillating:
                            print(f"  ⚠ Oscillation detected in {len(oscillating)} parameters")
                        if diverging:
                            print(f"  ⚠ Divergence detected in {len(diverging)} parameters")
                            early_termination_reason = f"Divergence detected in parameters: {[m.parameter_name for m in diverging[:3]]}"
                            break
                
                iteration_time = (time.time() - iteration_start_time) * 1000
                print(f"  Iteration {iteration} completed in {iteration_time:.2f}ms")
            
            # Final validation
            print("Running final validation...")
            final_validation = self.validator.validate_all(self.registry) if self.validator else None
            
            # Get final convergence metrics
            if convergence_check and iteration > 1:
                final_convergence_metrics = convergence_metrics
            else:
                final_convergence_metrics = []
            
            total_time = (time.time() - start_time) * 1000
            
            result = RealmSequenceResult(
                sequence_name=f"QFD_Realm_Sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                total_execution_time_ms=total_time,
                realms_executed=sequence_results,
                final_validation_report=final_validation,
                convergence_achieved=converged,
                convergence_metrics=final_convergence_metrics,
                iterations_completed=iteration,
                max_iterations=max_iterations,
                early_termination_reason=early_termination_reason
            )
            
            print(f"\nRealm sequence completed:")
            print(f"  Total time: {total_time:.2f}ms")
            print(f"  Iterations: {iteration}/{max_iterations}")
            print(f"  Realms executed: {len(sequence_results)}")
            print(f"  Converged: {converged}")
            if final_validation:
                print(f"  Final validation: {final_validation.overall_status.value}")
            
            return result
            
        finally:
            self.sequence_running = False
    
    def _check_dependencies(self, realm_name: str) -> List[str]:
        """Check if realm dependencies have been satisfied."""
        dependencies = self.realm_dependencies.get(realm_name, [])
        missing = []
        
        executed_realms = {result.realm_name for result in self.execution_history 
                          if result.status == RealmStatus.COMPLETED}
        
        for dep in dependencies:
            if dep not in executed_realms:
                missing.append(dep)
        
        return missing
    
    def _get_parameter_snapshot(self) -> Dict[str, Optional[float]]:
        """Get a snapshot of current parameter values."""
        return {name: param.value for name, param in self.registry.get_all_parameters().items()}
    
    def _analyze_convergence(self, params_before: Dict[str, Optional[float]], 
                           params_after: Dict[str, Optional[float]]) -> List[ConvergenceMetrics]:
        """Analyze parameter convergence between iterations."""
        convergence_metrics = []
        
        for param_name in params_after:
            current_value = params_after[param_name]
            previous_value = params_before.get(param_name)
            
            if current_value is None:
                continue
            
            # Get convergence threshold
            threshold = self.convergence_thresholds.get(param_name, self.default_convergence_threshold)
            
            # Calculate changes
            if previous_value is not None:
                change_magnitude = abs(current_value - previous_value)
                relative_change = change_magnitude / abs(previous_value) if previous_value != 0 else change_magnitude
            else:
                change_magnitude = float('inf')
                relative_change = float('inf')
            
            # Check convergence
            is_converged = change_magnitude < threshold
            
            # Get parameter history for oscillation/divergence detection
            param_obj = self.registry.get_parameter(param_name)
            history_length = len(param_obj.history) if param_obj else 0
            
            # Simple oscillation detection (would be more sophisticated in practice)
            oscillation_detected = False
            divergence_detected = False
            
            if param_obj and len(param_obj.history) >= 4:
                recent_values = [change.new_value for change in param_obj.history[-4:]]
                
                # Check for oscillation (alternating increases/decreases)
                if len(recent_values) >= 4:
                    diffs = [recent_values[i+1] - recent_values[i] for i in range(len(recent_values)-1)]
                    sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
                    oscillation_detected = sign_changes >= 2
                
                # Check for divergence (consistently increasing magnitude)
                if len(recent_values) >= 3:
                    magnitudes = [abs(v - recent_values[0]) for v in recent_values[1:]]
                    divergence_detected = all(magnitudes[i] > magnitudes[i-1] for i in range(1, len(magnitudes)))
            
            metrics = ConvergenceMetrics(
                parameter_name=param_name,
                current_value=current_value,
                previous_value=previous_value,
                change_magnitude=change_magnitude,
                relative_change=relative_change,
                convergence_threshold=threshold,
                is_converged=is_converged,
                oscillation_detected=oscillation_detected,
                divergence_detected=divergence_detected,
                history_length=history_length
            )
            
            convergence_metrics.append(metrics)
        
        return convergence_metrics
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of realm execution history."""
        if not self.execution_history:
            return {"message": "No realms executed yet"}
        
        total_time = sum(result.execution_time_ms for result in self.execution_history)
        successful_executions = [r for r in self.execution_history if r.status == RealmStatus.COMPLETED]
        failed_executions = [r for r in self.execution_history if r.status == RealmStatus.FAILED]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "failed_executions": len(failed_executions),
            "total_execution_time_ms": total_time,
            "average_execution_time_ms": total_time / len(self.execution_history),
            "realms_executed": [r.realm_name for r in successful_executions],
            "failed_realms": [r.realm_name for r in failed_executions],
            "total_parameters_modified": len(set().union(*[r.parameters_modified for r in successful_executions])),
            "total_constraints_added": sum(r.constraints_added for r in successful_executions)
        }
    
    def export_execution_log(self, filename: str) -> None:
        """Export execution history to JSON file."""
        export_data = {
            "execution_summary": self.get_execution_summary(),
            "execution_history": [
                {
                    "realm_name": result.realm_name,
                    "status": result.status.value,
                    "execution_time_ms": result.execution_time_ms,
                    "parameters_modified": result.parameters_modified,
                    "constraints_added": result.constraints_added,
                    "notes": result.notes,
                    "error_message": result.error_message,
                    "validation_status": result.validation_report.overall_status.value if result.validation_report else None,
                    "validation_violations": len(result.validation_report.get_all_violations()) if result.validation_report else 0,
                    "metadata": result.metadata
                }
                for result in self.execution_history
            ],
            "realm_dependencies": self.realm_dependencies,
            "convergence_thresholds": self.convergence_thresholds,
            "registered_realms": list(self.realm_functions.keys())
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def reset_execution_history(self) -> None:
        """Reset execution history (useful for testing)."""
        self.execution_history.clear()
        self.current_realm = None
        self.sequence_running = False