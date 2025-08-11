"""
Plugin validator that integrates the plugin system with the validation framework.

This module provides a validator that runs all registered plugins and integrates
their results into the main validation pipeline.
"""

from typing import List, Optional
import time

from .base_validator import BaseValidator, ValidationResult, ValidationStatus, ValidationViolation
from ..registry.parameter_registry import ParameterRegistry
from ..plugins.plugin_manager import PluginManager


class PluginValidator(BaseValidator):
    """
    Validator that runs all registered constraint plugins.
    
    This validator integrates the plugin system with the main validation framework,
    allowing plugin-defined constraints to be evaluated alongside built-in validators.
    """
    
    def __init__(self, plugin_manager: PluginManager, name: str = "Plugin Validator"):
        """
        Initialize the plugin validator.
        
        Args:
            plugin_manager: PluginManager instance with registered plugins
            name: Name for this validator
        """
        super().__init__(name)
        self.plugin_manager = plugin_manager
        self.conflict_resolution_strategy = "priority"  # Default strategy
        
    def set_conflict_resolution_strategy(self, strategy: str) -> None:
        """
        Set the conflict resolution strategy for plugin conflicts.
        
        Args:
            strategy: Resolution strategy ('priority', 'disable_lower', 'user_choice')
        """
        self.conflict_resolution_strategy = strategy
    
    def validate(self, registry: ParameterRegistry) -> ValidationResult:
        """
        Validate all plugin constraints and resolve conflicts.
        
        Args:
            registry: ParameterRegistry instance to validate
            
        Returns:
            ValidationResult with combined results from all plugins
        """
        start_time = time.time()
        
        # Create main validation result
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.VALID
        )
        
        # Get all active plugins
        active_plugins = self.plugin_manager.get_active_plugins()
        
        if not active_plugins:
            result.add_info("No active plugins to validate")
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        
        # Run all plugin validations
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(registry)
        
        # Combine results
        total_violations = 0
        total_warnings = 0
        
        for plugin_result in plugin_results:
            # Add violations from plugin result
            for violation in plugin_result.violations:
                result.add_violation(violation)
                total_violations += 1
            
            # Add warnings from plugin result
            for warning in plugin_result.warnings:
                result.add_warning(f"Plugin {plugin_result.validator_name}: {warning}")
                total_warnings += 1
            
            # Add info messages
            for info in plugin_result.info_messages:
                result.add_info(f"Plugin {plugin_result.validator_name}: {info}")
        
        # Check for plugin conflicts
        conflicts = self.plugin_manager.get_plugin_conflicts(registry)
        
        if conflicts:
            result.add_warning(f"Found {len(conflicts)} plugin conflicts")
            
            # Attempt to resolve conflicts
            try:
                resolution = self.plugin_manager.resolve_plugin_conflicts(
                    conflicts, self.conflict_resolution_strategy
                )
                
                # Report resolution actions
                if resolution.get("disabled_plugins"):
                    disabled = resolution["disabled_plugins"]
                    result.add_warning(f"Disabled conflicting plugins: {', '.join(disabled)}")
                
                if resolution.get("priority_overrides"):
                    for override in resolution["priority_overrides"]:
                        result.add_info(
                            f"Parameter {override['parameter']}: "
                            f"Plugin {override['winning_plugin']} takes priority"
                        )
                
                if resolution.get("warnings"):
                    for warning in resolution["warnings"]:
                        result.add_warning(warning)
                
            except Exception as e:
                result.add_violation(ValidationViolation(
                    parameter_name="plugin_conflicts",
                    constraint_realm="plugin_validator",
                    violation_type="conflict_resolution_failed",
                    message=f"Failed to resolve plugin conflicts: {str(e)}"
                ))
        
        # Update metadata
        result.metadata.update({
            "total_plugins": len(active_plugins),
            "plugin_violations": total_violations,
            "plugin_warnings": total_warnings,
            "plugin_conflicts": len(conflicts),
            "conflict_resolution_strategy": self.conflict_resolution_strategy
        })
        
        # Set parameters and constraints checked
        result.parameters_checked = len(registry.get_all_parameters())
        
        # Count constraints from all plugins
        total_constraints = 0
        for plugin in active_plugins.values():
            total_constraints += len(plugin.get_parameter_dependencies())
        result.constraints_checked = total_constraints
        
        # Set execution time
        result.execution_time_ms = (time.time() - start_time) * 1000
        
        # Add summary info
        if result.status == ValidationStatus.VALID:
            result.add_info(f"All {len(active_plugins)} plugins validated successfully")
        else:
            result.add_info(f"Plugin validation completed with {len(result.violations)} violations")
        
        return result
    
    def is_applicable(self, registry: ParameterRegistry) -> bool:
        """
        Check if plugin validation is applicable.
        
        Args:
            registry: ParameterRegistry instance
            
        Returns:
            True if there are active plugins to validate
        """
        return self.enabled and len(self.plugin_manager.get_active_plugins()) > 0
    
    def get_required_parameters(self) -> List[str]:
        """
        Get list of parameters required by all active plugins.
        
        Returns:
            Combined list of parameter names from all active plugins
        """
        required_params = set()
        
        for plugin in self.plugin_manager.get_active_plugins().values():
            required_params.update(plugin.get_parameter_dependencies())
        
        return list(required_params)
    
    def get_description(self) -> str:
        """
        Get description of this validator.
        
        Returns:
            Human-readable description
        """
        active_count = len(self.plugin_manager.get_active_plugins())
        total_count = len(self.plugin_manager.plugins)
        
        return (f"Plugin Validator: Runs {active_count}/{total_count} registered "
                f"constraint plugins with {self.conflict_resolution_strategy} conflict resolution")
    
    def get_plugin_summary(self) -> dict:
        """
        Get summary of registered plugins.
        
        Returns:
            Dictionary with plugin summary information
        """
        active_plugins = self.plugin_manager.get_active_plugins()
        all_plugins = self.plugin_manager.get_registered_plugins()
        
        summary = {
            "total_plugins": len(all_plugins),
            "active_plugins": len(active_plugins),
            "inactive_plugins": len(all_plugins) - len(active_plugins),
            "plugins_by_priority": {},
            "plugin_details": {}
        }
        
        # Group by priority
        for name, info in all_plugins.items():
            priority = info.priority.name
            if priority not in summary["plugins_by_priority"]:
                summary["plugins_by_priority"][priority] = []
            summary["plugins_by_priority"][priority].append(name)
        
        # Add plugin details
        for name, plugin in self.plugin_manager.plugins.items():
            info = plugin.plugin_info
            summary["plugin_details"][name] = {
                "version": info.version,
                "author": info.author,
                "description": info.description,
                "priority": info.priority.name,
                "active": info.active,
                "dependencies": info.dependencies,
                "parameter_dependencies": plugin.get_parameter_dependencies(),
                "constraint_description": plugin.get_constraint_description()
            }
        
        return summary


class PluginValidatorFactory:
    """
    Factory for creating plugin validators with different configurations.
    """
    
    @staticmethod
    def create_default_plugin_validator(plugin_manager: PluginManager) -> PluginValidator:
        """
        Create a plugin validator with default settings.
        
        Args:
            plugin_manager: PluginManager instance
            
        Returns:
            PluginValidator with default configuration
        """
        validator = PluginValidator(plugin_manager, "Default Plugin Validator")
        validator.set_conflict_resolution_strategy("priority")
        return validator
    
    @staticmethod
    def create_strict_plugin_validator(plugin_manager: PluginManager) -> PluginValidator:
        """
        Create a plugin validator with strict conflict resolution.
        
        Args:
            plugin_manager: PluginManager instance
            
        Returns:
            PluginValidator configured for strict validation
        """
        validator = PluginValidator(plugin_manager, "Strict Plugin Validator")
        validator.set_conflict_resolution_strategy("disable_lower")
        return validator
    
    @staticmethod
    def create_permissive_plugin_validator(plugin_manager: PluginManager) -> PluginValidator:
        """
        Create a plugin validator with permissive conflict resolution.
        
        Args:
            plugin_manager: PluginManager instance
            
        Returns:
            PluginValidator configured for permissive validation
        """
        validator = PluginValidator(plugin_manager, "Permissive Plugin Validator")
        validator.set_conflict_resolution_strategy("user_choice")
        return validator