"""
Plugin manager for extensible constraint implementations.

This module provides a plugin system that allows users to register custom
constraint functions and validation logic without modifying the core framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass
from enum import Enum
import importlib
import inspect
import logging

from ..registry.parameter_registry import ParameterRegistry
from ..validation.base_validator import ValidationResult, ValidationStatus, ValidationViolation


class PluginPriority(Enum):
    """Priority levels for plugin constraints."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PluginInfo:
    """Information about a registered plugin."""
    name: str
    version: str
    author: str
    description: str
    priority: PluginPriority
    dependencies: List[str]
    active: bool = True


class ConstraintPlugin(ABC):
    """
    Abstract base class for constraint plugins.
    
    All constraint plugins must inherit from this class and implement
    the required methods for constraint validation.
    """
    
    @property
    @abstractmethod
    def plugin_info(self) -> PluginInfo:
        """Return plugin information."""
        pass
    
    @abstractmethod
    def get_parameter_dependencies(self) -> List[str]:
        """
        Return list of parameter names this constraint depends on.
        
        Returns:
            List of parameter names required for this constraint
        """
        pass
    
    @abstractmethod
    def validate_constraint(self, registry: ParameterRegistry) -> ValidationResult:
        """
        Validate the constraint using current parameter values.
        
        Args:
            registry: ParameterRegistry with current parameter values
            
        Returns:
            ValidationResult indicating whether constraint is satisfied
        """
        pass
    
    @abstractmethod
    def get_constraint_description(self) -> str:
        """
        Return human-readable description of the constraint.
        
        Returns:
            String description of what this constraint enforces
        """
        pass
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the plugin with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass


class PluginManager:
    """
    Manager for constraint plugins.
    
    Handles plugin registration, loading, validation, and execution.
    Provides conflict resolution for overlapping plugin constraints.
    """
    
    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, ConstraintPlugin] = {}
        self.plugin_priorities: Dict[str, PluginPriority] = {}
        self.plugin_dependencies: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_plugin(self, plugin: ConstraintPlugin, 
                       config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a constraint plugin.
        
        Args:
            plugin: ConstraintPlugin instance to register
            config: Optional configuration for the plugin
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            info = plugin.plugin_info
            
            # Check if plugin already registered
            if info.name in self.plugins:
                self.logger.warning(f"Plugin {info.name} already registered, skipping")
                return False
            
            # Validate plugin dependencies
            if not self._validate_plugin_dependencies(plugin):
                self.logger.error(f"Plugin {info.name} has unmet dependencies")
                return False
            
            # Initialize plugin
            plugin.initialize(config)
            
            # Register plugin
            self.plugins[info.name] = plugin
            self.plugin_priorities[info.name] = info.priority
            self.plugin_dependencies[info.name] = plugin.get_parameter_dependencies()
            
            self.logger.info(f"Successfully registered plugin: {info.name} v{info.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register plugin: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a constraint plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        if plugin_name not in self.plugins:
            self.logger.warning(f"Plugin {plugin_name} not found")
            return False
        
        try:
            # Cleanup plugin
            self.plugins[plugin_name].cleanup()
            
            # Remove from registries
            del self.plugins[plugin_name]
            del self.plugin_priorities[plugin_name]
            del self.plugin_dependencies[plugin_name]
            
            self.logger.info(f"Successfully unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def load_plugin_from_module(self, module_path: str, 
                               plugin_class_name: str,
                               config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load and register a plugin from a Python module.
        
        Args:
            module_path: Python module path (e.g., 'my_plugins.custom_constraint')
            plugin_class_name: Name of the plugin class in the module
            config: Optional configuration for the plugin
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            # Import module
            module = importlib.import_module(module_path)
            
            # Get plugin class
            if not hasattr(module, plugin_class_name):
                self.logger.error(f"Class {plugin_class_name} not found in {module_path}")
                return False
            
            plugin_class = getattr(module, plugin_class_name)
            
            # Validate that it's a ConstraintPlugin
            if not issubclass(plugin_class, ConstraintPlugin):
                self.logger.error(f"{plugin_class_name} is not a ConstraintPlugin")
                return False
            
            # Create instance and register
            plugin_instance = plugin_class()
            return self.register_plugin(plugin_instance, config)
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {module_path}: {e}")
            return False
    
    def get_registered_plugins(self) -> Dict[str, PluginInfo]:
        """
        Get information about all registered plugins.
        
        Returns:
            Dictionary mapping plugin names to PluginInfo objects
        """
        return {name: plugin.plugin_info for name, plugin in self.plugins.items()}
    
    def get_active_plugins(self) -> Dict[str, ConstraintPlugin]:
        """
        Get all active plugins.
        
        Returns:
            Dictionary of active plugins
        """
        return {name: plugin for name, plugin in self.plugins.items() 
                if plugin.plugin_info.active}
    
    def validate_all_plugin_constraints(self, registry: ParameterRegistry) -> List[ValidationResult]:
        """
        Validate all registered plugin constraints.
        
        Args:
            registry: ParameterRegistry with current parameter values
            
        Returns:
            List of ValidationResult objects from all plugins
        """
        results = []
        
        # Sort plugins by priority (highest first)
        sorted_plugins = sorted(
            self.get_active_plugins().items(),
            key=lambda x: x[1].plugin_info.priority.value,
            reverse=True
        )
        
        for plugin_name, plugin in sorted_plugins:
            try:
                result = plugin.validate_constraint(registry)
                result.validator_name = f"plugin_{plugin_name}"
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Plugin {plugin_name} validation failed: {e}")
                # Create error result
                error_result = ValidationResult(
                    validator_name=f"plugin_{plugin_name}",
                    status=ValidationStatus.INVALID
                )
                error_result.add_violation(ValidationViolation(
                    parameter_name="N/A",
                    constraint_realm=f"plugin_{plugin_name}",
                    violation_type="plugin_error",
                    message=f"Plugin validation error: {e}"
                ))
                error_result.metadata = {"error": str(e)}
                results.append(error_result)
        
        return results
    
    def get_plugin_conflicts(self, registry: ParameterRegistry) -> List[Dict[str, Any]]:
        """
        Identify conflicts between plugin constraints.
        
        Args:
            registry: ParameterRegistry with current parameter values
            
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        validation_results = self.validate_all_plugin_constraints(registry)
        
        # Group results by parameter dependencies
        param_to_plugins = {}
        for plugin_name, plugin in self.get_active_plugins().items():
            for param in plugin.get_parameter_dependencies():
                if param not in param_to_plugins:
                    param_to_plugins[param] = []
                param_to_plugins[param].append(plugin_name)
        
        # Check for conflicts on shared parameters
        for param, plugin_names in param_to_plugins.items():
            if len(plugin_names) > 1:
                # Multiple plugins affect this parameter
                plugin_results = {name: None for name in plugin_names}
                
                for result in validation_results:
                    plugin_name = result.validator_name.replace("plugin_", "")
                    if plugin_name in plugin_results:
                        plugin_results[plugin_name] = result
                
                # Check if any plugins have conflicting requirements
                valid_plugins = [name for name, result in plugin_results.items() 
                               if result and result.is_valid()]
                invalid_plugins = [name for name, result in plugin_results.items() 
                                 if result and not result.is_valid()]
                
                if valid_plugins and invalid_plugins:
                    conflicts.append({
                        "parameter": param,
                        "conflict_type": "validation_conflict",
                        "valid_plugins": valid_plugins,
                        "invalid_plugins": invalid_plugins,
                        "priority_resolution": self._resolve_priority_conflict(
                            valid_plugins + invalid_plugins
                        )
                    })
        
        return conflicts
    
    def resolve_plugin_conflicts(self, conflicts: List[Dict[str, Any]], 
                                strategy: str = "priority") -> Dict[str, Any]:
        """
        Resolve conflicts between plugin constraints.
        
        Args:
            conflicts: List of conflicts from get_plugin_conflicts()
            strategy: Resolution strategy ('priority', 'disable_lower', 'user_choice')
            
        Returns:
            Dictionary with resolution actions taken
        """
        resolution_actions = {
            "disabled_plugins": [],
            "priority_overrides": [],
            "warnings": []
        }
        
        for conflict in conflicts:
            if strategy == "priority":
                # Use priority-based resolution
                priority_resolution = conflict.get("priority_resolution", {})
                highest_priority_plugin = priority_resolution.get("highest_priority")
                
                if highest_priority_plugin:
                    # Keep highest priority plugin, disable others
                    other_plugins = [p for p in conflict["valid_plugins"] + conflict["invalid_plugins"]
                                   if p != highest_priority_plugin]
                    
                    for plugin_name in other_plugins:
                        if plugin_name in self.plugins:
                            self.plugins[plugin_name].plugin_info.active = False
                            resolution_actions["disabled_plugins"].append(plugin_name)
                    
                    resolution_actions["priority_overrides"].append({
                        "parameter": conflict["parameter"],
                        "winning_plugin": highest_priority_plugin,
                        "disabled_plugins": other_plugins
                    })
            
            elif strategy == "disable_lower":
                # Disable all lower priority plugins in conflict
                invalid_plugins = conflict.get("invalid_plugins", [])
                for plugin_name in invalid_plugins:
                    if plugin_name in self.plugins:
                        self.plugins[plugin_name].plugin_info.active = False
                        resolution_actions["disabled_plugins"].append(plugin_name)
            
            elif strategy == "user_choice":
                # Add warning for user to resolve manually
                resolution_actions["warnings"].append(
                    f"Manual resolution required for parameter {conflict['parameter']} "
                    f"conflict between plugins: {conflict['valid_plugins'] + conflict['invalid_plugins']}"
                )
        
        return resolution_actions
    
    def export_plugin_info(self, output_path: str) -> None:
        """
        Export plugin information to JSON file.
        
        Args:
            output_path: Path to save plugin information
        """
        import json
        
        plugin_data = {}
        for name, plugin in self.plugins.items():
            info = plugin.plugin_info
            plugin_data[name] = {
                "name": info.name,
                "version": info.version,
                "author": info.author,
                "description": info.description,
                "priority": info.priority.name,
                "dependencies": info.dependencies,
                "active": info.active,
                "parameter_dependencies": plugin.get_parameter_dependencies(),
                "constraint_description": plugin.get_constraint_description()
            }
        
        with open(output_path, 'w') as f:
            json.dump(plugin_data, f, indent=2)
        
        self.logger.info(f"Plugin information exported to {output_path}")
    
    def _validate_plugin_dependencies(self, plugin: ConstraintPlugin) -> bool:
        """
        Validate that plugin dependencies are met.
        
        Args:
            plugin: Plugin to validate
            
        Returns:
            True if dependencies are met, False otherwise
        """
        info = plugin.plugin_info
        
        # Check if required dependencies are available
        for dep in info.dependencies:
            if dep not in self.plugins:
                self.logger.error(f"Plugin {info.name} requires {dep} which is not registered")
                return False
        
        return True
    
    def _resolve_priority_conflict(self, plugin_names: List[str]) -> Dict[str, Any]:
        """
        Resolve conflict based on plugin priorities.
        
        Args:
            plugin_names: List of conflicting plugin names
            
        Returns:
            Dictionary with priority resolution information
        """
        plugin_priorities = {}
        for name in plugin_names:
            if name in self.plugin_priorities:
                plugin_priorities[name] = self.plugin_priorities[name].value
        
        if not plugin_priorities:
            return {}
        
        highest_priority = max(plugin_priorities.values())
        highest_priority_plugins = [name for name, priority in plugin_priorities.items() 
                                  if priority == highest_priority]
        
        return {
            "highest_priority": highest_priority_plugins[0] if len(highest_priority_plugins) == 1 else None,
            "highest_priority_plugins": highest_priority_plugins,
            "priority_values": plugin_priorities,
            "tie": len(highest_priority_plugins) > 1
        }