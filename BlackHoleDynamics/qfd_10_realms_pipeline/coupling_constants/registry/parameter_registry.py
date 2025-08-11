"""
Parameter Registry for tracking coupling constants across QFD realms.

This module provides the core data structures and registry for managing
parameter states, constraints, and changes throughout the realm execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class ConstraintType(Enum):
    """Types of constraints that can be applied to parameters."""
    FIXED = "fixed"
    BOUNDED = "bounded"
    DERIVED = "derived"
    TARGET = "target"


@dataclass
class Constraint:
    """Represents a constraint on a parameter from a specific realm."""
    realm: str
    constraint_type: ConstraintType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: Optional[float] = None
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True

    def is_satisfied(self, value: float) -> bool:
        """Check if a parameter value satisfies this constraint."""
        if not self.active:
            return True
            
        if self.constraint_type == ConstraintType.FIXED:
            return self.target_value is not None and abs(value - self.target_value) < (self.tolerance or 1e-12)
        elif self.constraint_type == ConstraintType.BOUNDED:
            min_ok = self.min_value is None or value >= self.min_value
            max_ok = self.max_value is None or value <= self.max_value
            return min_ok and max_ok
        elif self.constraint_type == ConstraintType.TARGET:
            return self.target_value is not None and abs(value - self.target_value) < (self.tolerance or 1e-6)
        elif self.constraint_type == ConstraintType.DERIVED:
            # Derived constraints are checked by specific validation logic
            return True
        
        return True


@dataclass
class ParameterChange:
    """Records a change to a parameter value."""
    timestamp: datetime
    realm: str
    old_value: Optional[float]
    new_value: float
    reason: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ParameterState:
    """Complete state of a parameter including value, constraints, and history."""
    name: str
    value: Optional[float] = None
    uncertainty: Optional[float] = None
    constraints: List[Constraint] = field(default_factory=list)
    fixed_by_realm: Optional[str] = None
    last_modified: datetime = field(default_factory=datetime.utcnow)
    history: List[ParameterChange] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a new constraint to this parameter."""
        self.constraints.append(constraint)

    def update_value(self, new_value: float, realm: str, reason: str = "") -> None:
        """Update the parameter value and record the change."""
        change = ParameterChange(
            timestamp=datetime.utcnow(),
            realm=realm,
            old_value=self.value,
            new_value=new_value,
            reason=reason
        )
        self.history.append(change)
        self.value = new_value
        self.last_modified = change.timestamp

    def is_fixed(self) -> bool:
        """Check if this parameter is fixed by any realm."""
        return self.fixed_by_realm is not None

    def get_active_constraints(self) -> List[Constraint]:
        """Get all active constraints for this parameter."""
        return [c for c in self.constraints if c.active]

    def validate_constraints(self) -> List[str]:
        """Validate current value against all active constraints."""
        violations = []
        if self.value is None:
            return ["Parameter value not set"]

        for constraint in self.get_active_constraints():
            if not constraint.is_satisfied(self.value):
                violations.append(
                    f"Constraint violation from {constraint.realm}: "
                    f"{constraint.constraint_type.value} constraint not satisfied"
                )
        
        return violations


class ParameterRegistry:
    """Central registry for all coupling constants and their states."""
    
    def __init__(self):
        self._parameters: Dict[str, ParameterState] = {}
        self._realm_order: List[str] = []
        
    def register_parameter(self, param: ParameterState) -> None:
        """Register a new parameter in the registry."""
        self._parameters[param.name] = param
        
    def get_parameter(self, name: str) -> Optional[ParameterState]:
        """Get a parameter by name."""
        return self._parameters.get(name)
        
    def get_all_parameters(self) -> Dict[str, ParameterState]:
        """Get all registered parameters."""
        return self._parameters.copy()
        
    def update_parameter(self, name: str, value: float, realm: str, reason: str = "") -> None:
        """Update a parameter value."""
        if name not in self._parameters:
            # Auto-register parameter if it doesn't exist
            self._parameters[name] = ParameterState(name=name)
            
        param = self._parameters[name]
        
        # Check if parameter is fixed by another realm
        if param.is_fixed() and param.fixed_by_realm != realm:
            raise ValueError(
                f"Parameter {name} is fixed by realm {param.fixed_by_realm}, "
                f"cannot be modified by realm {realm}"
            )
            
        param.update_value(value, realm, reason)
        
    def add_constraint(self, param_name: str, constraint: Constraint) -> None:
        """Add a constraint to a parameter."""
        if param_name not in self._parameters:
            self._parameters[param_name] = ParameterState(name=param_name)
            
        self._parameters[param_name].add_constraint(constraint)
        
        # If this is a FIXED constraint, mark the parameter as fixed
        if constraint.constraint_type == ConstraintType.FIXED:
            self._parameters[param_name].fixed_by_realm = constraint.realm
            
    def get_conflicting_constraints(self) -> List[Dict[str, Any]]:
        """Find parameters with conflicting constraints."""
        conflicts = []
        
        for param_name, param in self._parameters.items():
            active_constraints = param.get_active_constraints()
            
            # Check for conflicting bounds
            min_bounds = [c for c in active_constraints if c.min_value is not None]
            max_bounds = [c for c in active_constraints if c.max_value is not None]
            fixed_values = [c for c in active_constraints if c.constraint_type == ConstraintType.FIXED]
            
            # Multiple fixed values
            if len(fixed_values) > 1:
                conflicts.append({
                    "parameter": param_name,
                    "type": "multiple_fixed_values",
                    "constraints": [{"realm": c.realm, "value": c.target_value} for c in fixed_values]
                })
                
            # Incompatible bounds
            if min_bounds and max_bounds:
                max_min = max(c.min_value for c in min_bounds)
                min_max = min(c.max_value for c in max_bounds)
                if max_min > min_max:
                    conflicts.append({
                        "parameter": param_name,
                        "type": "incompatible_bounds",
                        "max_min_bound": max_min,
                        "min_max_bound": min_max,
                        "min_constraints": [c.realm for c in min_bounds],
                        "max_constraints": [c.realm for c in max_bounds]
                    })
            
            # FIXED incompatible with BOUNDED
            for fx in fixed_values:
                for b in active_constraints:
                    if b.constraint_type == ConstraintType.BOUNDED and fx.target_value is not None:
                        lo, hi = (b.min_value, b.max_value)
                        if lo is not None and fx.target_value < lo:
                            conflicts.append({
                                "parameter": param_name,
                                "type": "fixed_outside_bounds",
                                "fixed_realm": fx.realm,
                                "fixed_value": fx.target_value,
                                "bounds_realm": b.realm,
                                "bounds": (lo, hi),
                                "violation": "below_minimum"
                            })
                        if hi is not None and fx.target_value > hi:
                            conflicts.append({
                                "parameter": param_name,
                                "type": "fixed_outside_bounds",
                                "fixed_realm": fx.realm,
                                "fixed_value": fx.target_value,
                                "bounds_realm": b.realm,
                                "bounds": (lo, hi),
                                "violation": "above_maximum"
                            })
            
            # FIXED incompatible with TARGET beyond tolerance
            target_constraints = [c for c in active_constraints if c.constraint_type == ConstraintType.TARGET]
            for fx in fixed_values:
                for t in target_constraints:
                    if fx.target_value is not None and t.target_value is not None:
                        tol = t.tolerance or 0.0
                        if abs(fx.target_value - t.target_value) > tol:
                            conflicts.append({
                                "parameter": param_name,
                                "type": "fixed_vs_target_mismatch",
                                "fixed_realm": fx.realm,
                                "fixed_value": fx.target_value,
                                "target_realm": t.realm,
                                "target_value": t.target_value,
                                "tolerance": tol,
                                "actual_difference": abs(fx.target_value - t.target_value)
                            })
                    
        return conflicts
        
    def validate_all_parameters(self) -> Dict[str, List[str]]:
        """Validate all parameters against their constraints."""
        validation_results = {}
        
        for param_name, param in self._parameters.items():
            violations = param.validate_constraints()
            if violations:
                validation_results[param_name] = violations
                
        return validation_results
        
    def export_state(self, format: str = "json") -> str:
        """Export the current registry state."""
        export_data = {}
        
        for param_name, param in self._parameters.items():
            export_data[param_name] = {
                "value": param.value,
                "uncertainty": param.uncertainty,
                "fixed_by_realm": param.fixed_by_realm,
                "last_modified": param.last_modified.isoformat(),
                "constraints": [
                    {
                        "realm": c.realm,
                        "type": c.constraint_type.value,
                        "min_value": c.min_value,
                        "max_value": c.max_value,
                        "target_value": c.target_value,
                        "tolerance": c.tolerance,
                        "notes": c.notes,
                        "active": c.active
                    }
                    for c in param.constraints
                ],
                "change_count": len(param.history),
                "metadata": param.metadata
            }
            
        if format.lower() == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")