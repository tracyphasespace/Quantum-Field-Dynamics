"""
YAML configuration loader for coupling constants.

Integrates with the existing qfd_params/defaults.yaml configuration
to populate the parameter registry with initial parameter definitions.
"""

import yaml
from typing import Dict, Any
from ..registry.parameter_registry import ParameterRegistry, ParameterState, Constraint, ConstraintType


def _ensure_numeric(x, key: str):
    """Ensure a value is numeric, with proper error handling."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x))
    except Exception as e:
        raise ValueError(f"{key} must be numeric, got {x!r}") from e


def _validate_parameter_config(param_name: str, param_config: Dict[str, Any]) -> None:
    """Validate parameter configuration structure."""
    # Check for required keys (can be extended)
    # For now, we're lenient but could add required fields
    
    # Validate numeric fields
    if 'min' in param_config:
        _ensure_numeric(param_config['min'], f"{param_name}.min")
    if 'max' in param_config:
        _ensure_numeric(param_config['max'], f"{param_name}.max")
    
    # Validate that min <= max if both present
    min_val = param_config.get('min')
    max_val = param_config.get('max')
    if min_val is not None and max_val is not None:
        min_num = _ensure_numeric(min_val, f"{param_name}.min")
        max_num = _ensure_numeric(max_val, f"{param_name}.max")
        if min_num > max_num:
            raise ValueError(f"Parameter {param_name}: min ({min_num}) > max ({max_num})")


def load_parameters_from_yaml(yaml_path: str, registry: ParameterRegistry) -> None:
    """
    Load parameters from YAML configuration into the registry.
    
    Args:
        yaml_path: Path to the YAML configuration file
        registry: ParameterRegistry instance to populate
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load parameters section
    parameters = config.get('parameters', {})
    for param_name, param_config in parameters.items():
        # Validate parameter configuration
        _validate_parameter_config(param_name, param_config)
        
        # Create parameter state
        param_state = ParameterState(
            name=param_name,
            metadata={
                'unit': param_config.get('unit', ''),
                'note': param_config.get('note', ''),
                'source': 'yaml_config'
            }
        )
        
        # Add bounded constraint if min/max are specified
        min_val = _ensure_numeric(param_config.get('min'), f"{param_name}.min")
        max_val = _ensure_numeric(param_config.get('max'), f"{param_name}.max")
                
        if min_val is not None or max_val is not None:
            constraint_notes = f"Configuration bounds for {param_name}"
            unit = param_config.get('unit', '')
            if unit:
                constraint_notes += f" (unit: {unit})"
            
            constraint = Constraint(
                realm="config",
                constraint_type=ConstraintType.BOUNDED,
                min_value=min_val,
                max_value=max_val,
                notes=constraint_notes
            )
            param_state.add_constraint(constraint)
        
        registry.register_parameter(param_state)
    
    # Load PPN targets as constraints
    ppn_targets = config.get('ppn_targets', {})
    if ppn_targets:
        gamma_target = ppn_targets.get('gamma')
        beta_target = ppn_targets.get('beta')
        gamma_tol = ppn_targets.get('tol_gamma', 1e-5)
        beta_tol = ppn_targets.get('tol_beta', 1e-4)
        
        if gamma_target is not None:
            gamma_constraint = Constraint(
                realm="ppn_config",
                constraint_type=ConstraintType.TARGET,
                target_value=gamma_target,
                tolerance=gamma_tol,
                notes="PPN gamma parameter target from configuration"
            )
            registry.add_constraint("PPN_gamma", gamma_constraint)
            
        if beta_target is not None:
            beta_constraint = Constraint(
                realm="ppn_config", 
                constraint_type=ConstraintType.TARGET,
                target_value=beta_target,
                tolerance=beta_tol,
                notes="PPN beta parameter target from configuration"
            )
            registry.add_constraint("PPN_beta", beta_constraint)
    
    # Load CMB targets as constraints
    cmb_targets = config.get('cmb_targets', {})
    if cmb_targets:
        t_cmb = cmb_targets.get('T_CMB_K')
        if t_cmb is not None:
            cmb_constraint = Constraint(
                realm="cmb_config",
                constraint_type=ConstraintType.FIXED,
                target_value=t_cmb,
                tolerance=1e-6,
                notes="CMB temperature fixed from configuration"
            )
            registry.add_constraint("T_CMB_K", cmb_constraint)


def get_parameter_summary(registry: ParameterRegistry) -> Dict[str, Any]:
    """
    Generate a summary of loaded parameters and constraints.
    
    Args:
        registry: ParameterRegistry instance
        
    Returns:
        Dictionary with parameter summary statistics
    """
    all_params = registry.get_all_parameters()
    
    summary = {
        'total_parameters': len(all_params),
        'parameters_with_constraints': 0,
        'total_constraints': 0,
        'constraint_types': {},
        'parameters_by_realm': {}
    }
    
    for param_name, param in all_params.items():
        if param.constraints:
            summary['parameters_with_constraints'] += 1
            summary['total_constraints'] += len(param.constraints)
            
            for constraint in param.constraints:
                constraint_type = constraint.constraint_type.value
                summary['constraint_types'][constraint_type] = summary['constraint_types'].get(constraint_type, 0) + 1
                
                realm = constraint.realm
                if realm not in summary['parameters_by_realm']:
                    summary['parameters_by_realm'][realm] = []
                if param_name not in summary['parameters_by_realm'][realm]:
                    summary['parameters_by_realm'][realm].append(param_name)
    
    return summary