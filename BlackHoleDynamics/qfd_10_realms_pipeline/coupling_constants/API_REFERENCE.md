# QFD Coupling Constants Framework - API Reference

## Core Classes

### ParameterRegistry
Central registry for managing coupling constants.

```python
from coupling_constants.registry.parameter_registry import ParameterRegistry

registry = ParameterRegistry()
registry.update_parameter("k_J", 1e-15, "realm0_cmb", "CMB constraint")
param = registry.get_parameter("k_J")
```

### CompositeValidator
Orchestrates multiple constraint validators.

```python
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.ppn_validator import PPNValidator

validator = CompositeValidator()
validator.add_validator(PPNValidator())
report = validator.validate_all(registry)
```

### DependencyMapper
Analyzes parameter dependencies.

```python
from coupling_constants.analysis.dependency_mapper import DependencyMapper

mapper = DependencyMapper(registry)
mapper.build_dependency_graph()
critical_path = mapper.find_critical_path()
```

### CouplingVisualizer
Generates visualizations.

```python
from coupling_constants.visualization.coupling_visualizer import CouplingVisualizer

visualizer = CouplingVisualizer(registry)
visualizer.plot_dependency_graph(mapper, "graph.png")
visualizer.create_dashboard(mapper, "dashboard")
```

### PluginManager
Manages constraint plugins.

```python
from coupling_constants.plugins.plugin_manager import PluginManager
from coupling_constants.plugins.constraint_plugins import VacuumStabilityPlugin

plugin_manager = PluginManager()
plugin_manager.register_plugin(VacuumStabilityPlugin())
results = plugin_manager.validate_all_plugin_constraints(registry)
```

## Command Line Interface

```bash
# Validate constraints
python qfd_coupling_cli.py validate --config qfd_params/defaults.yaml

# Full analysis
python qfd_coupling_cli.py analyze --config qfd_params/defaults.yaml --output-dir results --visualize

# Export data
python qfd_coupling_cli.py export --config qfd_params/defaults.yaml --format json --output data.json
```

## Integration

### RealmIntegrationManager
Integrates with QFD realm workflow.

```python
from coupling_constants.integration.realm_integration import RealmIntegrationManager

manager = RealmIntegrationManager("qfd_params/defaults.yaml", "analysis_output")
result = manager.execute_realm_with_integration("realm0_cmb", "realms.realm0_cmb")
report_path = manager.generate_analysis_report()
```

For complete API documentation, see the comprehensive examples in the main README.md file.