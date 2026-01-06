# QFD Coupling Constants Analysis Framework

A comprehensive framework for analyzing, validating, and visualizing coupling constants across all physical realms in the QFD (Quantum Field Dynamics) system.

## Overview

The QFD Coupling Constants Analysis Framework provides a centralized system for:
- **Parameter Management**: Track coupling constants with constraints and history
- **Constraint Validation**: Ensure physical consistency across all realms
- **Dependency Analysis**: Map parameter relationships and critical paths
- **Sensitivity Analysis**: Quantify parameter impact on observables
- **Visualization**: Generate dependency graphs, constraint plots, and dashboards
- **Integration**: Seamlessly work with existing QFD realm workflow
- **Extensibility**: Plugin system for custom physics constraints

## Quick Start

### Installation

```bash
# Install dependencies
pip install numpy matplotlib networkx pandas pyyaml

# The framework is ready to use - no additional installation required
```

### Basic Usage

```python
from coupling_constants.registry.parameter_registry import ParameterRegistry
from coupling_constants.config.yaml_loader import load_parameters_from_yaml
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.ppn_validator import PPNValidator

# Load parameters from configuration
registry = ParameterRegistry()
load_parameters_from_yaml("qfd_params/defaults.yaml", registry)

# Set up validation
validator = CompositeValidator("QFD Validator")
validator.add_validator(PPNValidator())

# Run validation
report = validator.validate_all(registry)
print(report.get_summary())
```

### Command Line Interface

```bash
# Validate constraints
python qfd_coupling_cli.py validate --config qfd_params/defaults.yaml

# Full analysis with visualizations
python qfd_coupling_cli.py analyze --config qfd_params/defaults.yaml --output-dir results --visualize --sensitivity

# Export for publication
python qfd_coupling_cli.py export --config qfd_params/defaults.yaml --format publication --output pub_data

# Generate dependency graph
python qfd_coupling_cli.py visualize --config qfd_params/defaults.yaml --type dependency --output-dir plots
```

## Architecture

### Core Components

```
coupling_constants/
├── registry/           # Parameter storage and constraint management
├── validation/         # Constraint validators (PPN, CMB, vacuum, etc.)
├── analysis/          # Dependency mapping, sensitivity analysis
├── visualization/     # Plotting, dashboards, export capabilities
├── plugins/           # Extensible constraint plugin system
├── integration/       # Realm workflow integration
├── cli/              # Command-line interface
└── config/           # Configuration loading and management
```

### Key Classes

- **`ParameterRegistry`**: Central parameter storage with constraint tracking
- **`CompositeValidator`**: Orchestrates multiple constraint validators
- **`DependencyMapper`**: Builds and analyzes parameter dependency graphs
- **`SensitivityAnalyzer`**: Performs parameter sensitivity analysis
- **`CouplingVisualizer`**: Generates plots and visualizations
- **`PluginManager`**: Manages extensible constraint plugins
- **`RealmIntegrationManager`**: Integrates with existing realm workflow

## Features

### Parameter Management
- **Centralized Registry**: Single source of truth for all coupling constants
- **Constraint Tracking**: Bounded, fixed, and target constraints with realm attribution
- **History Tracking**: Complete parameter change history with timestamps
- **Conflict Detection**: Automatic detection and resolution of constraint conflicts

### Validation Framework
- **Multi-Level Validation**: Bounds, physics constraints, and custom plugins
- **PPN Validation**: Solar system tests of General Relativity parameters
- **CMB Validation**: Cosmic microwave background consistency checks
- **Vacuum Constraints**: Ensure vacuum stability (n_vac = 1, minimal photon drag)
- **Plugin System**: Extensible constraints (photon mass, cosmological constant, etc.)

### Analysis Capabilities
- **Dependency Analysis**: Parameter relationship mapping and critical path identification
- **Sensitivity Analysis**: Numerical derivatives and Monte Carlo uncertainty propagation
- **Convergence Detection**: Monitor parameter convergence across realm iterations
- **Influence Matrix**: Quantify parameter impact on observables

### Visualization & Export
- **Dependency Graphs**: Interactive network visualizations of parameter relationships
- **Constraint Plots**: Visual representation of parameter bounds and current values
- **Parameter Evolution**: Time-series plots of parameter changes
- **Comprehensive Dashboards**: HTML dashboards with multiple visualizations
- **Multi-Format Export**: JSON, YAML, CSV, LaTeX tables, publication-ready formats

### Integration & Workflow
- **Realm Integration**: Seamless integration with existing QFD realm execution
- **CLI Interface**: Complete command-line tools for all functionality
- **Automated Reporting**: Generate comprehensive analysis reports
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Usage Examples

### Parameter Registry

```python
from coupling_constants.registry.parameter_registry import ParameterRegistry, Constraint, ConstraintType

# Create registry
registry = ParameterRegistry()

# Add parameter with constraint
registry.update_parameter("k_J", 1e-15, "realm0_cmb", "Photon drag parameter")

constraint = Constraint(
    realm="vacuum_physics",
    constraint_type=ConstraintType.BOUNDED,
    min_value=0.0,
    max_value=1e-10,
    notes="Photon drag must be negligible in vacuum"
)
registry.add_constraint("k_J", constraint)
```

### Validation

```python
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.ppn_validator import PPNValidator
from coupling_constants.validation.cmb_validator import CMBValidator

# Set up comprehensive validation
validator = CompositeValidator("QFD Comprehensive Validator")
validator.add_validator(PPNValidator())
validator.add_validator(CMBValidator())

# Run validation
report = validator.validate_all(registry)

# Check results
if report.overall_status.value == 'valid':
    print("✓ All constraints satisfied")
else:
    print(f"✗ {report.total_violations} constraint violations found")
    for violation in report.get_all_violations():
        print(f"  - {violation.message}")
```

### Dependency Analysis

```python
from coupling_constants.analysis.dependency_mapper import DependencyMapper

# Build dependency graph
mapper = DependencyMapper(registry)
mapper.build_dependency_graph()

# Find critical path
critical_path = mapper.find_critical_path()
print(f"Critical path: {' → '.join(critical_path)}")

# Identify parameter clusters
clusters = mapper.identify_parameter_clusters()
for cluster_id, params in clusters.items():
    print(f"Cluster {cluster_id}: {params}")
```

### Sensitivity Analysis

```python
from coupling_constants.analysis.sensitivity_analyzer import SensitivityAnalyzer

# Perform sensitivity analysis
analyzer = SensitivityAnalyzer(registry)

# Compute sensitivity for specific observable
sensitivity = analyzer.compute_parameter_sensitivity("PPN_gamma")

# Monte Carlo uncertainty analysis
analyzer.perform_monte_carlo_analysis(n_samples=1000)

# Rank parameters by impact
rankings = analyzer.rank_parameters_by_impact(["PPN_gamma", "PPN_beta"])
```

### Visualization

```python
from coupling_constants.visualization.coupling_visualizer import CouplingVisualizer

# Create visualizer
visualizer = CouplingVisualizer(registry)

# Generate dependency graph
visualizer.plot_dependency_graph(mapper, "dependency_graph.png")

# Plot parameter constraints
visualizer.plot_parameter_constraints("constraints.png")

# Create comprehensive dashboard
visualizer.create_dashboard(mapper, output_dir="dashboard")
```

### Plugin System

```python
from coupling_constants.plugins.plugin_manager import PluginManager
from coupling_constants.plugins.constraint_plugins import VacuumStabilityPlugin

# Set up plugin manager
plugin_manager = PluginManager()

# Register plugins
plugin_manager.register_plugin(VacuumStabilityPlugin())

# Run plugin validation
results = plugin_manager.validate_all_plugin_constraints(registry)

for result in results:
    status = "✓" if result.is_valid() else "✗"
    plugin_name = result.validator_name.replace("plugin_", "")
    print(f"{status} {plugin_name}: {result.status.value}")
```

### Integration with Realm Workflow

```python
from coupling_constants.integration.realm_integration import RealmIntegrationManager

# Initialize integration manager
manager = RealmIntegrationManager("qfd_params/defaults.yaml", "analysis_output")

# Register plugins
manager.register_plugin("vacuum_stability")
manager.register_plugin("photon_mass")

# Execute realm with integration
result = manager.execute_realm_with_integration(
    "realm0_cmb", 
    "realms.realm0_cmb"
)

# Generate comprehensive report
report_path = manager.generate_analysis_report(include_visualizations=True)
print(f"Analysis report: {report_path}")
```

## Configuration

### Parameter Configuration (YAML)

```yaml
parameters:
  k_J:
    min: 0.0
    max: 1e-6
    note: "Incoherent photon drag; must be ~0 locally"
  xi:
    min: 0.1
    max: 100.0
    note: "Coupling parameter"
  PPN_gamma:
    min: 0.99
    max: 1.01
    note: "PPN gamma parameter from solar system tests"
  T_CMB_K:
    min: 2.7
    max: 2.8
    note: "CMB temperature in Kelvin"
```

## Testing

The framework includes comprehensive testing:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test categories
python -m unittest tests.test_parameter_registry
python -m unittest tests.test_validation_framework
python -m unittest tests.physics_validation.test_known_parameter_sets
python -m unittest tests.test_complete_system_integration

# Run performance tests
python -m unittest tests.physics_validation.test_performance
```

## Contributing

### Adding New Validators

```python
from coupling_constants.validation.base_validator import BaseValidator, ValidationResult

class CustomValidator(BaseValidator):
    def __init__(self):
        super().__init__("Custom Physics Validator")
    
    def validate(self, registry):
        # Implement validation logic
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.VALID
        )
        return result
```

### Adding New Plugins

```python
from coupling_constants.plugins.plugin_manager import ConstraintPlugin, PluginInfo, PluginPriority

class CustomConstraintPlugin(ConstraintPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="custom_constraint",
            version="1.0.0",
            author="Your Name",
            description="Custom physics constraint",
            priority=PluginPriority.NORMAL,
            dependencies=[],
            active=True
        )
    
    def validate_constraint(self, registry):
        # Implement constraint validation
        pass
```

## API Reference

### Core Classes

#### ParameterRegistry
- `update_parameter(name, value, realm, reason)`: Update parameter value
- `add_constraint(name, constraint)`: Add constraint to parameter
- `get_parameter(name)`: Retrieve parameter state
- `get_all_parameters()`: Get all parameters
- `get_conflicting_constraints()`: Find constraint conflicts

#### CompositeValidator
- `add_validator(validator)`: Add validator to composite
- `validate_all(registry)`: Run all validators
- `get_validator_names()`: List registered validators

#### DependencyMapper
- `build_dependency_graph()`: Construct parameter dependency graph
- `find_critical_path()`: Identify critical parameter path
- `identify_parameter_clusters()`: Group related parameters
- `compute_influence_matrix()`: Calculate parameter influence

#### CouplingVisualizer
- `plot_dependency_graph(mapper, output_path)`: Generate dependency visualization
- `plot_parameter_constraints(output_path)`: Plot parameter bounds
- `create_dashboard(mapper, output_dir)`: Generate comprehensive dashboard

## License

This framework is part of the QFD project. See the main project license for details.

## Support

For questions, issues, or contributions, please refer to the main QFD project documentation and issue tracker.