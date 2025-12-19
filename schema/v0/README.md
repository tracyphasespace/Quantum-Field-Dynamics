# QFD RunSpec Schema v0

Complete schema system for reproducible Grand Solver runs.

## Overview

The RunSpec v0 system provides a **data contract** for QFD parameter fitting experiments. Every run is fully specified by a single JSON file that captures:

- **Model**: Physics equations, computational backend, grid configuration
- **Parameters**: Couplings, scales, nuisance terms with bounds and priors
- **Datasets**: Observational data with selection cuts and likelihood models
- **Objective**: Multi-dataset optimization with weights and regularization
- **Solver**: Optimization method, parallelization, random seeds
- **Provenance**: Git commit, versions, timestamps

## Files

### Core Schemas

- **`RunSpec.schema.json`** - Top-level specification for a complete run
- **`ModelSpec.schema.json`** - Physical model definition (equations, grid, backend)
- **`ParameterSpec.schema.json`** - Individual parameter with bounds, priors, units
- **`DatasetSpec.schema.json`** - Observational data source and likelihood
- **`ObjectiveSpec.schema.json`** - Optimization objective function
- **`ResultSpec.schema.json`** - Output specification for fit results

### Tools

- **`validate_runspec.py`** - Validator and resolver for RunSpec files
- **`examples/`** - Example RunSpec instances

## Quick Start

### 1. Validate a RunSpec

```bash
python validate_runspec.py examples/core_compression_fit.runspec.json --check-only
```

### 2. Resolve file references and fill git provenance

```bash
python validate_runspec.py examples/core_compression_fit.runspec.json \
    --resolve --fill-git --output resolved.json
```

### 3. Use in a solver

```python
import json
from validate_runspec import RunSpecValidator

# Load and validate
with open('my_run.runspec.json') as f:
    runspec = json.load(f)

validator = RunSpecValidator(Path('schema/v0'))
errors = validator.validate(runspec)
if errors:
    raise ValueError(f"Invalid RunSpec: {errors}")

# Resolve references
runspec = validator.resolve_references(runspec, Path('.'))

# Run solver
result = grand_solver.run(runspec)

# Save result
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)
```

## Schema Hierarchy

```
RunSpec
├── model: ModelSpec
│   ├── equations: List[EquationTerm]
│   ├── grid: GridSpec
│   └── backend: BackendConfig
├── parameters: List[ParameterSpec]
│   ├── Universal couplings (global)
│   ├── Observable-specific nuisance
│   └── Dataset-specific calibration
├── datasets: List[DatasetSpec]
│   ├── source: DataSource
│   ├── selection: SelectionCriteria
│   └── likelihood: LikelihoodModel
├── objective: ObjectiveSpec
│   ├── components: List[DatasetComponent]
│   └── regularization: List[RegularizationTerm]
└── solver: SolverConfig
    ├── method: OptimizationMethod
    └── parallel: ParallelConfig
```

## Example: Core Compression Fit

The `examples/core_compression_fit.runspec.json` demonstrates fitting nuclear surface/volume terms (c1, c2) to AME2020 binding energy data:

**Parameters being fit:**
- `c1` - Surface term (A^(2/3) scaling)
- `c2` - Volume correction (linear in A)
- `normalization_scale` - Dataset calibration nuisance

**Fixed parameters:**
- `V4 = 11 MeV` - Quartic potential depth (from proven value)
- `g_c = 0.985` - Geometric charge coupling (from proven value)

**Dataset:**
- AME2020 stable nuclides (4 ≤ A ≤ 240)
- Gaussian likelihood with independent errors
- 5-sigma outlier rejection

**Objective:**
- Chi-squared minimization
- L2 regularization on c1, c2
- Per-datapoint normalization

**Solver:**
- `scipy.optimize.minimize`
- 4-process parallelization
- Seed: 42

## Parameter Categories

### Universal Couplings (Global)

These appear in **multiple domains** and must be consistent across all solvers:

- `k_J` - Universal J·A interaction (nuclear, cosmo, astrophysics)
- `V4` - Quartic potential depth (nuclear, particle)
- `g_c` - Geometric charge coupling (nuclear, particle)
- `lambda_R` - Rotor coupling (particle)

### Observable-Specific Nuisance

Affect **one observable** across all datasets measuring it:

- `H0_calibration` - Hubble constant systematic (affects all CMB, SN, redshift)
- `alpha_EM` - Fine structure variation (affects all atomic transitions)

### Dataset-Specific Nuisance

Unique to **one dataset**:

- `normalization_scale` - Overall calibration for AME2020
- `zero_point_offset` - Systematic offset for Planck CMB

## Provenance Tracking

Every RunSpec captures complete provenance:

```json
{
  "git": {
    "commit": "a1b2c3d4",
    "dirty": false,
    "branch": "main"
  },
  "solver": {
    "seed": 42
  },
  "metadata": {
    "created": "2025-12-19T00:00:00Z",
    "author": "Tracy McPherson"
  }
}
```

Results reference the original RunSpec:

```json
{
  "run_id": "core_compression_fit_001",
  "runspec_path": "examples/core_compression_fit.runspec.json",
  "provenance": {
    "git_commit": "a1b2c3d4",
    "start_time": "2025-12-19T00:00:00Z",
    "python_version": "3.11.5"
  }
}
```

## Reproducibility Guarantee

Given a `RunSpec` + `git commit` + `random seed`, the **exact same result** can be reproduced:

1. Checkout the specified git commit
2. Load the RunSpec
3. Run the solver with specified seed
4. Verify output matches original `ResultSpec`

This enables:
- **Peer review**: Reviewers can reproduce fits exactly
- **Debugging**: Trace unexpected results to exact configuration
- **Publication**: Archive RunSpec + git tag for paper supplementary material
- **Comparison**: A/B test different models/solvers with identical data

## Design Principles

1. **Separation of Concerns**: Models, Parameters, Datasets, Objectives are independent
2. **Composability**: Build complex runs from simple reusable components
3. **Explicit over Implicit**: All settings declared, no hidden defaults
4. **File or Inline**: References can be external files or inline objects
5. **Validation First**: Schemas catch errors before expensive computation
6. **Provenance Always**: Every run tracks its origin

## Next Steps

### v0.2 Additions

- **DependencyGraph.schema.json** - Parameter dependency DAG
- **SensitivitySpec.schema.json** - Parameter sensitivity analysis
- **ComparisonSpec.schema.json** - Multi-run A/B testing
- **PublicationSpec.schema.json** - Publication-ready output bundles

### Integration

- Python solver entrypoints that consume RunSpec
- Observable adapters: `qfd.adapters.nuclear.predict_binding_energy`
- Model implementations: `qfd.solvers.nuclear.terms.*`
- Dataset loaders: `qfd.data.load_ame2020`

### Grand Solver Architecture

```
Grand Solver
├── Nuclear Solver (uses: V4, k_c2, c1, c2, g_c)
├── CMB Solver (uses: k_J, eta_prime, A_plasma)
├── Redshift Solver (uses: k_J, eta_prime, rho_vac)
├── Lepton Solver (uses: V2, V4, lambda_R, g_c)
└── BBH Solver (uses: k_J, G_eff, alpha_gravity)
```

Each solver:
1. Reads its parameters from shared `GrandUnifiedParameters`
2. Loads its datasets
3. Computes predictions via observable adapters
4. Returns chi-squared contribution

Grand Solver:
1. Loads RunSpec defining all domains
2. Spawns domain solvers in parallel
3. Aggregates chi-squared across domains
4. Optimizes shared parameters
5. Saves unified ResultSpec

## References

- QFD Appendix Z.2: Clifford algebra structure
- QFD Appendix N: Nuclear Genesis
- QFD Appendix P: Cosmology
- QFD Grand Solver Architecture: `projects/Lean4/QFD/GRAND_SOLVER_ARCHITECTURE.md`
