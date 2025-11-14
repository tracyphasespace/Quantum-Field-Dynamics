# QFD Supernova Pipeline Tests

Integration tests for the V15 supernova analysis pipeline.

## Running Tests

```bash
# Run all tests
pytest tests/test_pipeline.py -v

# Run specific test class
pytest tests/test_pipeline.py::TestParameterOrdering -v

# Run with detailed output
pytest tests/test_pipeline.py -v --tb=long
```

## Test Coverage

### TestParameterOrdering
Validates that parameter ordering is consistent between pipeline stages:
- `test_persn_params_array_conversion`: Round-trip NamedTuple ↔ array conversion
- `test_persn_params_model_order`: Validates `to_model_order()` reordering
- `test_stage1_order_matches_namedtuple`: Ensures Stage 1 array order matches NamedTuple definition

### TestNumericalStability
Validates that parameter values produce stable model outputs:
- `test_typical_parameters_no_overflow`: Typical parameters don't cause overflow
- `test_parameter_order_sensitivity`: Demonstrates that parameter order matters

### TestGlobalParams
Validates global cosmology parameter structures:
- `test_global_params_conversion`: Round-trip conversion
- `test_global_params_expected_ranges`: Parameter values in expected ranges

### TestStage1Stage2Handoff
Integration tests for Stage 1 → Stage 2 data transfer:
- `test_load_stage1_results_structure`: Validates Stage 1 output loading

## Why These Tests Matter

The V15 pipeline passes parameters between stages as raw NumPy arrays. This creates a subtle bug risk:

**Stage 1 saves**: `[t0, A_plasma, beta, ln_A]`
**Model expects**: `(t0, ln_A, A_plasma, beta)`

If Stage 2 forgets to reorder, beta=0.57 becomes ln_A (catastrophic underflow) and ln_A=18.5 becomes beta (catastrophic overflow). These tests catch this class of bugs automatically.

## Adding New Tests

When adding new per-SN parameters or modifying the physics model:
1. Update `pipeline_io.py` with new parameter definitions
2. Add tests to validate the new parameters
3. Run tests before starting expensive MCMC runs

## Continuous Integration

To prevent regressions, run tests before committing:

```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
pytest tests/test_pipeline.py -q
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```
