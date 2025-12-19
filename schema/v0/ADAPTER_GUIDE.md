# Observable Adapter Guide

## Overview

Observable adapters are the bridge between QFD theory and experimental data. They map theoretical parameters (c1, c2, V4, g_c) to observable predictions (binding energies, power spectra, etc.).

## Architecture

```
RunSpec.json
    ↓
    objective.components[*].observable_adapter
    ↓
    "qfd.adapters.nuclear.predict_binding_energy"
    ↓
    Dynamic Import
    ↓
    func(df, params, config) → predictions
    ↓
    Chi-squared calculation
```

## Adapter Signature

All adapters must follow this signature:

```python
def adapter_function(
    df: pd.DataFrame,           # Input data
    params: Dict[str, float],   # Hydrated parameters
    config: Optional[Dict[str, Any]] = None  # Dataset config
) -> np.ndarray:                # Predictions
    """
    Adapter docstring.

    Args:
        df: DataFrame with observables (A, Z, etc.)
        params: Parameter values from solver
            - May include "nuclear.c1" or just "c1"
            - Handle both for robustness
        config: Optional dataset-specific configuration

    Returns:
        np.ndarray: Predicted values matching target column
    """
```

## Writing an Adapter

### 1. Create Module

```bash
mkdir -p qfd/adapters/your_domain
touch qfd/adapters/your_domain/__init__.py
touch qfd/adapters/your_domain/your_observable.py
```

### 2. Implement Physics

```python
# qfd/adapters/your_domain/your_observable.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def predict_your_observable(
    df: pd.DataFrame,
    params: Dict[str, float],
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Predict your observable from QFD parameters.
    """
    # 1. Extract data
    x = df["x_column"].to_numpy()

    # 2. Extract parameters (handle both namespaced and bare names)
    param1 = params.get("domain.param1", params.get("param1"))
    param2 = params.get("param2")

    # 3. Calculate prediction
    y_pred = your_physics_formula(x, param1, param2)

    return y_pred
```

### 3. Export in `__init__.py`

```python
# qfd/adapters/your_domain/__init__.py
from .your_observable import predict_your_observable

__all__ = ["predict_your_observable"]
```

### 4. Update RunSpec

```json
{
  "objective": {
    "components": [
      {
        "dataset_id": "your_dataset",
        "observable_adapter": "qfd.adapters.your_domain.predict_your_observable",
        "weight": 1.0
      }
    ]
  }
}
```

## Example: Nuclear Binding Energy

```python
# qfd/adapters/nuclear/binding_energy.py
def predict_binding_energy(df, params, config=None):
    # Extract data
    A = df["A"].to_numpy()
    Z = df["Z"].to_numpy()

    # Extract parameters
    c1 = params.get("nuclear.c1", params.get("c1"))
    c2 = params.get("nuclear.c2", params.get("c2"))
    V4 = params.get("V4")
    g_c = params.get("g_c")

    # QFD Core Compression Law
    E_vol = c2 * A
    E_surf = -c1 * np.power(A, 2/3)
    E_coul = -0.71 * g_c * Z * (Z - 1) / np.power(A, 1/3)

    BE = V4 * (E_vol + E_surf + E_coul)
    return BE
```

**RunSpec**:
```json
{
  "datasets": [{
    "id": "ame2020",
    "source": "data/ame2020.csv",
    "columns": {
      "A": "mass_number",
      "Z": "proton_number",
      "target": "binding_energy_MeV"
    }
  }],
  "objective": {
    "components": [{
      "dataset_id": "ame2020",
      "observable_adapter": "qfd.adapters.nuclear.predict_binding_energy"
    }]
  }
}
```

## Multi-Domain Fitting

You can fit multiple observables simultaneously:

```json
{
  "objective": {
    "components": [
      {
        "dataset_id": "nuclear_data",
        "observable_adapter": "qfd.adapters.nuclear.predict_binding_energy",
        "weight": 1.0
      },
      {
        "dataset_id": "cmb_data",
        "observable_adapter": "qfd.adapters.cosmo.predict_cmb_power",
        "weight": 0.5
      }
    ]
  }
}
```

The solver will:
1. Call each adapter with appropriate data
2. Calculate χ² for each component
3. Sum with weights: χ²_total = w₁·χ²_nuclear + w₂·χ²_cmb

## Parameter Name Resolution

Adapters should handle both namespaced and bare parameter names:

```python
# Robust parameter extraction
def get_param(params, *names):
    """Try multiple parameter name variants."""
    for name in names:
        if name in params:
            return params[name]
    raise KeyError(f"None of {names} found in parameters")

# Usage
c1 = get_param(params, "nuclear.c1", "c1")
V4 = get_param(params, "V4")
```

## Column Name Flexibility

Use helper functions to find columns:

```python
def get_column(df, candidates):
    """Get first matching column."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return df[cols_lower[cand.lower()]]
    raise KeyError(f"None of {candidates} found in {df.columns}")

# Usage
A = get_column(df, ["A", "mass_number", "massnumber"])
```

## Testing Adapters

Include self-tests:

```python
def validate_adapter():
    """Self-test for adapter."""
    df = pd.DataFrame({"A": [12], "Z": [6]})
    params = {"c1": 1.0, "c2": 0.05, "V4": 11e6}

    result = predict_your_observable(df, params)

    assert result.shape == (1,)
    assert np.isfinite(result).all()
    print("✓ Adapter validated")
    return True

if __name__ == "__main__":
    validate_adapter()
```

## Best Practices

1. **Clear Documentation**: Explain physics model in docstring
2. **Robust Column Finding**: Support multiple column name variants
3. **Flexible Parameter Names**: Handle both "domain.param" and "param"
4. **Physical Units**: Document expected units (eV, MeV, etc.)
5. **Error Messages**: Clear errors when required data missing
6. **Self-Contained**: No global state, pure function
7. **Validated**: Include self-test in module

## Current Adapters

| Module | Observable | Status |
|--------|-----------|--------|
| `qfd.adapters.nuclear.predict_binding_energy` | Binding Energy | ✅ Implemented |
| `qfd.adapters.nuclear.predict_binding_energy_per_nucleon` | BE/A | ✅ Implemented |
| `qfd.adapters.cosmo.predict_cmb_power` | CMB C_ℓ | ⏳ Planned |
| `qfd.adapters.particle.predict_lepton_mass` | Lepton masses | ⏳ Planned |

## Debugging

Enable verbose output:

```python
def predict_binding_energy(df, params, config=None):
    verbose = config.get("verbose", False) if config else False

    if verbose:
        print(f"Adapter called with {len(df)} rows")
        print(f"Parameters: {params}")

    # ... calculation ...

    if verbose:
        print(f"Predicted: min={result.min()}, max={result.max()}")

    return result
```

Then in RunSpec:
```json
{
  "datasets": [{
    "id": "test",
    "verbose": true
  }]
}
```

## References

- Dynamic adapter pattern: `solve_v03.py:load_adapter()`
- Example implementation: `qfd/adapters/nuclear/binding_energy.py`
- Schema specification: `ObjectiveSpec.schema.json`
