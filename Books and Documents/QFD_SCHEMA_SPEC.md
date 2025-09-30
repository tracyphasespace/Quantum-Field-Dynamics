# QFD Result Schema & Snapshot Specification

## Overview

This document defines the standardized schema for all QFD solver outputs to ensure consistency across tools and eliminate per-tool workarounds.

## Genesis Constants (Single Source of Truth)

```python
from qfd_result_schema import genesis_constants

gc = genesis_constants()
# Returns:
# {
#     "alpha": 4.0,
#     "gamma_e_target": 6.0,
#     "reference_virial": 0.0472,
#     "discovery_date": "2024-overnight-sweep", 
#     "status": "proven_stable"
# }
```

## Standard JSON Result Schema

All QFD solvers must output JSON conforming to `QFDResultSchema`:

```json
{
  "alpha": 4.0,
  "gamma_e_target": 6.0,
  "beta": 3.0,
  "eta": 0.05,
  "kappa_time": 3.2,
  
  "mass_amu": 2.0,
  "charge": 1,
  "electrons": 1,
  
  "grid_points": 128,
  "max_radius": 14.0,
  "grid_dx": 0.21875,
  "spectral_cutoff": 0.36,
  "iters_outer": 1200,
  "tol_energy_rel": 1e-8,
  
  "converged": false,
  "E_model": -0.123456,
  "virial": 0.0472,
  
  "penalty_Q": 1.23e-06,
  "penalty_B": 2.34e-06,
  "penalty_center": 3.45e-07,
  
  "T_N": 0.456,
  "T_e": 0.123,
  "V_coul": -0.789,
  "V4_N_eff": -0.234,
  "V4_e": -0.012,
  "V_time_balance": 0.001,
  
  "virial_ok": true,
  "penalties_ok": true,
  "physical_success": true,
  
  "timestamp": "2024-01-15T14:30:00",
  "solver_version": "Genesis_v3.2"
}
```

## Standard .pt Snapshot Schema

All field snapshots must conform to `QFDSnapshotSpec`:

```python
{
  "psi_N": torch.Tensor,      # Nuclear field on CPU
  "psi_e": torch.Tensor,      # Electron field on CPU
  
  "grid_points": 128,         # Grid resolution
  "max_radius": 14.0,         # Domain size
  "grid_dx": 0.21875,         # Grid spacing
  
  "alpha": 4.0,               # Genesis Constants
  "gamma_e_target": 6.0,
  
  "mass_amu": 2.0,            # Matter specification
  "charge": 1,
  "electrons": 1,
  
  "solver_version": "Genesis_v3.2",
  "timestamp": "2024-01-15T14:30:00"
}
```

## Success Criteria (Physics-First)

Physical success is independent of convergence flags:

- **virial_ok**: `virial < 0.1`
- **penalties_ok**: `max(penalty_Q, penalty_B, penalty_center) < 1e-5`
- **physical_success**: `virial_ok AND penalties_ok`

## Tool Compliance

### ✅ Compliant Tools
- `Deuterium.py` (Genesis v3.2)
- `run_target_deuterium.py` 
- `test_genesis_constants.py`

### 🔄 Migration Needed
- `AutopilotHydrogen.py` (legacy discovery tool)
- `AllNightLong.py`
- Other legacy solvers

## Usage Examples

### Validate Schema
```python
from qfd_result_schema import validate_result_schema

with open("result.json") as f:
    data = json.load(f)
    
if validate_result_schema(data):
    print("✓ Schema valid")
else:
    print("✗ Schema validation failed")
```

### Convert Legacy Results
```python
from qfd_result_schema import QFDResultSchema

# Convert legacy Deuterium.py output
standard_result = QFDResultSchema.from_deuterium_result(legacy_data)
with open("standardized.json", "w") as f:
    json.dump(standard_result.to_dict(), f, indent=2)
```

### Load Snapshot for Visualization
```python
import torch

snapshot = torch.load("state_mass2.00_Z1_Ne1.pt")
psi_N = snapshot["psi_N"]
psi_e = snapshot["psi_e"]
grid_dx = snapshot["grid_dx"]
# All metadata guaranteed to be present
```

## Migration Guide

1. **Import schema**: `from qfd_result_schema import QFDResultSchema, genesis_constants`
2. **Use Genesis Constants**: Replace hardcoded values with `genesis_constants()`
3. **Standardize output**: Convert results using `QFDResultSchema.from_*_result()`
4. **Validate**: Use `validate_result_schema()` in tests
5. **Update visualization**: Expect standardized snapshot format

This eliminates schema drift and enables fully automated calibration pipelines.