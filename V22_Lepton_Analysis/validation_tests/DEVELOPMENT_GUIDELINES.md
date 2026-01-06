# Development Guidelines: V22 Lepton Analysis Validation Tests

## Performance and Progress Monitoring

### Always Use Progress Bars
- **For any loop > 5 iterations**, use `tqdm` for progress monitoring
- **For long-running optimizations**, add progress callbacks or periodic checkpoints
- **Example**:
  ```python
  from tqdm import tqdm

  for param in tqdm(param_grid, desc="Parameter scan", unit="point"):
      result = expensive_optimization(param)
  ```

### Parallel Processing
- **Hardware**: 8 logical cores (4 physical with hyperthreading), 7 GB RAM available
- **Use 8 workers** for scipy.optimize.differential_evolution (memory usage ~0.4 GB total):
  ```python
  result = differential_evolution(
      objective,
      bounds,
      workers=8,  # Parallel evaluation on 8 cores (~0.4 GB total)
      maxiter=100,
  )
  ```
- **Thread management**: If seeing too many threads (8+ per worker), set environment variables:
  ```bash
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  python your_script.py
  ```

### Output Buffering
- **Always flush output** in long loops to ensure progress visibility:
  ```python
  import sys

  for item in items:
      print(f"Processing {item}")
      sys.stdout.flush()
  ```

### Intermediate Results
- **Write intermediate outputs** every N iterations for runs > 10 minutes:
  ```python
  if iteration % checkpoint_interval == 0:
      with open(f"checkpoint_{iteration}.json", "w") as f:
          json.dump(current_state, f)
  ```

## Optimization Best Practices

### differential_evolution Settings
- **maxiter**: Use 100 (not 200) for initial exploration, 200 for final fits
- **workers**: Use 4 for parallelization
- **bounds**: Set reasonable bounds based on physics, not too tight
- **seed**: Always set for reproducibility (seed=42)
- **atol, tol**: Use 1e-8 for high-precision fits

### Multi-start Stability
- For critical fits, run with 5+ different seeds:
  ```python
  seeds = [42, 123, 456, 789, 1011]
  results = []
  for seed in seeds:
      result = fitter.fit(seed=seed)
      results.append(result)
  ```

## Code Organization

### Energy Functional Convention
- **Sign convention**: All penalty terms ADD (never subtract)
- **Standard form**: `E_total = E_circ + E_stab + E_grad`
- Each component is positive-definite

### Localization Envelope
- **Outside-only envelope**: Preserves interior velocities
  ```python
  def localization_envelope(self, r):
      g = np.ones_like(r)
      mask = r > self.R_v
      if np.any(mask):
          g[mask] = np.exp(-(((r[mask] - self.R_v) / self.delta_v) ** self.p))
      return g
  ```

### Mandatory Diagnostics
For any lepton fit, always report:
- **Fit quality**: χ², S_opt
- **Parameters**: β, per-lepton (U, R_c, A)
- **Energies**: Per-lepton (E_circ, E_stab, E_grad, E_total)
- **Profile metrics**: F_inner (fraction of E_circ from structured region)
- **Bound hits**: Flag any parameters at bounds

## File Naming and Output

### Results Directory Structure
```
results/V22/
├── logs/              # All stdout logs (.log files)
├── *.json            # Numerical results
└── plots/            # Generated figures
```

### Naming Convention
- **Scripts**: `{runN}_{description}_{variant}.py`
  - Example: `run2_emu_regression_corrected.py`
- **Logs**: `{script_name}.log`
  - Example: `run2_emu_corrected.log`
- **Results**: `{script_name}_results.json`
  - Example: `run2_emu_corrected_results.json`

### Always Save Results
- Write JSON results BEFORE printing summary
- Include metadata: timestamp, git commit, configuration frozen
- Example:
  ```python
  results = {
      "timestamp": datetime.now().isoformat(),
      "config": {"k": 1.5, "delta_v_factor": 0.5, "p": 6},
      "beta_min": beta_min,
      "chi2_min": chi2_min,
      "fit_results": detailed_results,
  }

  with open("results/V22/output.json", "w") as f:
      json.dump(results, f, indent=2)
  ```

## Acceptance Criteria Templates

### Standard Gates
For lepton regression runs, use these standard criteria:

1. **χ² not pathological**: `chi2 < 1e6` (orders of magnitude reduction from 1e8)
2. **Positive scale**: `S_opt > 0`
3. **Not all at bounds**: `≤ 1 parameter per lepton at bounds`
4. **Interior β**: `β_min` not at scan edges (margin 0.02)
5. **Multi-start stability**: `std(S_opt) / mean(S_opt) < 0.1` across 5 seeds

### Classification
- **PASS**: All criteria met
- **SOFT PASS**: χ² and S_opt good, but degeneracy detected (bound saturation, β edge)
- **FAIL**: χ² pathological or S_opt ≤ 0

## Git Commit Guidelines

### When to Commit
- After implementing major functional changes (e.g., sign convention fix)
- After successful validation runs (include results in commit)
- Before pivoting physics approach

### Commit Message Format
```
{type}: {brief summary}

{detailed explanation if needed}

Results: {key outcomes}
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`

Example:
```
fix: Correct energy sign convention (E_total = E_circ + E_stab + E_grad)

All penalty terms now add consistently. Previous convention had
E_stab subtracting, causing artificial electron near-cancellation.

Sanity check: Both e,μ now have E_total > 0 at reasonable parameters.
```

## Common Pitfalls

### Energy Calculations
- ❌ **Don't**: Use `E_total = E_circ - E_stab + E_grad` (inconsistent signs)
- ✓ **Do**: Use `E_total = E_circ + E_stab + E_grad` (all penalties add)

### Localization
- ❌ **Don't**: Use global exponential `exp(-(r/R_v)^p)` (suppresses boundary)
- ✓ **Do**: Use outside-only piecewise envelope (preserves interior)

### Progress Monitoring
- ❌ **Don't**: Run long optimizations without tqdm or checkpoints
- ✓ **Do**: Always add progress bars and intermediate saves

### Parallelization
- ❌ **Don't**: Use `workers=1` when 4+ cores available
- ✓ **Do**: Use `workers=4` for differential_evolution

### Bound Checking
- ❌ **Don't**: Trust optimizer if all parameters hit bounds
- ✓ **Do**: Check `abs(param - bound) < 1e-6` and flag as degeneracy

## Quick Reference

### Start New Run Script Template
```python
#!/usr/bin/env python3
import numpy as np
from scipy.optimize import differential_evolution
from tqdm import tqdm
import json
import sys

# Configuration (FROZEN)
K_LOCALIZATION = 1.5
DELTA_V_FACTOR = 0.5
P_ENVELOPE = 6

# Main loop with progress monitoring
for param in tqdm(param_grid, desc="Scan", unit="point"):
    result = optimize_with_config(param)
    print(f"Result: {result}")
    sys.stdout.flush()

# Save results
with open("results/V22/output.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Optimization Call Template
```python
result = differential_evolution(
    objective_function,
    bounds=[...],
    maxiter=100,
    seed=42,
    atol=1e-8,
    tol=1e-8,
    workers=4,
)
```

### Energy Calculation Template
```python
E_circ = circulation_energy(params)
E_stab = stabilization_energy(params)
E_grad = gradient_energy(params)
E_total = E_circ + E_stab + E_grad  # All terms ADD
```
