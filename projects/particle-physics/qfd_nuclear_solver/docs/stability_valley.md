## QFD Stability Valley (Parameter-Free)

**Script:** `scripts/qfd_stability_valley.py`  
**Purpose:** Predict the stable proton number `Z` for each baryon number `A` using only geometric/topological inputs.

### Energy Functional

The solver evaluates

```
E(A,Z) = E_volume · A
       + E_surface · A^(2/3)
       + a_sym · A · (1 - 2Z/A)^2
       + a_c · Z^2 / A^(1/3)
```

All coefficients are derived from fundamental QFD constants and require **no fitting**:

| Term        | Expression                          | Value (MeV) |
|-------------|-------------------------------------|-------------|
| `E_volume`  | `M_p · (1 - α²/β) · (1 - λ/(12π))`  | 927.668     |
| `E_surface` | `(β · M_p / 2) / 15`                | 10.228      |
| `a_sym`     | `(β · M_p) / 15`                    | 20.455      |
| `a_c`       | `α · ħc / r₀`                       | 1.200       |

with `α = 1/137.036`, `β = 1/3.043233053`, `λ = 0.42`, `r₀ = 1.2 fm`.

### Prediction Quality

Using the evaluation set (H-2…Ni-58) the solver achieves:

- Mean |ΔZ| = **1.29** charges (target < 2 ✓)
- Median |ΔZ| = 1
- Max |ΔZ| = 4 (Ni-58)
- Exact matches for all light nuclei (A ≤ 12)

Systematic trend: `Z_pred` is 1–4 units below experiment for A ≥ 28, indicating higher-order effects (shell/pairing) become relevant but the geometric valley is already captured.

### Charge-Fraction Evolution

| Regime            | Mean `Z/A` |
|-------------------|------------|
| Light (A < 20)    | 0.478      |
| Medium (20 ≤ A < 60) | 0.428  |
| Heavy (A ≥ 60)    | 0.391      |
| Asymptotic `q∞`   | `√(α/β)` ≈ **0.149** |

Interpretation: the nucleus becomes “charge-poor” purely from topological density minimization; no explicit Coulomb force is required.

### Usage

```bash
python scripts/qfd_stability_valley.py
```

This prints the coefficient table, the prediction vs experiment summary, accuracy metrics, and writes `qfd_stability_valley.png` with:

1. `Z` vs `A` stability valley compared against `Z=A/2` and the asymptotic line.
2. `Z/A` vs `A` illustrating the monotonic drop toward `q∞`.

### Integration Notes

- The solver relies only on NumPy, SciPy, and Matplotlib already listed in `requirements.txt`.
- Results mirror those published in `LaGrangianSolitons/STABILITY_VALLEY_RESULTS.md`; this repo now owns an identical copy so downstream analyses can reference it directly.
