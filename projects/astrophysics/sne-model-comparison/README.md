# SNe Ia Model Comparison Framework

**Purpose**: Head-to-head comparison of QFD cosmological models against Pantheon+ supernova data.

## The Challenge

Our new Lean4-derived photon decay model predicts a different distance-redshift relationship than the older phenomenological models. We need to test both against real data.

## Models Under Test

### Model A: Phenomenological (Old)
- **Source**: `hubble_constant_validation.py`
- **Formula**: `Δm = α × z^β` where α=0.85, β=0.6 (fitted)
- **Basis**: Empirical fit to match ΛCDM
- **Status**: Known to work, but no physics derivation

### Model B: Lean4-Derived (New)
- **Source**: `hubble_validation_lean4.py`
- **Formula**: `ln(1+z) = κ×D` where κ = H₀/c
- **Basis**: Helicity-locked soliton decay (SolitonQuantization.lean)
- **Parameters**: β from Golden Loop, κ derived (not fitted)
- **Status**: Physics-based, needs validation

### Model C: J·A Interaction (v22)
- **Source**: `qfd-sn-v22/src/qfd_sn/cosmology.py`
- **Formula**: `D = z×c/k_J` with plasma veil `η'×z`
- **Basis**: Static spacetime with J·A energy loss
- **Status**: Independent QFD approach

### Model D: ΛCDM (Reference)
- **Formula**: Standard Friedmann with Ω_m=0.3, Ω_Λ=0.7
- **Basis**: Expanding universe with dark energy
- **Status**: Observational consensus model

## Data Source

**Pantheon+ Sample** (Scolnic et al. 2022)
- 1701 Type Ia supernovae
- Redshift range: 0.001 < z < 2.3
- Located in: `qfd-supernova-v15/data/pantheon_plus_*.csv`

## Key Predictions

| Model | Distance D(z) | Extra dimming |
|-------|--------------|---------------|
| A (Old) | D = z×c/H₀ | α×z^0.6 |
| B (Lean4) | D = ln(1+z)/κ | Built into D(z) |
| C (J·A) | D = z×c/k_J | η'×z |
| D (ΛCDM) | Friedmann integral | Built into D_L(z) |

## Comparison Metrics

1. **RMS residual**: `sqrt(mean((μ_obs - μ_model)²))`
2. **Reduced χ²**: `χ²/dof` (with measurement errors)
3. **Residual trend**: Is there z-dependent bias?
4. **Parameter count**: Fewer fitted parameters = better

## Running the Comparison

```bash
cd /home/tracy/development/QFD_SpectralGap/projects/astrophysics/sne-model-comparison
python3 scripts/compare_models.py
```

## Expected Outcomes

1. If Model B (Lean4) fits as well as Model A (Old):
   - Replace phenomenological with physics-based
   - Update all downstream code

2. If Model B fits worse:
   - Investigate why helicity-decay doesn't match
   - May need additional physics (scattering, etc.)

3. If Model C (J·A) provides best fit:
   - J·A interaction is preferred mechanism
   - Integrate with Lean4 formalism

## References

- GoldenLoop.lean: β derivation
- SolitonQuantization.lean: E = ℏω from helicity lock
- VacuumParameters.lean: λ = m_proton, stiffness parameters
- Pantheon+ paper: Scolnic et al. 2022, ApJ 938, 113
