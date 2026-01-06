# Complete Energy Functional - MCMC Implementation

**Goal**: Resolve 3% β offset (V22: β≈3.15 vs Golden Loop: β=3.058) by including gradient density and emergent time terms.

---

## Quick Start

### Installation

```bash
cd /home/tracy/development/QFD_SpectralGap/complete_energy_functional

# Install dependencies
pip install numpy scipy emcee corner matplotlib h5py
```

### Run Quick Test (5 minutes)

```bash
python -c "from mcmc_stage1_gradient import quick_test; quick_test(n_steps=100)"
```

### Run Full Stage 1 MCMC (~5 hours)

```python
from mcmc_stage1_gradient import run_stage1_mcmc, analyze_stage1_results

# Run MCMC
samples, sampler = run_stage1_mcmc(
    n_walkers=44,      # 4× n_dim for good mixing
    n_steps=10000,     # Production steps
    n_burn=2000,       # Burn-in steps
    n_cores=8,         # Parallel cores
    output_file='results/stage1_chains.h5'
)

# Analyze results
results = analyze_stage1_results(samples, sampler, output_dir='results')

# Key question: Did β → 3.058?
print(f"β posterior: {results['β']['median']:.4f} ± {results['β']['std']:.4f}")
print(f"Target:      3.0580")
```

---

## Module Structure

```
complete_energy_functional/
├── __init__.py                 # Package initialization
├── functionals.py              # Energy functional implementations
├── solvers.py                  # Variational solvers (Euler-Lagrange)
├── mcmc_stage1_gradient.py     # Stage 1: Gradient-only MCMC
├── mcmc_stage2_temporal.py     # Stage 2: Add temporal term (TODO)
├── mcmc_stage3_em.py           # Stage 3: Full EM functional (TODO)
├── priors.py                   # Prior specifications (TODO)
├── likelihoods.py              # Likelihood functions (TODO)
├── convergence.py              # Diagnostics and plots (TODO)
└── results/                    # Output directory
    ├── stage1_chains.h5        # HDF5 chains from emcee
    ├── stage1_results.json     # Summary statistics
    ├── stage1_corner_xi_beta.png
    └── stage1_traces.png
```

---

## Energy Functional Hierarchy

### Level 0: V22 Baseline (Current)
```
E = ∫ β(δρ)² dV
```
**Result**: β ≈ 3.15 (3% offset)

### Level 1: Gradient Density (Stage 1)
```
E = ∫ [½ξ|∇ρ|² + β(δρ)²] dV
```
**New parameter**: ξ (gradient stiffness)
**Goal**: Test if ξ term resolves β offset

### Level 2: Emergent Time (Stage 2)
```
E = ∫ [½ξ|∇ρ|² + β(δρ)² + τ(∂ρ/∂t)²] dV
```
**New parameter**: τ (temporal stiffness)
**Goal**: Include time evolution from Cl(3,3)→Cl(3,1)

### Level 3: Full EM Functional (Stage 3)
```
E = ∫ [½ξ|∇ρ|² + β(δρ)² + τ(∂ρ/∂t)² + E_EM[ρ]] dV
```
**New term**: E_EM from first principles (Appendix G)
**Goal**: Eliminate empirical C_μ normalization

---

## Parameters

### Shared (Cross-Lepton)
- **ξ**: Gradient stiffness (dimensionless, ~1)
- **β**: Vacuum stiffness (dimensionless, target=3.058)
- **τ**: Temporal stiffness (Stage 2, dimensionless, ~1)

### Per-Lepton (3 sets for e, μ, τ)
- **R**: Vortex radius (meters, ~10⁻¹³)
- **U**: Circulation velocity (fraction of c, 0.1-0.9)
- **A**: Amplitude normalization (dimensionless)

### Total Dimensions
- Stage 1: 11D (ξ, β, R×3, U×3, A×3)
- Stage 2: 12D (+ τ)
- Stage 3: 12-13D (+ γ_EM, depending on implementation)

---

## Expected Outcomes

### Scenario 1: Gradient Term Resolves Offset ✓
- **Result**: β_posterior ≈ 3.058 ± 0.02
- **Interpretation**: V22 offset was due to missing ∇ρ term
- **Next**: Validate ξ value from first principles

### Scenario 2: Temporal Term Needed ⚠
- **Result**: Stage 1 shows β ≈ 3.1, Stage 2 → β ≈ 3.058
- **Interpretation**: Both gradient and time terms contribute
- **Next**: Search for breathing mode signatures

### Scenario 3: EM Functional Required ✗
- **Result**: Stages 1-2 insufficient
- **Interpretation**: Need Appendix G full EM response
- **Next**: Implement first-principles EM functional

### Scenario 4: Multiple Minima ⚠
- **Result**: MCMC finds multiple β values
- **Interpretation**: Fundamental degeneracy remains
- **Next**: Add independent observables (charge radius, g-2)

---

## Computational Requirements

### Stage 1: Gradient-Only
- **Parameters**: 11D
- **Walkers**: 44 (4× parameters)
- **Steps**: 10,000 + 2,000 burn-in
- **Runtime**: ~5 hours (8 cores)

### Memory
- **HDF5 chains**: ~200 MB
- **Working memory**: ~2 GB

### Optimization
- Parallel evaluation across walkers
- Cache Hill vortex solutions
- Use adaptive step sizes
- Consider surrogate model if too slow

---

## Convergence Diagnostics

### Gelman-Rubin R̂
- Run 4-8 independent chains
- Require R̂ < 1.01 for all parameters
- Compare within-chain vs between-chain variance

### Effective Sample Size (ESS)
- Account for autocorrelation
- Need ESS > 100 per parameter
- Check using `emcee.autocorr`

### Visual Checks
- Trace plots: mixing and stationarity
- Corner plots: degeneracies
- Acceptance fraction: 0.2-0.5 optimal

---

## Usage Examples

### Example 1: Check V22 Limit (ξ→0)

```python
from functionals import v22_baseline_functional
from solvers import hill_vortex_profile

# Should reproduce V22 result
β_v22 = 3.15
R, U, A = 1e-13, 0.5, 1.0

r = np.linspace(0, 10*R, 500)
ρ = hill_vortex_profile(r, R, U, A)

E_v22 = v22_baseline_functional(ρ, r, β_v22)
print(f"V22 energy: {E_v22:.6e}")
```

### Example 2: Test Gradient Term

```python
from functionals import gradient_energy_functional

ξ = 1.0  # Try gradient stiffness
β = 3.058  # Golden Loop value

E_total, E_grad, E_comp = gradient_energy_functional(ρ, r, ξ, β)

print(f"Gradient contribution: {E_grad:.6e} ({100*E_grad/E_total:.1f}%)")
print(f"Compression contribution: {E_comp:.6e} ({100*E_comp/E_total:.1f}%)")
```

### Example 3: Solve Euler-Lagrange

```python
from solvers import solve_euler_lagrange, integrate_energy

# Solve for equilibrium density
r, ρ_eq = solve_euler_lagrange(ξ=1.0, β=3.058, R=1e-13, U=0.5)

# Compute energy
E = integrate_energy(ξ=1.0, β=3.058, ρ=ρ_eq, r=r)
print(f"Equilibrium energy: {E:.6e}")
```

---

## Known Issues / TODO

### Current Limitations

1. **Unit Conversion**: Energy → mass conversion is placeholder
   - Need proper natural units ℏ=c=1
   - Convert Joules → MeV/c²

2. **Boundary Conditions**: Hill vortex matching needs refinement
   - Core boundary at r=R
   - Asymptotic decay to ρ_vac

3. **Solver Stability**: Relaxation may not converge for extreme parameters
   - Add adaptive relaxation parameter
   - Implement shooting method as backup

4. **Model Uncertainty**: σ_model is calibrated from V22
   - Should be self-consistent
   - May need iterative refinement

### Future Enhancements

- [ ] Implement Stage 2 (temporal term)
- [ ] Implement Stage 3 (EM functional from Appendix G)
- [ ] Add charge radius constraint
- [ ] Add magnetic moment observables
- [ ] Bayesian model comparison (evidence calculation)
- [ ] Surrogate model for fast evaluation
- [ ] Gradient-based MCMC (PyMC, NUTS)

---

## References

1. **V22_Lepton_Analysis/FINAL_STATUS_SUMMARY.md**
   - Current β ≈ 3.15 result
   - Identifiability analysis
   - Cross-lepton coupling

2. **COMPLETE_ENERGY_FUNCTIONAL.md**
   - Full theoretical framework
   - Parameter interpretation
   - Expected outcomes

3. **QFD/EmergentAlgebra.lean**
   - Cl(3,3) → Cl(3,1) emergence
   - Spacetime selection mechanism
   - Centralizer construction

4. **QFD Book Appendix Z.4**
   - Emergent time from internal rotor
   - Bivector symmetry breaking

5. **QFD Book Appendix G**
   - First-principles EM response
   - Magnetic moment derivation

---

## Support

For questions or issues:
1. Check COMPLETE_ENERGY_FUNCTIONAL.md for theoretical background
2. Run quick_test() to verify installation
3. Check HDF5 backend for chain corruption
4. Monitor memory usage during long runs

---

**Status**: Stage 1 ready for testing
**Next**: Run full MCMC to test gradient term hypothesis

---
