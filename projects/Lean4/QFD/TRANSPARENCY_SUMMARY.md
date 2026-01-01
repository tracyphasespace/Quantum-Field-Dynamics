# QFD Lepton Soliton Model — Transparency Summary

## Inputs vs Outputs

| Quantity | Source | Notes |
|----------|--------|-------|
| Fine structure α | Experimental constant 1/137.035999 | Input to the Golden Loop |
| c₁, c₂ nuclear coefficients | Fit to NuMass.csv (5 842 nuclides) | Produced by `projects/particle-physics/nuclide-prediction` |
| β = 3.058230856 | Derived in `FineStructure.lean` from α, c₁, c₂ | Depends on the global nuclide fit |
| ξ, τ | Inferred by fitting Hill-vortex energy functional to lepton masses (Stage 2 MCMC) | ≈ 1 after Compton-scale correction |
| α_circ | Currently calibrated from the muon g−2 run; converges to ≈ e/(2π) | Goal: derive directly from geometry |

## Model summary

- Hill spherical vortex profile with Compton radius \(R = \hbar/(mc)\).
- Energy functional \(E = \int [\tfrac12 ξ \lvert\nablaρ\rvert^2 + β (δρ)^2 + τ (\partialρ/\partial t)^2]\,dV\) evaluated on the Hill profile (no self-consistent solver yet).
- Parameters (β, ξ, τ) are set as above; R is fixed by each lepton’s mass. The model reproduces \(L \approx \hbar/2\) at \(U \approx 0.876c\) — a consistency check rather than an independent prediction.

## What is calibrated vs checked

- **Spin:** Matches ℏ/2 after setting β, ξ, τ, and R; demonstrates self-consistency.
- **g−2:** \(V_4 = -ξ/β\) gives −0.327 when ξ = 1, β = 3.058 (Schwinger’s value). The muon anomaly match requires tuning α_circ. Neither is yet parameter-free.
- **Tau regime:** Requires higher-order terms (V₆); unresolved.

## Strengths

- α and β tie back to the nuclide scaling law, linking nuclear and lepton sectors.
- Once parameters are fixed, the model reproduces spin/anomaly magnitudes across generations, supporting the geometric soliton picture.

## Current gaps

- ξ and α_circ still come from fits; to claim predictive power they need independent derivations or constraints.
- Three parameters fitted to three masses → no hold-out validation in the mass sector.
- g−2 statements should be framed as consistency checks until the dependence on muon data is removed.

## Next steps

1. Update README/VALIDATION docs to spell out inputs vs derived checks and link to this summary.
2. Check in the Compton-scale correction + Stage 2 MCMC scripts so ξ ≈ 1 can be reproduced.
3. Anchor ξ (and α_circ) via another sector if possible, then treat g−2 as a prediction.
4. Rephrase the g−2 discussion: “matches experiment when ξ ≈ 1” instead of “predicts with no free parameters.”
