# PLAN.md – Lean4/QFD Documentation & Validation Roadmap

## 1. Consolidate historical status files
- Archive the contents of `archive/historical_status/*.md` into a single, dated summary (e.g., `STATUS_LOG.md`).
- Keep only the most recent snapshot for each topic (charge, gravity, neutrino, etc.) in the active docs.
- Record which files were merged and where their content now lives.

## 2. Update active documentation
- `README.md` / `VALIDATION_SUMMARY.md`: clearly distinguish **inputs** (α, c₁, c₂, R) from **derived checks** (spin, V₄, λ ≈ mₚ).
- Add a “Transparency” appendix summarizing β/ξ/α_circ provenance.
- Replace “validated first-principles theory” wording with “current status: model reproduces observations once parameters are fixed.”

## 3. Nuclide fit hold-out procedure
- Implement the 1% hold-out test inside `nuclide-prediction`: fit c₁,c₂ on A ≥ cut and evaluate residuals on the lowest A rows; include summary in the output artifacts.
- Document the hold-out results in the main release notes (proton not used during calibration).

## 4. Lepton model reproducibility
- Check in the Compton-scale correction notebook / script used for Stage 2 (β-ξ-τ) so others can rerun the MCMC.
- Automate generation of the spin table and g−2 consistency plots.
- Add tests that fail if ξ, τ drift far from ~1 when Compton scaling is applied.

## 5. Clarify g−2 discussion
- Reword g−2 sections to “matches experiment when ξ ≈ 1” rather than “predicts with no free parameters.”
- Note explicitly that α_circ was calibrated to the muon and converged to ≈ e/(2π), and list it as future work to derive directly.

## 6. Lean proof housekeeping
- Run `lake build` with `LEAN_CODING_GUIDE` checklist; note current sorry counts (Koide final step, etc.).
- Document which Lean files correspond to the active pipeline (e.g., CoreCompression, FineStructure) and which are historical.

## 7. Release packaging
- For v1.0-RC1, bundle only the current doc set: `README.md`, `RELEASE_NOTES_v1.0-RC1.md`, `NUCLEAR_ELECTRONIC_BRIDGE.md`, plus the transparency appendix.
- Move legacy marketing-oriented write-ups into `archive/historical_status` and link to the new consolidated log.

This plan keeps the necessary information while preventing future proliferation of redundant narrative files.
