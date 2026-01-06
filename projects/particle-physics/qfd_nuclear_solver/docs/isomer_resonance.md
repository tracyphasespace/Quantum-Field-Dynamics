## Isomer Resonance Sweep

- **Script**: `scripts/qfd_isomer_ladder.py`
- **Dataset**: `data/stable_nuclides_qfd.csv` (163 stable nuclides lifted from the LaGrangianSolitons suite)
- **Model**: Uses the parameter-free coefficients (`E_volume`, `E_surface`, `a_sym`) with **0.50 Coulomb shielding** and discrete isomer bonuses (`0.5×E_surface` per closure + `0.25×E_surface` pairing, boosted ×1.5 when both Z and N hit a node).

### Current Status

Running `python scripts/qfd_isomer_ladder.py` (discrete search over all integer `Z`) now yields:

- Mean |ΔZ| = **0.613** across 163 nuclides
- Median |ΔZ| = 0, Max |ΔZ| = 4
- Exact matches: **102/163 (62.6%)**
- Light nuclei: mean |ΔZ| = 0.28 (80% exact)
- Medium nuclei (40≤A<100): mean |ΔZ| = 0.74 (54% exact)
- Heavy band (100≤A<200): mean |ΔZ| = 0.84 (54% exact)
- Superheavy set (A≥200): 8/8 exact

This mirrors the “revitalized” dashboard trend from `NUCLIDE_SPACE_ANALYSIS.md`; remaining gaps are confined to a handful of mid-Z isotopes (e.g., Zn–Ge, Xe) where additional structure (e.g., refined closure list or variable bonuses) may still be needed.

### Next Steps

1. **Fine-tune closure table**: experiment with sub-closures (e.g., N=14, 32) or variable bonuses to lift the remaining Zn/Ge/Xe misses.
2. **Automate plots**: port the 2×2 diagnostic plot (`nuclide_space_exploration.png`) so this repo can regenerate the same success/failure maps.
3. **Clifford map prep**: gather the algebraic derivations showing how the resonance nodes (2, 8, 20, 28, 50, 82, 126) emerge from the Cl(3,3) / Cl(6,2) lattice; this will feed into the forthcoming “Clifford Isomer Map” document.
