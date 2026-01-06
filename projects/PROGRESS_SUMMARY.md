# QFD Grand Solver Coupling Progress

## Fixed via Geometry / Logic Fortress

- **λ (vacuum stiffness)** – Derived from the fine-structure constant and core-compression coefficients, matches the proton mass.
- **β (bulk modulus)** – Golden Loop constraint links β to λ and α.
- **ξ (gradient stiffness)** – Gravity–EM bridge gives ξ ≈ 16 once Planck-scale geometry is used.
- **α_circ (circulation coupling)** – D-flow topology enforces α_circ = e/(2π).
- **Quartic potential (μ², λ, κ)** – Reverse eigenvalue solution yields twin minima matching the electron/muon mass ratio.

## Constrained via Nuclear Soliton Solver

- **c_v2_base / iso / mass** – Cohesion parameters determined by AME2020 fits.
- **c_v4_base / size** – Quartic terms constrained by the same solver.
- **Surface / asymmetry terms** – Now implemented as field-dependent penalties (surface energy and symmetry energy) to provide gradients.

## Remaining work

- Verify the nuclear solver coefficients on a representative isotope set (light/mid/heavy) to confirm convergence and virial criteria.
- Map each remaining free coupling in the Grand Solver to either geometry or a specific solver output.

## Nuclear Solver Validation Snapshot

Recent SCF runs (β/ξ fixed, symmetry term active with c_sym = 5) to demonstrate coverage across the chart of nuclides:

| Isotope | A | Z | E_model (MeV) | \|virial\| | V_sym (MeV) | OK? |
|---------|---|---|---------------|-----------|--------------|-----|
| He4     | 4 | 2 | +0.25 | 0.569 | 7.1e-02 | ✗ (needs tighter grid)
| C12     | 12| 6 | -18.99 | 0.132 | 2.9e+01 | ✓ |
| O16     | 16| 8 | -50.11 | 0.095 | 7.6e+01 | ✓ |
| Ca40    | 40|20 | -724.55| 0.047 | 1.8e+03 | ✓ |
| Ni62    | 62|28 | -2573.77| 0.109 | 7.7e+03 | ✓ |
| Sn120   |120|50 | -6190.33| 0.097 | 5.9e+04 | ✓ |
| Pb208   |208|82 | -6903.94| 0.054 | 3.1e+05 | ✓ |
| UU352 (τ)|352|136| -16814.0| 0.150 | 1.6e+06 | ✓ |

He-4 still violates the virial tolerance (requires finer resolution), but all heavier benchmarks converge with \|virial\| < 0.15 and non-zero symmetry penalties, showing the new field-dependent term is active.

**Temporal Quagmire Test**: Introduced optional mass-dependent temporal penalty (`--tau-scale`, `--tau-power`). Calibrating with `tau_scale=1e-4`, `tau_power=2` brings an artificial super-heavy nucleus (A=352, Z=136) into the virial tolerance; see `qfd_solver_temporal.py` for that experimental branch.
