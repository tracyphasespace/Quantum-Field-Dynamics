# QFD Coupling Lockdown Plan

1. **Validate Nuclear Solver Inputs**
   - Run the Phase-9 SCF solver on a representative isotope set (light/mid/heavy).
   - Check virial, convergence, the new symmetry term, and calibrate the optional temporal penalty (see `qfd_solver_temporal.py`) for super-heavy masses if needed.

2. **Map Remaining Couplings**
   - For every Grand Solver parameter, specify whether it is:
     - Fixed by geometry/Logic Fortress.
     - Exported from the nuclear soliton solver.
     - Still unresolved.
   - Update the schema/run-specs accordingly.

3. **Cross-Sector Consistency Tests**
   - Re-run the gravity bridge, circulation proof, and quartic potential solver with the finalized constants.
   - Ensure the cross-check scripts (xi, alpha_circ, reverse potential) live under `projects/Lean4/projects/solvers` with reproducible output.

4. **Grand Solver Integration**
   - Feed the locked couplings into the master RunSpec / parameter files.
   - Demonstrate a Grand Solver run with no free fit parameters.
