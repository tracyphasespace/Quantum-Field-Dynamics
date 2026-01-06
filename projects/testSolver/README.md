# Test Solver Experiments

## Context
- Mirror of `schema/v0/experiments` copied to `projects/testSolver` so we can run tests without touching locked schema tree.
- Goal: run the "Grand Solver" configuration (`grand_solver_lockdown.json`) plus other RunSpecs via `schema/v0/solve.py`.

## Current blockers
1. `schema/v0/solve.py` expects the schema at `schema/v0/schema/RunSpec.schema.json`, but only `schema/v0/RunSpec.schema.json` exists. Result: `FileNotFoundError` when launching the solver.
2. Need either to (a) adjust `solve.py` to look up the correct path or (b) create `schema/v0/schema/RunSpec.schema.json` (copy of the top-level schema) before running.

## Next steps
- Decide whether to duplicate the schema file or modify `solve.py` to reference the existing location.
- After schema path is fixed, re-run: `python3 schema/v0/solve.py projects/testSolver/grand_solver_lockdown.json`.
- Capture outputs under `projects/testSolver/results/` (create directory if needed) so they stay isolated from the main schema tree.

## Notes
- Working directory: `/home/tracy/development/QFD_SpectralGap`.
- Sandbox: workspace-write; network restricted.
