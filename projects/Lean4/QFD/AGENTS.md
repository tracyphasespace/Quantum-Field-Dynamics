# Repository Guidelines

## Project Structure & Module Organization
Lean sources sit at the top level (`EmergentAlgebra.lean`, `SpectralGap.lean`, `ToyModel.lean`, etc.) and inside domain folders such as `Classical/`, `Relativity/`, `Computing/`, and `Test/`. Documentation and indices (`README.md`, `PROOF_INDEX.md`, `THEOREM_STATEMENTS.txt`, `BookContent/`) describe where every theorem belongs—update them whenever you introduce a file or move a result. Keep drafts in `archive/` or `sketches/`, but promote completed proofs into the main tree so `lake build QFD` exercises them. Temporary explorations go in `Test/TrivialProof.lean` until they become reusable lemmas.

## Build, Test, and Development Commands
- `cd /home/tracy/development/QFD_SpectralGap && lake update && lake build QFD` — installs Mathlib, honors `leanprover/lean4:v4.27.0-rc1`, and compiles every module.
- `lake env lean --make SpectralGap.lean` — re-checks a single file while you iterate.
- `lake env lean --make Test/TrivialProof.lean` — compiles regression snippets before they enter the library.
Use `lake fmt` only if your local setup matches the project configuration; otherwise rely on editor formatting to avoid churn.

## Coding Style & Naming Conventions
Follow Mathlib style: two-space indentation, `CamelCase` names for definitions and lemmas, and descriptive docstrings introduced with `/-! ... -/`. Prefer additive prefixes (`BivectorGenerator.ofMatrix`) over suffixes, and reuse domain terms already present (`StabilityOperator`, `HasQuantizedTopology`). Keep imports minimal, expose helper lemmas with `@[simp]` only when they decrease proof search, and explain complex tactic blocks with a short comment rather than inlined prose.

## Testing Guidelines
Successful compilation is the regression suite, so do not merge with `sorry`. Add `example` blocks near critical lemmas, then mirror long-running checks inside `Test/TrivialProof.lean` so reviewers can execute them independently. When modifying foundational files such as `EmergentAlgebra.lean` or `SpectralGap.lean`, run a fresh `lake build QFD` and note which downstream files were touched in your PR description. Every new structure should be accompanied by at least one lemma demonstrating non-trivial behavior (bounds, invariants, or simp-normal forms).

## Commit & Pull Request Guidelines
Git history favors short, imperative subjects with optional prefixes (`docs: Update repository statistics…`, `Add Harmonic Nuclear Model…`). Adopt the same style, keep the first line under 72 characters, and mention affected modules in the body. Pull requests should summarize the theorem or operator you added, cite documentation you updated (`PROOF_INDEX.md`, `GRAND_SOLVER_ARCHITECTURE.md`, etc.), and list the commands you ran (`lake build QFD`, targeted `lean --make`). Flag breaking API changes, new dependencies, or expected warnings explicitly so automation and reviewers can reproduce your environment quickly.
