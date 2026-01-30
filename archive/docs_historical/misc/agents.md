# Jules Policy for Quantum-Field-Dynamics

## Default Behavior
- Operate in **read-only mode** unless explicitly instructed otherwise.
- Do **not** install dependencies, run tests, or modify files by default.
- Never commit changes automatically without explicit approval.

## Shell-Only Tasks
- Any task beginning with `bash -lc` is a **shell-only task**.
- For shell-only tasks, run the exact command provided and return **stdout/stderr only**.
- Do not interpret, expand, or re-plan these commands into multi-step workflows.

## Code Edits
- Only propose or apply code changes when the user prompt includes the words:
  - **"edit"**, **"fix"**, **"commit"**, or **"modify"**.
- If no such keyword is present, do not attempt code edits.

## Tests
- Do not auto-run test suites (`pytest`, `unittest`, `run_tests.sh`, etc.) unless explicitly asked.

## Dependency Management
- Do not install Python packages, update pip, or create new environments unless explicitly asked.

## Communication
- When in doubt, ask for clarification before taking actions that modify the environment or repo.
