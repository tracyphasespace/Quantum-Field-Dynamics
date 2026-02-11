# Using PhysLean in this Project

The `PhysLean` library has been successfully added as a dependency to this project. This document provides instructions for Code CLI tools (and human developers) on how to utilize it.

## 1. Accessing PhysLean Modules

To use modules from `PhysLean` in your Lean 4 `.lean` files, simply use the `import` command. The base module name is `PhysLean`.

Example:
```lean
import PhysLean.Physics.QuantumMechanics -- To import Quantum Mechanics formalizations
import PhysLean.Classical.Maxwell -- To import Maxwell's equations
```
You can explore the `PhysLean` repository structure to find specific modules.

## 2. Documentation

For detailed documentation on the contents of `PhysLean`, including available definitions, theorems, and examples, please refer to:

-   **PhysLean GitHub Repository:** [https://github.com/HEPLean/PhysLean](https://github.com/HEPLean/PhysLean)
    The `README.md` file on the GitHub repository is the primary source of high-level documentation and entry points.

You may need to clone the repository locally (outside this project's dependency structure, for browsing purposes) or use a web browser to navigate the directory structure on GitHub to discover specific file paths.

## 3. Common Usage for Code CLI Tools

When formalizing new physics concepts or verifying existing ones:

-   **Search:** Use `search_file_content` or `grep` within the `_lake/packages/PhysLean` directory to find relevant definitions (`def`), theorems (`theorem`), or axioms (`axiom`).
    Example: `search_file_content --dir_path _lake/packages/PhysLean "QuantumHarmonicOscillator"`
-   **Import:** Once a relevant module is identified, add the corresponding `import` statement to your `.lean` file.
-   **Utilize:** Directly `apply`, `exact`, `rw`, or `simp` theorems and definitions from `PhysLean` in your proofs.

This integration allows for the direct utilization of `PhysLean`'s formalized physics within your project's Lean 4 codebase.
