#!/usr/bin/env python3
"""
CCL Constraint Validator

Validates that JSON RunSpec parameter bounds are consistent with
the proven Lean 4 constraints in QFD/Nuclear/CoreCompressionLaw.lean.

This is the bridge between theory and implementation - ensuring that
the solver only searches within the mathematically valid parameter space.
"""

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

class CCLConstraintValidator:
    """
    Validates CCL parameters against proven Lean 4 bounds.

    Proven bounds from CoreCompressionLaw.lean:
      c1 ∈ (0.0, 1.5)     [Surface tension positivity & Coulomb balance]
      c2 ∈ [0.2, 0.5]     [Volume packing fraction]
    """

    # Proven bounds from Lean 4
    PROVEN_BOUNDS = {
        "c1": {"lower": 0.0, "upper": 1.5, "lower_inclusive": False, "upper_inclusive": False},
        "c2": {"lower": 0.2, "upper": 0.5, "lower_inclusive": True, "upper_inclusive": True}
    }

    def __init__(self, lean_file: Optional[Path] = None):
        """
        Initialize validator.

        Args:
            lean_file: Path to CoreCompressionLaw.lean (for extracting bounds dynamically)
        """
        self.lean_file = lean_file or Path("projects/Lean4/QFD/Nuclear/CoreCompressionLaw.lean")
        self.proven_bounds = self.PROVEN_BOUNDS.copy()

    def parse_lean_bounds(self) -> Dict[str, Dict]:
        """
        Parse proven bounds from Lean 4 source code.

        Extracts constraints from structure CCLConstraints:
          c1_positive : p.c1.val > 0.0
          c1_bounded : p.c1.val < 1.5
          c2_lower : p.c2.val ≥ 0.2
          c2_upper : p.c2.val ≤ 0.5

        Returns:
            Dictionary of parameter bounds
        """
        if not self.lean_file.exists():
            print(f"⚠️  Lean file not found: {self.lean_file}")
            print(f"   Using hardcoded proven bounds")
            return self.proven_bounds

        with open(self.lean_file) as f:
            content = f.read()

        # Extract CCLConstraints structure
        constraints_match = re.search(
            r'structure CCLConstraints.*?where(.*?)(?:^/-!|\n\n)',
            content,
            re.MULTILINE | re.DOTALL
        )

        if not constraints_match:
            print(f"⚠️  Could not parse CCLConstraints from {self.lean_file}")
            return self.proven_bounds

        constraints_text = constraints_match.group(1)

        # Parse bounds
        bounds = {}

        # c1 bounds
        c1_lower_match = re.search(r'p\.c1\.val\s*>\s*([0-9.]+)', constraints_text)
        c1_upper_match = re.search(r'p\.c1\.val\s*<\s*([0-9.]+)', constraints_text)

        if c1_lower_match and c1_upper_match:
            bounds["c1"] = {
                "lower": float(c1_lower_match.group(1)),
                "upper": float(c1_upper_match.group(1)),
                "lower_inclusive": False,  # > is strict
                "upper_inclusive": False   # < is strict
            }

        # c2 bounds
        c2_lower_match = re.search(r'p\.c2\.val\s*≥\s*([0-9.]+)', constraints_text)
        c2_upper_match = re.search(r'p\.c2\.val\s*≤\s*([0-9.]+)', constraints_text)

        if c2_lower_match and c2_upper_match:
            bounds["c2"] = {
                "lower": float(c2_lower_match.group(1)),
                "upper": float(c2_upper_match.group(1)),
                "lower_inclusive": True,   # ≥ is inclusive
                "upper_inclusive": True    # ≤ is inclusive
            }

        print(f"✅ Parsed proven bounds from {self.lean_file}")
        return bounds if bounds else self.proven_bounds

    def validate_param_bounds(
        self,
        param_name: str,
        json_bounds: Tuple[float, float],
        proven_bounds: Dict
    ) -> Tuple[bool, str]:
        """
        Validate that JSON bounds are a subset of proven bounds.

        Args:
            param_name: Parameter name (e.g., "c1", "c2")
            json_bounds: (lower, upper) from JSON config
            proven_bounds: Proven bounds from Lean 4

        Returns:
            (is_valid, message) tuple
        """
        json_lower, json_upper = json_bounds

        proven_lower = proven_bounds["lower"]
        proven_upper = proven_bounds["upper"]
        lower_incl = proven_bounds["lower_inclusive"]
        upper_incl = proven_bounds["upper_inclusive"]

        # Check if JSON bounds are within proven bounds
        valid_lower = (json_lower > proven_lower if not lower_incl else json_lower >= proven_lower)
        valid_upper = (json_upper < proven_upper if not upper_incl else json_upper <= proven_upper)

        if not valid_lower or not valid_upper:
            lower_op = "≥" if lower_incl else ">"
            upper_op = "≤" if upper_incl else "<"

            msg = (f"{param_name}: JSON [{json_lower}, {json_upper}] violates "
                   f"proven bounds ({proven_lower} {lower_op} {param_name} {upper_op} {proven_upper})")
            return False, msg

        return True, f"{param_name}: ✓ [{json_lower}, {json_upper}] ⊆ proven bounds"

    def validate_runspec(self, runspec_path: Path) -> bool:
        """
        Validate a RunSpec JSON file against proven Lean constraints.

        Args:
            runspec_path: Path to RunSpec JSON file

        Returns:
            True if valid, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Validating RunSpec against Lean 4 proven constraints")
        print(f"{'='*60}")
        print(f"RunSpec: {runspec_path}")
        print(f"Lean proof: {self.lean_file}")

        # Load JSON
        with open(runspec_path) as f:
            runspec = json.load(f)

        # Parse Lean bounds
        proven_bounds = self.parse_lean_bounds()

        print(f"\nProven bounds from Lean 4:")
        for param, bounds in proven_bounds.items():
            lower_op = "≥" if bounds["lower_inclusive"] else ">"
            upper_op = "≤" if bounds["upper_inclusive"] else "<"
            print(f"  {param}: {bounds['lower']} {lower_op} {param} {upper_op} {bounds['upper']}")

        # Validate each parameter
        print(f"\nValidating JSON parameters:")
        all_valid = True

        for param_spec in runspec.get("parameters", []):
            param_name = param_spec["name"].split(".")[-1]  # Get "c1" from "nuclear.c1"

            if param_name in proven_bounds:
                json_bounds = param_spec.get("bounds")

                if json_bounds is None:
                    print(f"  ⚠️  {param_name}: No bounds specified in JSON")
                    continue

                is_valid, msg = self.validate_param_bounds(
                    param_name,
                    tuple(json_bounds),
                    proven_bounds[param_name]
                )

                print(f"  {msg}")
                all_valid = all_valid and is_valid

        # Check if fitted values satisfy constraints
        if "fit" in runspec or runspec_path.parent.name.startswith("exp_"):
            # This is a results file, check fitted values
            print(f"\nChecking fitted values:")

            # Try to load results_summary.json if this is an experiment directory
            results_dir = runspec_path.parent
            if results_dir.name.startswith("exp_"):
                summary_path = results_dir / "results_summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        results = json.load(f)

                    fitted_params = results.get("fit", {}).get("params_best", {})

                    for param_full, value in fitted_params.items():
                        param_name = param_full.split(".")[-1]

                        if param_name in proven_bounds:
                            bounds = proven_bounds[param_name]
                            lower_op = "≥" if bounds["lower_inclusive"] else ">"
                            upper_op = "≤" if bounds["upper_inclusive"] else "<"

                            lower_ok = (value > bounds["lower"] if not bounds["lower_inclusive"]
                                        else value >= bounds["lower"])
                            upper_ok = (value < bounds["upper"] if not bounds["upper_inclusive"]
                                        else value <= bounds["upper"])

                            if lower_ok and upper_ok:
                                print(f"  ✅ {param_name} = {value:.6f} satisfies proven constraints")
                            else:
                                print(f"  ❌ {param_name} = {value:.6f} VIOLATES proven constraints!")
                                print(f"     Required: {bounds['lower']} {lower_op} {param_name} {upper_op} {bounds['upper']}")
                                all_valid = False

        print(f"\n{'='*60}")
        if all_valid:
            print(f"✅ VALIDATION PASSED")
            print(f"   All parameters satisfy proven Lean 4 constraints")
        else:
            print(f"❌ VALIDATION FAILED")
            print(f"   Some parameters violate proven constraints")
        print(f"{'='*60}\n")

        return all_valid


def main():
    """Validate CCL RunSpecs against Lean 4 proven constraints."""
    import sys

    validator = CCLConstraintValidator()

    # Validate Phase 1 results
    phase1_runspec = Path("schema/v0/experiments/ccl_ame2020_production.json")
    phase1_results = Path("results/exp_2025_ccl_ame2020_production")

    if phase1_runspec.exists():
        print("\n" + "="*60)
        print("PHASE 1 VALIDATION")
        print("="*60)
        valid = validator.validate_runspec(phase1_runspec)

        if not valid:
            sys.exit(1)

    # Check if additional RunSpecs specified on command line
    if len(sys.argv) > 1:
        for runspec_path in sys.argv[1:]:
            path = Path(runspec_path)
            if path.exists():
                validator.validate_runspec(path)
            else:
                print(f"⚠️  RunSpec not found: {runspec_path}")


if __name__ == "__main__":
    main()
