#!/usr/bin/env python3
"""
Lean4 ↔ JSON Schema Consistency Checker

Validates that JSON parameter definitions are consistent with Lean4 type-safe
schema definitions.

Usage:
    python check_lean_json_consistency.py [--strict]
    python check_lean_json_consistency.py experiments/ccl_fit_v1.json
    python check_lean_json_consistency.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class Dimensions:
    """Physical dimensions: [L^l M^m T^t Q^q]"""
    length: int
    mass: int
    time: int
    charge: int

    @staticmethod
    def parse_lean(lean_str: str) -> Optional['Dimensions']:
        """Parse Lean dimension notation: ⟨1, 0, -1, 0⟩"""
        match = re.search(r'⟨\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*⟩', lean_str)
        if match:
            return Dimensions(
                length=int(match.group(1)),
                mass=int(match.group(2)),
                time=int(match.group(3)),
                charge=int(match.group(4))
            )
        return None

    def to_string(self) -> str:
        """Convert to human-readable string"""
        parts = []
        if self.length != 0:
            parts.append(f"L^{self.length}" if self.length != 1 else "L")
        if self.mass != 0:
            parts.append(f"M^{self.mass}" if self.mass != 1 else "M")
        if self.time != 0:
            parts.append(f"T^{self.time}" if self.time != 1 else "T")
        if self.charge != 0:
            parts.append(f"Q^{self.charge}" if self.charge != 1 else "Q")
        return " ".join(parts) if parts else "dimensionless"

    def __eq__(self, other):
        return (self.length == other.length and
                self.mass == other.mass and
                self.time == other.time and
                self.charge == other.charge)


@dataclass
class LeanParameter:
    """Parameter from Lean4 schema"""
    name: str
    type_name: str  # Unitless, Energy, Mass, etc.
    dimensions: Optional[Dimensions]
    comment: str


@dataclass
class LeanConstraint:
    """Constraint from Lean4 schema"""
    param_name: str
    bounds: Optional[Tuple[float, float]]
    constraints: List[str]


@dataclass
class JsonParameter:
    """Parameter from JSON config"""
    name: str
    value: float
    role: str
    bounds: Optional[Tuple[float, float]]
    units: str
    frozen: bool


@dataclass
class ConsistencyIssue:
    """A consistency problem between Lean4 and JSON"""
    severity: str  # error, warning, info
    category: str
    message: str
    lean_detail: Optional[str] = None
    json_detail: Optional[str] = None


# ----------------------------
# Lean4 Parser
# ----------------------------

class LeanSchemaParser:
    """Parse Lean4 schema files to extract parameters and constraints"""

    def __init__(self, lean_dir: Path):
        self.lean_dir = lean_dir

    def parse_couplings(self) -> Dict[str, List[LeanParameter]]:
        """Parse QFD/Schema/Couplings.lean"""
        couplings_file = self.lean_dir / "QFD" / "Schema" / "Couplings.lean"
        if not couplings_file.exists():
            raise FileNotFoundError(f"Lean4 schema not found: {couplings_file}")

        with open(couplings_file) as f:
            content = f.read()

        # Parse parameter structures
        structures = {}

        # Find all structure definitions
        struct_pattern = r'structure\s+(\w+Params)\s+where\s+(.*?)(?=\nstructure|\nend\s|$)'
        for match in re.finditer(struct_pattern, content, re.DOTALL):
            struct_name = match.group(1)
            struct_body = match.group(2)

            params = []
            # Parse each parameter line: name : Type -- comment
            param_pattern = r'(\w+)\s*:\s*(\w+(?:\s*⟨[^⟩]+⟩)?)\s*(?:--\s*(.*))?'
            for param_match in re.finditer(param_pattern, struct_body):
                name = param_match.group(1)
                type_str = param_match.group(2).strip()
                comment = param_match.group(3).strip() if param_match.group(3) else ""

                # Extract dimensions if present
                dims = Dimensions.parse_lean(type_str)

                params.append(LeanParameter(
                    name=name,
                    type_name=type_str.split()[0] if ' ' in type_str else type_str,
                    dimensions=dims,
                    comment=comment
                ))

            structures[struct_name] = params

        return structures

    def parse_constraints(self) -> Dict[str, List[LeanConstraint]]:
        """Parse QFD/Schema/Constraints.lean"""
        constraints_file = self.lean_dir / "QFD" / "Schema" / "Constraints.lean"
        if not constraints_file.exists():
            return {}

        with open(constraints_file) as f:
            content = f.read()

        constraints = {}

        # Find all constraint structures
        struct_pattern = r'structure\s+(\w+Constraints)\s+\(p\s*:\s*(\w+Params)\)\s*:\s*Prop\s+where\s+(.*?)(?=\nstructure|\ndef\s|\nend\s|$)'
        for match in re.finditer(struct_pattern, content, re.DOTALL):
            constraint_name = match.group(1)
            param_struct = match.group(2)
            constraint_body = match.group(3)

            param_constraints = {}

            # Parse range constraints: param_range : lo < p.param.val ∧ p.param.val < hi
            range_pattern = r'(\w+)_range\s*:\s*([0-9.e+-]+)\s*<\s*p\.(\w+)\.val\s*∧\s*p\.\w+\.val\s*<\s*([0-9.e+-]+)'
            for range_match in re.finditer(range_pattern, constraint_body):
                param_name = range_match.group(3)
                lo = float(range_match.group(2))
                hi = float(range_match.group(4))

                if param_name not in param_constraints:
                    param_constraints[param_name] = LeanConstraint(
                        param_name=param_name,
                        bounds=(lo, hi),
                        constraints=[]
                    )
                else:
                    param_constraints[param_name].bounds = (lo, hi)

            # Parse positivity constraints: param_positive : p.param.val > 0
            pos_pattern = r'(\w+)_positive\s*:\s*p\.(\w+)\.val\s*>\s*0'
            for pos_match in re.finditer(pos_pattern, constraint_body):
                param_name = pos_match.group(2)
                if param_name not in param_constraints:
                    param_constraints[param_name] = LeanConstraint(
                        param_name=param_name,
                        bounds=None,
                        constraints=["positive"]
                    )
                else:
                    param_constraints[param_name].constraints.append("positive")

            constraints[param_struct] = list(param_constraints.values())

        return constraints


# ----------------------------
# JSON Parser
# ----------------------------

class JsonSchemaParser:
    """Parse JSON experiment configs to extract parameters"""

    def __init__(self, json_path: Path):
        self.json_path = json_path

    def parse_parameters(self) -> List[JsonParameter]:
        """Parse parameters from JSON config"""
        with open(self.json_path) as f:
            config = json.load(f)

        params = []
        for p in config.get("parameters", []):
            # Skip if missing required fields (might be a reference)
            if not isinstance(p, dict) or "name" not in p or "role" not in p:
                continue

            bounds = None
            if "bounds" in p and p["bounds"]:
                b = p["bounds"]
                # Handle both list [min, max] and dict {"min": x, "max": y}
                if isinstance(b, list):
                    bounds = (b[0], b[1])
                elif isinstance(b, dict):
                    bounds = (b.get("min"), b.get("max"))

            # Use init value if value not specified
            value = p.get("value", p.get("init", 0.0))

            params.append(JsonParameter(
                name=p["name"],
                value=value,
                role=p["role"],
                bounds=bounds,
                units=p.get("units", ""),
                frozen=p.get("frozen", False)
            ))

        return params


# ----------------------------
# Consistency Checker
# ----------------------------

class ConsistencyChecker:
    """Check consistency between Lean4 and JSON schemas"""

    def __init__(self, lean_dir: Path, strict: bool = False):
        self.lean_parser = LeanSchemaParser(lean_dir)
        self.strict = strict
        self.issues: List[ConsistencyIssue] = []

    def check_json_config(self, json_path: Path) -> List[ConsistencyIssue]:
        """Check a single JSON config against Lean4 schema"""
        self.issues = []

        # Parse Lean4 schema
        lean_params = self.lean_parser.parse_couplings()
        lean_constraints = self.lean_parser.parse_constraints()

        # Parse JSON config
        json_parser = JsonSchemaParser(json_path)
        json_params = json_parser.parse_parameters()

        # Build lookup tables
        all_lean_params = {}
        for struct_name, params in lean_params.items():
            for param in params:
                # Map nuclear.c1 -> c1, etc.
                all_lean_params[param.name] = param

        json_param_map = {p.name.split('.')[-1]: p for p in json_params}

        # Check 1: All JSON params exist in Lean
        self._check_json_params_in_lean(json_params, all_lean_params)

        # Check 2: Bounds consistency
        self._check_bounds_consistency(json_params, all_lean_params, lean_constraints)

        # Check 3: Units consistency
        self._check_units_consistency(json_params, all_lean_params)

        # Check 4: Missing Lean params in JSON (warning only)
        self._check_lean_coverage(all_lean_params, json_param_map)

        return self.issues

    def _check_json_params_in_lean(self, json_params: List[JsonParameter], lean_params: Dict[str, LeanParameter]):
        """Verify all JSON parameters exist in Lean schema"""
        for jp in json_params:
            # Extract short name (nuclear.c1 -> c1)
            short_name = jp.name.split('.')[-1]

            # Allow nuisance/calibration parameters to be JSON-only
            # These are dataset-specific terms, not fundamental couplings
            if jp.role in ["nuisance", "calibration"] and short_name not in lean_params:
                self.issues.append(ConsistencyIssue(
                    severity="info",
                    category="json_only_nuisance",
                    message=f"JSON parameter '{jp.name}' is nuisance/calibration only (not in Lean)",
                    json_detail=f"role={jp.role}, units={jp.units}"
                ))
                continue

            if short_name not in lean_params:
                self.issues.append(ConsistencyIssue(
                    severity="error",
                    category="missing_lean_definition",
                    message=f"JSON parameter '{jp.name}' not found in Lean4 schema",
                    json_detail=f"role={jp.role}, units={jp.units}"
                ))

    def _check_bounds_consistency(self, json_params: List[JsonParameter],
                                    lean_params: Dict[str, LeanParameter],
                                    lean_constraints: Dict[str, List[LeanConstraint]]):
        """Check that JSON bounds contain Lean bounds"""

        # Build constraint lookup
        all_constraints = {}
        for struct_name, constraints in lean_constraints.items():
            for c in constraints:
                all_constraints[c.param_name] = c

        for jp in json_params:
            short_name = jp.name.split('.')[-1]

            if short_name not in all_constraints:
                continue  # No Lean constraints for this param

            lean_c = all_constraints[short_name]

            if lean_c.bounds is None:
                continue  # No specific bounds in Lean

            lean_lo, lean_hi = lean_c.bounds

            if jp.bounds is None:
                self.issues.append(ConsistencyIssue(
                    severity="warning",
                    category="missing_bounds",
                    message=f"Parameter '{jp.name}' has Lean bounds but no JSON bounds",
                    lean_detail=f"Lean: ({lean_lo}, {lean_hi})",
                    json_detail="JSON: no bounds"
                ))
                continue

            json_lo, json_hi = jp.bounds

            # Check containment: JSON bounds should contain Lean bounds
            if json_lo > lean_lo or json_hi < lean_hi:
                severity = "error" if self.strict else "warning"
                self.issues.append(ConsistencyIssue(
                    severity=severity,
                    category="bounds_mismatch",
                    message=f"Parameter '{jp.name}': JSON bounds do not contain Lean bounds",
                    lean_detail=f"Lean: ({lean_lo}, {lean_hi})",
                    json_detail=f"JSON: [{json_lo}, {json_hi}]"
                ))

            # Check if frozen param has unnecessarily wide bounds
            if jp.frozen and (json_lo != json_hi):
                self.issues.append(ConsistencyIssue(
                    severity="info",
                    category="frozen_with_bounds",
                    message=f"Parameter '{jp.name}' is frozen but has non-trivial bounds",
                    json_detail=f"frozen=true, bounds=[{json_lo}, {json_hi}]"
                ))

    def _check_units_consistency(self, json_params: List[JsonParameter], lean_params: Dict[str, LeanParameter]):
        """Check units naming consistency"""

        # Units mapping
        lean_to_json = {
            "Unitless": ["dimensionless", "unitless"],
            "Energy": ["eV", "MeV", "GeV", "J"],
            "Mass": ["kg", "eV", "MeV"],  # Mass can be in eV in particle physics
            "Length": ["m", "fm", "km"],
            "Time": ["s", "ms"],
            "Density": ["kg/m^3"],
            "Velocity": ["m/s", "km/s", "km/s/Mpc"]
        }

        for jp in json_params:
            short_name = jp.name.split('.')[-1]

            if short_name not in lean_params:
                continue

            lp = lean_params[short_name]

            # Check if JSON units match Lean type
            if lp.type_name in lean_to_json:
                expected_units = lean_to_json[lp.type_name]

                if jp.units not in expected_units:
                    # Check for common synonyms
                    if lp.type_name == "Unitless" and jp.units in ["dimensionless", "unitless"]:
                        # This is fine, but warn about inconsistency
                        self.issues.append(ConsistencyIssue(
                            severity="info",
                            category="units_naming",
                            message=f"Parameter '{jp.name}': Units naming inconsistency",
                            lean_detail=f"Lean: {lp.type_name}",
                            json_detail=f"JSON: '{jp.units}' (recommend 'dimensionless')"
                        ))
                    else:
                        self.issues.append(ConsistencyIssue(
                            severity="warning",
                            category="units_mismatch",
                            message=f"Parameter '{jp.name}': Units may not match Lean type",
                            lean_detail=f"Lean type: {lp.type_name} (expects: {expected_units})",
                            json_detail=f"JSON units: '{jp.units}'"
                        ))

    def _check_lean_coverage(self, lean_params: Dict[str, LeanParameter], json_param_map: Dict[str, JsonParameter]):
        """Check if all Lean params are in JSON (warning only)"""

        missing = []
        for name, lp in lean_params.items():
            if name not in json_param_map:
                missing.append(name)

        if missing and len(missing) < len(lean_params):  # Don't warn if JSON is empty
            self.issues.append(ConsistencyIssue(
                severity="info",
                category="incomplete_json",
                message=f"JSON config does not include all Lean parameters",
                lean_detail=f"Missing: {', '.join(missing)}"
            ))


# ----------------------------
# Reporter
# ----------------------------

class ConsistencyReporter:
    """Format and display consistency issues"""

    def __init__(self, issues: List[ConsistencyIssue]):
        self.issues = issues

    def print_report(self):
        """Print human-readable report"""
        if not self.issues:
            print("✅ All consistency checks passed!")
            return

        # Group by severity
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        info = [i for i in self.issues if i.severity == "info"]

        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for issue in errors:
                self._print_issue(issue)

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                self._print_issue(issue)

        if info:
            print(f"\nℹ️  INFO ({len(info)}):")
            for issue in info:
                self._print_issue(issue)

        # Summary
        print(f"\n{'='*60}")
        print(f"Summary: {len(errors)} errors, {len(warnings)} warnings, {len(info)} info")

        if errors:
            print("\n❌ FAILED: Critical consistency errors found")
            return 1
        elif warnings:
            print("\n⚠️  PASSED WITH WARNINGS: Some issues need attention")
            return 0
        else:
            print("\n✅ PASSED: All checks passed")
            return 0

    def _print_issue(self, issue: ConsistencyIssue):
        """Print a single issue"""
        print(f"\n  [{issue.category}] {issue.message}")
        if issue.lean_detail:
            print(f"    Lean4: {issue.lean_detail}")
        if issue.json_detail:
            print(f"    JSON:  {issue.json_detail}")


# ----------------------------
# CLI
# ----------------------------

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Check Lean4 ↔ JSON schema consistency")
    parser.add_argument("config", nargs="?",
                       help="JSON config file to check (default: experiments/ccl_fit_v1.json)")
    parser.add_argument("--strict", action="store_true",
                       help="Treat bounds mismatches as errors instead of warnings")
    parser.add_argument("--all", action="store_true",
                       help="Check all JSON files in experiments/ and examples/")
    parser.add_argument("--lean-dir", type=Path,
                       default=Path(__file__).parent.parent.parent / "projects" / "Lean4",
                       help="Path to Lean4 project root")

    args = parser.parse_args(argv[1:])

    # Determine which configs to check
    configs = []
    if args.all:
        schema_dir = Path(__file__).parent
        for pattern in ["experiments/*.json", "examples/*.json"]:
            configs.extend(schema_dir.glob(pattern))
    elif args.config:
        configs = [Path(args.config)]
    else:
        # Default
        configs = [Path(__file__).parent / "experiments" / "ccl_fit_v1.json"]

    if not configs:
        print("No config files found to check", file=sys.stderr)
        return 2

    # Check each config
    checker = ConsistencyChecker(args.lean_dir, strict=args.strict)

    all_issues = []
    for config in configs:
        if not config.exists():
            print(f"Config not found: {config}", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f"Checking: {config.name}")
        print('='*60)

        issues = checker.check_json_config(config)
        all_issues.extend(issues)

        reporter = ConsistencyReporter(issues)
        reporter.print_report()

    return 1 if any(i.severity == "error" for i in all_issues) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
