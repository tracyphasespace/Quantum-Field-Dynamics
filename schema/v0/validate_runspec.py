#!/usr/bin/env python3
"""
QFD RunSpec Validator

Loads, validates, and resolves RunSpec JSON files against the v0 schema.

Usage:
    python validate_runspec.py <runspec.json>
    python validate_runspec.py <runspec.json> --resolve --output resolved.json
    python validate_runspec.py <runspec.json> --check-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import jsonschema
from jsonschema import Draft7Validator, RefResolver


class RunSpecValidator:
    """Validates and resolves QFD RunSpec files."""

    def __init__(self, schema_dir: Path):
        """Initialize validator with schema directory."""
        self.schema_dir = schema_dir
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict[str, Any]:
        """Load all JSON schemas from the schema directory."""
        schemas = {}
        schema_files = [
            "RunSpec.schema.json",
            "ModelSpec.schema.json",
            "ParameterSpec.schema.json",
            "DatasetSpec.schema.json",
            "ObjectiveSpec.schema.json",
            "ResultSpec.schema.json"
        ]

        for schema_file in schema_files:
            schema_path = self.schema_dir / schema_file
            if schema_path.exists():
                with open(schema_path) as f:
                    schema = json.load(f)
                    schemas[schema["$id"]] = schema
            else:
                print(f"Warning: Schema file not found: {schema_path}", file=sys.stderr)

        return schemas

    def validate(self, runspec: Dict[str, Any]) -> List[str]:
        """
        Validate a RunSpec against the schema.

        Returns:
            List of validation errors (empty if valid)
        """
        if not self.schemas:
            return ["No schemas loaded"]

        # Get RunSpec schema
        runspec_schema_id = "https://qfd.physics/schema/v0/RunSpec.schema.json"
        if runspec_schema_id not in self.schemas:
            return ["RunSpec schema not found"]

        schema = self.schemas[runspec_schema_id]

        # Create resolver for $ref resolution
        resolver = RefResolver(
            base_uri=runspec_schema_id,
            referrer=schema,
            store=self.schemas
        )

        # Validate
        validator = Draft7Validator(schema, resolver=resolver)
        errors = []

        for error in validator.iter_errors(runspec):
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"{path}: {error.message}")

        return errors

    def resolve_references(self, runspec: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """
        Resolve file references in RunSpec to inline objects.

        Args:
            runspec: RunSpec dictionary
            base_dir: Base directory for resolving relative paths

        Returns:
            Resolved RunSpec with all file references loaded inline
        """
        resolved = runspec.copy()

        # Resolve model reference
        if isinstance(resolved.get("model"), str):
            model_path = base_dir / resolved["model"]
            with open(model_path) as f:
                resolved["model"] = json.load(f)

        # Resolve parameter references
        if "parameters" in resolved:
            resolved_params = []
            for param in resolved["parameters"]:
                if isinstance(param, str):
                    param_path = base_dir / param
                    with open(param_path) as f:
                        resolved_params.append(json.load(f))
                else:
                    resolved_params.append(param)
            resolved["parameters"] = resolved_params

        # Resolve dataset references
        if "datasets" in resolved:
            resolved_datasets = []
            for dataset in resolved["datasets"]:
                if isinstance(dataset, str):
                    dataset_path = base_dir / dataset
                    with open(dataset_path) as f:
                        resolved_datasets.append(json.load(f))
                else:
                    resolved_datasets.append(dataset)
            resolved["datasets"] = resolved_datasets

        # Resolve objective reference
        if isinstance(resolved.get("objective"), str):
            objective_path = base_dir / resolved["objective"]
            with open(objective_path) as f:
                resolved["objective"] = json.load(f)

        return resolved

    def fill_git_provenance(self, runspec: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in git provenance information at runtime."""
        import subprocess

        updated = runspec.copy()

        if "git" not in updated:
            updated["git"] = {}

        try:
            # Get current commit
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            updated["git"]["commit"] = commit

            # Check if dirty
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            updated["git"]["dirty"] = bool(status)

            # Get current branch
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            updated["git"]["branch"] = branch

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: Could not retrieve git provenance", file=sys.stderr)

        return updated

    def check_parameter_consistency(self, runspec: Dict[str, Any]) -> List[str]:
        """
        Check for common consistency issues in parameter specifications.

        Returns:
            List of warnings/errors
        """
        issues = []

        if "parameters" not in runspec:
            return issues

        for param in runspec["parameters"]:
            if not isinstance(param, dict):
                continue

            name = param.get("name", "unknown")

            # Check bounds consistency
            if "bounds" in param:
                bounds = param["bounds"]
                if "min" in bounds and "max" in bounds:
                    if bounds["min"] >= bounds["max"]:
                        issues.append(f"Parameter '{name}': min bound >= max bound")

                # Check init within bounds
                if "init" in param:
                    init = param["init"]
                    if isinstance(init, (int, float)):
                        if "min" in bounds and init < bounds["min"]:
                            issues.append(f"Parameter '{name}': init < min bound")
                        if "max" in bounds and init > bounds["max"]:
                            issues.append(f"Parameter '{name}': init > max bound")

            # Check frozen parameters don't need bounds
            if param.get("frozen") and "bounds" in param:
                issues.append(f"Parameter '{name}': frozen=true but bounds specified (redundant)")

            # Check sensitivity classification
            if param.get("sensitivity") == "high" and param.get("frozen"):
                issues.append(f"Parameter '{name}': high sensitivity but frozen (unusual)")

        return issues


def main():
    parser = argparse.ArgumentParser(description="Validate QFD RunSpec files")
    parser.add_argument("runspec", type=Path, help="Path to RunSpec JSON file")
    parser.add_argument("--schema-dir", type=Path,
                       default=Path(__file__).parent,
                       help="Directory containing schema files")
    parser.add_argument("--resolve", action="store_true",
                       help="Resolve file references to inline objects")
    parser.add_argument("--output", type=Path,
                       help="Output path for resolved RunSpec")
    parser.add_argument("--check-only", action="store_true",
                       help="Only validate, don't resolve or output")
    parser.add_argument("--fill-git", action="store_true",
                       help="Fill git provenance from current repository state")

    args = parser.parse_args()

    # Load RunSpec
    if not args.runspec.exists():
        print(f"Error: RunSpec file not found: {args.runspec}", file=sys.stderr)
        return 1

    with open(args.runspec) as f:
        runspec = json.load(f)

    # Initialize validator
    validator = RunSpecValidator(args.schema_dir)

    # Validate
    print(f"Validating {args.runspec}...")
    errors = validator.validate(runspec)

    if errors:
        print("\nValidation FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("✓ Schema validation passed")

    # Check parameter consistency
    consistency_issues = validator.check_parameter_consistency(runspec)
    if consistency_issues:
        print("\nParameter consistency warnings:")
        for issue in consistency_issues:
            print(f"  ⚠ {issue}")

    if args.check_only:
        return 0

    # Fill git provenance if requested
    if args.fill_git:
        print("\nFilling git provenance...")
        runspec = validator.fill_git_provenance(runspec)
        if runspec.get("git", {}).get("commit"):
            print(f"  Commit: {runspec['git']['commit'][:8]}")
            print(f"  Branch: {runspec['git'].get('branch', 'unknown')}")
            print(f"  Dirty: {runspec['git'].get('dirty', 'unknown')}")

    # Resolve references if requested
    if args.resolve:
        print("\nResolving file references...")
        base_dir = args.runspec.parent
        try:
            runspec = validator.resolve_references(runspec, base_dir)
            print("✓ References resolved")
        except FileNotFoundError as e:
            print(f"Error resolving reference: {e}", file=sys.stderr)
            return 1

    # Output resolved RunSpec
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(runspec, f, indent=2)
        print(f"\n✓ Resolved RunSpec written to {args.output}")
    else:
        print("\nResolved RunSpec:")
        print(json.dumps(runspec, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
