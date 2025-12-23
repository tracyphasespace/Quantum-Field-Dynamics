import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

#!/usr/bin/env python3
"""
Schema Validation Script for Black Hole Rift Dynamics

Validates that config.py matches blackhole_rift_charge_rotation.json schema:
1. All schema parameters present in config
2. All parameter values within bounds
3. All constraints satisfied
4. All Lean references documented

Usage:
    python validate_config_vs_schema.py
"""

import json
from pathlib import Path
from config import SimConfig


def main():
    """Run full validation suite"""

    print("=" * 80)
    print("BLACK HOLE RIFT DYNAMICS: Schema Validation")
    print("=" * 80)
    print()

    # Load schema
    schema_path = Path(__file__).parents[3] / "schema" / "v0" / "experiments" / "blackhole_rift_charge_rotation.json"

    if not schema_path.exists():
        print(f"❌ ERROR: Schema file not found at {schema_path}")
        return 1

    with open(schema_path, 'r') as f:
        schema = json.load(f)

    print(f"✅ Loaded schema: {schema['experiment_id']}")
    print(f"   Version: {schema['schema_version']}")
    print(f"   Model: {schema['model']['id']}")
    print()

    # Create config instance
    config = SimConfig()
    config.__post_init__()

    # ========================================
    # Test 1: Parameter Coverage
    # ========================================

    print("Test 1: Parameter Coverage")
    print("-" * 80)

    missing_params = []
    extra_params = []
    param_count = 0

    schema_params = {p["name"]: p for p in schema["parameters"]}

    for param_name, param_def in schema_params.items():
        if param_def.get("role") == "derived":
            continue  # Skip derived parameters

        config_name = param_name.upper()
        if not hasattr(config, config_name):
            missing_params.append(param_name)
        else:
            param_count += 1

    if missing_params:
        print(f"❌ Missing {len(missing_params)} parameters:")
        for p in missing_params:
            print(f"   - {p}")
    else:
        print(f"✅ All {param_count} schema parameters present in config")

    print()

    # ========================================
    # Test 2: Parameter Bounds
    # ========================================

    print("Test 2: Parameter Bounds")
    print("-" * 80)

    bounds_violations = []

    for param_name, param_def in schema_params.items():
        if param_def.get("role") == "derived":
            continue

        if "bounds" not in param_def:
            continue  # No bounds to check

        config_name = param_name.upper()
        if not hasattr(config, config_name):
            continue  # Already caught in Test 1

        config_value = getattr(config, config_name)
        bounds = param_def["bounds"]

        if not (bounds[0] <= config_value <= bounds[1]):
            bounds_violations.append({
                "param": param_name,
                "value": config_value,
                "bounds": bounds
            })

    if bounds_violations:
        print(f"❌ Found {len(bounds_violations)} bounds violations:")
        for v in bounds_violations:
            print(f"   - {v['param']} = {v['value']} (bounds: {v['bounds']})")
    else:
        print("✅ All parameter values within schema bounds")

    print()

    # ========================================
    # Test 3: Critical Constraints
    # ========================================

    print("Test 3: Critical Constraints")
    print("-" * 80)

    constraints = config.validate_constraints()

    failed_constraints = [name for name, passed in constraints.items() if not passed]

    if failed_constraints:
        print(f"❌ Failed {len(failed_constraints)} constraints:")
        for c in failed_constraints:
            print(f"   - {c}")
    else:
        print(f"✅ All {len(constraints)} constraints satisfied:")
        for name in constraints.keys():
            print(f"   ✓ {name}")

    print()

    # ========================================
    # Test 4: Frozen Constants
    # ========================================

    print("Test 4: Frozen Constants (CODATA)")
    print("-" * 80)

    frozen_constants = {
        "Q_ELECTRON": -1.602176634e-19,
        "M_ELECTRON": 9.1093837015e-31,
        "Q_PROTON": 1.602176634e-19,
        "M_PROTON": 1.67262192369e-27,
        "K_COULOMB": 8.9875517923e9,
    }

    frozen_mismatches = []
    for const_name, expected_value in frozen_constants.items():
        actual_value = getattr(config, const_name)
        if abs(actual_value - expected_value) / abs(expected_value) > 1e-10:
            frozen_mismatches.append({
                "param": const_name,
                "expected": expected_value,
                "actual": actual_value
            })

    if frozen_mismatches:
        print(f"❌ Found {len(frozen_mismatches)} frozen constant mismatches:")
        for m in frozen_mismatches:
            print(f"   - {m['param']}: {m['actual']} (expected {m['expected']})")
    else:
        print(f"✅ All {len(frozen_constants)} frozen constants match CODATA")

    print()

    # ========================================
    # Test 5: Lean References
    # ========================================

    print("Test 5: Lean Theorem References")
    print("-" * 80)

    lean_params = [p for p in schema["parameters"] if "lean_reference" in p]
    print(f"✅ {len(lean_params)} parameters linked to Lean theorems:")

    for param in lean_params[:5]:  # Show first 5
        print(f"   - {param['name']:25s} → {param['lean_reference']}")

    if len(lean_params) > 5:
        print(f"   ... and {len(lean_params) - 5} more")

    print()

    # ========================================
    # Test 6: Derived Parameters
    # ========================================

    print("Test 6: Derived Parameters")
    print("-" * 80)

    print(f"✅ r_core_BH1 = {config.r_core_BH1:.3f}")
    print(f"✅ mass_ratio (m_p/m_e) = {config.mass_ratio_proton_electron:.1f}")
    print(f"✅ r_g (BH1) = {config.gravitational_radius_BH1:.3e} m")
    print(f"✅ r_g (BH2) = {config.gravitational_radius_BH2:.3e} m")

    print()

    # ========================================
    # Test 7: Full Schema Validation
    # ========================================

    print("Test 7: Full Schema Validation")
    print("-" * 80)

    validation_report = config.validate_against_schema(schema_path)

    if validation_report["status"] == "valid":
        print("✅ Configuration is VALID against schema")
    else:
        print(f"❌ Configuration is INVALID: {len(validation_report['violations'])} violations")
        for violation in validation_report["violations"]:
            print(f"   - {violation}")

    print()

    # ========================================
    # Summary
    # ========================================

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_tests_passed = (
        len(missing_params) == 0 and
        len(bounds_violations) == 0 and
        len(failed_constraints) == 0 and
        len(frozen_mismatches) == 0 and
        validation_report["status"] == "valid"
    )

    if all_tests_passed:
        print("✅ ALL TESTS PASSED - Configuration is schema-compliant!")
        print()
        print("Next steps:")
        print("  1. Implement rotation_dynamics.py module")
        print("  2. Extend core.py for 3D scalar field")
        print("  3. Update simulation.py with charge dynamics")
        return 0
    else:
        print("❌ SOME TESTS FAILED - See details above")
        print()
        print("Fix violations before proceeding with implementation")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
