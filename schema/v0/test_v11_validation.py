#!/usr/bin/env python3
"""
Test suite for Grand Solver v1.1 validation hardening.

Tests:
1. Duplicate parameter names
2. Duplicate dataset IDs
3. Bounds-incompatible solver
4. Non-positive sigma values
5. Non-finite sigma values (NaN)
6. Non-finite sigma values (Inf)
"""

import json
import sys
import tempfile
import os
import subprocess

def test_duplicate_params():
    """Test: Duplicate parameter names should be rejected"""
    runspec = {
        "schema_version": "v0.1",
        "experiment_id": "test_dup_params",
        "model": {"id": "test"},
        "parameters": [
            {"name": "c1", "value": 1.0, "role": "coupling"},
            {"name": "c1", "value": 2.0, "role": "coupling"}  # DUPLICATE!
        ],
        "datasets": [
            {"id": "data1", "source": "test.csv", "columns": {"target": "y"}}
        ],
        "objective": {
            "type": "chi_squared",
            "components": [{"dataset_id": "data1", "observable_adapter": "test.adapter"}]
        },
        "solver": {"method": "scipy.minimize", "options": {"algo": "L-BFGS-B"}}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(runspec, f)
        path = f.name

    result = subprocess.run(
        ["python", "schema/v0/solve_v03.py", path],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True
    )

    os.unlink(path)

    return "Duplicate parameter names" in result.stderr

def test_duplicate_datasets():
    """Test: Duplicate dataset IDs should be rejected"""
    runspec = {
        "schema_version": "v0.1",
        "experiment_id": "test_dup_datasets",
        "model": {"id": "test"},
        "parameters": [
            {"name": "c1", "value": 1.0, "role": "coupling"}
        ],
        "datasets": [
            {"id": "data1", "source": "test1.csv", "columns": {"target": "y"}},
            {"id": "data1", "source": "test2.csv", "columns": {"target": "y"}}  # DUPLICATE!
        ],
        "objective": {
            "type": "chi_squared",
            "components": [{"dataset_id": "data1", "observable_adapter": "test.adapter"}]
        },
        "solver": {"method": "scipy.minimize", "options": {"algo": "L-BFGS-B"}}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(runspec, f)
        path = f.name

    result = subprocess.run(
        ["python", "schema/v0/solve_v03.py", path],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True
    )

    os.unlink(path)

    return "Duplicate dataset IDs" in result.stderr

def test_bounds_incompatible_solver():
    """Test: Non-bounds-compatible solver with bounded parameters should fail"""
    # Create test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("A,y\n1,2.0\n2,3.0\n")
        data_path = f.name

    runspec = {
        "schema_version": "v0.1",
        "experiment_id": "test_bounds_incompatible",
        "model": {"id": "test"},
        "parameters": [
            {"name": "c1", "value": 1.0, "role": "coupling", "bounds": [0.5, 1.5]}  # HAS BOUNDS!
        ],
        "datasets": [
            {"id": "data1", "source": data_path, "columns": {"A": "A", "target": "y"}}
        ],
        "objective": {
            "type": "chi_squared",
            "components": [{"dataset_id": "data1", "observable_adapter": "qfd.adapters.nuclear.predict_binding_energy"}]
        },
        "solver": {
            "method": "scipy.minimize",
            "options": {"algo": "BFGS"}  # DOES NOT SUPPORT BOUNDS!
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(runspec, f)
        path = f.name

    result = subprocess.run(
        ["python", "schema/v0/solve_v03.py", path],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True
    )

    os.unlink(path)
    os.unlink(data_path)

    return "does not support bounds" in result.stderr

def test_sigma_nonpositive():
    """Test: Non-positive sigma values (≤0) should be rejected"""
    # Create test data with zero sigma
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("A,Z,y,sigma\n")
        f.write("4,2,28.3,0.1\n")
        f.write("12,6,92.2,0.0\n")  # SIGMA = 0!
        data_path = f.name

    runspec = {
        "schema_version": "v0.1",
        "experiment_id": "test_sigma_zero",
        "model": {"id": "test"},
        "parameters": [
            {"name": "c1", "value": 1.0, "role": "coupling"}
        ],
        "datasets": [
            {
                "id": "data1",
                "source": data_path,
                "columns": {"A": "A", "Z": "Z", "target": "y", "sigma": "sigma"}
            }
        ],
        "objective": {
            "type": "chi_squared",
            "components": [{"dataset_id": "data1", "observable_adapter": "qfd.adapters.nuclear.predict_binding_energy"}]
        },
        "solver": {"method": "scipy.minimize", "options": {"algo": "L-BFGS-B"}}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(runspec, f)
        path = f.name

    result = subprocess.run(
        ["python", "schema/v0/solve_v03.py", path],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True
    )

    os.unlink(path)
    os.unlink(data_path)

    return "non-positive values" in result.stderr

def test_sigma_nan():
    """Test: Non-finite sigma values (NaN) should be rejected"""
    # Create test data with NaN sigma
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("A,Z,y,sigma\n")
        f.write("4,2,28.3,0.1\n")
        f.write("12,6,92.2,NaN\n")  # SIGMA = NaN!
        data_path = f.name

    runspec = {
        "schema_version": "v0.1",
        "experiment_id": "test_sigma_nan",
        "model": {"id": "test"},
        "parameters": [
            {"name": "c1", "value": 1.0, "role": "coupling"}
        ],
        "datasets": [
            {
                "id": "data1",
                "source": data_path,
                "columns": {"A": "A", "Z": "Z", "target": "y", "sigma": "sigma"}
            }
        ],
        "objective": {
            "type": "chi_squared",
            "components": [{"dataset_id": "data1", "observable_adapter": "qfd.adapters.nuclear.predict_binding_energy"}]
        },
        "solver": {"method": "scipy.minimize", "options": {"algo": "L-BFGS-B"}}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(runspec, f)
        path = f.name

    result = subprocess.run(
        ["python", "schema/v0/solve_v03.py", path],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True
    )

    os.unlink(path)
    os.unlink(data_path)

    return "non-finite values" in result.stderr

def test_sigma_inf():
    """Test: Non-finite sigma values (Inf) should be rejected"""
    # Create test data with Inf sigma
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("A,Z,y,sigma\n")
        f.write("4,2,28.3,0.1\n")
        f.write("12,6,92.2,inf\n")  # SIGMA = Inf!
        data_path = f.name

    runspec = {
        "schema_version": "v0.1",
        "experiment_id": "test_sigma_inf",
        "model": {"id": "test"},
        "parameters": [
            {"name": "c1", "value": 1.0, "role": "coupling"}
        ],
        "datasets": [
            {
                "id": "data1",
                "source": data_path,
                "columns": {"A": "A", "Z": "Z", "target": "y", "sigma": "sigma"}
            }
        ],
        "objective": {
            "type": "chi_squared",
            "components": [{"dataset_id": "data1", "observable_adapter": "qfd.adapters.nuclear.predict_binding_energy"}]
        },
        "solver": {"method": "scipy.minimize", "options": {"algo": "L-BFGS-B"}}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(runspec, f)
        path = f.name

    result = subprocess.run(
        ["python", "schema/v0/solve_v03.py", path],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True
    )

    os.unlink(path)
    os.unlink(data_path)

    return "non-finite values" in result.stderr

def main():
    print("=" * 60)
    print("QFD Grand Solver v1.1 Validation Test Suite")
    print("=" * 60)

    tests = [
        ("Duplicate parameter names", test_duplicate_params),
        ("Duplicate dataset IDs", test_duplicate_datasets),
        ("Bounds-incompatible solver", test_bounds_incompatible_solver),
        ("Non-positive sigma (σ=0)", test_sigma_nonpositive),
        ("Non-finite sigma (σ=NaN)", test_sigma_nan),
        ("Non-finite sigma (σ=Inf)", test_sigma_inf)
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTesting: {name}...", end=" ")
        try:
            passed = test_func()
            if passed:
                print("✅ PASS")
                results.append(True)
            else:
                print("❌ FAIL")
                results.append(False)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
