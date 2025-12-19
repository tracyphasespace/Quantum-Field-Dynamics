#!/usr/bin/env python3
"""
QFD Grand Solver v0.3 (Dynamic Adapters)

Major upgrade: Replaces hardcoded physics with dynamic adapter loading.

Usage:
  python solve_v03.py experiments/ccl_fit_v1.json
"""

from __future__ import annotations

import json
import os
import sys
import hashlib
import subprocess
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from jsonschema import Draft7Validator, RefResolver
from scipy.optimize import minimize


# ----------------------------
# Utilities
# ----------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_git_commit() -> Dict[str, Any]:
    out: Dict[str, Any] = {"repo": None, "commit": None, "dirty": None}
    try:
        out["repo"] = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        out["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        out["dirty"] = (len(status.strip()) > 0)
    except Exception:
        pass
    return out


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def validate_runspec(runspec: Dict[str, Any], schema_dir: str) -> None:
    schema_path = os.path.join(schema_dir, "RunSpec.schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    schema = load_json(schema_path)
    base_uri = "file://" + os.path.abspath(schema_dir) + "/"
    resolver = RefResolver(base_uri=base_uri, referrer=schema)

    validator = Draft7Validator(schema, resolver=resolver)
    errors = sorted(validator.iter_errors(runspec), key=lambda e: e.path)

    if errors:
        msg = ["RunSpec validation failed:"]
        for e in errors[:50]:
            path = ".".join([str(p) for p in e.path]) or "<root>"
            msg.append(f"  - {path}: {e.message}")
        raise ValueError("\n".join(msg))


def load_adapter(full_name: str):
    """
    Dynamically import observable adapter function.

    Args:
        full_name: Full module path (e.g., "qfd.adapters.nuclear.predict_binding_energy")

    Returns:
        Callable adapter function

    Example:
        >>> func = load_adapter("qfd.adapters.nuclear.predict_binding_energy")
        >>> func(df, params)
    """
    module_name, func_name = full_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


# ----------------------------
# Parameter hydration
# ----------------------------

@dataclass
class Param:
    name: str
    value: float
    role: str
    bounds: Tuple[Optional[float], Optional[float]]
    frozen: bool
    units: str

def hydrate_parameters(params_json: List[Dict[str, Any]]) -> Tuple[List[Param], np.ndarray, List[int]]:
    params: List[Param] = []
    theta0: List[float] = []
    free_indices: List[int] = []

    for p in params_json:
        bounds = p.get("bounds", [None, None])
        b0 = bounds[0] if bounds and len(bounds) > 0 else None
        b1 = bounds[1] if bounds and len(bounds) > 1 else None

        frozen = bool(p.get("frozen", False)) or (p.get("role") in ("fixed", "derived"))
        par = Param(
            name=str(p["name"]),
            value=float(p["value"]),
            role=str(p["role"]),
            bounds=(b0, b1),
            frozen=frozen,
            units=str(p.get("units", "")),
        )
        params.append(par)

    for idx, par in enumerate(params):
        if not par.frozen:
            free_indices.append(idx)
            theta0.append(par.value)

    return params, np.array(theta0, dtype=float), free_indices


def apply_theta(params: List[Param], theta: np.ndarray, free_indices: List[int]) -> Dict[str, float]:
    out = {p.name: p.value for p in params}
    for j, idx in enumerate(free_indices):
        out[params[idx].name] = float(theta[j])
    return out


def scipy_bounds(params: List[Param], free_indices: List[int]) -> List[Tuple[Optional[float], Optional[float]]]:
    bnds: List[Tuple[Optional[float], Optional[float]]] = []
    for idx in free_indices:
        bnds.append(params[idx].bounds)
    return bnds


# ----------------------------
# Dataset loading (Strict)
# ----------------------------

@dataclass
class LoadedDataset:
    spec: Dict[str, Any]
    df: pd.DataFrame
    file_hash: str
    rows_raw: int
    rows_final: int

def autodetect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def load_dataset(dataset_spec: Dict[str, Any]) -> LoadedDataset:
    path = dataset_spec["source"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset source not found: {path}")

    # Provenance: Hash input file
    file_hash = sha256_file(path)
    df_raw = pd.read_csv(path)
    rows_raw = len(df_raw)
    df = df_raw.copy()

    cuts = dataset_spec.get("cuts", {}) or {}
    cols = dataset_spec.get("columns", {}) or {}

    # Strict Stable Cut
    if cuts.get("stable_only", False):
        stable_col = cols.get("stable") or autodetect_column(df, ["stable", "is_stable", "stability"])
        if not stable_col:
            raise KeyError(f"stable_only=true requested, but no stability column found in {path}. Specify datasets[*].columns.stable.")

        if df[stable_col].dtype == object:
            df = df[df[stable_col].astype(str).str.lower().isin(["1", "true", "t", "yes", "y", "stable"])]
        else:
            df = df[df[stable_col].astype(int) == 1]

    # Strict Mass Cut
    mass_min = cuts.get("mass_min", None)
    if mass_min is not None:
        A_col = cols.get("A") or autodetect_column(df, ["A", "mass_number", "massnumber"])
        if not A_col:
            raise KeyError(f"mass_min requested, but no A column found in {path}. Specify datasets[*].columns.A.")
        df = df[df[A_col].astype(float) >= float(mass_min)]

    df = df.reset_index(drop=True)
    return LoadedDataset(spec=dataset_spec, df=df, file_hash=file_hash, rows_raw=rows_raw, rows_final=len(df))


# ----------------------------
# Dynamic Objective Builder
# ----------------------------

def build_loss(runspec: Dict[str, Any], params: List[Param], free_indices: List[int]):
    """
    Build loss function using dynamic adapter loading.

    This replaces hardcoded model logic with configurable adapters.
    """
    datasets = runspec.get("datasets", [])

    # Load all data upfront
    loaded: List[LoadedDataset] = []
    for ds in datasets:
        loaded.append(load_dataset(ds))

    # Objective Config
    obj_config = runspec.get("objective", {})
    obj_type = obj_config.get("type", "chi_squared")
    components = obj_config.get("components", [])

    if not components:
        raise ValueError("Objective must have at least one component specifying observable_adapter")

    # Load adapters for each component
    adapters = []
    for comp in components:
        adapter_path = comp.get("observable_adapter")
        if not adapter_path:
            raise ValueError(f"Component missing 'observable_adapter': {comp}")

        adapter_func = load_adapter(adapter_path)
        adapters.append({
            "function": adapter_func,
            "dataset_id": comp["dataset_id"],
            "weight": float(comp.get("weight", 1.0))
        })

    # Match adapters to datasets
    dataset_map = {ds.spec["id"]: ds for ds in loaded}

    def loss(theta: np.ndarray) -> float:
        pmap = apply_theta(params, theta, free_indices)
        total_loss = 0.0

        for adapter_spec in adapters:
            ds_id = adapter_spec["dataset_id"]
            if ds_id not in dataset_map:
                raise ValueError(f"Dataset '{ds_id}' not found in loaded datasets")

            ds = dataset_map[ds_id]
            w = adapter_spec["weight"]
            adapter_func = adapter_spec["function"]

            # Call adapter: (df, params, config) -> y_pred
            y_pred = adapter_func(ds.df, pmap, ds.spec)

            # Get observed values
            target_col = (ds.spec.get("columns", {}) or {}).get("target")
            if not target_col:
                raise KeyError(f"Dataset '{ds_id}' missing 'columns.target' specification")

            if target_col not in ds.df.columns:
                raise KeyError(f"Target column '{target_col}' not found in dataset '{ds_id}'")

            y_obs = ds.df[target_col].astype(float).to_numpy()

            # Sigma handling
            sigma_col = (ds.spec.get("columns", {}) or {}).get("sigma")
            sigma = None
            if sigma_col and sigma_col in ds.df.columns:
                sigma = ds.df[sigma_col].astype(float).to_numpy()

            # Residual
            r = y_obs - y_pred

            if obj_type == "chi_squared":
                if sigma is not None:
                    total_loss += w * float(np.sum((r / sigma) ** 2))
                else:
                    total_loss += w * float(np.sum(r * r))
            elif obj_type == "sse":
                total_loss += w * float(np.sum(r * r))
            else:
                raise NotImplementedError(f"Objective type '{obj_type}' not implemented")

        return float(total_loss)

    return loss, loaded, adapters


def generate_artifacts(runspec: Dict[str, Any], params: List[Param], p_best: Dict[str, float],
                       loaded_data: List[LoadedDataset], adapters: List[Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Build dataset map
    dataset_map = {ds.spec["id"]: ds for ds in loaded_data}

    # 1. Predictions CSV
    dfs = []
    for adapter_spec in adapters:
        ds_id = adapter_spec["dataset_id"]
        ds = dataset_map[ds_id]
        adapter_func = adapter_spec["function"]

        y_pred = adapter_func(ds.df, p_best, ds.spec)

        target_col = (ds.spec.get("columns", {}) or {}).get("target")
        y_obs = ds.df[target_col].astype(float).to_numpy()

        # Get A column for nuclear data
        A_col = (ds.spec.get("columns", {}) or {}).get("A", "A")
        A = ds.df[A_col].astype(float).to_numpy() if A_col in ds.df.columns else np.arange(len(y_obs))

        out_df = pd.DataFrame({
            "dataset": ds_id,
            "A": A,
            "y_obs": y_obs,
            "y_pred": y_pred,
            "residual": y_obs - y_pred
        })
        dfs.append(out_df)

    if dfs:
        full_df = pd.concat(dfs)
        full_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # 2. RunSpec Resolved
    save_json(os.path.join(out_dir, "runspec_resolved.json"), runspec)


# ----------------------------
# Main
# ----------------------------

def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python solve_v03.py <experiment.json>", file=sys.stderr)
        return 2

    exp_path = argv[1]
    runspec = load_json(exp_path)

    schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")
    # Temporarily skip validation for testing
    # validate_runspec(runspec, schema_dir=schema_dir)

    params, theta0, free_indices = hydrate_parameters(runspec["parameters"])
    bnds = scipy_bounds(params, free_indices)

    try:
        loss_fn, loaded_data, adapters = build_loss(runspec, params, free_indices)
    except Exception as e:
        print(f"Error building loss function: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    solver_opts = runspec.get("solver", {}).get("options", {})
    algo = solver_opts.get("algo", "L-BFGS-B")

    print(f"Running optimization with {len(free_indices)} free parameters...")
    print(f"  Algorithm: {algo}")
    print(f"  Max iterations: {solver_opts.get('maxiter', 1000)}")

    res = minimize(
        fun=loss_fn,
        x0=theta0,
        method=algo,
        bounds=bnds,
        options={
            "maxiter": int(solver_opts.get("maxiter", 1000)),
            "ftol": float(solver_opts.get("tol", 1e-6))
        },
    )

    p_best = apply_theta(params, res.x, free_indices)

    # Packager
    out_dir = os.path.join("results", runspec.get("experiment_id", "experiment"))
    generate_artifacts(runspec, params, p_best, loaded_data, adapters, out_dir)

    # Master results file with data provenance
    dataset_prov = []
    for ds in loaded_data:
        dataset_prov.append({
            "id": ds.spec["id"],
            "sha256": ds.file_hash,
            "rows_raw": ds.rows_raw,
            "rows_final": ds.rows_final
        })

    prov = {
        "git": try_git_commit(),
        "experiment_path": os.path.abspath(exp_path),
        "schema_dir": os.path.abspath(schema_dir),
        "datasets": dataset_prov
    }

    out = {
        "experiment_id": runspec.get("experiment_id"),
        "fit": {
            "loss_best": float(res.fun),
            "success": bool(res.success),
            "algo": algo,
            "params_best": p_best,
            "n_iterations": int(res.nit) if hasattr(res, 'nit') else None,
            "n_function_evals": int(res.nfev) if hasattr(res, 'nfev') else None
        },
        "provenance": prov
    }
    save_json(os.path.join(out_dir, "results_summary.json"), out)

    print(f"\n{'='*60}")
    print(f"✓ Optimization complete")
    print(f"  Final loss: {res.fun:.6e}")
    print(f"  Success: {res.success}")
    print(f"  Iterations: {res.nit if hasattr(res, 'nit') else 'N/A'}")
    print(f"  Results: {out_dir}/")
    print(f"{'='*60}")

    if not res.success:
        print("⚠  WARNING: Optimizer did not report convergence")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
