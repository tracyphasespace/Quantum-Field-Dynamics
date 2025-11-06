# tools/write_stage2_summary.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Mapping, Any
import numpy as np

@dataclass
class Standardizer:
    """Feature standardization used in Stage 2:
       φ_std = (φ - mean) / scale
       alpha_pred = alpha0 + sum_i c[i] * φ_std[i]
    """
    means: np.ndarray  # shape (3,)
    scales: np.ndarray # shape (3,)

def _summ(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std(ddof=1)),
        "median": float(np.median(arr)),
        "q05":    float(np.quantile(arr, 0.05)),
        "q95":    float(np.quantile(arr, 0.95)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "n":      int(arr.size),
    }

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    if a.ndim == 1 and b.ndim == 1 and a.size == b.size and a.size > 1:
        return float(np.corrcoef(a, b)[0, 1])
    return float("nan")

def _stack_c(samples: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return c with shape (nsamples, 3) regardless of how it's stored."""
    if "c" in samples:  # preferred
        c = np.asarray(samples["c"])
        if c.ndim == 2 and c.shape[1] == 3:
            return c
    # tolerate c0/c1/c2 naming
    keys = [k for k in samples.keys() if k in ("c0","c1","c2")]
    if len(keys) == 3:
        return np.column_stack([samples["c0"], samples["c1"], samples["c2"]])
    # tolerate physical-only (back out standardized from them if std given)
    phys_keys = [k for k in ("k_J","eta_prime","xi") if k in samples]
    if len(phys_keys) == 3:
        # caller must provide standardizer; returned c is a placeholder here
        raise KeyError("Samples contain only physical params; supply standardized 'c' or (c0,c1,c2).")
    raise KeyError("Could not find standardized coefficients 'c' (or c0,c1,c2) in samples.")

def backtransform_physical(
    c: np.ndarray, alpha0: np.ndarray, std: Standardizer
) -> Dict[str, np.ndarray]:
    """Map standardized (c, alpha0) → physical (k_J, eta_prime, xi, alpha0_phys).
       alpha_pred = alpha0 + sum_i c_i * (φ_i - μ_i)/s_i
                  = [alpha0 - sum_i c_i * μ_i/s_i] + sum_i (c_i/s_i) φ_i
       => physical coeffs are c_i/s_i, and the physical offset equals the bracket.
    """
    means = std.means.reshape(1, 3)
    scales = std.scales.reshape(1, 3)
    k_phys = c / scales                        # shape (ns,3)
    alpha0_phys = alpha0 - (c * (means / scales)).sum(axis=1)
    return {
        "k_J":        k_phys[:, 0],
        "eta_prime":  k_phys[:, 1],
        "xi":         k_phys[:, 2],
        "alpha0_phys": alpha0_phys,
    }

def write_stage2_summary(
    out_json_path: str,
    samples: Mapping[str, np.ndarray],
    standardizer: Standardizer,
    survey_names: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    out_json_path : str
        File path to write the JSON summary.
    samples : dict-like
        Posterior draws; must include 'alpha0' and either 'c' or ('c0','c1','c2').
        May include 'sigma_alpha' (shape [nsamples, S]) and 'nu'.
    standardizer : Standardizer
        Means/scales used to construct standardized features.
    survey_names : list[str], optional
        Names for per-survey sigma entries; len must match sigma_alpha's last dim.

    Returns
    -------
    dict : the JSON object that was written (useful for tests).
    """
    # Pull standardized
    c = _stack_c(samples)                        # (ns, 3)
    alpha0 = np.asarray(samples["alpha0"])       # (ns,)
    # Back-transform
    phys = backtransform_physical(c, alpha0, standardizer)

    # Summaries
    out = {
        "meta": {
            "n_samples": int(alpha0.size),
            "standardizer": {
                "means":  standardizer.means.tolist(),
                "scales": standardizer.scales.tolist(),
            },
            "survey_names": survey_names or [],
        },
        "standardized": {
            "alpha0": _summ(alpha0),
            "c0": _summ(c[:, 0]),
            "c1": _summ(c[:, 1]),
            "c2": _summ(c[:, 2]),
            "corr": {
                "c0_c1": _corr(c[:,0], c[:,1]),
                "c0_c2": _corr(c[:,0], c[:,2]),
                "c1_c2": _corr(c[:,1], c[:,2]),
            },
        },
        "physical": {
            "alpha0": _summ(phys["alpha0_phys"]),
            "k_J": _summ(phys["k_J"]),
            "eta_prime": _summ(phys["eta_prime"]),
            "xi": _summ(phys["xi"]),
            "corr": {
                "kJ_xi": _corr(phys["k_J"], phys["xi"]),
                "kJ_eta": _corr(phys["k_J"], phys["eta_prime"]),
                "eta_xi": _corr(phys["eta_prime"], phys["xi"]),
            },
        },
        "noise": {},
    }

    # Optional noise pieces
    if "sigma_alpha" in samples:
        sig = np.asarray(samples["sigma_alpha"])  # (ns, S) or (ns,) if scalar
        if sig.ndim == 1:
            out["noise"]["sigma_alpha"] = {"all": _summ(sig)}
        elif sig.ndim == 2:
            S = sig.shape[1]
            names = survey_names or [f"survey_{i}" for i in range(S)]
            if len(names) != S:
                raise ValueError("survey_names length must match sigma_alpha dimension.")
            per = {names[i]: _summ(sig[:, i]) for i in range(S)}
            out["noise"]["sigma_alpha"] = per
    if "nu" in samples:
        out["noise"]["nu"] = _summ(samples["nu"])

    # Also include small "quick-look" section
    out["quicklook"] = {
        "alpha0_phys_median": float(np.median(phys["alpha0_phys"])),
        "alpha0_phys_mean":   float(np.mean(phys["alpha0_phys"])),
    }

    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2)
    return out
