#!/usr/bin/env python3
"""
GPT_fit_lifetimes_v3.py

Closed-form calibration of (beta, k0) for the unified model.
Includes internal chi normalization and robust handling for grid/single modes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Protocol

from src.probes.zeeman_probe import zeeman_probe, ZeemanResult


# Constants for physics calculations
ELECTRON_CHARGE = 1.602176634e-19  # Coulombs
REDUCED_PLANCK_CONSTANT = 1.054571817e-34  # J s
EV_TO_JOULE = 1.602176634e-19  # J/eV

class ZeemanBundleWrapper:
    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle
        self._summary = bundle.get("summary", {})
        self._manifest = bundle.get("manifest", {})
        self._constants = self._summary.get("constants", {}) or {}
        self._zeeman_probe_config = self._manifest.get("probes", {}).get("zeeman", {})
        # Provide a mutable attribute 'species' to satisfy the HasZeeman protocol
        self.species: str = self._summary.get("particle", "unknown")

    def energy_at_B(self, B: float) -> float:
        initial_energy   = self._summary.get("energy")
        magnetic_moment  = self._constants.get("magnetic_moment") or self._summary.get("magnetic_moment")

        if initial_energy is None or magnetic_moment is None:
            raise ValueError("Missing energy or magnetic_moment in bundle summary for Zeeman probe.")

        return initial_energy - magnetic_moment * B

    @property
    def B0(self) -> float:
        return self._zeeman_probe_config.get("B0", 0.0)

    def dE_dB_to_anomaly(self, dE_dB: float) -> float:
        g_factor = self._constants.get("g_factor") or self._summary.get("g_factor")
        if g_factor is not None:
            return (g_factor - 2.0) / 2.0
        else:
            raise ValueError("Missing g_factor in bundle constants for Zeeman probe anomaly calculation.")


# --------------------------- I/O helpers ---------------------------

def load_bundle_data(manifest_path: Path) -> Dict[str, Any]:
    """Load a bundle manifest + its referenced summary and optional extras (features)."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    bundle: Dict[str, Any] = {"manifest": manifest, "summary": None, "features": {}}

    summary_rel = manifest.get("summary")
    if not summary_rel:
        raise FileNotFoundError(f"Manifest missing 'summary' key: {manifest_path}")

    summary_path = manifest_path.parent / summary_rel
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        bundle["summary"] = json.load(f)

    extras_rel = manifest.get("extras") or manifest.get("features")
    if extras_rel:
        features_path = manifest_path.parent / extras_rel
        if features_path.is_file():
            with open(features_path, "r", encoding="utf-8") as f:
                bundle["features"] = json.load(f)

    return bundle


# --- begin compat loader ---
from pathlib import Path
import json

def _load_obj(base: Path, obj):
    if isinstance(obj, str):
        return json.loads((base / obj).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return obj
    return {}

def load_bundle_compat(manifest_path) -> dict:
    mp = Path(manifest_path)
    M = json.loads(mp.read_text(encoding="utf-8"))
    base = mp.parent

    summary  = _load_obj(base, M.get("summary"))
    extras   = _load_obj(base, M.get("extras"))    # legacy name used by fitter
    features = _load_obj(base, M.get("features"))  # modern name used by builder

    # merge: let explicit extras win; fill any gaps from features
    for k, v in features.items():
        extras.setdefault(k, v)

    # back-fill expected summary aliases from extras (keeps your current logic working)
    if "Hkin_final" not in summary and "K" in extras:
        summary["Hkin_final"] = float(extras["K"])
    if "Q_proxy_final" not in summary and "q_star" in extras:
        summary["Q_proxy_final"] = float(extras["q_star"])

    merged = dict(extras)  # already has 'features' merged in above
    return {
        "manifest": M,
        "summary": summary,
        "features": merged,  # <- expose as 'features' for the fitter
        "extras": merged,
        "species": M.get("species"),
    }
# --- end compat loader ---


def extract_feature(bundle: Dict[str, Any],
                    key: str,
                    summary_key: str,
                    exit_code: int) -> Tuple[float, str]:
    """
    Extract scalar from features first (if applicable), then summary.
    Raise ValueError(message, exit_code) if missing/non-numeric.
    """
    source = "missing"
    val = None

    # Prioritize features.json for specific keys
    if key in ["U", "R_eff", "I", "K", "q_star", "L_mag", "Hcsr"]:
        val = bundle.get("features", {}).get(key)
        if val is not None:
            source = "features"

    if val is None: # If not found in features, try summary
        val = bundle.get("summary", {}).get(summary_key)
        if val is not None:
            source = "summary"

    if val is None:
        raise ValueError(f"Feature '{key}' not found in summary('{summary_key}') or extras.", exit_code)

    try:
        return float(val), source
    except Exception:
        raise ValueError(f"Feature '{key}' is not numeric (got: {val!r}).", exit_code)


# NEW: anchors shouldn't depend on chi-mode or Q_proxy
def extract_anchor_features(bundle_data: dict) -> dict:
    U,  _  = extract_feature(bundle_data, 'U',     'U_final',      8)
    Re, _  = extract_feature(bundle_data, 'R_eff', 'R_eff_final',  8)
    Ie, _  = extract_feature(bundle_data, 'I',     'I_final',      8)
    Ke, _  = extract_feature(bundle_data, 'K',     'Hkin_final',   8)
    return {'U0': U, 'Re': Re, 'Ie': Ie, 'Ke': Ke}


def safe_exp(x: float, hi: float = 700.0, lo: float = -745.0):
    """
    Returns (value, overflow_flag, clamped_input).
    exp(x) overflows ~>709 on float64; underflows below ~-745.
    """
    if x > hi:
        return math.exp(hi), True, hi
    if x < lo:
        return 0.0, True, lo
    return math.exp(x), False, x

def _need(features: dict, keys: list[str], mode: str):
    missing = [k for k in keys if k not in features]
    if missing:
        raise ValueError(
            f"chi-mode '{mode}' requires features {missing} in every bundle; "
            f"please add them to the v2 manifests."
        )

def _assert_numeric(name, val):
    import math
    if val is None or (isinstance(val, float) and math.isnan(val)):
        raise ValueError(f"feature '{name}' is NaN/None; please regenerate bundles")

def extract_all_features(bundle: Dict[str, Any], chi_mode: str) -> Dict[str, float]:
    """
    Extract U, R_eff, I, K, chi for a bundle according to the requested chi_mode.
    chi uses |Q_proxy_final| by default (and optionally L_mag or Hcsr if requested mode needs it).
    """
    try:
        U, _      = extract_feature(bundle, "U",     "U_final",     8)
        R_eff, _  = extract_feature(bundle, "R_eff", "R_eff_final", 8)
        I, _      = extract_feature(bundle, "I",     "I_final",     8)
        K, _      = extract_feature(bundle, "K",     "Hkin_final",  8)

        # Use extract_feature for q_star as well
        Q_val, _ = extract_feature(bundle, "q_star", "Q_proxy_final", 8)
        Q = abs(float(Q_val))

        # Assert numeric for key features
        _assert_numeric("U", U)
        _assert_numeric("R_eff", R_eff)
        _assert_numeric("I", I)
        _assert_numeric("K", K)
        _assert_numeric("q_star", Q) # Q is the numeric value of q_star

        # Use _need for feature checks
        if chi_mode == "q_over_I":
            chi = Q / I
        elif chi_mode == "qR_over_I":
            chi = (Q * R_eff) / I
        elif chi_mode == "LdotQ_over_I":
            L_mag, _ = extract_feature(bundle, "L_mag", "L_mag", 8)
            chi = abs(L_mag) * Q / I
        elif chi_mode == "hcsr_over_I":
            H_csr, _ = extract_feature(bundle, "Hcsr", "Hcsr_final", 8)
            chi = abs(H_csr) / I
        else:
            raise ValueError(f"Unknown chi_mode '{chi_mode}'.", 8)

        return {"U": U, "R_eff": R_eff, "I": I, "K": K, "chi": chi}

    except ValueError as e:
        msg = e.args[0] if e.args else "unknown"
        code = e.args[1] if len(e.args) > 1 else 8
        raise ValueError(f"missing_features: {msg}", code)


def calculate_ln_gamma_star(feat: Dict[str, float],
                            anchors: Dict[str, float],
                            exponents: Tuple[float, float, float, float]) -> float:
    """
    ln Γ* = aU ln(U/U0) + aR ln(R/Re) + aI ln(I/Ie) + aK ln(Ke/K)
    """
    aU, aR, aI, aK = exponents
    lnU_U0 = math.log(feat["U"] / anchors["U0"])
    lnR_Re = math.log(feat["R_eff"] / anchors["Re"])
    lnI_Ie = math.log(feat["I"] / anchors["Ie"])
    lnKe_K = math.log(anchors["Ke"] / feat["K"])
    return (aU * lnU_U0) + (aR * lnR_Re) + (aI * lnI_Ie) + (aK * lnKe_K)


def write_output(output_path: Path, data: Dict[str, Any], pretty: bool) -> None:
    if not output_path:
        return
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)  # <-- only the parent dir
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4 if pretty else None)


def parse_grid(grid_str: str) -> Dict[str, List[float]]:
    """
    Parse a grid like: "aU=6,7,8,9,10;aR=0.5,1.0,1.5;aI=0.25,0.5,0.75;aK=0.5,1.0,1.5"
    """
    grid: Dict[str, List[float]] = {}
    for part in grid_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Malformed grid token (missing '='): {part!r}", 2)
        key, values_str = part.split("=", 1)
        key = key.strip()
        vals: List[float] = []
        for v in values_str.split(","):
            v = v.strip()
            if v:
                vals.append(float(v))
        if not vals:
            raise ValueError(f"No values provided for grid key '{key}'.", 2)
        grid[key] = vals

    # require canonical keys
    for req in ("aU", "aR", "aI", "aK"):
        if req not in grid:
            raise ValueError(f"Grid missing required key '{req}'.", 2)
    return grid

import hashlib, pathlib, subprocess

def _sha256(path: str) -> str:
    p = pathlib.Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

# --------------------------- Main ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QFD Unified Lifetime Fitter v3 (with chi normalization)")
    parser.add_argument("--muon", required=True, help="Path to muon manifest.json")
    parser.add_argument("--tau", required=True, help="Path to tau manifest.json")

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--exponents", nargs=4, type=float, metavar=("aU", "aR", "aI", "aK"))
    g.add_argument("--grid", type=str,
                   help='Grid like "aU=6,7,8,9,10;aR=0.5,1.0,1.5;aI=0.25,0.5,0.75;aK=0.5,1.0,1.5"')

    a = parser.add_mutually_exclusive_group(required=True)
    a.add_argument("--anchor-bundle", dest="anchor_bundle",
                   help="Path to electron anchor bundle manifest.json")
    a.add_argument("--anchors", nargs=4, type=float, metavar=("U0", "Re", "Ie", "Ke"),
                   help="Anchor values directly")

    parser.add_argument("--chi-mode", default="qR_over_I",
                        choices=["q_over_I", "qR_over_I", "hcsr_over_I", "LdotQ_over_I"])
    parser.add_argument("--electron", help="Path to electron manifest for sanity check")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", help="Path to write output JSON")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--emit-zeeman", action="store_true",
                        help="Run zeeman_probe on each bundle and attach results.")
    parser.add_argument("--emit-features", action="store_true",
                        help="Compute L_mag, Hcsr and write back into results diagnostics.")

    args = parser.parse_args()

    output_path = Path(args.out) if args.out else (Path.cwd() / "calibration_report_v3.json")
    output: Dict[str, Any] = {
        "analysis_metadata": {
            "model": "unified_stability_v3",
            "script": "GPT_fit_lifetimes_v3.py",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "device": args.device,
            "chi_mode": args.chi_mode, # Added chi_mode to meta
        },
        "inputs": {
            "muon_manifest": args.muon,
            "tau_manifest": args.tau,
            "electron_manifest": args.electron,
            "chi_mode": args.chi_mode,
        },
        "fit_result": {},
        "species": {},
        "status": "ok",
        "error": None,
    }

    # Load bundles
    try:
        electron_bundle_data = load_bundle_compat(args.electron)
        muon_bundle_data     = load_bundle_compat(args.muon)
        tau_bundle_data      = load_bundle_compat(args.tau)
        anchor_bundle_data   = load_bundle_compat(args.anchor_bundle or args.electron)

    except Exception as e:
        output["status"] = "error"
        output["error"] = str(e)
        write_output(output_path, output, getattr(args, "pretty", False))
        sys.exit(10)

    # Optional Probes
    if args.emit_zeeman:
        output["zeeman_results"] = {}
        try:
            # Muon Zeeman probe
            muon_zeeman_wrapper = ZeemanBundleWrapper(muon_bundle_data)
            muon_zeeman_result = zeeman_probe(muon_zeeman_wrapper)
            output["zeeman_results"]["muon"] = {
                "species": muon_zeeman_result.species,
                "dE_dB": muon_zeeman_result.dE_dB,
                "a_anom": muon_zeeman_result.a_anom,
            }

            # Tau Zeeman probe
            tau_zeeman_wrapper = ZeemanBundleWrapper(tau_bundle_data)
            tau_zeeman_result = zeeman_probe(tau_zeeman_wrapper)
            output["zeeman_results"]["tau"] = {
                "species": tau_zeeman_result.species,
                "dE_dB": tau_zeeman_result.dE_dB,
                "a_anom": tau_zeeman_result.a_anom,
            }

            # Electron Zeeman probe (if electron bundle is loaded)
            if electron_bundle_data:
                electron_zeeman_wrapper = ZeemanBundleWrapper(electron_bundle_data)
                electron_zeeman_result = zeeman_probe(electron_zeeman_wrapper)
                output["zeeman_results"]["electron"] = {
                    "species": electron_zeeman_result.species,
                    "dE_dB": electron_zeeman_result.dE_dB,
                    "a_anom": electron_zeeman_result.a_anom,
                }

        except Exception as e:
            output["status"] = "error"
            output["error"] = f"Zeeman probe error: {str(e)}"
            write_output(output_path, output, getattr(args, "pretty", False))
            sys.exit(10)

    # Anchors
    # Get anchors
    if args.anchor_bundle:
        output["inputs"]["anchors_source"] = "anchor-bundle"
        anchor_bundle_data = load_bundle_compat(args.anchor_bundle)  # use compat here too
        anchors = extract_anchor_features(anchor_bundle_data)  # uses only U,R,I,K
    else:
        output["inputs"]["anchors_source"] = "cli"
        anchors = {'U0': args.anchors[0], 'Re': args.anchors[1], 'Ie': args.anchors[2], 'Ke': args.anchors[3]}
    output["inputs"]["anchors"] = anchors

    # Exponent sets
    if args.grid:
        output["inputs"]["exponent_mode"] = "grid"
        grid_spec = parse_grid(args.grid)
        output["inputs"]["grid"] = grid_spec
        keys, values = zip(*grid_spec.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        exp_tuples: List[Tuple[float, float, float, float]] = [
            (float(c["aU"]), float(c["aR"]), float(c["aI"]), float(c["aK"])) for c in combos
        ]
    else:
        output["inputs"]["exponent_mode"] = "single"
        au, ar, ai, ak = [float(x) for x in args.exponents]  # type: ignore[arg-type]
        output["inputs"]["exponents_requested"] = {"aU": au, "aR": ar, "aI": ai, "aK": ak}
        exp_tuples = [(au, ar, ai, ak)]

    # Experimental lifetimes (s)
    tau_mu_exp = 2.196_981_1e-6
    tau_tau_exp = 2.903e-13
    y_mu = math.log(tau_mu_exp)
    y_tau = math.log(tau_tau_exp)

    results: List[Dict[str, Any]] = []

    try:
        for exps in exp_tuples:
            mu = extract_all_features(muon_bundle_data, args.chi_mode)
            ta = extract_all_features(tau_bundle_data,  args.chi_mode)
            el: Optional[Dict[str, float]] = None

            # Gather chi values; include electron for normalization if provided
            chis = [mu["chi"], ta["chi"]]
            if electron_bundle_data:
                el = extract_all_features(electron_bundle_data, args.chi_mode)
                chis.append(el["chi"])

            chi_mean = sum(chis) / len(chis)
            chi_std  = math.sqrt(sum((x - chi_mean) ** 2 for x in chis) / len(chis))

            def z(x: float) -> float:
                return (x - chi_mean) / chi_std if chi_std >= 1e-9 else (x - chi_mean)

            mu["chi_normalized"] = z(mu["chi"])
            ta["chi_normalized"] = z(ta["chi"])
            if el is not None:
                el["chi_normalized"] = z(el["chi"])

            # Geometric terms
            lnG_mu = calculate_ln_gamma_star(mu, anchors, exps)
            lnG_ta = calculate_ln_gamma_star(ta, anchors, exps)

            dchi_norm = ta["chi_normalized"] - mu["chi_normalized"]
            if abs(dchi_norm) < 1e-6:
                continue  # too ill-conditioned for a stable closed-form fit

            beta  = ((y_tau - lnG_ta) - (y_mu - lnG_mu)) / dchi_norm
            ln_k0 = y_mu - lnG_mu - beta * mu["chi_normalized"]

            ln_tau_mu_pred = ln_k0 + lnG_mu + beta * mu["chi_normalized"]
            ln_tau_ta_pred = ln_k0 + lnG_ta + beta * ta["chi_normalized"]
            sse = (ln_tau_mu_pred - y_mu) ** 2 + (ln_tau_ta_pred - y_tau) ** 2

            tau_e_pred: Optional[float] = None
            lnG_e: Optional[float] = None
            if el is not None:
                lnG_e = calculate_ln_gamma_star(el, anchors, exps)
                ln_tau_e_pred = ln_k0 + lnG_e + beta * el["chi_normalized"]
                tau_e_pred = math.exp(ln_tau_e_pred)

            results.append({
                "exponents": exps,
                "beta": float(beta),
                "ln_k0": float(ln_k0),
                "sse": float(sse),
                "delta_chi_norm": float(dchi_norm),
                "mu_features": mu,
                "tau_features": ta,
                "electron_features": el,
                "ln_gamma_mu": float(lnG_mu),
                "ln_gamma_tau": float(lnG_ta),
                "ln_gamma_e": None if lnG_e is None else float(lnG_e),
                "tau_e_pred": tau_e_pred,
                "chi_mean": float(chi_mean),
                "chi_std_dev": float(chi_std),
            })

        if not results:
            raise ValueError("All exponent tuples were unsolvable (|Δχ_norm| too small).", 7)

        # Best by SSE
        best = min(results, key=lambda r: r["sse"])
        aU, aR, aI, aK = best["exponents"]

        # k0 from ln_k0 (safe)
        k0_val, k0_over, _ = safe_exp(best['ln_k0'])

        output["fit_result"] = {
            "best_exponents": dict(zip(['aU','aR','aI','aK'], best['exponents'])),
            "beta": best['beta'],
            "k0": k0_val,
            "ln_k0": best['ln_k0'],
            "sse": best['sse'],
            "diagnostics": {
                "delta_chi": best['delta_chi_norm'], # Renamed delta_chi_norm to delta_chi
                "beta_sign_ok": best['beta'] > 0,
                "chi_mean": best['chi_mean'], # Use best['chi_mean'] instead of chi_mean
                "chi_std_dev": best['chi_std_dev'], # Use best['chi_std_dev'] instead of chi_std_dev
                "overflow": {"k0": k0_over}
            }
        }

        # Per-species
        # MUON
        ln_tau_pred_mu = best['ln_k0'] + best['ln_gamma_mu'] + best['beta'] * best['mu_features']['chi_normalized']
        tau_mu, of_mu, _ = safe_exp(ln_tau_pred_mu)
        output["species"]["muon"] = {
            "features": best['mu_features'],
            "ln_gamma_star": best['ln_gamma_mu'],
            "ln_tau_exp": y_mu,
            "ln_tau_pred": ln_tau_pred_mu,
            "tau_pred_seconds": tau_mu,
            "overflow": of_mu
        }
        # TAU
        ln_tau_pred_tau = best['ln_k0'] + best['ln_gamma_tau'] + best['beta'] * best['tau_features']['chi_normalized']
        tau_tau, of_tau, _ = safe_exp(ln_tau_pred_tau)
        output["species"]["tau"] = {
            "features": best['tau_features'],
            "ln_gamma_star": best['ln_gamma_tau'],
            "ln_tau_exp": y_tau,
            "ln_tau_pred": ln_tau_pred_tau,
            "tau_pred_seconds": tau_tau,
            "overflow": of_tau
        }

        if electron_bundle_data:
            # reuse electron_features_for_norm built earlier; rebuild features for gamma
            electron_features = extract_all_features(electron_bundle_data, args.chi_mode)
            ln_gamma_e = calculate_ln_gamma_star(electron_features, anchors, best['exponents'])
            # use the freshly-computed electron_features (not the loop-local 'el'), and protect against tiny std dev
            chi_std_den = best['chi_std_dev'] if best['chi_std_dev'] >= 1e-9 else 1.0
            ln_tau_pred_e = best['ln_k0'] + ln_gamma_e + best['beta'] * (electron_features['chi'] - best['chi_mean']) / chi_std_den
            tau_e, of_e, _ = safe_exp(ln_tau_pred_e)
            output["fit_result"]["diagnostics"]["electron_check"] = {
                "evaluated": True,
                "tau_pred_seconds": tau_e,
                "ln_tau_pred": ln_tau_pred_e,
                "overflow": of_e,
                "warn_unstable_electron": (not of_e) and (tau_e < 1e20)
            }
        else:
            output["fit_result"]["diagnostics"]["electron_check"] = {
                "evaluated": False, "tau_pred_seconds": None, "overflow": False, "warn_unstable_electron": False
            }

    except ValueError as e:
        code = e.args[1] if len(e.args) > 1 and isinstance(e.args[1], int) else 10
        output["status"] = "error"
        output["error"] = str(e.args[0] if e.args else e)
        # ensure output is written on controlled failures
        write_output(output_path, output, getattr(args, "pretty", False))
        sys.exit(code)

    except Exception as e:
        output["status"] = "error"
        output["error"] = str(e)
        # ensure output is written on unexpected failures
        write_output(output_path, output, getattr(args, "pretty", False))
        sys.exit(10)

    finally:
        # ensure stable top-level Zeeman block for downstream tools
        if "zeeman_results" not in output: # Use 'output' instead of 'out_json'
            zr = output.get("zeeman") \
                 or output.get("fit_result", {}).get("zeeman") \
                 or output.get("diagnostics", {}).get("zeeman")
            if zr is not None:
                output["zeeman_results"] = zr

        try:
            git_rev = subprocess.run(["git","rev-parse","HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:
            git_rev = None

        output.setdefault("provenance", {})
        output["provenance"]["git_rev"] = git_rev
        output["provenance"]["manifests"] = {
            "electron": {"path": args.electron, "sha256": _sha256(args.electron)},
            "muon":     {"path": args.muon,     "sha256": _sha256(args.muon)},
            "tau":      {"path": args.tau,      "sha256": _sha256(args.tau)},
        }

        # always attempt to write the output summary (best-effort)
        write_output(output_path, output, getattr(args, "pretty", False))


if __name__ == "__main__":
    main()
