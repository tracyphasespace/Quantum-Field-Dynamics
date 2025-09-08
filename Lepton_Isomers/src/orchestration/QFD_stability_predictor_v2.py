#!/usr/bin/env python3
"""
GPT_stability_predictor_v2.py

Compute a QFD lifetime prediction for a single converged bundle.
"""

import argparse
import json
import sys
import math
from pathlib import Path
from datetime import datetime

# --- Data Loading ---

def load_bundle_data(manifest_path: Path) -> dict:
    """Loads all data associated with a bundle from its manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    bundle_data = {'manifest': manifest, 'summary': None, 'features': {}}
    
    summary_path = manifest_path.parent / manifest['summary']
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with open(summary_path, 'r') as f:
        bundle_data['summary'] = json.load(f)

    if 'extras' in manifest and manifest['extras']:
        features_path = manifest_path.parent / manifest['extras']
        if features_path.is_file():
            with open(features_path, 'r') as f:
                bundle_data['features'] = json.load(f)

    return bundle_data

# --- Feature Extraction ---

def extract_feature(bundle_data, key, summary_key, exit_code):
    """Generic feature extractor with fallback logic."""
    source = "missing"
    val = bundle_data.get('summary', {}).get(summary_key)
    if val is not None:
        source = "summary"
    else:
        val = bundle_data.get('features', {}).get(key)
        if val is not None:
            source = "features"

    if val is None:
        # In a full implementation, field computation would happen here.
        raise ValueError(f"Feature '{key}' could not be found.", exit_code)
    
    return val, source

def extract_all_features(bundle_data: dict, chi_mode: str) -> dict:
    """Extracts all required features for a bundle."""
    features = {}
    sources = {}

    try:
        features['U'], sources['U'] = extract_feature(bundle_data, 'U', 'U_final', 2)
        features['R_eff'], sources['R_eff'] = extract_feature(bundle_data, 'R_eff', 'R_eff_final', 3)
        features['I'], sources['I'] = extract_feature(bundle_data, 'I', 'I_final', 3)
        features['K'], sources['K'] = extract_feature(bundle_data, 'K', 'Hkin_final', 4)

        q_star = bundle_data.get('summary', {}).get('Q_proxy_final')
        if q_star is None:
            raise ValueError("Q_proxy_final not found in summary.", 8)

        if chi_mode == 'q_over_I':
            features['chi'] = abs(q_star) / features['I']
        elif chi_mode == 'qR_over_I':
            features['chi'] = abs(q_star) * features['R_eff'] / features['I']
        elif chi_mode == 'LdotQ_over_I':
            l_mag, _ = extract_feature(bundle_data, 'L_mag', 'L_mag', 5)
            features['chi'] = abs(l_mag) * abs(q_star) / features['I']
        elif chi_mode == 'hcsr_over_I':
            h_csr, _ = extract_feature(bundle_data, 'H_csr', 'Hcsr_final', 8)
            features['chi'] = abs(h_csr) / features['I']
        sources['chi'] = "computed"

    except ValueError as e:
        if e.args[1] == 2: raise ValueError("missing_U", 2)
        if e.args[1] == 3: raise ValueError("missing_I", 3)
        if e.args[1] == 4: raise ValueError("missing_K", 4)
        if e.args[1] == 5: raise ValueError("missing_chi", 5)
        raise e

    return {"values": features, "sources": sources}

def write_output(output_path, data, pretty):
    """Writes the output JSON file."""
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4 if pretty else None)

def main():
    parser = argparse.ArgumentParser(description="QFD Unified Stability Predictor v2")
    parser.add_argument("--manifest", required=True, help="Path to the bundle’s manifest.")
    parser.add_argument("--exponents", required=True, nargs=4, type=float, metavar=('aU', 'aR', 'aI', 'aK'), help="Four floats for the geometric exponents.")
    parser.add_argument("--species", help="Tag used in the output (default inferred from manifest).", default=None)
    anchor_group = parser.add_mutually_exclusive_group(required=True)
    anchor_group.add_argument("--anchor-bundle", help="Path to the electron anchor bundle manifest.")
    anchor_group.add_argument("--anchors", nargs=4, type=float, metavar=('U0', 'Re', 'Ie', 'Ke'), help="Four floats for U0, Re, Ie, Ke.")
    parser.add_argument("--chi-mode", default='q_over_I', choices=['q_over_I', 'qR_over_I', 'LdotQ_over_I', 'hcsr_over_I'], help="Mode for calculating the chi parameter.")
    parser.add_argument("--k0", type=float, help="Value for the k0 parameter.")
    parser.add_argument("--beta", type=float, help="Value for the beta parameter.")
    parser.add_argument("--device", default='cpu', choices=['cpu', 'cuda'], help="Device for any field-derived integrals.")
    parser.add_argument("--out", help="Path to write output JSON.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the output JSON.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.out) if args.out else manifest_path.parent / "stability_report_v2.json"

    output_data = {
        "analysis_metadata": {
            "model": "unified_stability_v2",
            "script": "GPT_stability_predictor_v2.py",
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "device": args.device,
            "species": args.species
        },
        "inputs": {
            "manifest": str(manifest_path),
            "exponents": dict(zip(['aU', 'aR', 'aI', 'aK'], args.exponents)),
            "chi_mode": args.chi_mode,
            "params": {"k0": args.k0, "beta": args.beta}
        },
        "features": {},
        "prediction": {},
        "status": "ok",
        "error": None
    }

    try:
        target_bundle = load_bundle_data(manifest_path)
        if not args.species and 'species' in target_bundle['manifest']:
            output_data["analysis_metadata"]["species"] = target_bundle['manifest']['species']

        target_features_data = extract_all_features(target_bundle, args.chi_mode)
        target_features = target_features_data["values"]
        output_data["features"] = target_features
        output_data["features"]["sources"] = target_features_data["sources"]

        anchors = {}
        if args.anchor_bundle:
            output_data["inputs"]["anchors_source"] = "anchor-bundle"
            anchor_bundle_data = load_bundle_data(Path(args.anchor_bundle))
            anchor_features_data = extract_all_features(anchor_bundle_data, args.chi_mode)
            anchor_features = anchor_features_data["values"]
            anchors = {'U0': anchor_features['U'], 'Re': anchor_features['R_eff'], 'Ie': anchor_features['I'], 'Ke': anchor_features['K']}
        else:
            output_data["inputs"]["anchors_source"] = "cli"
            anchors = {'U0': args.anchors[0], 'Re': args.anchors[1], 'Ie': args.anchors[2], 'Ke': args.anchors[3]}
        output_data["inputs"]["anchors"] = anchors

        # Calculate logs and gamma_geom_star
        aU, aR, aI, aK = args.exponents
        lnU_U0 = math.log(target_features['U'] / anchors['U0'])
        lnR_Re = math.log(target_features['R_eff'] / anchors['Re'])
        lnI_Ie = math.log(target_features['I'] / anchors['Ie'])
        lnKe_K = math.log(anchors['Ke'] / target_features['K'])
        
        ln_gamma_geom_star = (aU * lnU_U0) + (aR * lnR_Re) + (aI * lnI_Ie) + (aK * lnKe_K)
        gamma_geom_star = math.exp(ln_gamma_geom_star)

        output_data["features"]["logs"] = {
            "lnU_U0": lnU_U0, "lnR_Re": lnR_Re, "lnI_Ie": lnI_Ie, "lnKe_K": lnKe_K,
            "gamma_geom_star": gamma_geom_star, "ln_gamma_geom_star": ln_gamma_geom_star
        }

        # Predict lifetime
        has_params = args.k0 is not None and args.beta is not None
        output_data["prediction"]["has_params"] = has_params
        if has_params:
            ln_k0 = math.log(args.k0)
            ln_tau_pred = ln_k0 + ln_gamma_geom_star + args.beta * target_features['chi']
            tau_pred_seconds = math.exp(ln_tau_pred)
            output_data["prediction"]["ln_tau_pred"] = ln_tau_pred
            output_data["prediction"]["tau_pred_seconds"] = tau_pred_seconds

        # Print summary
        summary_line = f"STAB2 {output_data['analysis_metadata']['species']} \u0393*={gamma_geom_star:.3e} chi={target_features['chi']:.3e}"
        if has_params:
            summary_line += f" -> \u03c4={tau_pred_seconds:.3e}s"
        print(summary_line)

    except ValueError as e:
        exit_code = e.args[1] if len(e.args) > 1 else 10
        output_data["status"] = e.args[0]
        output_data["error"] = f"Feature error: {e.args[0]}"
        print(f"STAB2 ERROR: {output_data['error']}", file=sys.stderr)
        write_output(output_path, output_data, args.pretty)
        sys.exit(exit_code)
    except Exception as e:
        output_data["status"] = "error"
        output_data["error"] = str(e)
        print(f"STAB2 ERROR: {e}", file=sys.stderr)
        write_output(output_path, output_data, args.pretty)
        sys.exit(10)

    write_output(output_path, output_data, args.pretty)

if __name__ == "__main__":
    main()