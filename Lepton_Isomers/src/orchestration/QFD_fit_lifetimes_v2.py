#!/usr/bin/env python3
"""
GPT_fit_lifetimes_v2.py

Closed-form calibration of (beta, k0) for the unified model.
"""

import argparse
import json
import sys
import math
import itertools
from pathlib import Path
from datetime import datetime, timezone

# --- Shared Functions (from predictor) ---

def load_bundle_data(manifest_path: Path) -> dict:
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

def extract_feature(bundle_data, key, summary_key, exit_code):
    source = "missing"
    val = bundle_data.get('summary', {}).get(summary_key)
    if val is not None:
        source = "summary"
    else:
        val = bundle_data.get('features', {}).get(key)
        if val is not None:
            source = "features"
    if val is None:
        raise ValueError(f"Feature '{key}' could not be found.", exit_code)
    return val, source

def extract_all_features(bundle_data: dict, chi_mode: str) -> dict:
    features = {}
    try:
        features['U'], _ = extract_feature(bundle_data, 'U', 'U_final', 8)
        features['R_eff'], _ = extract_feature(bundle_data, 'R_eff', 'R_eff_final', 8)
        features['I'], _ = extract_feature(bundle_data, 'I', 'I_final', 8)
        features['K'], _ = extract_feature(bundle_data, 'K', 'Hkin_final', 8)
        q_star = bundle_data.get('summary', {}).get('Q_proxy_final')
        if q_star is None: raise ValueError("Q_proxy_final not found in summary.", 8)
        if chi_mode == 'q_over_I':
            features['chi'] = abs(q_star) / features['I']
        elif chi_mode == 'qR_over_I':
            features['chi'] = abs(q_star) * features['R_eff'] / features['I']
        elif chi_mode == 'LdotQ_over_I':
            l_mag, _ = extract_feature(bundle_data, 'L_mag', 'L_mag', 8)
            features['chi'] = abs(l_mag) * abs(q_star) / features['I']
        elif chi_mode == 'hcsr_over_I':
            h_csr, _ = extract_feature(bundle_data, 'H_csr', 'Hcsr_final', 8)
            features['chi'] = abs(h_csr) / features['I']
    except ValueError as e:
        raise ValueError(f"missing_features: {e.args[0]}", 8)
    return features

def calculate_ln_gamma_star(features, anchors, exponents):
    aU, aR, aI, aK = exponents
    lnU_U0 = math.log(features['U'] / anchors['U0'])
    lnR_Re = math.log(features['R_eff'] / anchors['Re'])
    lnI_Ie = math.log(features['I'] / anchors['Ie'])
    lnKe_K = math.log(anchors['Ke'] / features['K'])
    return (aU * lnU_U0) + (aR * lnR_Re) + (aI * lnI_Ie) + (aK * lnKe_K)

def write_output(output_path, data, pretty):
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4 if pretty else None)

# --- Main Script Logic ---

def parse_grid(grid_str: str) -> dict:
    grid = {}
    parts = grid_str.split(';')
    for part in parts:
        key, values_str = part.split('=')
        grid[key.strip()] = [float(v) for v in values_str.split(',')]
    return grid

def main():
    parser = argparse.ArgumentParser(description="QFD Unified Lifetime Fitter v2")
    parser.add_argument("--muon", required=True, help="Path to muon manifest.")
    parser.add_argument("--tau", required=True, help="Path to tau manifest.")
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument("--exponents", nargs=4, type=float, metavar=('aU', 'aR', 'aI', 'aK'))
    exp_group.add_argument("--grid", type=str)
    anchor_group = parser.add_mutually_exclusive_group(required=True)
    anchor_group.add_argument("--anchor-bundle", help="Path to electron anchor bundle.")
    anchor_group.add_argument("--anchors", nargs=4, type=float, metavar=('U0', 'Re', 'Ie', 'Ke'))
    parser.add_argument("--chi-mode", default="qR_over_I", choices=["q_over_I","qR_over_I","hcsr_over_I","LdotQ_over_I"])
    parser.add_argument("--electron", help="Path to electron manifest for consistency check.")
    parser.add_argument("--device", default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--out", help="Path to write output JSON.")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.out) if args.out else Path.cwd() / "calibration_report.json"
    output_data = {
        "analysis_metadata": {
            "model": "unified_stability_v2",
            "script": "GPT_fit_lifetimes_v2.py",
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "device": args.device
        },
        "inputs": {
            "muon_manifest": args.muon,
            "tau_manifest": args.tau,
            "electron_manifest": args.electron,
            "chi_mode": args.chi_mode
        },
        "fit_result": {},
        "species": {},
        "status": "ok",
        "error": None
    }

    try:
        # Load bundles
        muon_bundle = load_bundle_data(Path(args.muon))
        tau_bundle = load_bundle_data(Path(args.tau))
        electron_bundle = load_bundle_data(Path(args.electron)) if args.electron else None

        # Get anchors
        if args.anchor_bundle:
            output_data["inputs"]["anchors_source"] = "anchor-bundle"
            anchor_bundle_data = load_bundle_data(Path(args.anchor_bundle))
            anchor_features = extract_all_features(anchor_bundle_data, args.chi_mode)
            anchors = {'U0': anchor_features['U'], 'Re': anchor_features['R_eff'], 'Ie': anchor_features['I'], 'Ke': anchor_features['K']}
        else:
            output_data["inputs"]["anchors_source"] = "cli"
            if args.anchors and len(args.anchors) == 4:
                anchors = {'U0': args.anchors[0], 'Re': args.anchors[1], 'Ie': args.anchors[2], 'Ke': args.anchors[3]}
            else:
                raise ValueError("Invalid --anchors provided. Expected 4 values.", 9)
        output_data["inputs"]["anchors"] = anchors

        # Get exponent combinations
        if args.grid:
            output_data["inputs"]["exponent_mode"] = "grid"
            grid_spec = parse_grid(args.grid)
            output_data["inputs"]["grid"] = grid_spec
            keys, values = zip(*grid_spec.items())
            exponent_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
            exponent_tuples = [tuple(c[k] for k in ['aU','aR','aI','aK']) for c in exponent_combos]
        else:
            output_data["inputs"]["exponent_mode"] = "single"
            output_data["inputs"]["exponents_requested"] = dict(zip(['aU', 'aR', 'aI', 'aK'], args.exponents))
            exponent_tuples = [tuple(args.exponents)]

        # Main loop
        results = []
        y_mu = math.log(2.1969811e-6)
        y_tau = math.log(2.903e-13)
        for exponents in exponent_tuples:
            mu_features = extract_all_features(muon_bundle, args.chi_mode)
            tau_features = extract_all_features(tau_bundle, args.chi_mode)
            ln_gamma_mu = calculate_ln_gamma_star(mu_features, anchors, exponents)
            ln_gamma_tau = calculate_ln_gamma_star(tau_features, anchors, exponents)

            y_mu = math.log(2.1969811e-6)
            y_tau = math.log(2.903e-13)

            delta_chi = tau_features['chi'] - mu_features['chi']
            if abs(delta_chi) < 1e-9:
                continue # Skip unsolvable combination

            beta = ((y_tau - ln_gamma_tau) - (y_mu - ln_gamma_mu)) / delta_chi
            ln_k0 = y_mu - ln_gamma_mu - beta * mu_features['chi']

            ln_tau_pred_mu = ln_k0 + ln_gamma_mu + beta * mu_features['chi']
            ln_tau_pred_tau = ln_k0 + ln_gamma_tau + beta * tau_features['chi']
            sse = (ln_tau_pred_mu - y_mu)**2 + (ln_tau_pred_tau - y_tau)**2
            results.append({'exponents': exponents, 'beta': beta, 'ln_k0': ln_k0, 'sse': sse, 'delta_chi': delta_chi, 'mu_features': mu_features, 'tau_features': tau_features, 'ln_gamma_mu': ln_gamma_mu, 'ln_gamma_tau': ln_gamma_tau})

        if not results:
            raise ValueError("All exponent tuples were unsolvable (delta_chi was too small).", 7)

        best_result = min(results, key=lambda x: x['sse'])

        # Populate output
        output_data["fit_result"] = {
            "best_exponents": dict(zip(['aU', 'aR', 'aI', 'aK'], best_result['exponents'])),
            "beta": best_result['beta'], "k0": math.exp(best_result['ln_k0']), "ln_k0": best_result['ln_k0'], "sse": best_result['sse'],
            "diagnostics": {"delta_chi": best_result['delta_chi'], "beta_sign_ok": best_result['beta'] > 0}
        }
        output_data["species"]["muon"] = {"features": best_result['mu_features'], "ln_gamma_star": best_result['ln_gamma_mu'], "ln_tau_exp": y_mu, "ln_tau_pred": best_result['ln_k0'] + best_result['ln_gamma_mu'] + best_result['beta'] * best_result['mu_features']['chi'], "tau_pred_seconds": math.exp(best_result['ln_k0'] + best_result['ln_gamma_mu'] + best_result['beta'] * best_result['mu_features']['chi'])}
        output_data["species"]["tau"] = {"features": best_result['tau_features'], "ln_gamma_star": best_result['ln_gamma_tau'], "ln_tau_exp": y_tau, "ln_tau_pred": best_result['ln_k0'] + best_result['ln_gamma_tau'] + best_result['beta'] * best_result['tau_features']['chi'], "tau_pred_seconds": math.exp(best_result['ln_k0'] + best_result['ln_gamma_tau'] + best_result['beta'] * best_result['tau_features']['chi'])}

        if electron_bundle:
            electron_features = extract_all_features(electron_bundle, args.chi_mode)
            ln_gamma_electron = calculate_ln_gamma_star(electron_features, anchors, best_result['exponents'])
            ln_tau_pred_electron = best_result['ln_k0'] + ln_gamma_electron + best_result['beta'] * electron_features['chi']
            tau_e_pred = math.exp(ln_tau_pred_electron)
            output_data["fit_result"]["diagnostics"]["electron_check"] = {"evaluated": True, "tau_pred_seconds": tau_e_pred, "warn_unstable_electron": tau_e_pred < 1e20}
        else:
            output_data["fit_result"]["diagnostics"]["electron_check"] = {"evaluated": False, "tau_pred_seconds": None, "warn_unstable_electron": False}

        # Print summary
        tau_e_pred_str = f"{output_data['fit_result']['diagnostics']['electron_check']['tau_pred_seconds']:.2e}" if output_data['fit_result']['diagnostics']['electron_check']['tau_pred_seconds'] is not None else "N/A"
        summary_line = f"FIT2 best: {' '.join([f'{k}={v}' for k,v in output_data['fit_result']['best_exponents'].items()])} | beta={output_data['fit_result']['beta']:.2f} | k0={output_data['fit_result']['k0']:.2e} s | dchi={output_data['fit_result']['diagnostics']['delta_chi']:.2e} | tau_e={tau_e_pred_str}s | SSE={output_data['fit_result']['sse']:.2e}"
        print(summary_line)
        if output_data['fit_result']['diagnostics']['beta_sign_ok'] is False:
            print("WARNING: beta is negative or zero. Consider flipping chi sign or revisiting chi-mode.")

    except ValueError as e:
        exit_code = e.args[1] if len(e.args) > 1 and len(e.args) > 1 else 10
        output_data["status"] = e.args[0] if len(e.args) > 0 else "ValueError"
        output_data["error"] = f"Feature error: {e.args[0]}" if len(e.args) > 0 else "ValueError occurred"
        write_output(output_path, output_data, args.pretty)
        sys.exit(exit_code)
    except Exception as e:
        output_data["status"] = "error"
        output_data["error"] = str(e)
        write_output(output_path, output_data, args.pretty)
        sys.exit(10)

    write_output(output_path, output_data, args.pretty)

if __name__ == "__main__":
    main()