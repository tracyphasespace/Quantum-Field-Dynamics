#!/usr/bin/env python3
"""
Create v2 Bundles from Rosetta Run
==================================

This script migrates the data from the 'rosetta_run_outputs' directory
into the new v2 bundle structure, creating the necessary manifests and
renaming files to match the new contract.

Usage:
    python examples/create_v2_bundles.py
"""

import json
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

# ---- helpers: put near top of file ----
COMPLETE_KEYS = {"q_star","R_eff","I","K","effective_radius","inertia"}

def is_complete_summary(j: dict) -> bool:
    if not isinstance(j, dict):
        return False
    # look in features, base, or root
    cands = []
    for node in (j.get("features"), j.get("base"), j):
        if isinstance(node, dict):
            have = {k for k in node.keys() if k in COMPLETE_KEYS and node.get(k) not in (None, 0, 0.0, "NaN")}
            cands.append(len(have))
    return max(cands, default=0) >= 3

def merge_dict(dst: dict, src: dict) -> dict:
    for k,v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge_dict(dst[k], v)
        elif k not in dst or dst[k] in (None, 0, 0.0, "NaN", "", []):
            dst[k] = v
    return dst




def coalesce(d, *names, default=None):
    for n in names:
        v = d.get(n)
        if isinstance(v, (int,float)) and v == v:
            return float(v)
    return default

# search in features/base/root
def lookup(summary, *keys, default=None):
    for node in (summary.get("features",{}), summary.get("base",{}), summary):
        val = coalesce(node, *keys, default=None)
        if val is not None:
            return val
    return default


def compute_L_features(bundle):
    # L_mag and LdotQ_over_I are not directly available in summary.json
    # Setting to 0.0 as placeholders for now.
    return SimpleNamespace(
        L_mag = 0.0,
        LdotQ_over_I = 0.0
    )

def compute_csr_features(bundle):
    summary_data = bundle["summary"]
    Hcsr = summary_data.get("Hcsr_final", 0.0)
    I = summary_data.get("I_final", 0.0)

    hcsr_over_I = Hcsr / I if I != 0 else 0.0 # Avoid division by zero

    return SimpleNamespace(
        Hcsr = Hcsr,
        hcsr_over_I = hcsr_over_I
    )

def find_latest_rosetta_results(base_dir: Path, particle: str) -> Path:
    """Find the directory of the latest ladder step for a particle."""
    particle_dir = base_dir / particle
    if not particle_dir.exists():
        raise FileNotFoundError(f"Rosetta directory not found: {particle_dir}")

    ladder_dirs = sorted([d for d in particle_dir.iterdir() if d.is_dir() and d.name.startswith('ladder_')], reverse=True)
    if not ladder_dirs:
        raise FileNotFoundError(f"No ladder directories found in {particle_dir}")
    return ladder_dirs[0]

def main():
    """Create the v2 bundles."""
    source_dir = Path("rosetta_run_outputs")
    dest_base_dir = Path("v2_bundles")

    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' not found.")
        return 1

    print(f"Creating v2 bundles in '{dest_base_dir}'...")

    particle_map = {
        "electron": "electron_511keV_v1",
        "muon": "muon_105MeV_v1",
        "tau": "tau_1777MeV_v1"
    }

    for particle, bundle_name in particle_map.items():
        try:
            print(f"Processing {particle}...")
            latest_run_dir = find_latest_rosetta_results(source_dir, particle)
            
            # Create destination directory
            bundle_dest_dir = dest_base_dir / bundle_name
            bundle_dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy and rename summary file
            source_results_json = latest_run_dir / "results.json"
            dest_summary_json = bundle_dest_dir / "summary.json"

            # if a summary.json already exists and looks more complete, KEEP it
            if dest_summary_json.exists():
                try:
                    existing = json.loads(dest_summary_json.read_text(encoding="utf-8"))
                    if is_complete_summary(existing):
                        # do not overwrite; optionally merge results.json into it
                        try:
                            results = json.loads(source_results_json.read_text(encoding="utf-8"))
                            merged = merge_dict(existing, results if isinstance(results, dict) else {})
                            dest_summary_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                    else:
                        shutil.copy2(source_results_json, dest_summary_json)  # overwrite only if old one was incomplete
                except Exception:
                    shutil.copy2(source_results_json, dest_summary_json)
            else:
                shutil.copy2(source_results_json, dest_summary_json)

            # Copy any associated .npy files
            for npy_file in latest_run_dir.glob("*.npy"):
                shutil.copy(npy_file, bundle_dest_dir)

            # after summary.json is finalized:
            summary = json.loads((bundle_dest_dir/"summary.json").read_text(encoding="utf-8"))

            

            features = {
                "q_star": lookup("q_star","q","q_abs","qMag","qnorm"),
                "R_eff" : lookup("R_eff","Reff","effective_radius","R_eff_final","R"),
                "I"     : lookup("I","current","Ie","I_e","I_final"),
                "K"     : lookup("K","kappa","Ke","K_e"),
                "Hcsr"  : lookup("Hcsr","Hcsr_final"), # Add Hcsr here
            }

            # add L_mag/Hcsr placeholders or compute if you have inputs
            features.setdefault("L_mag", lookup("L_mag", default=1.0))

            # write features.json and reference it
            (bundle_dest_dir/"features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")

            # ensure manifest points to features.json (extras or features field)
            manifest = json.loads((bundle_dest_dir/"bundle_manifest.json").read_text(encoding="utf-8"))
            manifest["features"] = str((bundle_dest_dir/"features.json").name)
            # or: manifest.setdefault("extras", {})["features"] = "features.json"
            (bundle_dest_dir/"bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            print(f"  -> Successfully created bundle: {bundle_dest_dir}")

        except FileNotFoundError as e:
            print(f"  -> SKIPPING: {e}")
            continue

    print("\nBundle creation process complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())