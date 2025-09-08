#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build minimal, self-consistent v2 bundles for (electron, muon, tau).

What this guarantees (so QFD_fit_lifetimes_v3.py runs cleanly):
- features.json has: q_star, R_eff, I, K, L_mag, Hcsr   (all numbers)
- summary.json  has: U_final, energy, magnetic_moment   (all numbers)
- bundle_manifest.json references those files and sets species.

These are 'physics-lite' placeholders meant to unblock the pipeline.
You can later regenerate with real values without changing the fitter.

Usage (from repo root):
  python tools/build_min_bundles.py
  # or tweak defaults:
  python tools/build_min_bundles.py --outdir v2_bundles --overwrite ^
    --qstar 1.0 --Re 1.0,1.05,0.90 --Ie 1.0,0.95,1.10 --K 1.0 ^
    --U0 1.0 --energy 1.0 --mu_mag 1.0 --Lmag 1.0 --Hcsr 1.0

Emits:
  v2_bundles/<species>_<label>/features.json
  v2_bundles/<species>_<label>/summary.json
  v2_bundles/<species>_<label>/bundle_manifest.json
"""

from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, List

SPECIES = ("electron", "muon", "tau")

@dataclass
class FeatureSet:
    q_star: float
    R_eff: float
    I: float
    K: float
    L_mag: float
    Hcsr: float

@dataclass
class Summary:
    created_utc: str
    species: str
    provenance: Dict[str, str]
    # Scalars the fitter / Zeeman expect:
    U_final: float
    energy: float
    magnetic_moment: float

def parse_triplet(v: str, label: str) -> Tuple[float, float, float]:
    """
    Parse 'a,b,c' into tuple of floats; if single value given ('x'), expand to (x,x,x).
    """
    parts = [p.strip() for p in v.split(",")]
    if len(parts) == 1:
        try:
            x = float(parts[0])
        except ValueError:
            raise SystemExit(f"--{label}: could not parse '{v}' as float")
        return (x, x, x)
    if len(parts) != 3:
        raise SystemExit(f"--{label}: expected 1 or 3 comma-separated numbers, got '{v}'")
    try:
        a, b, c = map(float, parts)
    except ValueError:
        raise SystemExit(f"--{label}: could not parse '{v}' as three floats")
    return (a, b, c)

def write_json(path: Path, obj: Dict, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent), encoding="utf-8")

def build_bundle(root: Path,
                 species: str,
                 label: str,
                 features: FeatureSet,
                 summary_vals: Summary,
                 overwrite: bool) -> Path:
    """
    Write features.json, summary.json, bundle_manifest.json under:
      <root>/<species>_<label>/
    Return path to bundle_manifest.json
    """
    bundle_dir = root / f"{species}_{label}"
    features_path = bundle_dir / "features.json"
    summary_path  = bundle_dir / "summary.json"
    manifest_path = bundle_dir / "bundle_manifest.json"

    if not overwrite and manifest_path.exists():
        raise SystemExit(f"Refusing to overwrite existing bundle: {manifest_path} (use --overwrite)")

    write_json(features_path, asdict(features))
    write_json(summary_path, asdict(summary_vals))

    manifest = {
        "schema": "v2_bundle_manifest",
        "species": species,
        # Keep these relative; our orchestrators join against the bundle_dir:
        "features": features_path.name,
        "summary": summary_path.name,
        "extras": {},  # reserved for future
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
    }
    write_json(manifest_path, manifest)
    return manifest_path

def main():
    ap = argparse.ArgumentParser(description="Build minimal v2 bundles with complete keys for fitter + Zeeman.")
    ap.add_argument("--outdir", default="v2_bundles", help="Output directory for bundles (default: v2_bundles)")
    ap.add_argument("--label", default="v1", help="Bundle label suffix, e.g. 'v1' → electron_511keV_v1 (default: v1)")

    # Core feature scalars (triplets map to e, mu, tau in that order)
    ap.add_argument("--qstar", default="1.0", help="q_star value (single or triplet 'e,mu,tau'). Default 1.0")
    ap.add_argument("--Re",    default="1.0,1.05,0.90", help="R_eff for (e,mu,tau). Default '1.0,1.05,0.90'")
    ap.add_argument("--Ie",    default="1.0,0.95,1.10", help="I for (e,mu,tau). Default '1.0,0.95,1.10'")
    ap.add_argument("--K",     default="1.0", help="K value (single or triplet). Default 1.0")
    ap.add_argument("--Lmag",  default="1.0", help="L_mag value (single or triplet). Default 1.0")
    ap.add_argument("--Hcsr",  default="1.0", help="Hcsr value (single or triplet). Default 1.0")

    # Summary scalars required by fitter/Zeeman
    ap.add_argument("--U0",     default="1.0", help="U_final (single or triplet). Default 1.0")
    ap.add_argument("--energy", default="1.0", help="Energy for Zeeman calc (single or triplet). Default 1.0")
    ap.add_argument("--mu_mag", default="1.0", help="Magnetic moment for Zeeman calc (single or triplet). Default 1.0")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing bundles if present.")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    # Parse per-species values
    q_e, q_mu, q_tau     = parse_triplet(args.qstar, "qstar")
    Re_e, Re_mu, Re_tau  = parse_triplet(args.Re, "Re")
    Ie_e, Ie_mu, Ie_tau  = parse_triplet(args.Ie, "Ie")
    K_e, K_mu, K_tau     = parse_triplet(args.K, "K")
    L_e, L_mu, L_tau     = parse_triplet(args.Lmag, "Lmag")
    H_e, H_mu, H_tau     = parse_triplet(args.Hcsr, "Hcsr")

    U_e, U_mu, U_tau         = parse_triplet(args.U0, "U0")
    En_e, En_mu, En_tau      = parse_triplet(args.energy, "energy")
    Mu_e, Mu_mu, Mu_tau      = parse_triplet(args.mu_mag, "mu_mag")

    per_species = {
        "electron": {
            "features": FeatureSet(q_star=q_e,   R_eff=Re_e,  I=Ie_e,  K=K_e,  L_mag=L_e,  Hcsr=H_e),
            "summary":  (U_e, En_e, Mu_e),
        },
        "muon": {
            "features": FeatureSet(q_star=q_mu,  R_eff=Re_mu, I=Ie_mu, K=K_mu, L_mag=L_mu, Hcsr=H_mu),
            "summary":  (U_mu, En_mu, Mu_mu),
        },
        "tau": {
            "features": FeatureSet(q_star=q_tau, R_eff=Re_tau, I=Ie_tau, K=K_tau, L_mag=L_tau, Hcsr=H_tau),
            "summary":  (U_tau, En_tau, Mu_tau),
        },
    }

    built: List[Path] = []
    for sp in SPECIES:
        f: FeatureSet = per_species[sp]["features"]
        U_final, energy, magnetic_moment = per_species[sp]["summary"]

        summary = Summary(
            created_utc = datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            species     = sp,
            provenance  = {"source": "build_min_bundles", "note": "minimal placeholders"},
            U_final     = float(U_final),
            energy      = float(energy),
            magnetic_moment = float(magnetic_moment),
        )

        # name bundles like: electron_511keV_v1, muon_105MeV_v1, tau_1777MeV_v1
        # If you prefer plain sp_label, change the mapping here.
        label_map = {
            "electron": f"511keV_{args.label}",
            "muon":     f"105MeV_{args.label}",
            "tau":      f"1777MeV_{args.label}",
        }
        manifest_path = build_bundle(
            root=outdir,
            species=sp,
            label=label_map[sp],
            features=f,
            summary_vals=summary,
            overwrite=args.overwrite,
        )
        built.append(manifest_path)

    print("built bundles:")
    for p in built:
        # Print paths relative to CWD for easy copy-paste into PowerShell variables
        print(str(p))

if __name__ == "__main__":
    main()
