#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robustness Runner (v1a, Windows-safe)
- Auto-finds QFD/GPT fitter if --fitter not provided.
- Launches subprocess with argument list (no POSIX quotes).
- Timezone-aware UTC stamp.

Usage (smoke test with a prebuilt tau list):
  python src/orchestration/robustness_runner.py ^
    --bundle-list data\tau_list.txt ^
    --trials 1 ^
    --chi-modes qR_over_I q_over_I ^
    --exponents 6 0.5 0.5 1.5 ^
    --anchor-bundle v2_bundles-electron_511keV_v1\bundle_manifest.json ^
    --muon-bundle   v2_bundles-muon_105MeV_v1\bundle_manifest.json ^
    --electron-bundle v2_bundles-electron_511keV_v1\bundle_manifest.json ^
    --device cuda ^
    --expected-beta 98.84 99.07 ^
    --beta-tol 1.0 ^
    --expected-k0 1.448e21 1.701e21 ^
    --k0-rel-tol 0.05 ^
    --sse-max 1e-28 ^
    --out results\robustness ^
    --emit-csv
"""
from __future__ import annotations
import argparse, csv, dataclasses, json, math, os, pathlib, random, subprocess, sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

Path = pathlib.Path

@dataclasses.dataclass
class TrialFit:
    trial: int
    seed: int
    chi_mode: str
    beta: float
    k0: float
    sse: float
    delta_chi: float
    chi_std: float
    tau_e: float
    overflow_k0: bool
    manifest_tau: str
    fit_json: str
    ok: bool
    error: Optional[str] = None

@dataclasses.dataclass
class ModeSummary:
    chi_mode: str
    beta_mean: float
    beta_std: float
    beta_min: float
    beta_max: float
    k0_mean: float
    k0_std: float
    k0_min: float
    k0_max: float
    pass_rate: float
    passed: bool
    ref_beta: Optional[float] = None
    ref_k0: Optional[float] = None
    beta_tol: Optional[float] = None
    k0_rel_tol: Optional[float] = None
    sse_max: Optional[float] = None

def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _find_fitter() -> Path:
    # Prefer QFD_* then GPT_* inside the repo
    here = Path(".")
    q = list(here.rglob("QFD_fit_lifetimes_v3.py"))
    if q: return q[0]
    g = list(here.rglob("GPT_fit_lifetimes_v3.py"))
    if g: return g[0]
    raise SystemExit("Could not find QFD_fit_lifetimes_v3.py or GPT_fit_lifetimes_v3.py. Pass --fitter explicitly.")

def regenerate_bundle(shell_tpl: str, species: str, seed: int, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = shell_tpl.format(species=species, seed=seed, outdir=str(outdir))
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"regen failed (seed={seed}) rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    manifest = ""
    for line in proc.stdout.splitlines()[::-1]:
        if line.strip():
            manifest = line.strip()
            break
    if not manifest or not Path(manifest).exists():
        raise RuntimeError(f"regen did not yield a manifest path (seed={seed}). Got: '{manifest}'\nSTDOUT:\n{proc.stdout}")
    return Path(manifest)

def run_fit(fitter: Path, manifest_tau: Path, muon: Path, electron: Path,
            chi_mode: str, exponents: List[float], anchor_bundle: Path,
            device: str, out_json: Path, pretty: bool = True) -> Dict[str, Any]:
    args = [
        sys.executable, str(fitter),
        "--muon", str(muon),
        "--tau", str(manifest_tau),
        "--exponents", *[str(x) for x in exponents],
        "--anchor-bundle", str(anchor_bundle),
        "--chi-mode", chi_mode,
        "--electron", str(electron),
        "--device", device,
        "--out", str(out_json),
    ]
    if pretty:
        args.append("--pretty")
    proc = subprocess.run(args, shell=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fit failed for {chi_mode} (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    if not out_json.exists():
        raise RuntimeError(f"fit did not emit JSON at {out_json}")
    return json.loads(out_json.read_text(encoding="utf-8"))

def stats(xs: List[float]):
    n = max(1, len(xs))
    m = sum(xs)/n
    var = sum((x-m)**2 for x in xs)/n
    return m, math.sqrt(var), (min(xs) if xs else float("nan")), (max(xs) if xs else float("nan"))

def verify_mode(rows: List[TrialFit], ref_beta: Optional[float], beta_tol: float,
                ref_k0: Optional[float], k0_rel_tol: float, sse_max: float) -> ModeSummary:
    chi_mode = rows[0].chi_mode
    betas = [r.beta for r in rows if r.ok]
    k0s   = [r.k0 for r in rows if r.ok]
    passes = 0
    for r in rows:
        ok = r.ok
        if ok and sse_max is not None: ok = ok and (r.sse <= sse_max)
        if ok and ref_beta is not None and beta_tol is not None:
            ok = ok and (abs(r.beta - ref_beta) <= beta_tol)
        if ok and ref_k0 is not None and k0_rel_tol is not None and ref_k0 != 0:
            ok = ok and (abs(r.k0 - ref_k0)/abs(ref_k0) <= k0_rel_tol)
        r.ok = ok
        if ok: passes += 1
    bm, bs, bmin, bmax = stats(betas)
    km, ks, kmin, kmax = stats(k0s)
    passed = (passes == len(rows)) and len(rows) > 0
    return ModeSummary(
        chi_mode=chi_mode,
        beta_mean=bm, beta_std=bs, beta_min=bmin, beta_max=bmax,
        k0_mean=km,   k0_std=ks,   k0_min=kmin,  k0_max=kmax,
        pass_rate=(passes/len(rows) if rows else 0.0),
        passed=passed,
        ref_beta=ref_beta, ref_k0=ref_k0,
        beta_tol=beta_tol, k0_rel_tol=k0_rel_tol, sse_max=sse_max
    )

def main():
    ap = argparse.ArgumentParser(description="Robustness runner for lepton lifetime fits.")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1337)
    ap.add_argument("--species", default="tau", choices=["tau"])
    ap.add_argument("--chi-modes", nargs="+", required=True)
    ap.add_argument("--exponents", nargs=4, type=float, required=True)
    ap.add_argument("--anchor-bundle", type=Path, required=True)
    ap.add_argument("--muon-bundle", type=Path, required=True)
    ap.add_argument("--electron-bundle", type=Path, required=True)
    ap.add_argument("--fitter", type=Path, required=False, help="If omitted, auto-detect QFD/GPT fitter.")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--out", type=Path, default=Path("results") / "robustness")
    ap.add_argument("--emit-csv", action="store_true")

    ap.add_argument("--regen-shell", type=str, help="Command template printing manifest path. Placeholders: {species},{seed},{outdir}")
    ap.add_argument("--bundle-list", type=Path, help="Text file with one manifest path per line (prebuilt bundles)")

    ap.add_argument("--expected-beta", nargs="*", type=float, help="Reference beta per chi-mode (order must match --chi-modes)")
    ap.add_argument("--beta-tol", type=float, default=1.0)
    ap.add_argument("--expected-k0", nargs="*", type=float, help="Reference k0 per chi-mode (order must match --chi-modes)")
    ap.add_argument("--k0-rel-tol", type=float, default=0.05)
    ap.add_argument("--sse-max", type=float, default=1e-28)
    args = ap.parse_args()

    stamp = _now_utc_stamp()
    root = args.out / stamp
    (root / "runs").mkdir(parents=True, exist_ok=True)

    # auto-detect fitter if not supplied
    fitter = Path(args.fitter) if args.fitter else _find_fitter()

    # resolve references
    ref_beta = {}
    ref_k0   = {}
    if args.expected_beta:
        if len(args.expected_beta) != len(args.chi_modes):
            raise SystemExit("--expected-beta length must match --chi-modes")
        ref_beta = dict(zip(args.chi_modes, args.expected_beta))
    if args.expected_k0:
        if len(args.expected_k0) != len(args.chi_modes):
            raise SystemExit("--expected-k0 length must match --chi-modes")
        ref_k0 = dict(zip(args.chi_modes, args.expected_k0))

    # prepare manifests
    manifests: List[Path] = []
    if args.bundle_list and args.bundle_list.exists():
        for line in args.bundle_list.read_text(encoding="utf-8").splitlines():
            p = Path(line.strip())
            if p.exists(): manifests.append(p)
    elif args.regen_shell:
        pass
    else:
        raise SystemExit("Provide either --regen-shell or --bundle-list")

    trials = args.trials if args.regen_shell else min(args.trials, len(manifests))
    all_rows: List[TrialFit] = []

    for t in range(trials):
        seed = args.seed0 + t
        if args.regen_shell:
            trial_dir = (root / "runs" / f"trial_{t:03d}")
            trial_dir.mkdir(parents=True, exist_ok=True)
            try:
                manifest_tau = regenerate_bundle(args.regen_shell, args.species, seed, trial_dir)
            except Exception as e:
                for m in args.chi_modes:
                    all_rows.append(TrialFit(t, seed, m, float("nan"), float("nan"), float("inf"),
                                             float("nan"), float("nan"), float("nan"),
                                             False, "", str(trial_dir/ f"trial_{t:03d}_{m}.json"), False, str(e)))
                continue
        else:
            manifest_tau = manifests[t]

        for m in args.chi_modes:
            out_json = root / "runs" / f"trial_{t:03d}_{m}.json"
            try:
                J = run_fit(
                    fitter=fitter, manifest_tau=manifest_tau,
                    muon=args.muon_bundle, electron=args.electron_bundle,
                    chi_mode=m, exponents=args.exponents,
                    anchor_bundle=args.anchor_bundle, device=args.device,
                    out_json=out_json, pretty=True
                )
                fit = J.get("fit_result", {})
                diag = fit.get("diagnostics", {})
                echeck = diag.get("electron_check", {})
                row = TrialFit(
                    trial=t, seed=seed, chi_mode=m,
                    beta=float(fit.get("beta", float("nan"))),
                    k0=float(fit.get("k0", float("nan"))),
                    sse=float(fit.get("sse", float("inf"))),
                    delta_chi=float(diag.get("delta_chi", float("nan"))),
                    chi_std=float(diag.get("chi_std_dev", float("nan"))),
                    tau_e=float(echeck.get("tau_pred_seconds", float("nan"))),
                    overflow_k0=bool(diag.get("overflow", {}).get("k0", False)),
                    manifest_tau=str(manifest_tau),
                    fit_json=str(out_json),
                    ok=True,
                )
            except Exception as e:
                row = TrialFit(
                    trial=t, seed=seed, chi_mode=m, beta=float("nan"), k0=float("nan"), sse=float("inf"),
                    delta_chi=float("nan"), chi_std=float("nan"), tau_e=float("nan"),
                    overflow_k0=False, manifest_tau=str(manifest_tau), fit_json=str(out_json),
                    ok=False, error=str(e)
                )
            all_rows.append(row)

    by_mode: Dict[str, List[TrialFit]] = {}
    for r in all_rows:
        by_mode.setdefault(r.chi_mode, []).append(r)

    summaries: List[ModeSummary] = []
    overall_pass = True
    for m, rows in by_mode.items():
        ms = verify_mode(
            rows=rows,
            ref_beta=ref_beta.get(m),
            beta_tol=args.beta_tol,
            ref_k0=ref_k0.get(m),
            k0_rel_tol=args.k0_rel_tol,
            sse_max=args.sse_max
        )
        summaries.append(ms)
        overall_pass = overall_pass and ms.passed

    report = {
        "meta": {
            "timestamp_utc": stamp,
            "trials": trials,
            "species": args.species,
            "chi_modes": args.chi_modes,
            "exponents": args.exponents,
            "anchor_bundle": str(args.anchor_bundle),
            "muon_bundle": str(args.muon_bundle),
            "electron_bundle": str(args.electron_bundle),
            "fitter": str(fitter),
            "device": args.device,
            "regen_shell": args.regen_shell or None,
            "bundle_list": str(args.bundle_list) if args.bundle_list else None,
            "tolerances": {
                "beta_tol": args.beta_tol,
                "k0_rel_tol": args.k0_rel_tol,
                "sse_max": args.sse_max,
                "expected_beta": ref_beta or None,
                "expected_k0": ref_k0 or None
            }
        },
        "summary": [dataclasses.asdict(s) for s in summaries],
        "trials": [dataclasses.asdict(r) for r in all_rows],
        "passed": overall_pass,
    }
    out_json = Path(args.out) / stamp / "robustness_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out_json))
    sys.exit(0 if overall_pass else 1)

if __name__ == "__main__":
    main()