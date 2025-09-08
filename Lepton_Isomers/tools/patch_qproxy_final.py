import json
from pathlib import Path

root = Path("v2_bundles")
patched = 0
for bdir in root.iterdir():
    sfile = bdir / "summary.json"
    ffile = bdir / "features.json"
    if not (sfile.exists() and ffile.exists()):
        continue
    S = json.loads(sfile.read_text(encoding="utf-8"))
    F = json.loads(ffile.read_text(encoding="utf-8"))

    # Anchors the fitter expects (fallback from features)
    S["U_final"]     = float(S.get("U_final",     F.get("U",     1.0)))
    S["R_eff_final"] = float(S.get("R_eff_final", F.get("R_eff", 1.0)))
    S["I_final"]     = float(S.get("I_final",     F.get("I",     1.0)))
    S["Hkin_final"]  = float(S.get("Hkin_final",  F.get("K",     1.0)))

    # NEW: q* lives in features; the fitter wants summary.Q_proxy_final
    S["Q_proxy_final"] = float(S.get("Q_proxy_final", F.get("q_star", 1.0)))

    # Minimal Zeeman inputs so --emit-zeeman doesn't error
    S.setdefault("energy", 1.0)
    S.setdefault("magnetic_moment", 1.0)

    sfile.write_text(json.dumps(S, indent=2), encoding="utf-8")
    patched += 1

print(f"Patched {patched} bundle summaries.")
