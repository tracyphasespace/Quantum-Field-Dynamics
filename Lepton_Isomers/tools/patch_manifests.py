import json, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

root = Path("v2_bundles")
manifests = list(root.rglob("bundle_manifest.json"))

def patch_one(p: Path):
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return (str(p), False, f"read_error:{e}")

    j.setdefault("features", {})
    f = j["features"]
    changed = False

    # --- L_mag ---
    if "L_mag" not in f:
        src = "placeholder"
        try:
            if all(isinstance(f.get(k), (int,float)) for k in ("Lx","Ly","Lz")):
                Lx,Ly,Lz = (float(f[k]) for k in ("Lx","Ly","Lz"))
                f["L_mag"] = (Lx*Lx + Ly*Ly + Lz*Lz) ** 0.5
                src = "computed_from_LxLyLz"
            elif isinstance(f.get("L_vec"), list) and len(f["L_vec"])==3 and all(isinstance(v,(int,float)) for v in f["L_vec"]):
                x,y,z = map(float, f["L_vec"])
                f["L_mag"] = (x*x + y*y + z*z) ** 0.5
                src = "computed_from_L_vec"
            else:
                f["L_mag"] = 1.0
        except Exception:
            f["L_mag"] = 1.0
        j.setdefault("feature_meta", {}).setdefault("L_mag", {})["source"] = src
        changed = True

    # --- Hcsr ---
    if "Hcsr" not in f:
        src = "placeholder"
        for k in ("Hcsr_raw","H_csr","HCSR","Hcsr_est"):
            if isinstance(f.get(k),(int,float)):
                f["Hcsr"] = float(f[k]); src = f"copied_from_{k}"; break
        else:
            f["Hcsr"] = 1.0
        j.setdefault("feature_meta", {}).setdefault("Hcsr", {})["source"] = src
        changed = True

    if changed:
        bak = p.with_suffix(p.suffix + ".bak")
        try:
            if not bak.exists():
                shutil.copy2(p, bak)
        except Exception:
            pass
        p.write_text(json.dumps(j, indent=2), encoding="utf-8")
    return (str(p), changed, None)

with ThreadPoolExecutor() as ex:
    results = list(ex.map(patch_one, manifests))

patched = sum(1 for _,c,_ in results if c)
print(f"Patched {patched}/{len(results)} manifests")
