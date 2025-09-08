import json
from pathlib import Path

def patch_features():
    base_dir = Path("v2_bundles")
    
    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' not found.")
        return

    print(f"Patching features.json files in '{base_dir}'...")

    for bundle_dir in base_dir.iterdir():
        if bundle_dir.is_dir():
            print(f"Processing bundle: {bundle_dir.name}")
            summary_path = bundle_dir / "summary.json"
            features_path = bundle_dir / "features.json"

            if not summary_path.exists():
                print(f"  -> SKIPPING: summary.json not found in {bundle_dir.name}")
                continue

            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)

                # Extract values from summary.json
                q_star = summary_data.get("constants", {}).get("physics_constants", {}).get("q_star", 0.0)
                R_eff = summary_data.get("R_eff_final", 0.0)
                I = summary_data.get("I_final", 0.0)
                Hcsr = summary_data.get("Hcsr_final", 0.0)

                hcsr_over_I = Hcsr / I if I != 0 else 0.0 # Avoid division by zero

                # Prepare features.json content
                features_content = {
                    "q_star": q_star,
                    "R_eff": R_eff,
                    "I": I,
                    "Hcsr": Hcsr,
                    "hcsr_over_I": hcsr_over_I,
                    "L_mag": 0.0, # Placeholder as not directly available in summary.json
                    "LdotQ_over_I": 0.0, # Placeholder as not directly available in summary.json
                    "K": 0.0      # Placeholder as not directly available in summary.json
                }

                with open(features_path, 'w', encoding='utf-8') as f:
                    json.dump(features_content, f, indent=4)
                
                print(f"  -> Successfully patched features.json for {bundle_dir.name}")

            except json.JSONDecodeError:
                print(f"  -> SKIPPING: Could not decode JSON from {summary_path}")
            except Exception as e:
                print(f"  -> An error occurred while processing {bundle_dir.name}: {e}")

    print("\nPatching process complete.")

if __name__ == "__main__":
    patch_features()
