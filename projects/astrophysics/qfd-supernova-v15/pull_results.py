#!/usr/bin/env python3
"""
pull_results.py - Cross-platform script to sync QFD V15 results to local machine

Usage: python pull_results.py [--local-dir /path/to/destination]

This script:
1. Pulls latest changes from the working branch
2. Copies all figures, reports, and documentation to a local directory
3. Creates an index file for easy navigation
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_command(cmd, check=True):
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if check:
            print(f"⚠ Command failed: {cmd}")
            print(f"Error: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Sync QFD V15 publication figures and results to local machine"
    )
    parser.add_argument(
        "--local-dir",
        default=str(Path.home() / "QFD_Results"),
        help="Destination directory (default: ~/QFD_Results)"
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    repo_root = Path(__file__).parent

    print("═" * 60)
    print("  QFD V15 Results Sync Script (Python)")
    print("═" * 60)
    print()

    # Get current branch
    os.chdir(repo_root)
    current_branch = run_command("git rev-parse --abbrev-ref HEAD")
    print(f"✓ Current branch: {current_branch}")

    # Pull latest changes
    print(f"\n[1/4] Pulling latest changes from remote...")
    pull_result = run_command(f"git pull origin {current_branch}", check=False)
    if pull_result:
        print(f"✓ Pull complete")
    else:
        print(f"⚠ Already up to date or conflicts present")

    # Get latest commit
    latest_commit = run_command("git log -1 --oneline")
    print(f"✓ Latest commit: {latest_commit}")

    # Create local directories
    print(f"\n[2/4] Creating local results directory...")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "figures").mkdir(exist_ok=True)
    (local_dir / "reports").mkdir(exist_ok=True)
    (local_dir / "validation_plots").mkdir(exist_ok=True)
    print(f"✓ Created: {local_dir}")

    # Track copied files
    copied_stats = {
        'figures': 0,
        'validation_plots': 0,
        'reports': 0,
        'data_files': 0
    }

    # Copy publication figures
    print(f"\n[3/4] Copying publication figures...")
    figures_src = repo_root / "results/mock_stage3/figures"
    if figures_src.exists():
        for fig in figures_src.glob("*.png"):
            shutil.copy2(fig, local_dir / "figures")
            copied_stats['figures'] += 1
        print(f"✓ Copied {copied_stats['figures']} figures to {local_dir / 'figures'}")
    else:
        print(f"⚠ No mock_stage3 figures found")

    # Copy validation plots
    print(f"\n[3.5/4] Copying validation plots...")
    validation_src = repo_root / "validation_plots"
    if validation_src.exists():
        for fig in validation_src.glob("*.png"):
            shutil.copy2(fig, local_dir / "validation_plots")
            copied_stats['validation_plots'] += 1
        print(f"✓ Copied {copied_stats['validation_plots']} validation plots")

    # Copy reports
    print(f"\n[4/4] Copying reports and data files...")
    reports_src = repo_root / "results/mock_stage3/reports"
    if reports_src.exists():
        for report in reports_src.glob("*.csv"):
            shutil.copy2(report, local_dir / "reports")
            copied_stats['reports'] += 1
        print(f"✓ Copied {copied_stats['reports']} CSV reports")

    # Copy Stage 3 results
    stage3_csv = repo_root / "results/mock_stage3/stage3_results.csv"
    if stage3_csv.exists():
        shutil.copy2(stage3_csv, local_dir)
        copied_stats['data_files'] += 1
        print(f"✓ Copied stage3_results.csv")

    # Copy posterior samples
    posterior_csv = repo_root / "results/mock_stage3/posterior_samples.csv"
    if posterior_csv.exists():
        shutil.copy2(posterior_csv, local_dir)
        copied_stats['data_files'] += 1
        print(f"✓ Copied posterior_samples.csv")

    # Copy documentation
    print(f"\n[Bonus] Copying documentation...")
    docs_to_copy = [
        "PUBLICATION_FIGURES_SUMMARY.md",
        "docs/PUBLICATION_TEMPLATE.md",
        "docs/CODE_VERIFICATION.md",
        "HOTFIX_VALIDATION.md",
    ]
    for doc in docs_to_copy:
        doc_path = repo_root / doc
        if doc_path.exists():
            shutil.copy2(doc_path, local_dir / doc_path.name)
    print(f"✓ Copied documentation files")

    # Create index file
    print(f"\n[Final] Creating results index...")
    index_content = f"""# QFD V15 Results - Local Copy

**Synced on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Git branch**: {current_branch}
**Latest commit**: {latest_commit}

## Directory Structure

```
{local_dir}/
├── figures/                    # Publication figures (Fig 4, 5, 6, 8)
├── validation_plots/           # Validation plots (Fig 1, 2, 3)
├── reports/                    # Per-survey diagnostic CSVs
├── stage3_results.csv          # Stage 3 residuals (300 SNe)
├── posterior_samples.csv       # MCMC samples (2000 draws)
├── PUBLICATION_FIGURES_SUMMARY.md
├── PUBLICATION_TEMPLATE.md
├── CODE_VERIFICATION.md
└── HOTFIX_VALIDATION.md
```

## Available Figures

### Publication Figures (300 DPI)
"""

    # List figures
    for fig in sorted((local_dir / "figures").glob("*.png")):
        size_mb = fig.stat().st_size / 1024  # KB
        index_content += f"- **{fig.name}** ({size_mb:.0f}K)\n"

    index_content += "\n### Validation Plots\n"
    for fig in sorted((local_dir / "validation_plots").glob("*.png")):
        size_mb = fig.stat().st_size / 1024  # KB
        index_content += f"- **{fig.name}** ({size_mb:.0f}K)\n"

    index_content += f"""
## Quick Links

**View figures**: Open `{local_dir / 'figures'}` in your file browser

**Read summary**: Open `PUBLICATION_FIGURES_SUMMARY.md` in your markdown viewer

**Review data**: Load CSVs from `reports/` directory in Excel/Python/R

## Re-sync

To pull latest updates, run:
```bash
cd {repo_root}
python pull_results.py
```

Or with custom destination:
```bash
python pull_results.py --local-dir /path/to/destination
```

---
*Generated by pull_results.py*
"""

    (local_dir / "INDEX.md").write_text(index_content)
    print(f"✓ Created INDEX.md")

    # Final summary
    print(f"\n{'═' * 60}")
    print(f"✓ SUCCESS! Results synced to: {local_dir}")
    print(f"{'═' * 60}")
    print()
    print(f"📊 What's available:")
    print(f"   - {copied_stats['figures']} publication figures")
    print(f"   - {copied_stats['validation_plots']} validation plots")
    print(f"   - {copied_stats['reports']} CSV reports")
    print(f"   - {copied_stats['data_files']} data files")
    print()
    print(f"📂 Open results folder:")
    print(f"   {local_dir}")
    print()
    print(f"📖 Read the index:")
    print(f"   {local_dir / 'INDEX.md'}")
    print()
    print(f"🔄 To re-sync later, just run: python pull_results.py")
    print()


if __name__ == "__main__":
    main()
