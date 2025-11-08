#!/usr/bin/env python3
"""
Simple git sync script that actually works.
Discards auto-generated files and pulls cleanly.
"""
import subprocess
import sys
import os

def run(cmd):
    """Run command and return success status."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0

def main():
    branch = "claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3"

    print("=" * 60)
    print("Git Sync Script")
    print("=" * 60)

    # Discard changes to auto-generated files
    print("\n1. Discarding auto-generated files...")
    run("git checkout -- '**/*_provenance.json' 2>/dev/null || true")
    run("git checkout -- 'projects/astrophysics/qfd-supernova-v15/figures/*_provenance.json' 2>/dev/null || true")

    # Pull latest
    print("\n2. Pulling latest changes...")
    if not run(f"git pull origin {branch}"):
        print("\nERROR: Pull failed. Check errors above.")
        return 1

    print("\n" + "=" * 60)
    print("✓ Sync complete!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
