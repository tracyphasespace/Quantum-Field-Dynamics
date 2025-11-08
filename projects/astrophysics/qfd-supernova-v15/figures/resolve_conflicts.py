#!/usr/bin/env python3
"""
Resolve git conflicts with auto-generated files.

When you run 'make all', it generates PDFs and provenance JSONs.
If Claude also generated these files, you'll get merge conflicts.
This script resolves them automatically.
"""

import subprocess
import sys

def run(cmd):
    """Run a shell command and return success status."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("Resolving conflicts with auto-generated files")
    print("=" * 60)

    # Check if we're in a merge/pull conflict
    result = subprocess.run("git status", shell=True, capture_output=True, text=True)
    status = result.stdout

    if "You have unmerged paths" in status or "both modified" in status:
        print("\n✓ Detected merge conflict")
        print("\nResolving by keeping local auto-generated files...")

        # For provenance files, use theirs (Claude's code changes)
        run("git checkout --theirs '*_provenance.json' 2>/dev/null || true")

        # For PDFs, use ours (your local generated versions)
        run("git checkout --ours '*.pdf' 2>/dev/null || true")

        # Add resolved files
        run("git add '*_provenance.json' *.pdf")

        print("\n✓ Conflicts resolved!")
        print("\nNext steps:")
        print("  1. Run: git commit -m 'Resolve merge conflict with auto-generated files'")
        print("  2. Run: git push")

    elif "Your branch is behind" in status:
        print("\n✓ No conflicts, just need to pull")
        print("\nDiscarding local auto-generated files and pulling...")

        # Discard local changes to auto-generated files
        run("git checkout -- '*_provenance.json' 2>/dev/null || true")

        # Pull latest
        branch = "claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3"
        if run(f"git pull origin {branch}"):
            print("\n✓ Pull successful!")
            print("\nYou can now run 'make all' to regenerate figures with latest code")
        else:
            print("\n✗ Pull failed - check errors above")
            return 1

    elif "nothing to commit, working tree clean" in status:
        print("\n✓ No conflicts detected - working tree is clean!")
        print("\nIf you want to get Claude's latest changes:")
        print("  Run: python sync_simple.py")

    else:
        print("\nCurrent git status:")
        print(status)
        print("\nTo manually resolve:")
        print("  1. Discard provenance changes: git checkout -- '*_provenance.json'")
        print("  2. Pull latest: git pull origin claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3")
        print("  3. Regenerate: make all")

    return 0

if __name__ == '__main__':
    sys.exit(main())
