#!/bin/bash
# ============================================================
# Sync Script: Push Local Changes & Pull Claude's Updates
# Branch: claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3
# Works on: WSL, Linux, Mac, Git Bash
# ============================================================

set -e  # Exit on error (except where we handle errors)

BRANCH="claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3"

echo ""
echo "============================================================"
echo "Syncing with Claude's Branch"
echo "============================================================"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "ERROR: Not in a git repository!"
    echo "Please run this script from your Quantum-Field-Dynamics directory."
    exit 1
fi

# Show current branch
echo "Current branch: $(git branch --show-current)"
echo ""

# Show current status
echo "Current status:"
git status --short
echo ""

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    echo "You have uncommitted changes."
    read -p "Do you want to commit them? (y/n): " commit_choice

    if [[ "$commit_choice" =~ ^[Yy]$ ]]; then
        echo ""
        read -p "Enter commit message: " commit_msg

        echo ""
        echo "Adding all changes..."
        git add .

        echo "Committing changes..."
        if git commit -m "$commit_msg"; then
            echo "✓ Changes committed successfully"
        else
            echo "WARNING: Commit failed. Continuing anyway..."
        fi
        echo ""
    else
        echo "Skipping commit. Changes will remain uncommitted."
        echo ""
    fi
else
    echo "No uncommitted changes detected."
    echo ""
fi

# Fetch latest from remote
echo "Fetching latest from remote..."
if git fetch origin; then
    echo "✓ Fetch complete"
else
    echo "ERROR: Failed to fetch from remote!"
    exit 1
fi
echo ""

# Check out the Claude branch
echo "Checking out Claude's branch..."
if git checkout "$BRANCH"; then
    echo "✓ On Claude's branch"
else
    echo "ERROR: Failed to checkout branch!"
    exit 1
fi
echo ""

# Pull latest changes
echo "Pulling latest changes from Claude..."
if git pull origin "$BRANCH"; then
    echo "✓ Pull complete"
else
    echo "ERROR: Failed to pull changes!"
    echo "You may have merge conflicts. Please resolve them manually."
    exit 1
fi
echo ""

# Push any local commits
echo "Pushing your local commits (if any)..."
if git push origin "$BRANCH" 2>/dev/null; then
    echo "✓ Push complete"
else
    echo "NOTE: Nothing to push (or push failed - this is normal if you have no new commits)"
fi
echo ""

# Show final status
echo "============================================================"
echo "Final Status"
echo "============================================================"
git status
echo ""

# Show recent commits
echo "============================================================"
echo "Recent Commits (last 5)"
echo "============================================================"
git log --oneline --graph --decorate -5
echo ""

# Show what files changed
echo "============================================================"
echo "Files Changed in Latest Pull"
echo "============================================================"
git diff --name-status HEAD@{1} HEAD 2>/dev/null || echo "(No changes detected)"
echo ""

echo "============================================================"
echo "✓ Sync Complete!"
echo "============================================================"
echo ""
echo "Your repository is now up to date with Claude's latest changes."
echo "You can now view the files in:"
echo "  - projects/astrophysics/qfd-supernova-v15/"
echo ""
echo "Latest figures are in:"
echo "  - projects/astrophysics/qfd-supernova-v15/figures/"
echo ""
