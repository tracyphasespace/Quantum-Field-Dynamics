#!/bin/bash
# ============================================================================
# sync_to_universe.sh - Sync publishable files to QFD-Universe repo
# ============================================================================
#
# This script syncs only the files listed in PUBLISH_MANIFEST.txt to the
# public-facing QFD-Universe repository. Development files stay private.
#
# Usage:
#   ./sync_to_universe.sh           # Dry run (preview only)
#   ./sync_to_universe.sh --commit  # Sync and commit to Universe repo
#   ./sync_to_universe.sh --push    # Sync, commit, and push
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/QFD"
UNIVERSE_DIR="/home/tracy/development/QFD-Universe/formalization/QFD"
UNIVERSE_ROOT="/home/tracy/development/QFD-Universe/formalization"
MANIFEST="$SCRIPT_DIR/PUBLISH_MANIFEST.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=============================================="
echo "QFD Universe Sync Tool"
echo "=============================================="
echo ""

# Check prerequisites
if [ ! -f "$MANIFEST" ]; then
    echo -e "${RED}ERROR: PUBLISH_MANIFEST.txt not found${NC}"
    exit 1
fi

if [ ! -d "$UNIVERSE_ROOT" ]; then
    echo -e "${RED}ERROR: Universe repo not found at $UNIVERSE_ROOT${NC}"
    exit 1
fi

echo "Source:      $SOURCE_DIR"
echo "Destination: $UNIVERSE_DIR"
echo "Manifest:    $MANIFEST"
echo ""

# Count files that will sync
echo -e "${CYAN}Analyzing manifest...${NC}"
LEAN_COUNT=$(find "$SOURCE_DIR" -name "*.lean" | wc -l)
echo "Total .lean files in source: $LEAN_COUNT"
echo ""

# Determine mode
MODE="dry-run"
if [ "$1" == "--commit" ] || [ "$1" == "--push" ]; then
    MODE="$1"
fi

if [ "$MODE" == "dry-run" ]; then
    echo -e "${YELLOW}DRY RUN - No changes will be made${NC}"
    echo "Use --commit to sync, or --push to sync and push to GitHub"
    echo ""
    echo "Files that would sync:"
    echo "----------------------"

    rsync -avnc --delete \
        --filter="merge $MANIFEST" \
        "$SOURCE_DIR/" "$UNIVERSE_DIR/" 2>/dev/null | grep -E "^[^.]" | head -80

    echo ""
    echo -e "${YELLOW}(Showing first 80 entries - run with --commit to execute)${NC}"
else
    echo -e "${GREEN}SYNCING...${NC}"
    echo ""

    # Create destination if needed
    mkdir -p "$UNIVERSE_DIR"

    # Sync using the manifest filter
    rsync -av --delete \
        --filter="merge $MANIFEST" \
        "$SOURCE_DIR/" "$UNIVERSE_DIR/"

    # Also sync build files to formalization root
    echo ""
    echo "Syncing build files..."
    cp "$SCRIPT_DIR/lakefile.toml" "$UNIVERSE_ROOT/"
    cp "$SCRIPT_DIR/lean-toolchain" "$UNIVERSE_ROOT/"
    [ -f "$SCRIPT_DIR/lake-manifest.json" ] && cp "$SCRIPT_DIR/lake-manifest.json" "$UNIVERSE_ROOT/"

    echo ""
    echo -e "${GREEN}Sync complete!${NC}"

    # Count synced files
    SYNCED_COUNT=$(find "$UNIVERSE_DIR" -name "*.lean" | wc -l)
    echo "Synced $SYNCED_COUNT .lean files to Universe repo"

    # Commit if requested
    if [ "$MODE" == "--commit" ] || [ "$MODE" == "--push" ]; then
        echo ""
        echo "Committing to Universe repo..."
        cd "$UNIVERSE_ROOT"
        git add -A

        # Check if there are changes
        if git diff --cached --quiet; then
            echo "No changes to commit."
        else
            # Get stats
            STATS=$(git diff --cached --stat | tail -1)
            TIMESTAMP=$(date +"%Y-%m-%d %H:%M")

            git commit -m "sync: Update from development repo ($TIMESTAMP)

$STATS

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
            echo -e "${GREEN}Committed!${NC}"
        fi

        # Push if requested
        if [ "$MODE" == "--push" ]; then
            echo ""
            echo "Pushing to GitHub..."
            git push
            echo -e "${GREEN}Pushed to GitHub!${NC}"
        fi
    fi
fi

echo ""
echo "Done."
