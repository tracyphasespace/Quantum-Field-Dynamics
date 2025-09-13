#!/bin/bash
# Version bumping utility script for QFD CMB Module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to display usage
usage() {
    echo "Usage: $0 [major|minor|patch|prerelease] [prerelease-name]"
    echo ""
    echo "Examples:"
    echo "  $0 patch                    # 1.0.0 -> 1.0.1"
    echo "  $0 minor                    # 1.0.0 -> 1.1.0"
    echo "  $0 major                    # 1.0.0 -> 2.0.0"
    echo "  $0 prerelease alpha         # 1.0.0 -> 1.0.0-alpha.1"
    echo ""
    exit 1
}

# Check arguments
if [ $# -lt 1 ]; then
    usage
fi

BUMP_TYPE="$1"
PRERELEASE="${2:-}"

# Validate bump type
case "$BUMP_TYPE" in
    major|minor|patch|prerelease)
        ;;
    *)
        echo "Error: Invalid bump type '$BUMP_TYPE'"
        usage
        ;;
esac

# Run version manager
echo "Current version: $(python "$SCRIPT_DIR/version_manager.py" get)"

if [ "$BUMP_TYPE" = "prerelease" ] && [ -n "$PRERELEASE" ]; then
    python "$SCRIPT_DIR/version_manager.py" bump --bump-type "$BUMP_TYPE" --prerelease "$PRERELEASE"
else
    python "$SCRIPT_DIR/version_manager.py" bump --bump-type "$BUMP_TYPE"
fi

echo "Version bump completed successfully!"