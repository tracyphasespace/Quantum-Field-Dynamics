#!/bin/bash
# Release automation script for QFD CMB Module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to display usage
usage() {
    echo "Usage: $0 [prepare|validate|tag] [options]"
    echo ""
    echo "Commands:"
    echo "  prepare BUMP_TYPE [PRERELEASE]  Prepare a new release"
    echo "  validate                        Validate release readiness"
    echo "  tag [VERSION] [MESSAGE]         Create and push git tag"
    echo ""
    echo "Bump types: major, minor, patch, prerelease"
    echo ""
    echo "Examples:"
    echo "  $0 validate                     # Check if ready for release"
    echo "  $0 prepare patch                # Prepare patch release"
    echo "  $0 prepare minor                # Prepare minor release"
    echo "  $0 prepare prerelease alpha     # Prepare alpha prerelease"
    echo "  $0 tag                          # Tag current version"
    echo "  $0 tag 1.0.0 'Release 1.0.0'   # Tag specific version"
    echo ""
    exit 1
}

# Check arguments
if [ $# -lt 1 ]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    validate)
        echo "🔍 Validating release readiness..."
        python "$SCRIPT_DIR/prepare_release.py" validate
        echo "✅ Ready for release!"
        ;;
    
    prepare)
        if [ $# -lt 1 ]; then
            echo "Error: BUMP_TYPE required for prepare command"
            usage
        fi
        
        BUMP_TYPE="$1"
        PRERELEASE="${2:-}"
        
        echo "🚀 Preparing $BUMP_TYPE release..."
        
        if [ -n "$PRERELEASE" ]; then
            python "$SCRIPT_DIR/prepare_release.py" prepare --bump-type "$BUMP_TYPE" --prerelease "$PRERELEASE"
        else
            python "$SCRIPT_DIR/prepare_release.py" prepare --bump-type "$BUMP_TYPE"
        fi
        
        echo ""
        echo "📝 Review the changes and then run:"
        echo "   git diff --cached"
        echo "   git commit -m 'Prepare release'"
        echo "   $0 tag"
        ;;
    
    tag)
        VERSION="${1:-}"
        MESSAGE="${2:-}"
        
        echo "🏷️  Creating release tag..."
        
        if [ -n "$VERSION" ] && [ -n "$MESSAGE" ]; then
            python "$SCRIPT_DIR/prepare_release.py" tag --version "$VERSION" --message "$MESSAGE"
        elif [ -n "$VERSION" ]; then
            python "$SCRIPT_DIR/prepare_release.py" tag --version "$VERSION"
        else
            python "$SCRIPT_DIR/prepare_release.py" tag
        fi
        
        echo "✅ Tag created and pushed!"
        echo "🚀 GitHub Actions will now build and publish the release."
        ;;
    
    *)
        echo "Error: Unknown command '$COMMAND'"
        usage
        ;;
esac