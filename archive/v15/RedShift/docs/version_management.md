# Version Management

This document describes the version management system for the QFD CMB Module, which follows [Semantic Versioning](https://semver.org/) principles.

## Semantic Versioning

The project uses semantic versioning with the format `MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]`:

- **MAJOR**: Incremented for incompatible API changes
- **MINOR**: Incremented for backwards-compatible functionality additions
- **PATCH**: Incremented for backwards-compatible bug fixes
- **PRERELEASE**: Optional identifier for pre-release versions (e.g., `alpha.1`, `beta.2`, `rc.1`)
- **BUILD**: Optional build metadata (not used in this project)

## Version Management Tools

### Python Script

The main version management is handled by `scripts/version_manager.py`:

```bash
# Get current version
python scripts/version_manager.py get

# Set specific version
python scripts/version_manager.py set --version "1.2.3"

# Bump version
python scripts/version_manager.py bump --bump-type patch
python scripts/version_manager.py bump --bump-type minor
python scripts/version_manager.py bump --bump-type major
python scripts/version_manager.py bump --bump-type prerelease --prerelease alpha

# Check version consistency across files
python scripts/version_manager.py check
```

### Shell Scripts

For convenience, wrapper scripts are provided:

**Linux/macOS:**
```bash
# Make executable (first time only)
chmod +x scripts/bump_version.sh

# Bump versions
./scripts/bump_version.sh patch      # 1.0.0 -> 1.0.1
./scripts/bump_version.sh minor      # 1.0.0 -> 1.1.0
./scripts/bump_version.sh major      # 1.0.0 -> 2.0.0
./scripts/bump_version.sh prerelease alpha  # 1.0.0 -> 1.0.0-alpha.1
```

**Windows:**
```cmd
REM Bump versions
scripts\bump_version.bat patch
scripts\bump_version.bat minor
scripts\bump_version.bat major
scripts\bump_version.bat prerelease alpha
```

## Version Storage

Versions are stored in multiple files and must be kept consistent:

1. **`qfd_cmb/__init__.py`**: Primary version source
   ```python
   __version__ = "1.0.0"
   ```

2. **`pyproject.toml`**: Modern Python packaging
   ```toml
   [project]
   version = "1.0.0"
   ```

3. **`setup.py`**: Reads version from `__init__.py` automatically

## Automated Checks

The CI pipeline automatically checks version consistency across all files:

- Runs on every push and pull request
- Fails the build if versions are inconsistent
- Uses `python scripts/version_manager.py check`

## Release Workflow

### Development Releases

For development and testing:

```bash
# Create alpha release
./scripts/bump_version.sh prerelease alpha  # -> 1.0.0-alpha.1

# Increment alpha
./scripts/bump_version.sh prerelease alpha  # -> 1.0.0-alpha.2

# Create beta release
./scripts/bump_version.sh prerelease beta   # -> 1.0.0-beta.1

# Create release candidate
./scripts/bump_version.sh prerelease rc     # -> 1.0.0-rc.1
```

### Production Releases

For production releases:

```bash
# Bug fixes
./scripts/bump_version.sh patch  # 1.0.0 -> 1.0.1

# New features (backwards compatible)
./scripts/bump_version.sh minor  # 1.0.0 -> 1.1.0

# Breaking changes
./scripts/bump_version.sh major  # 1.0.0 -> 2.0.0
```

## Integration with Git

### Tagging Releases

After bumping version, create a git tag:

```bash
# Bump version
./scripts/bump_version.sh minor

# Get new version
NEW_VERSION=$(python scripts/version_manager.py get)

# Create and push tag
git add .
git commit -m "Bump version to $NEW_VERSION"
git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"
git push origin main --tags
```

### Branch Strategy

- **main**: Stable releases only
- **develop**: Development work
- **feature/***: Feature branches
- **release/***: Release preparation branches

Version bumps should typically happen on:
- **release branches**: For preparing new releases
- **main branch**: For hotfixes
- **develop branch**: For development releases

## Best Practices

### When to Bump Versions

- **Patch**: Bug fixes, documentation updates, internal refactoring
- **Minor**: New features, new APIs, deprecations (with backwards compatibility)
- **Major**: Breaking changes, removed APIs, architectural changes

### Pre-release Versions

Use pre-release versions for:
- **alpha**: Early development, unstable API
- **beta**: Feature-complete, testing phase
- **rc** (release candidate): Final testing before release

### Version Consistency

- Always use the version management tools
- Never manually edit version numbers
- Run `python scripts/version_manager.py check` before committing
- CI will catch inconsistencies automatically

## Troubleshooting

### Version Inconsistency

If versions become inconsistent:

```bash
# Check which files are inconsistent
python scripts/version_manager.py check

# Fix by setting all to the same version
python scripts/version_manager.py set --version "1.0.0"
```

### Invalid Version Format

The version manager validates semantic version format. Valid examples:
- `1.0.0`
- `2.1.3`
- `1.0.0-alpha.1`
- `2.0.0-beta.2`
- `1.5.0-rc.1`

Invalid examples:
- `1.0` (missing patch)
- `v1.0.0` (no 'v' prefix)
- `1.0.0.1` (too many components)

### Script Permissions

On Linux/macOS, make scripts executable:

```bash
chmod +x scripts/bump_version.sh
chmod +x scripts/version_manager.py
```

## API Reference

### VersionManager Class

The `VersionManager` class provides programmatic access to version management:

```python
from scripts.version_manager import VersionManager

vm = VersionManager()

# Get current version
current = vm.get_current_version()

# Validate version format
is_valid = vm.validate_version("1.0.0")

# Parse version components
major, minor, patch, pre, build = vm.parse_version("1.0.0-alpha.1")

# Bump version
new_version = vm.bump_version("minor")

# Update all files
vm.update_all_versions("1.2.0")

# Check consistency
is_consistent = vm.check_version_consistency()
```