# Release Process

This document describes the automated release process for the QFD CMB Module, including version management, changelog generation, and publishing to PyPI.

## Overview

The release process is fully automated using GitHub Actions and includes:

1. **Version Management**: Semantic versioning with consistency checks
2. **Automated Testing**: Multi-platform testing before release
3. **Changelog Generation**: Automatic changelog from git commits
4. **GitHub Releases**: Automated release creation with assets
5. **PyPI Publishing**: Automatic package publishing
6. **Pre-release Support**: Alpha, beta, and release candidate versions

## Release Types

### Stable Releases

- **Patch Release** (`1.0.0` → `1.0.1`): Bug fixes and minor improvements
- **Minor Release** (`1.0.0` → `1.1.0`): New features, backwards compatible
- **Major Release** (`1.0.0` → `2.0.0`): Breaking changes

### Pre-releases

- **Alpha** (`1.0.0-alpha.1`): Early development, unstable API
- **Beta** (`1.0.0-beta.1`): Feature-complete, testing phase
- **Release Candidate** (`1.0.0-rc.1`): Final testing before stable release

## Quick Start

### Using Release Scripts

**Linux/macOS:**
```bash
# Make scripts executable (first time only)
chmod +x scripts/release.sh scripts/bump_version.sh

# Validate release readiness
./scripts/release.sh validate

# Prepare a patch release
./scripts/release.sh prepare patch

# Review and commit changes
git diff --cached
git commit -m "Prepare release v1.0.1"

# Create and push tag (triggers release)
./scripts/release.sh tag
```

**Windows:**
```cmd
REM Validate release readiness
scripts\release.bat validate

REM Prepare a patch release
scripts\release.bat prepare patch

REM Review and commit changes
git diff --cached
git commit -m "Prepare release v1.0.1"

REM Create and push tag (triggers release)
scripts\release.bat tag
```

### Manual Process

```bash
# 1. Validate release readiness
python scripts/prepare_release.py validate

# 2. Bump version and update changelog
python scripts/prepare_release.py prepare --bump-type patch

# 3. Review changes
git diff --cached

# 4. Commit changes
git commit -m "Prepare release v1.0.1"

# 5. Create and push tag
python scripts/prepare_release.py tag

# 6. GitHub Actions will handle the rest automatically
```

## Detailed Workflow

### 1. Pre-Release Validation

Before creating a release, the system validates:

- ✅ Git working directory is clean
- ✅ Currently on a release branch (`main`, `master`, or `develop`)
- ✅ All tests pass
- ✅ Package builds successfully
- ✅ Version consistency across files

```bash
# Check release readiness
./scripts/release.sh validate
```

### 2. Version Bumping

The version manager automatically updates versions in:
- `qfd_cmb/__init__.py`
- `pyproject.toml`
- Validates consistency with `setup.py`

```bash
# Bump patch version (1.0.0 → 1.0.1)
./scripts/release.sh prepare patch

# Bump minor version (1.0.0 → 1.1.0)
./scripts/release.sh prepare minor

# Bump major version (1.0.0 → 2.0.0)
./scripts/release.sh prepare major

# Create prerelease (1.0.0 → 1.0.0-alpha.1)
./scripts/release.sh prepare prerelease alpha
```

### 3. Changelog Generation

The system automatically generates changelog entries from git commits:

- Categorizes commits by type (Features, Bug Fixes, etc.)
- Uses conventional commit format when available
- Links to full changelog on GitHub
- Updates `CHANGELOG.md` with new entries

### 4. Git Tagging

Creating a git tag triggers the automated release:

```bash
# Tag current version
./scripts/release.sh tag

# Tag specific version with message
./scripts/release.sh tag 1.0.1 "Bug fix release"
```

### 5. Automated Release Pipeline

When a tag is pushed, GitHub Actions automatically:

1. **Validates** the release (version consistency, tests)
2. **Tests** across multiple platforms and Python versions
3. **Builds** the package and validates it
4. **Generates** changelog from git history
5. **Creates** GitHub release with changelog
6. **Publishes** to PyPI (stable releases) or Test PyPI (pre-releases)
7. **Notifies** of success or failure

## GitHub Actions Workflows

### Release Workflow (`.github/workflows/release.yml`)

Triggered by:
- **Git tags**: `v*` (e.g., `v1.0.0`)
- **Manual dispatch**: With version input

Jobs:
1. **validate**: Version consistency and format validation
2. **test**: Multi-platform testing (Ubuntu, macOS, Windows)
3. **build**: Package building and validation
4. **generate-changelog**: Automatic changelog generation
5. **create-release**: GitHub release creation
6. **publish-pypi**: PyPI publishing (stable releases)
7. **publish-test-pypi**: Test PyPI publishing (pre-releases)
8. **notify**: Success/failure notifications

### Manual Release Trigger

You can also trigger releases manually from GitHub:

1. Go to **Actions** → **Release** workflow
2. Click **Run workflow**
3. Enter version number (e.g., `1.0.1`)
4. Choose if it's a pre-release
5. Click **Run workflow**

## PyPI Publishing

### Stable Releases

Stable releases are automatically published to [PyPI](https://pypi.org/project/qfd-cmb/):
- Triggered by version tags without pre-release identifiers
- Uses PyPI API token stored in GitHub secrets
- Package becomes immediately available via `pip install qfd-cmb`

### Pre-releases

Pre-releases are published to [Test PyPI](https://test.pypi.org/project/qfd-cmb/):
- Triggered by version tags with pre-release identifiers (`-alpha`, `-beta`, `-rc`)
- Allows testing the package before stable release
- Install with: `pip install -i https://test.pypi.org/simple/ qfd-cmb`

## Configuration

### GitHub Secrets

Required secrets for automated publishing:

- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `TEST_PYPI_API_TOKEN`: Test PyPI API token for pre-release publishing

### Trusted Publishing (Recommended)

For enhanced security, configure [trusted publishing](https://docs.pypi.org/trusted-publishers/):

1. Configure trusted publisher on PyPI
2. Remove `password` from workflow
3. Use `id-token: write` permission

## Branch Strategy

### Recommended Workflow

```
main (stable releases)
├── develop (development)
├── feature/new-feature
├── release/1.1.0 (release preparation)
└── hotfix/critical-bug
```

### Release Branches

- **main/master**: Stable releases only
- **develop**: Development work, pre-releases
- **release/***: Release preparation branches
- **hotfix/***: Critical bug fixes

## Examples

### Patch Release (Bug Fix)

```bash
# On main branch
git checkout main
git pull origin main

# Validate and prepare
./scripts/release.sh validate
./scripts/release.sh prepare patch

# Review and commit
git diff --cached
git commit -m "Prepare release v1.0.1"

# Create tag and trigger release
./scripts/release.sh tag
```

### Minor Release (New Features)

```bash
# On develop branch
git checkout develop
git pull origin develop

# Prepare release branch
git checkout -b release/1.1.0

# Prepare release
./scripts/release.sh prepare minor
git commit -m "Prepare release v1.1.0"

# Merge to main
git checkout main
git merge release/1.1.0

# Tag and release
./scripts/release.sh tag
```

### Pre-release (Alpha)

```bash
# On develop branch
./scripts/release.sh prepare prerelease alpha
git commit -m "Prepare alpha release v1.1.0-alpha.1"
./scripts/release.sh tag
```

## Troubleshooting

### Version Inconsistency

```bash
# Check which files have inconsistent versions
python scripts/version_manager.py check

# Fix by setting all to the same version
python scripts/version_manager.py set --version "1.0.0"
```

### Failed Release

If a release fails:

1. Check GitHub Actions logs
2. Fix the issue
3. Delete the failed tag: `git tag -d v1.0.1 && git push origin :refs/tags/v1.0.1`
4. Re-run the release process

### PyPI Publishing Issues

- Verify API tokens are correctly set in GitHub secrets
- Check that version doesn't already exist on PyPI
- Ensure package builds successfully locally

### Rollback Release

To rollback a release:

1. Delete the GitHub release
2. Delete the git tag: `git tag -d v1.0.1 && git push origin :refs/tags/v1.0.1`
3. Revert version changes: `git revert <commit-hash>`
4. Consider yanking from PyPI if necessary

## Best Practices

### Commit Messages

Use conventional commit format for better changelog generation:

```
feat: add new CMB analysis function
fix: resolve numerical instability in projector
docs: update API documentation
test: add integration tests for visibility module
refactor: optimize kernel computation
```

### Release Timing

- **Patch releases**: As needed for bug fixes
- **Minor releases**: Monthly or when significant features are ready
- **Major releases**: Quarterly or for breaking changes
- **Pre-releases**: Weekly during active development

### Testing

- Always run full test suite before release
- Test on multiple platforms if possible
- Validate examples and documentation
- Check backwards compatibility

### Documentation

- Update documentation before release
- Ensure examples work with new version
- Update API documentation for new features
- Review and update README if needed

## Monitoring

### Release Health

Monitor release health through:
- GitHub Actions success/failure rates
- PyPI download statistics
- User feedback and issue reports
- Automated test results

### Metrics

Track key metrics:
- Release frequency
- Time from commit to release
- Test coverage
- Download counts
- Issue resolution time