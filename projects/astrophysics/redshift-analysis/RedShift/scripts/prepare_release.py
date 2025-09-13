#!/usr/bin/env python3
"""
Release preparation script for QFD CMB Module

This script helps prepare releases by:
1. Updating version numbers
2. Updating CHANGELOG.md
3. Creating git tags
4. Validating release readiness
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from version_manager import VersionManager


class ReleaseManager:
    """Manages release preparation and validation."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize release manager.
        
        Args:
            project_root: Path to project root. If None, uses current directory.
        """
        self.project_root = Path(project_root or os.getcwd())
        self.version_manager = VersionManager(project_root)
        self.changelog_path = self.project_root / "CHANGELOG.md"
    
    def run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command.
        
        Args:
            cmd: Command to run as list of strings.
            check: Whether to raise exception on non-zero exit code.
            
        Returns:
            CompletedProcess result.
        """
        print(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root, check=check, 
                            capture_output=True, text=True)
    
    def get_git_status(self) -> str:
        """Get git status output."""
        try:
            result = self.run_command(["git", "status", "--porcelain"])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""
    
    def is_git_clean(self) -> bool:
        """Check if git working directory is clean."""
        try:
            return len(self.get_git_status()) == 0
        except:
            # If git is not available or not a git repo, consider it "clean"
            return True
    
    def get_current_branch(self) -> str:
        """Get current git branch name."""
        try:
            result = self.run_command(["git", "branch", "--show-current"])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "main"  # Default assumption
    
    def get_latest_tag(self) -> Optional[str]:
        """Get the latest git tag."""
        try:
            result = self.run_command(["git", "describe", "--tags", "--abbrev=0"], check=False)
            if result.returncode == 0:
                return result.stdout.strip()
        except subprocess.CalledProcessError:
            pass
        return None
    
    def get_commits_since_tag(self, tag: Optional[str] = None) -> List[str]:
        """Get commits since the given tag (or all commits if no tag)."""
        if tag:
            range_spec = f"{tag}..HEAD"
        else:
            # Get last 50 commits if no tag
            range_spec = "HEAD~50..HEAD"
        
        try:
            result = self.run_command(["git", "log", range_spec, "--oneline", "--no-merges"])
            return [line.strip() for line in result.stdout.split('\n') if line.strip()]
        except subprocess.CalledProcessError:
            return []
    
    def update_changelog(self, version: str, commits: List[str]) -> None:
        """Update CHANGELOG.md with new version entry.
        
        Args:
            version: Version being released.
            commits: List of commit messages since last release.
        """
        # Read existing changelog
        if self.changelog_path.exists():
            existing_content = self.changelog_path.read_text(encoding='utf-8')
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        
        # Generate new entry
        date_str = datetime.now().strftime("%Y-%m-%d")
        new_entry = f"## [{version}] - {date_str}\n\n"
        
        # Categorize commits
        categories = {
            'Added': [],
            'Changed': [],
            'Fixed': [],
            'Removed': [],
            'Security': [],
            'Other': []
        }
        
        for commit in commits:
            commit_msg = ' '.join(commit.split()[1:])  # Remove hash
            commit_lower = commit_msg.lower()
            
            if any(word in commit_lower for word in ['feat:', 'feature:', 'add']):
                categories['Added'].append(commit_msg)
            elif any(word in commit_lower for word in ['fix:', 'bugfix:', 'bug']):
                categories['Fixed'].append(commit_msg)
            elif any(word in commit_lower for word in ['change:', 'update:', 'modify']):
                categories['Changed'].append(commit_msg)
            elif any(word in commit_lower for word in ['remove:', 'delete:', 'drop']):
                categories['Removed'].append(commit_msg)
            elif any(word in commit_lower for word in ['security:', 'sec:']):
                categories['Security'].append(commit_msg)
            else:
                categories['Other'].append(commit_msg)
        
        # Build changelog entry
        for category, items in categories.items():
            if items:
                new_entry += f"### {category}\n\n"
                for item in items:
                    new_entry += f"- {item}\n"
                new_entry += "\n"
        
        if not any(categories.values()):
            new_entry += "- Initial release\n\n"
        
        # Insert new entry after header
        lines = existing_content.split('\n')
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('## ') and '[' in line:
                header_end = i
                break
            elif i > 10:  # Safety check - don't search too far
                header_end = len(lines)
                break
        
        if header_end == 0:
            # No existing entries, add after header
            for i, line in enumerate(lines):
                if line.strip() == '' and i > 0:
                    header_end = i + 1
                    break
        
        # Insert new entry
        new_lines = lines[:header_end] + [new_entry] + lines[header_end:]
        
        # Write updated changelog
        self.changelog_path.write_text('\n'.join(new_lines), encoding='utf-8')
        print(f"Updated {self.changelog_path}")
    
    def validate_release_readiness(self) -> List[str]:
        """Validate that the project is ready for release.
        
        Returns:
            List of validation errors (empty if ready).
        """
        errors = []
        warnings = []
        
        # Check git status (only if in a git repository)
        try:
            if not self.is_git_clean():
                errors.append("Git working directory is not clean")
            
            # Check branch
            current_branch = self.get_current_branch()
            if current_branch not in ['main', 'master', 'develop']:
                warnings.append(f"Not on a typical release branch (current: {current_branch})")
        except:
            # Not a git repository - skip git checks
            warnings.append("Not in a git repository - skipping git checks")
        
        # Check version consistency
        try:
            if not self.version_manager.check_version_consistency():
                errors.append("Version inconsistency detected")
        except Exception as e:
            errors.append(f"Version validation failed: {e}")
        
        # Check if tests pass (optional - skip if pytest not available or no tests)
        test_dir = self.project_root / "tests"
        if test_dir.exists() and any(test_dir.glob("test_*.py")):
            try:
                # Check if pytest is available
                result = self.run_command(["python", "-c", "import pytest"], check=False)
                if result.returncode == 0:
                    result = self.run_command(["python", "-m", "pytest", "--tb=short", "-x"], check=False)
                    if result.returncode != 0:
                        errors.append("Tests are failing")
                else:
                    warnings.append("pytest not available - cannot validate tests")
            except (FileNotFoundError, subprocess.CalledProcessError):
                warnings.append("Cannot run tests - pytest not available")
        else:
            warnings.append("No test files found - skipping test validation")
        
        # Check if package builds (optional - skip if build tools not available)
        try:
            # Check if build module is available
            result = self.run_command(["python", "-c", "import build"], check=False)
            if result.returncode == 0:
                result = self.run_command(["python", "-m", "build", "--outdir", "dist-test"], check=False)
                if result.returncode != 0:
                    errors.append("Package build failed")
                else:
                    # Clean up test build
                    import shutil
                    test_dist = self.project_root / "dist-test"
                    if test_dist.exists():
                        shutil.rmtree(test_dist)
            else:
                warnings.append("build module not available - cannot validate package build")
        except (FileNotFoundError, subprocess.CalledProcessError):
            warnings.append("Cannot validate package build - build tools not available")
        
        # Print warnings
        if warnings:
            print("⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        return errors
    
    def prepare_release(self, bump_type: str, prerelease: Optional[str] = None, 
                       skip_validation: bool = False) -> str:
        """Prepare a new release.
        
        Args:
            bump_type: Type of version bump ('major', 'minor', 'patch', 'prerelease').
            prerelease: Prerelease identifier for prerelease bumps.
            skip_validation: Skip release readiness validation.
            
        Returns:
            New version string.
            
        Raises:
            ValueError: If release preparation fails.
        """
        # Validate release readiness
        if not skip_validation:
            errors = self.validate_release_readiness()
            if errors:
                print("❌ Release validation failed:")
                for error in errors:
                    print(f"  - {error}")
                raise ValueError("Release not ready")
        
        # Get current version and commits
        current_version = self.version_manager.get_current_version()
        latest_tag = self.get_latest_tag()
        commits = self.get_commits_since_tag(latest_tag)
        
        print(f"Current version: {current_version}")
        print(f"Latest tag: {latest_tag or 'None'}")
        print(f"Commits since last release: {len(commits)}")
        
        # Bump version
        new_version = self.version_manager.bump_version(bump_type, prerelease)
        self.version_manager.update_all_versions(new_version)
        
        print(f"Bumped version to: {new_version}")
        
        # Update changelog
        self.update_changelog(new_version, commits)
        
        # Stage changes
        self.run_command(["git", "add", "qfd_cmb/__init__.py", "pyproject.toml", "CHANGELOG.md"])
        
        print(f"✅ Release {new_version} prepared successfully!")
        print("\nNext steps:")
        print(f"1. Review changes: git diff --cached")
        print(f"2. Commit changes: git commit -m 'Prepare release {new_version}'")
        print(f"3. Create tag: git tag -a v{new_version} -m 'Release {new_version}'")
        print(f"4. Push changes: git push origin main --tags")
        
        return new_version
    
    def create_tag(self, version: str, message: Optional[str] = None) -> None:
        """Create and push a git tag for the release.
        
        Args:
            version: Version to tag.
            message: Tag message. If None, uses default format.
        """
        if not message:
            message = f"Release {version}"
        
        tag_name = f"v{version}"
        
        # Create tag
        self.run_command(["git", "tag", "-a", tag_name, "-m", message])
        print(f"Created tag: {tag_name}")
        
        # Push tag
        self.run_command(["git", "push", "origin", tag_name])
        print(f"Pushed tag: {tag_name}")


def main():
    """Main CLI interface for release management."""
    parser = argparse.ArgumentParser(description="Prepare project release")
    parser.add_argument(
        "action",
        choices=["prepare", "validate", "tag"],
        help="Action to perform"
    )
    parser.add_argument(
        "--bump-type",
        choices=["major", "minor", "patch", "prerelease"],
        help="Type of version bump (for 'prepare' action)"
    )
    parser.add_argument(
        "--prerelease",
        help="Prerelease identifier (for prerelease bumps)"
    )
    parser.add_argument(
        "--version",
        help="Version to tag (for 'tag' action)"
    )
    parser.add_argument(
        "--message",
        help="Tag message (for 'tag' action)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip release readiness validation"
    )
    parser.add_argument(
        "--project-root",
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    try:
        rm = ReleaseManager(args.project_root)
        
        if args.action == "validate":
            errors = rm.validate_release_readiness()
            if errors:
                print("❌ Release validation failed:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            else:
                print("✅ Release validation passed")
        
        elif args.action == "prepare":
            if not args.bump_type:
                print("Error: --bump-type required for 'prepare' action")
                sys.exit(1)
            
            new_version = rm.prepare_release(
                args.bump_type, 
                args.prerelease, 
                args.skip_validation
            )
            print(f"Prepared release: {new_version}")
        
        elif args.action == "tag":
            if not args.version:
                # Use current version
                args.version = rm.version_manager.get_current_version()
            
            rm.create_tag(args.version, args.message)
            print(f"Tagged release: v{args.version}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()