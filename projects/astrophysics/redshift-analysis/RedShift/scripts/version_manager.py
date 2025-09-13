#!/usr/bin/env python3
"""
Version management utility for QFD CMB Module

This script provides utilities for managing semantic versioning across
the project files, including __init__.py, setup.py, and pyproject.toml.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Tuple, Optional


class VersionManager:
    """Manages semantic versioning across project files."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize version manager.
        
        Args:
            project_root: Path to project root. If None, uses current directory.
        """
        self.project_root = Path(project_root or os.getcwd())
        self.init_file = self.project_root / "qfd_cmb" / "__init__.py"
        self.setup_file = self.project_root / "setup.py"
        self.pyproject_file = self.project_root / "pyproject.toml"
    
    def get_current_version(self) -> str:
        """Get current version from __init__.py.
        
        Returns:
            Current version string.
            
        Raises:
            ValueError: If version cannot be found or parsed.
        """
        if not self.init_file.exists():
            raise ValueError(f"__init__.py not found at {self.init_file}")
        
        content = self.init_file.read_text(encoding='utf-8')
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        
        if not match:
            raise ValueError("Version not found in __init__.py")
        
        return match.group(1)
    
    def validate_version(self, version: str) -> bool:
        """Validate semantic version format.
        
        Args:
            version: Version string to validate.
            
        Returns:
            True if version is valid semantic version.
        """
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'
        return bool(re.match(pattern, version))
    
    def parse_version(self, version: str) -> Tuple[int, int, int, Optional[str], Optional[str]]:
        """Parse semantic version into components.
        
        Args:
            version: Version string to parse.
            
        Returns:
            Tuple of (major, minor, patch, prerelease, build).
            
        Raises:
            ValueError: If version format is invalid.
        """
        if not self.validate_version(version):
            raise ValueError(f"Invalid semantic version format: {version}")
        
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'
        match = re.match(pattern, version)
        
        return (
            int(match.group(1)),  # major
            int(match.group(2)),  # minor
            int(match.group(3)),  # patch
            match.group(4),       # prerelease
            match.group(5)        # build
        )
    
    def bump_version(self, bump_type: str, prerelease: Optional[str] = None) -> str:
        """Bump version according to semantic versioning rules.
        
        Args:
            bump_type: Type of bump ('major', 'minor', 'patch', 'prerelease').
            prerelease: Prerelease identifier for prerelease bumps.
            
        Returns:
            New version string.
            
        Raises:
            ValueError: If bump type is invalid or current version is malformed.
        """
        current = self.get_current_version()
        major, minor, patch, current_pre, build = self.parse_version(current)
        
        if bump_type == 'major':
            return f"{major + 1}.0.0"
        elif bump_type == 'minor':
            return f"{major}.{minor + 1}.0"
        elif bump_type == 'patch':
            return f"{major}.{minor}.{patch + 1}"
        elif bump_type == 'prerelease':
            if not prerelease:
                prerelease = 'alpha'
            
            if current_pre:
                # Extract numeric part and increment
                pre_match = re.match(r'([a-zA-Z]+)\.?(\d+)?', current_pre)
                if pre_match:
                    pre_name = pre_match.group(1)
                    pre_num = int(pre_match.group(2) or 0) + 1
                    return f"{major}.{minor}.{patch}-{pre_name}.{pre_num}"
            
            return f"{major}.{minor}.{patch}-{prerelease}.1"
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
    
    def update_version_in_file(self, file_path: Path, new_version: str) -> None:
        """Update version in a specific file.
        
        Args:
            file_path: Path to file to update.
            new_version: New version string.
        """
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            return
        
        content = file_path.read_text(encoding='utf-8')
        
        if file_path.name == "__init__.py":
            # Update __version__ = "x.y.z"
            content = re.sub(
                r'__version__\s*=\s*["\'][^"\']+["\']',
                f'__version__ = "{new_version}"',
                content
            )
        elif file_path.name == "pyproject.toml":
            # Update version = "x.y.z"
            content = re.sub(
                r'version\s*=\s*["\'][^"\']+["\']',
                f'version = "{new_version}"',
                content
            )
        elif file_path.name == "setup.py":
            # setup.py reads version from __init__.py, so no direct update needed
            print(f"Skipping {file_path} - version read from __init__.py")
            return
        
        file_path.write_text(content, encoding='utf-8')
        print(f"Updated version to {new_version} in {file_path}")
    
    def update_all_versions(self, new_version: str) -> None:
        """Update version in all project files.
        
        Args:
            new_version: New version string.
        """
        if not self.validate_version(new_version):
            raise ValueError(f"Invalid version format: {new_version}")
        
        files_to_update = [self.init_file, self.pyproject_file]
        
        for file_path in files_to_update:
            self.update_version_in_file(file_path, new_version)
    
    def check_version_consistency(self) -> bool:
        """Check if versions are consistent across all files.
        
        Returns:
            True if all versions match, False otherwise.
        """
        init_version = self.get_current_version()
        
        # Check pyproject.toml
        if self.pyproject_file.exists():
            content = self.pyproject_file.read_text(encoding='utf-8')
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                pyproject_version = match.group(1)
                if pyproject_version != init_version:
                    print(f"Version mismatch: __init__.py={init_version}, pyproject.toml={pyproject_version}")
                    return False
        
        print(f"All versions consistent: {init_version}")
        return True


def main():
    """Main CLI interface for version management."""
    parser = argparse.ArgumentParser(description="Manage project version")
    parser.add_argument(
        "action",
        choices=["get", "set", "bump", "check"],
        help="Action to perform"
    )
    parser.add_argument(
        "--version",
        help="Version to set (for 'set' action)"
    )
    parser.add_argument(
        "--bump-type",
        choices=["major", "minor", "patch", "prerelease"],
        help="Type of version bump (for 'bump' action)"
    )
    parser.add_argument(
        "--prerelease",
        help="Prerelease identifier (for prerelease bumps)"
    )
    parser.add_argument(
        "--project-root",
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    try:
        vm = VersionManager(args.project_root)
        
        if args.action == "get":
            print(vm.get_current_version())
        
        elif args.action == "set":
            if not args.version:
                print("Error: --version required for 'set' action")
                sys.exit(1)
            vm.update_all_versions(args.version)
            print(f"Version set to {args.version}")
        
        elif args.action == "bump":
            if not args.bump_type:
                print("Error: --bump-type required for 'bump' action")
                sys.exit(1)
            
            new_version = vm.bump_version(args.bump_type, args.prerelease)
            vm.update_all_versions(new_version)
            print(f"Version bumped to {new_version}")
        
        elif args.action == "check":
            if vm.check_version_consistency():
                print("✓ All versions are consistent")
                sys.exit(0)
            else:
                print("✗ Version inconsistency detected")
                sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()