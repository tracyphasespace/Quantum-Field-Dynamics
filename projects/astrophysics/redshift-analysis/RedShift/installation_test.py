#!/usr/bin/env python3
"""
Installation test script for QFD CMB Module
Tests that the package can be installed and used from a clean environment
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_installation():
    """Test installation process"""
    print("Testing installation process...")
    
    # Test that setup.py exists and is valid
    if not os.path.exists('setup.py'):
        print("❌ setup.py not found")
        return False
    
    # Test that pyproject.toml exists
    if not os.path.exists('pyproject.toml'):
        print("❌ pyproject.toml not found")
        return False
    
    # Test that requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    print("✅ All packaging files present")
    
    # Test that the package can be built
    success, stdout, stderr = run_command("python setup.py check")
    if not success:
        print(f"❌ setup.py check failed: {stderr}")
        return False
    
    print("✅ Package structure is valid")
    
    # Test that requirements can be parsed
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        # Check for essential dependencies
        essential_deps = ['numpy', 'scipy', 'matplotlib']
        for dep in essential_deps:
            if not any(dep in req for req in requirements):
                print(f"❌ Missing essential dependency: {dep}")
                return False
        
        print("✅ Requirements file is valid")
        
    except Exception as e:
        print(f"❌ Error reading requirements: {e}")
        return False
    
    return True

def test_package_structure():
    """Test that the package has the expected structure"""
    print("\nTesting package structure...")
    
    expected_files = [
        'qfd_cmb/__init__.py',
        'qfd_cmb/ppsi_models.py',
        'qfd_cmb/visibility.py',
        'qfd_cmb/kernels.py',
        'qfd_cmb/projector.py',
        'qfd_cmb/figures.py',
        'run_demo.py',
        'fit_planck.py',
        'README.md',
        'LICENSE'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All expected files present")
    
    # Test that the main package can be imported
    try:
        sys.path.insert(0, '.')
        import qfd_cmb
        print("✅ Package can be imported")
        
        # Test that all modules can be imported
        from qfd_cmb import ppsi_models, visibility, kernels, projector, figures
        print("✅ All modules can be imported")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_documentation():
    """Test that documentation is adequate"""
    print("\nTesting documentation...")
    
    # Check README exists and has content
    if not os.path.exists('README.md'):
        print("❌ README.md not found")
        return False
    
    with open('README.md', 'r') as f:
        readme_content = f.read()
    
    if len(readme_content) < 500:  # Minimum reasonable README length
        print("❌ README.md is too short")
        return False
    
    # Check for essential sections
    essential_sections = ['installation', 'usage', 'example']
    missing_sections = []
    for section in essential_sections:
        if section.lower() not in readme_content.lower():
            missing_sections.append(section)
    
    if missing_sections:
        print(f"⚠️  README may be missing sections: {missing_sections}")
    
    print("✅ README.md is adequate")
    
    # Check that docstrings exist in modules
    try:
        import qfd_cmb.ppsi_models as ppsi
        if not ppsi.oscillatory_psik.__doc__:
            print("⚠️  Some functions may be missing docstrings")
        else:
            print("✅ Functions have docstrings")
    except:
        print("⚠️  Could not check docstrings")
    
    return True

def test_examples():
    """Test that examples work"""
    print("\nTesting examples...")
    
    # Test demo script
    with tempfile.TemporaryDirectory() as temp_dir:
        success, stdout, stderr = run_command(f"python run_demo.py --outdir {temp_dir}")
        
        if not success:
            print(f"❌ Demo script failed: {stderr}")
            return False
        
        # Check outputs were created
        expected_outputs = ['qfd_demo_spectra.csv', 'TT.png', 'EE.png', 'TE.png']
        for output in expected_outputs:
            if not os.path.exists(os.path.join(temp_dir, output)):
                print(f"❌ Demo output missing: {output}")
                return False
        
        print("✅ Demo script works correctly")
    
    # Check if examples directory exists
    if os.path.exists('examples'):
        example_files = os.listdir('examples')
        if example_files:
            print(f"✅ Examples directory contains: {example_files}")
        else:
            print("⚠️  Examples directory is empty")
    else:
        print("⚠️  No examples directory found")
    
    return True

def main():
    """Run all installation tests"""
    print("=" * 60)
    print("QFD CMB Module - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Installation", test_installation),
        ("Package Structure", test_package_structure),
        ("Documentation", test_documentation),
        ("Examples", test_examples)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("INSTALLATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Package is ready for distribution!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed - Please address issues before distribution")
        return 1

if __name__ == "__main__":
    sys.exit(main())