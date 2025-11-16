#!/usr/bin/env python3
"""
Documentation review script for QFD CMB Module
Reviews all documentation for accuracy, completeness, and consistency
"""

import os
import sys
import ast
import inspect
import subprocess
import tempfile
from pathlib import Path
import re

def check_readme():
    """Review README.md for completeness and accuracy"""
    print("Reviewing README.md...")
    
    if not os.path.exists('README.md'):
        return {"error": "README.md not found"}
    
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    
    # Check length
    results['adequate_length'] = len(content) > 1000
    
    # Check for essential sections
    essential_sections = [
        'installation', 'usage', 'example', 'requirements', 
        'license', 'citation', 'description'
    ]
    
    missing_sections = []
    for section in essential_sections:
        if section.lower() not in content.lower():
            missing_sections.append(section)
    
    results['missing_sections'] = missing_sections
    results['has_essential_sections'] = len(missing_sections) == 0
    
    # Check for code examples
    code_blocks = content.count('```')
    results['has_code_examples'] = code_blocks >= 2  # At least one code block
    
    # Check for installation instructions
    results['has_installation'] = any(word in content.lower() for word in ['pip install', 'conda install', 'setup.py'])
    
    # Check for badges (CI, coverage, etc.)
    results['has_badges'] = '![' in content and 'badge' in content.lower()
    
    return results

def check_docstrings():
    """Check docstrings in all Python modules"""
    print("Reviewing docstrings...")
    
    results = {}
    modules_to_check = [
        'qfd_cmb/ppsi_models.py',
        'qfd_cmb/visibility.py', 
        'qfd_cmb/kernels.py',
        'qfd_cmb/projector.py',
        'qfd_cmb/figures.py'
    ]
    
    for module_path in modules_to_check:
        if not os.path.exists(module_path):
            continue
            
        module_name = os.path.basename(module_path).replace('.py', '')
        results[module_name] = {}
        
        # Parse the module
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            results[module_name]['parse_error'] = True
            continue
        
        # Check module docstring
        module_docstring = ast.get_docstring(tree)
        results[module_name]['has_module_docstring'] = module_docstring is not None
        
        # Check function docstrings
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        function_docs = {}
        
        for func in functions:
            if not func.name.startswith('_'):  # Skip private functions
                docstring = ast.get_docstring(func)
                function_docs[func.name] = {
                    'has_docstring': docstring is not None,
                    'docstring_length': len(docstring) if docstring else 0
                }
        
        results[module_name]['functions'] = function_docs
        
        # Summary stats
        total_functions = len(function_docs)
        documented_functions = sum(1 for f in function_docs.values() if f['has_docstring'])
        results[module_name]['documentation_coverage'] = documented_functions / total_functions if total_functions > 0 else 0
    
    return results

def check_examples():
    """Review example code and tutorials"""
    print("Reviewing examples and tutorials...")
    
    results = {}
    
    # Check examples directory
    if os.path.exists('examples'):
        example_files = os.listdir('examples')
        results['examples_directory_exists'] = True
        results['example_files'] = example_files
        
        # Test example scripts
        python_examples = [f for f in example_files if f.endswith('.py')]
        results['python_examples'] = python_examples
        
        # Try to run basic_usage.py if it exists
        if 'basic_usage.py' in python_examples:
            try:
                result = subprocess.run(
                    [sys.executable, 'examples/basic_usage.py'], 
                    capture_output=True, text=True, timeout=30
                )
                results['basic_usage_runs'] = result.returncode == 0
                if result.returncode != 0:
                    results['basic_usage_error'] = result.stderr
            except subprocess.TimeoutExpired:
                results['basic_usage_runs'] = False
                results['basic_usage_error'] = "Timeout"
            except Exception as e:
                results['basic_usage_runs'] = False
                results['basic_usage_error'] = str(e)
        
        # Check for Jupyter notebooks
        notebooks = [f for f in example_files if f.endswith('.ipynb')]
        results['has_notebooks'] = len(notebooks) > 0
        results['notebooks'] = notebooks
        
    else:
        results['examples_directory_exists'] = False
    
    # Check main demo script
    if os.path.exists('run_demo.py'):
        results['demo_script_exists'] = True
        
        # Test demo script with help flag
        try:
            result = subprocess.run(
                [sys.executable, 'run_demo.py', '--help'], 
                capture_output=True, text=True, timeout=10
            )
            results['demo_help_works'] = result.returncode == 0
        except Exception as e:
            results['demo_help_works'] = False
            results['demo_help_error'] = str(e)
    
    return results

def check_api_documentation():
    """Check API documentation generation"""
    print("Reviewing API documentation...")
    
    results = {}
    
    # Check if docs directory exists
    if os.path.exists('docs'):
        results['docs_directory_exists'] = True
        
        # Check for Sphinx configuration
        if os.path.exists('docs/conf.py'):
            results['sphinx_config_exists'] = True
            
            # Try to build documentation
            try:
                result = subprocess.run(
                    ['sphinx-build', '-b', 'html', 'docs', 'docs/_build/html'],
                    capture_output=True, text=True, timeout=60, cwd='.'
                )
                results['docs_build_success'] = result.returncode == 0
                if result.returncode != 0:
                    results['docs_build_error'] = result.stderr
            except FileNotFoundError:
                results['docs_build_success'] = False
                results['docs_build_error'] = "Sphinx not installed"
            except subprocess.TimeoutExpired:
                results['docs_build_success'] = False
                results['docs_build_error'] = "Build timeout"
            except Exception as e:
                results['docs_build_success'] = False
                results['docs_build_error'] = str(e)
        else:
            results['sphinx_config_exists'] = False
    else:
        results['docs_directory_exists'] = False
    
    return results

def check_contributing_guidelines():
    """Check contributing guidelines and development documentation"""
    print("Reviewing contributing guidelines...")
    
    results = {}
    
    # Check CONTRIBUTING.md
    if os.path.exists('CONTRIBUTING.md'):
        results['contributing_exists'] = True
        
        with open('CONTRIBUTING.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        results['contributing_length'] = len(content)
        
        # Check for essential sections
        essential_sections = ['setup', 'development', 'testing', 'pull request', 'code style']
        missing_sections = []
        for section in essential_sections:
            if section.lower() not in content.lower():
                missing_sections.append(section)
        
        results['contributing_missing_sections'] = missing_sections
        results['contributing_complete'] = len(missing_sections) == 0
    else:
        results['contributing_exists'] = False
    
    # Check CHANGELOG.md
    if os.path.exists('CHANGELOG.md'):
        results['changelog_exists'] = True
        
        with open('CHANGELOG.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        results['changelog_has_versions'] = bool(re.search(r'\[?\d+\.\d+\.\d+\]?', content))
    else:
        results['changelog_exists'] = False
    
    # Check LICENSE
    if os.path.exists('LICENSE'):
        results['license_exists'] = True
        
        with open('LICENSE', 'r', encoding='utf-8') as f:
            content = f.read()
        
        results['license_length'] = len(content)
        results['license_adequate'] = len(content) > 500  # Reasonable license length
    else:
        results['license_exists'] = False
    
    return results

def validate_code_examples():
    """Validate that code examples in documentation actually work"""
    print("Validating code examples...")
    
    results = {}
    
    # Check README code examples
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        # Extract Python code blocks
        python_blocks = re.findall(r'```python\n(.*?)\n```', readme_content, re.DOTALL)
        results['readme_python_blocks'] = len(python_blocks)
        
        # Try to validate simple import statements
        valid_examples = 0
        for block in python_blocks:
            if 'import' in block and 'qfd_cmb' in block:
                try:
                    # Create a temporary file and try to run it
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(block)
                        temp_file = f.name
                    
                    result = subprocess.run(
                        [sys.executable, temp_file], 
                        capture_output=True, text=True, timeout=10
                    )
                    
                    if result.returncode == 0:
                        valid_examples += 1
                    
                    os.unlink(temp_file)
                    
                except Exception:
                    pass
        
        results['valid_readme_examples'] = valid_examples
    
    return results

def main():
    """Run comprehensive documentation review"""
    print("=" * 60)
    print("QFD CMB Module - Documentation Review")
    print("=" * 60)
    
    all_results = {}
    
    # Run all checks
    checks = [
        ("README", check_readme),
        ("Docstrings", check_docstrings),
        ("Examples", check_examples),
        ("API Documentation", check_api_documentation),
        ("Contributing Guidelines", check_contributing_guidelines),
        ("Code Examples", validate_code_examples)
    ]
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            results = check_func()
            all_results[check_name.lower().replace(' ', '_')] = results
            
            # Print summary for this check
            if isinstance(results, dict):
                passed = sum(1 for v in results.values() if v is True)
                failed = sum(1 for v in results.values() if v is False)
                print(f"  ✅ {passed} checks passed, ❌ {failed} checks failed")
            
        except Exception as e:
            print(f"  ❌ Error during {check_name} check: {e}")
            all_results[check_name.lower().replace(' ', '_')] = {"error": str(e)}
    
    # Overall summary
    print("\n" + "=" * 60)
    print("DOCUMENTATION REVIEW SUMMARY")
    print("=" * 60)
    
    # Key findings
    readme_results = all_results.get('readme', {})
    if readme_results.get('has_essential_sections'):
        print("✅ README has all essential sections")
    else:
        missing = readme_results.get('missing_sections', [])
        print(f"⚠️  README missing sections: {missing}")
    
    docstring_results = all_results.get('docstrings', {})
    total_coverage = 0
    module_count = 0
    for module, data in docstring_results.items():
        if isinstance(data, dict) and 'documentation_coverage' in data:
            total_coverage += data['documentation_coverage']
            module_count += 1
    
    if module_count > 0:
        avg_coverage = total_coverage / module_count
        print(f"📚 Average docstring coverage: {avg_coverage:.1%}")
    
    examples_results = all_results.get('examples', {})
    if examples_results.get('examples_directory_exists'):
        print("✅ Examples directory exists")
        if examples_results.get('basic_usage_runs'):
            print("✅ Basic usage example runs successfully")
        else:
            print("⚠️  Basic usage example has issues")
    
    api_docs_results = all_results.get('api_documentation', {})
    if api_docs_results.get('docs_build_success'):
        print("✅ API documentation builds successfully")
    elif api_docs_results.get('docs_directory_exists'):
        print("⚠️  Documentation directory exists but build failed")
    else:
        print("⚠️  No API documentation found")
    
    contributing_results = all_results.get('contributing_guidelines', {})
    if contributing_results.get('contributing_complete'):
        print("✅ Contributing guidelines are complete")
    elif contributing_results.get('contributing_exists'):
        print("⚠️  Contributing guidelines exist but may be incomplete")
    else:
        print("⚠️  No contributing guidelines found")
    
    # Save detailed results
    import json
    with open('documentation_review_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: documentation_review_results.json")
    
    # Determine overall status
    critical_issues = 0
    
    if not readme_results.get('has_essential_sections'):
        critical_issues += 1
    
    if avg_coverage < 0.5 if module_count > 0 else True:
        critical_issues += 1
    
    if not examples_results.get('basic_usage_runs', True):
        critical_issues += 1
    
    if critical_issues == 0:
        print("\n✅ DOCUMENTATION REVIEW PASSED - Documentation is ready for publication")
        return 0
    else:
        print(f"\n⚠️  DOCUMENTATION REVIEW - {critical_issues} critical issues found")
        return 1

if __name__ == "__main__":
    sys.exit(main())