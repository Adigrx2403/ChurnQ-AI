"""
Test runner for the Churn Analysis System.
This script runs all tests and provides a comprehensive test report.
"""

import sys
import unittest
import warnings
from pathlib import Path
import importlib.util

# Suppress warnings during testing
warnings.filterwarnings('ignore')

# Add project directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'config'))

def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"⚠️  Could not import {module_name}: {e}")
        return None

def run_data_processing_tests():
    """Run data processing tests."""
    print("🔄 Running Data Processing Tests...")
    
    try:
        # Import the test module
        test_module = import_module_from_file(
            'test_data_processing',
            project_root / 'tests' / 'test_data_processing.py'
        )
        
        if test_module is None:
            print("❌ Could not load data processing tests")
            return False
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test cases
        for name in dir(test_module):
            obj = getattr(test_module, name)
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                suite.addTests(loader.loadTestsFromTestCase(obj))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error running data processing tests: {e}")
        return False

def run_model_tests():
    """Run model tests."""
    print("\n🔄 Running Model Tests...")
    
    try:
        # Import the test module
        test_module = import_module_from_file(
            'test_models',
            project_root / 'tests' / 'test_models.py'
        )
        
        if test_module is None:
            print("❌ Could not load model tests")
            return False
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test cases
        for name in dir(test_module):
            obj = getattr(test_module, name)
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                suite.addTests(loader.loadTestsFromTestCase(obj))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error running model tests: {e}")
        return False

def run_integration_tests():
    """Run integration tests."""
    print("\n🔄 Running Integration Tests...")
    
    try:
        # Import the test module
        test_module = import_module_from_file(
            'test_integration',
            project_root / 'tests' / 'test_integration.py'
        )
        
        if test_module is None:
            print("❌ Could not load integration tests")
            return False
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test cases
        for name in dir(test_module):
            obj = getattr(test_module, name)
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                suite.addTests(loader.loadTestsFromTestCase(obj))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error running integration tests: {e}")
        return False

def run_system_health_check():
    """Run system health check."""
    print("\n🔄 Running System Health Check...")
    
    health_results = {}
    
    # Check project structure
    required_dirs = ['src', 'config', 'models', 'data']
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        health_results[f"{dir_name}_directory"] = dir_path.exists()
    
    # Check key files
    key_files = {
        'app.py': project_root / 'app.py',
        'requirements.txt': project_root / 'requirements.txt',
        'processed_data': project_root / 'processed_ecommerce_data.csv',
        'config': project_root / 'config' / 'config.py'
    }
    
    for file_key, file_path in key_files.items():
        health_results[f"{file_key}_exists"] = file_path.exists()
    
    # Check model files
    models_dir = project_root / 'models'
    if models_dir.exists():
        model_files = list(models_dir.glob('*.joblib'))
        health_results['trained_models'] = len(model_files) > 0
    else:
        health_results['trained_models'] = False
    
    # Print health report
    print("📊 System Health Report:")
    for check, status in health_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check.replace('_', ' ').title()}")
    
    # Calculate health score
    health_score = sum(health_results.values()) / len(health_results) * 100
    print(f"\n🏥 Overall System Health: {health_score:.1f}%")
    
    return health_score >= 80

def main():
    """Main test runner function."""
    print("🚀 Starting Churn Analysis System Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Run system health check
    test_results['system_health'] = run_system_health_check()
    
    # Run all test suites
    test_results['data_processing'] = run_data_processing_tests()
    test_results['models'] = run_model_tests() 
    test_results['integration'] = run_integration_tests()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status_icon = "✅" if result else "❌"
        print(f"{status_icon} {test_name.replace('_', ' ').title()}: {'PASSED' if result else 'FAILED'}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate >= 75:
        print("🎉 System is ready for production!")
        return True
    else:
        print("⚠️  System needs attention before production deployment.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
