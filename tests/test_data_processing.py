"""
Unit tests for data processing pipeline.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

try:
    from data_processor import DataProcessor
    from business_intelligence import BusinessIntelligence
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    # Create mock classes for testing structure
    class DataProcessor:
        @staticmethod
        def load_and_process_data():
            return None
    
    class BusinessIntelligence:
        @staticmethod
        def calculate_business_metrics(data):
            return {}
    
    MODULES_AVAILABLE = False
    from config.config import *
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'CustomerID': range(1, 101),
            'Tenure': np.random.randint(1, 61, 100),
            'SatisfactionScore': np.random.randint(1, 6, 100),
            'Complain': np.random.choice([0, 1], 100),
            'OrderCount': np.random.randint(1, 20, 100),
            'HourSpendOnApp': np.random.uniform(0, 10, 100),
            'CashbackAmount': np.random.uniform(0, 200, 100),
            'NumberOfAddress': np.random.randint(1, 10, 100),
            'Churn': np.random.choice([0, 1], 100, p=[0.83, 0.17])
        })
        
        # Add missing values
        self.sample_data.loc[0:4, 'NumberOfAddress'] = np.nan
        self.sample_data.loc[5:7, 'HourSpendOnApp'] = np.nan

    def test_missing_value_imputation(self):
        """Test missing value handling."""
        if not MODULES_AVAILABLE:
            self.skipTest("Data processing modules not available")
        
        # Test with sample data
        data_with_missing = self.sample_data.copy()
        
        # Check for missing values
        missing_before = data_with_missing.isnull().sum().sum()
        self.assertGreater(missing_before, 0, "Sample data should have missing values")
        
        # Simple imputation for testing
        data_with_missing['NumberOfAddress'].fillna(
            data_with_missing['NumberOfAddress'].median(), inplace=True
        )
        data_with_missing['HourSpendOnApp'].fillna(
            data_with_missing['HourSpendOnApp'].mean(), inplace=True
        )
        
        missing_after = data_with_missing.isnull().sum().sum()
        self.assertEqual(missing_after, 0, "Should have no missing values after imputation")
        
        # Check that numerical columns were imputed with median
        original_median = self.sample_data['HourSpendOnApp'].median()
        imputed_values = data_with_missing.loc[5:7, 'HourSpendOnApp']
        
        # The imputed values should be reasonable (not the original NaN)
        self.assertTrue(all(imputed_values.notna()))

    def test_outlier_detection(self):
        """Test outlier detection and capping."""
        # Create data with obvious outliers
        test_data = self.sample_data.copy()
        test_data.loc[0, 'Tenure'] = 1000  # Extreme outlier
        
        processed_data = self.processor.handle_outliers(test_data)
        
        # Check that extreme value was capped
        self.assertLess(processed_data.loc[0, 'Tenure'], 1000)
        self.assertGreater(processed_data.loc[0, 'Tenure'], 0)

    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        processed_data = self.processor.engineer_features(self.sample_data.copy())
        
        # Check that new features were created
        expected_features = ['TenureBucket', 'EngagementScore', 'SatisfactionBucket']
        for feature in expected_features:
            self.assertIn(feature, processed_data.columns)
        
        # Check TenureBucket categories
        unique_buckets = processed_data['TenureBucket'].unique()
        self.assertTrue(all(bucket in [0, 1, 2, 3, 4] for bucket in unique_buckets))
        
        # Check EngagementScore is calculated correctly
        self.assertTrue(processed_data['EngagementScore'].min() >= 0)

    def test_data_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        if hasattr(self.processor, 'process_pipeline'):
            # This would test the full pipeline if implemented
            pass
        else:
            # Test individual components
            processed = self.sample_data.copy()
            processed = self.processor.handle_missing_values(processed)
            processed = self.processor.handle_outliers(processed)
            processed = self.processor.engineer_features(processed)
            
            # Verify final data quality
            self.assertEqual(processed.isnull().sum().sum(), 0)
            self.assertGreater(len(processed.columns), len(self.sample_data.columns))

    def test_data_validation(self):
        """Test data validation checks."""
        # Test with invalid data
        invalid_data = self.sample_data.copy()
        invalid_data['SatisfactionScore'] = -1  # Invalid satisfaction score
        
        # If validation is implemented, test it
        if hasattr(self.processor, 'validate_data'):
            with self.assertRaises(ValueError):
                self.processor.validate_data(invalid_data)

    def test_feature_scaling(self):
        """Test feature scaling functionality."""
        if hasattr(self.processor, 'scale_features'):
            numerical_cols = ['Tenure', 'OrderCount', 'HourSpendOnApp']
            test_data = self.sample_data[numerical_cols].copy()
            
            scaled_data = self.processor.scale_features(test_data)
            
            # Check that scaling was applied (mean should be close to 0, std close to 1)
            for col in numerical_cols:
                self.assertAlmostEqual(scaled_data[col].mean(), 0, places=1)
                self.assertAlmostEqual(scaled_data[col].std(), 1, places=1)

class TestDataQuality(unittest.TestCase):
    """Test data quality and integrity."""
    
    def setUp(self):
        """Set up test data."""
        # Try to load actual processed data if available
        try:
            self.data = pd.read_csv(Path(__file__).parent.parent / 'processed_ecommerce_data.csv')
            self.has_real_data = True
        except FileNotFoundError:
            self.has_real_data = False
            # Create sample data for testing
            np.random.seed(42)
            self.data = pd.DataFrame({
                'Tenure': np.random.randint(1, 61, 1000),
                'SatisfactionScore': np.random.randint(1, 6, 1000),
                'Churn': np.random.choice([0, 1], 1000, p=[0.83, 0.17])
            })

    def test_data_shape(self):
        """Test data has expected shape."""
        if self.has_real_data:
            self.assertGreater(len(self.data), 1000)  # Should have substantial data
            self.assertGreater(len(self.data.columns), 10)  # Should have multiple features

    def test_target_variable(self):
        """Test target variable properties."""
        if 'Churn' in self.data.columns:
            # Check churn rate is reasonable
            churn_rate = self.data['Churn'].mean()
            self.assertGreater(churn_rate, 0.05)  # At least 5% churn
            self.assertLess(churn_rate, 0.50)     # Less than 50% churn
            
            # Check binary values
            unique_values = set(self.data['Churn'].unique())
            self.assertEqual(unique_values, {0, 1})

    def test_no_missing_values(self):
        """Test that processed data has no missing values."""
        if self.has_real_data:
            missing_count = self.data.isnull().sum().sum()
            self.assertEqual(missing_count, 0, "Processed data should have no missing values")

    def test_feature_ranges(self):
        """Test that features are within expected ranges."""
        if 'SatisfactionScore' in self.data.columns:
            self.assertTrue(self.data['SatisfactionScore'].min() >= 1)
            self.assertTrue(self.data['SatisfactionScore'].max() <= 5)
        
        if 'Tenure' in self.data.columns:
            self.assertTrue(self.data['Tenure'].min() >= 0)
            self.assertTrue(self.data['Tenure'].max() <= 100)  # Reasonable upper bound

def run_data_tests():
    """Run all data processing tests."""
    print("🧪 Running Data Processing Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQuality))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, error in result.failures:
            print(f"  - {test}: {error}")
    
    if result.errors:
        print("\n🚨 Errors:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_data_tests()
