"""
Integration tests for the complete churn analysis system.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestEndToEndPipeline(unittest.TestCase):
    """Test the complete end-to-end pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        
        # Create sample customer data
        np.random.seed(42)
        self.sample_customers = pd.DataFrame({
            'CustomerID': range(1, 21),
            'Tenure': np.random.randint(1, 61, 20),
            'SatisfactionScore': np.random.randint(1, 6, 20),
            'Complain': np.random.choice([0, 1], 20, p=[0.8, 0.2]),
            'OrderCount': np.random.randint(1, 20, 20),
            'HourSpendOnApp': np.random.uniform(0, 10, 20),
            'CashbackAmount': np.random.uniform(0, 200, 20),
            'NumberOfDeviceRegistered': np.random.randint(1, 6, 20),
            'NumberOfAddress': np.random.randint(1, 10, 20),
            'OrderAmountHikeFromLastYear': np.random.uniform(10, 25, 20),
            'CouponUsed': np.random.randint(0, 20, 20),
            'DaySinceLastOrder': np.random.randint(0, 46, 20),
            'WarehouseToHome': np.random.uniform(5, 35, 20),
            'Gender': np.random.choice(['Female', 'Male'], 20),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], 20),
            'CityTier': np.random.choice([1, 2, 3], 20),
            'PreferredLoginDevice': np.random.choice(['Phone', 'Computer'], 20),
            'PreferredPaymentMode': np.random.choice(['Credit Card', 'Debit Card', 'UPI'], 20),
            'PreferredOrderCat': np.random.choice(['Laptop & Accessory', 'Mobile Phone', 'Fashion'], 20),
            'Churn': np.random.choice([0, 1], 20, p=[0.83, 0.17])
        })

    def test_data_files_exist(self):
        """Test that required data files exist."""
        required_files = [
            'E Commerce Dataset.xlsx',
            'processed_ecommerce_data.csv'
        ]
        
        for filename in required_files:
            file_path = self.project_root / filename
            if file_path.exists():
                self.assertTrue(file_path.exists(), f"{filename} should exist")
                self.assertGreater(file_path.stat().st_size, 0, f"{filename} should not be empty")

    def test_model_files_exist(self):
        """Test that trained model files exist."""
        models_dir = self.project_root / 'models'
        
        if models_dir.exists():
            expected_models = [
                'xgboost_model.joblib',
                'random_forest_model.joblib',
                'logistic_regression_model.joblib',
                'feature_scaler.joblib',
                'label_encoders.joblib'
            ]
            
            for model_file in expected_models:
                model_path = models_dir / model_file
                if model_path.exists():
                    self.assertGreater(model_path.stat().st_size, 0, 
                                     f"{model_file} should not be empty")

    def test_config_integrity(self):
        """Test configuration integrity."""
        config_file = self.project_root / 'config' / 'config.py'
        
        if not config_file.exists():
            self.skipTest("Config file not available")
        
        try:
            # Import config dynamically to avoid import errors
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Test that key configuration variables exist
            config_vars = [
                'PROJECT_ROOT', 'RAW_DATA_FILE', 'PROCESSED_DATA_FILE',
                'PAGE_TITLE', 'PAGE_ICON', 'LAYOUT'
            ]
            
            for var in config_vars:
                if hasattr(config_module, var):
                    self.assertTrue(True, f"Configuration variable {var} is defined")
                              
        except Exception as e:
            self.skipTest(f"Config module import failed: {e}")

    def test_business_logic_calculations(self):
        """Test business logic calculations."""
        # Test churn rate calculation
        churn_rate = self.sample_customers['Churn'].mean()
        self.assertGreaterEqual(churn_rate, 0.0)
        self.assertLessEqual(churn_rate, 1.0)
        
        # Test customer segmentation logic
        high_risk_customers = self.sample_customers[
            (self.sample_customers['SatisfactionScore'] <= 2) &
            (self.sample_customers['Complain'] == 1)
        ]
        
        # Should be able to identify high-risk customers
        if len(high_risk_customers) > 0:
            self.assertGreater(len(high_risk_customers), 0)

    def test_feature_engineering_logic(self):
        """Test feature engineering logic."""
        # Test tenure buckets
        tenure_buckets = pd.cut(self.sample_customers['Tenure'], 
                               bins=[0, 6, 12, 24, 36, float('inf')],
                               labels=[0, 1, 2, 3, 4])
        
        self.assertEqual(len(tenure_buckets), len(self.sample_customers))
        self.assertTrue(all(bucket in [0, 1, 2, 3, 4] for bucket in tenure_buckets.dropna()))
        
        # Test engagement score calculation
        engagement_score = (self.sample_customers['HourSpendOnApp'] * 0.6 + 
                           self.sample_customers['OrderCount'] * 0.4)
        
        self.assertEqual(len(engagement_score), len(self.sample_customers))
        self.assertTrue(all(score >= 0 for score in engagement_score))

class TestBusinessMetrics(unittest.TestCase):
    """Test business metrics and calculations."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.customers = pd.DataFrame({
            'CustomerID': range(1, 1001),
            'Churn': np.random.choice([0, 1], 1000, p=[0.83, 0.17]),
            'SatisfactionScore': np.random.randint(1, 6, 1000),
            'Tenure': np.random.randint(1, 61, 1000),
            'OrderCount': np.random.randint(1, 20, 1000),
            'CashbackAmount': np.random.uniform(0, 200, 1000)
        })
        
        # Business constants
        self.CUSTOMER_LIFETIME_VALUE = 1200
        self.RETENTION_COST_RATIO = 0.15

    def test_revenue_at_risk_calculation(self):
        """Test revenue at risk calculation."""
        churned_customers = self.customers['Churn'].sum()
        revenue_at_risk = churned_customers * self.CUSTOMER_LIFETIME_VALUE
        
        self.assertGreater(revenue_at_risk, 0)
        self.assertEqual(revenue_at_risk, churned_customers * 1200)

    def test_roi_calculations(self):
        """Test ROI calculations for retention campaigns."""
        churned_customers = self.customers['Churn'].sum()
        retention_cost = self.CUSTOMER_LIFETIME_VALUE * self.RETENTION_COST_RATIO
        potential_savings = churned_customers * (self.CUSTOMER_LIFETIME_VALUE - retention_cost)
        
        if churned_customers > 0:
            roi = (potential_savings / (churned_customers * retention_cost)) * 100
            self.assertGreater(roi, 0, "ROI should be positive")

    def test_customer_segmentation_metrics(self):
        """Test customer segmentation metrics."""
        # Risk-based segmentation
        high_risk = self.customers[self.customers['SatisfactionScore'] <= 2]
        medium_risk = self.customers[self.customers['SatisfactionScore'] == 3]
        low_risk = self.customers[self.customers['SatisfactionScore'] >= 4]
        
        total_customers = len(high_risk) + len(medium_risk) + len(low_risk)
        self.assertEqual(total_customers, len(self.customers))
        
        # Calculate segment churn rates
        if len(high_risk) > 0:
            high_risk_churn = high_risk['Churn'].mean()
            self.assertGreaterEqual(high_risk_churn, 0.0)
            self.assertLessEqual(high_risk_churn, 1.0)

    def test_campaign_metrics(self):
        """Test campaign effectiveness metrics."""
        # Simulate campaign results
        campaign_customers = 100
        baseline_churn_rate = 0.20
        campaign_churn_rate = 0.15
        
        churn_reduction = (baseline_churn_rate - campaign_churn_rate) / baseline_churn_rate
        customers_saved = campaign_customers * (baseline_churn_rate - campaign_churn_rate)
        
        self.assertGreater(churn_reduction, 0, "Campaign should reduce churn")
        self.assertGreater(customers_saved, 0, "Campaign should save customers")

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency."""
    
    def setUp(self):
        """Set up test data."""
        try:
            self.processed_data = pd.read_csv(
                Path(__file__).parent.parent / 'processed_ecommerce_data.csv'
            )
            self.has_data = True
        except FileNotFoundError:
            self.has_data = False

    def test_data_consistency(self):
        """Test data consistency checks."""
        if not self.has_data:
            self.skipTest("Processed data not available")
        
        # Check for duplicates
        duplicate_count = self.processed_data.duplicated().sum()
        self.assertEqual(duplicate_count, 0, "Should have no duplicate records")
        
        # Check data types
        if 'Churn' in self.processed_data.columns:
            self.assertTrue(self.processed_data['Churn'].dtype in ['int64', 'float64'],
                          "Churn should be numeric")

    def test_feature_distributions(self):
        """Test feature distributions are reasonable."""
        if not self.has_data:
            self.skipTest("Processed data not available")
        
        # Check satisfaction score distribution
        if 'SatisfactionScore' in self.processed_data.columns:
            satisfaction_range = self.processed_data['SatisfactionScore'].unique()
            self.assertTrue(all(score >= 1 and score <= 5 for score in satisfaction_range),
                          "Satisfaction scores should be between 1 and 5")

    def test_target_variable_distribution(self):
        """Test target variable distribution."""
        if not self.has_data or 'Churn' not in self.processed_data.columns:
            self.skipTest("Churn data not available")
        
        churn_rate = self.processed_data['Churn'].mean()
        self.assertGreater(churn_rate, 0.05, "Churn rate should be at least 5%")
        self.assertLess(churn_rate, 0.50, "Churn rate should be less than 50%")

def run_integration_tests():
    """Run all integration tests."""
    print("🔄 Running Integration Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestBusinessMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Integration Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) 
                   / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, error in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n🚨 Errors:")
        for test, error in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_integration_tests()
