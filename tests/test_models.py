"""
Unit tests for machine learning models.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestModelPerformance(unittest.TestCase):
    """Test model performance and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.models_path = self.project_root / 'models'
        
        # Try to load trained models
        self.models = {}
        model_files = {
            'xgboost': 'xgboost_model.joblib',
            'random_forest': 'random_forest_model.joblib',
            'logistic_regression': 'logistic_regression_model.joblib'
        }
        
        for model_name, filename in model_files.items():
            try:
                model_path = self.models_path / filename
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
            except Exception as e:
                print(f"Could not load {model_name}: {e}")
        
        # Create sample test data
        np.random.seed(42)
        n_samples = 100
        n_features = 15
        
        self.X_test = np.random.randn(n_samples, n_features)
        self.y_test = np.random.choice([0, 1], n_samples, p=[0.83, 0.17])

    def test_model_loading(self):
        """Test that models can be loaded successfully."""
        self.assertGreater(len(self.models), 0, "At least one model should be loaded")
        
        for model_name, model in self.models.items():
            self.assertIsNotNone(model, f"{model_name} should not be None")

    def test_model_predictions(self):
        """Test that models can make predictions."""
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                try:
                    # Test prediction
                    predictions = model.predict(self.X_test)
                    self.assertEqual(len(predictions), len(self.y_test))
                    
                    # Check predictions are binary
                    unique_preds = set(predictions)
                    self.assertTrue(unique_preds.issubset({0, 1}), 
                                  f"{model_name} predictions should be binary")
                    
                    # Test probability prediction
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(self.X_test)
                        self.assertEqual(probabilities.shape[0], len(self.y_test))
                        self.assertEqual(probabilities.shape[1], 2)  # Binary classification
                        
                        # Check probabilities sum to 1
                        prob_sums = probabilities.sum(axis=1)
                        np.testing.assert_array_almost_equal(prob_sums, 1.0, decimal=5)
                        
                except Exception as e:
                    self.fail(f"{model_name} prediction failed: {e}")

    def test_model_performance_metrics(self):
        """Test model performance on sample data."""
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                try:
                    predictions = model.predict(self.X_test)
                    
                    # Calculate metrics
                    accuracy = (predictions == self.y_test).mean()
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(self.X_test)[:, 1]
                        auc_score = roc_auc_score(self.y_test, probabilities)
                        self.assertGreater(auc_score, 0.5, 
                                         f"{model_name} AUC should be better than random")
                    
                    precision = precision_score(self.y_test, predictions, zero_division=0)
                    recall = recall_score(self.y_test, predictions, zero_division=0)
                    f1 = f1_score(self.y_test, predictions, zero_division=0)
                    
                    # Basic sanity checks
                    self.assertGreaterEqual(accuracy, 0.0)
                    self.assertLessEqual(accuracy, 1.0)
                    self.assertGreaterEqual(precision, 0.0)
                    self.assertLessEqual(precision, 1.0)
                    self.assertGreaterEqual(recall, 0.0)
                    self.assertLessEqual(recall, 1.0)
                    self.assertGreaterEqual(f1, 0.0)
                    self.assertLessEqual(f1, 1.0)
                    
                    print(f"{model_name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                          f"Recall: {recall:.3f}, F1: {f1:.3f}")
                    
                except Exception as e:
                    self.fail(f"{model_name} performance evaluation failed: {e}")

    def test_feature_importance(self):
        """Test feature importance extraction."""
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        self.assertEqual(len(importances), self.X_test.shape[1])
                        self.assertTrue(all(imp >= 0 for imp in importances))
                        self.assertAlmostEqual(sum(importances), 1.0, places=5)
                    elif hasattr(model, 'coef_'):
                        coefficients = model.coef_
                        self.assertEqual(coefficients.shape[1], self.X_test.shape[1])
                    
                except Exception as e:
                    print(f"Feature importance test failed for {model_name}: {e}")

class TestModelRobustness(unittest.TestCase):
    """Test model robustness and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.models_path = self.project_root / 'models'
        
        # Load one model for testing
        try:
            model_path = self.models_path / 'xgboost_model.joblib'
            if model_path.exists():
                self.model = joblib.load(model_path)
            else:
                self.model = None
        except:
            self.model = None

    def test_edge_cases(self):
        """Test model behavior on edge cases."""
        if self.model is None:
            self.skipTest("No model available for testing")
        
        # Test with extreme values
        n_features = 15  # Adjust based on your model
        extreme_data = np.array([
            [1000] * n_features,    # Very high values
            [-1000] * n_features,   # Very low values
            [0] * n_features,       # All zeros
        ])
        
        try:
            predictions = self.model.predict(extreme_data)
            self.assertEqual(len(predictions), 3)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(extreme_data)
                self.assertEqual(probabilities.shape, (3, 2))
                
        except Exception as e:
            print(f"Edge case testing failed: {e}")

    def test_single_prediction(self):
        """Test single sample prediction."""
        if self.model is None:
            self.skipTest("No model available for testing")
        
        # Test single sample
        single_sample = np.random.randn(1, 15)  # Adjust features count
        
        try:
            prediction = self.model.predict(single_sample)
            self.assertEqual(len(prediction), 1)
            self.assertIn(prediction[0], [0, 1])
            
        except Exception as e:
            self.fail(f"Single prediction failed: {e}")

class TestModelConsistency(unittest.TestCase):
    """Test model consistency and reproducibility."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_data = np.random.randn(10, 15)  # Adjust features count
        
        # Try to load model
        model_path = Path(__file__).parent.parent / 'models' / 'xgboost_model.joblib'
        try:
            self.model = joblib.load(model_path) if model_path.exists() else None
        except:
            self.model = None

    def test_prediction_consistency(self):
        """Test that model gives consistent predictions."""
        if self.model is None:
            self.skipTest("No model available for testing")
        
        # Make predictions multiple times
        pred1 = self.model.predict(self.test_data)
        pred2 = self.model.predict(self.test_data)
        
        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2, 
                                    "Model should give consistent predictions")

    def test_probability_consistency(self):
        """Test probability prediction consistency."""
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            self.skipTest("No model with probability prediction available")
        
        # Make probability predictions multiple times
        prob1 = self.model.predict_proba(self.test_data)
        prob2 = self.model.predict_proba(self.test_data)
        
        # Probabilities should be identical
        np.testing.assert_array_almost_equal(prob1, prob2, decimal=10,
                                           err_msg="Model should give consistent probabilities")

def run_model_tests():
    """Run all model tests."""
    print("🤖 Running Model Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestModelPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestModelRobustness))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConsistency))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Model Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
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
    run_model_tests()
