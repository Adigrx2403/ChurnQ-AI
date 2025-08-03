"""
Test configuration and shared fixtures for churn analysis tests.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

@pytest.fixture
def sample_customer_data():
    """Create sample customer data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Tenure': np.random.randint(1, 61, n_samples),
        'SatisfactionScore': np.random.randint(1, 6, n_samples),
        'Complain': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'OrderCount': np.random.randint(1, 20, n_samples),
        'HourSpendOnApp': np.random.uniform(0, 10, n_samples),
        'CashbackAmount': np.random.uniform(0, 200, n_samples),
        'NumberOfDeviceRegistered': np.random.randint(1, 6, n_samples),
        'NumberOfAddress': np.random.randint(1, 10, n_samples),
        'OrderAmountHikeFromLastYear': np.random.uniform(10, 25, n_samples),
        'CouponUsed': np.random.randint(0, 20, n_samples),
        'DaySinceLastOrder': np.random.randint(0, 46, n_samples),
        'WarehouseToHome': np.random.uniform(5, 35, n_samples),
        'Gender': np.random.choice(['Female', 'Male'], n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'CityTier': np.random.choice([1, 2, 3], n_samples),
        'PreferredLoginDevice': np.random.choice(['Phone', 'Computer'], n_samples),
        'PreferredPaymentMode': np.random.choice(['Credit Card', 'Debit Card', 'UPI'], n_samples),
        'PreferredOrderCat': np.random.choice(['Laptop & Accessory', 'Mobile Phone', 'Fashion'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.83, 0.17])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values to test preprocessing
    df.loc[df.sample(5).index, 'NumberOfAddress'] = np.nan
    df.loc[df.sample(3).index, 'HourSpendOnApp'] = np.nan
    
    return df

@pytest.fixture
def processed_data():
    """Sample processed data for model testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples, p=[0.83, 0.17])
    
    feature_names = [
        'Tenure', 'SatisfactionScore', 'Complain', 'OrderCount',
        'HourSpendOnApp', 'CashbackAmount', 'NumberOfDeviceRegistered',
        'TenureBucket', 'EngagementScore', 'ValueScore',
        'SatisfactionBucket', 'HasComplaint', 'HighAppUsage',
        'MultiDevice', 'RecentOrder'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='Churn')
    
    return X_df, y_series

@pytest.fixture
def model_config():
    """Sample model configuration."""
    return {
        'random_state': 42,
        'test_size': 0.2,
        'cv_folds': 5,
        'target_column': 'Churn'
    }
