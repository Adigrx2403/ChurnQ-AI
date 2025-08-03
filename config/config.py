# Customer Churn Analysis - Configuration
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Data files
RAW_DATA_FILE = "E Commerce Dataset.xlsx"
PROCESSED_DATA_FILE = "processed_ecommerce_data.csv"
FEATURE_ENGINEERED_FILE = "feature_engineered_data.csv"

# Model files
LOGISTIC_MODEL_FILE = "logistic_regression_model.joblib"
RF_MODEL_FILE = "random_forest_model.joblib"
XGBOOST_MODEL_FILE = "xgboost_model.joblib"
SCALER_FILE = "feature_scaler.joblib"
ENCODER_FILE = "label_encoders.joblib"
FEATURE_COLUMNS_FILE = "feature_columns.joblib"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering parameters
TENURE_BUCKETS = [0, 6, 12, 24, 36, float('inf')]
TENURE_LABELS = ['0-6M', '6-12M', '12-24M', '24-36M', '36M+']

SATISFACTION_BUCKETS = [0, 2, 3, 4, 5]
SATISFACTION_LABELS = ['Very Low', 'Low', 'Medium', 'High']

# Business metrics
CUSTOMER_LIFETIME_VALUE = 500  # Average CLV in dollars
RETENTION_COST_RATIO = 0.2     # Cost of retention vs CLV
ACQUISITION_COST = 100         # Customer acquisition cost

# Streamlit configuration
PAGE_TITLE = "E-Commerce Customer Churn Analysis"
PAGE_ICON = "📊"
LAYOUT = "wide"

# Colors for visualizations
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'churn': '#d62728',
    'no_churn': '#2ca02c'
}
