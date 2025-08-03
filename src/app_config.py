"""
Configuration object for the Streamlit app to avoid caching issues.
"""

class AppConfig:
    """Configuration class for the Streamlit application."""
    
    def __init__(self):
        # Import constants from config module
        from config.config import (
            RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
            TENURE_BUCKETS, TENURE_LABELS,
            SATISFACTION_BUCKETS, SATISFACTION_LABELS,
            CUSTOMER_LIFETIME_VALUE, RETENTION_COST_RATIO, ACQUISITION_COST,
            LOGISTIC_MODEL_FILE, RF_MODEL_FILE, XGBOOST_MODEL_FILE,
            SCALER_FILE, ENCODER_FILE, FEATURE_COLUMNS_FILE,
            PROCESSED_DATA_FILE
        )
        
        self.RANDOM_STATE = RANDOM_STATE
        self.TEST_SIZE = TEST_SIZE
        self.VALIDATION_SIZE = VALIDATION_SIZE
        self.TENURE_BUCKETS = TENURE_BUCKETS
        self.TENURE_LABELS = TENURE_LABELS
        self.SATISFACTION_BUCKETS = SATISFACTION_BUCKETS
        self.SATISFACTION_LABELS = SATISFACTION_LABELS
        self.CUSTOMER_LIFETIME_VALUE = CUSTOMER_LIFETIME_VALUE
        self.RETENTION_COST_RATIO = RETENTION_COST_RATIO
        self.ACQUISITION_COST = ACQUISITION_COST
        self.LOGISTIC_MODEL_FILE = LOGISTIC_MODEL_FILE
        self.RF_MODEL_FILE = RF_MODEL_FILE
        self.XGBOOST_MODEL_FILE = XGBOOST_MODEL_FILE
        self.SCALER_FILE = SCALER_FILE
        self.ENCODER_FILE = ENCODER_FILE
        self.FEATURE_COLUMNS_FILE = FEATURE_COLUMNS_FILE
        self.PROCESSED_DATA_FILE = PROCESSED_DATA_FILE
