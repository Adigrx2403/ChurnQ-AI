"""
Data processing and feature engineering module for customer churn analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Class for data processing and feature engineering."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load data from Excel file."""
        try:
            # Load the E Comm sheet
            df = pd.read_excel(filepath, sheet_name='E Comm')
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def get_data_summary(self, df):
        """Get comprehensive data summary."""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'churn_distribution': df['Churn'].value_counts().to_dict() if 'Churn' in df.columns else None,
            'churn_rate': df['Churn'].mean() if 'Churn' in df.columns else None
        }
        return summary
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        df_processed = df.copy()
        
        # Strategy for different columns
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # For numeric columns, fill with median
        for col in numeric_columns:
            if col != 'CustomerID' and df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                print(f"Filled {col} missing values with median: {median_val}")
        
        # For categorical columns, fill with mode
        for col in categorical_columns:
            if df_processed[col].isnull().sum() > 0:
                mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
                df_processed[col].fillna(mode_val, inplace=True)
                print(f"Filled {col} missing values with mode: {mode_val}")
        
        return df_processed
    
    def handle_outliers(self, df, columns=None):
        """Handle outliers using IQR method."""
        df_processed = df.copy()
        
        if columns is None:
            # Select numeric columns excluding ID and target
            columns = df_processed.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col not in ['CustomerID', 'Churn']]
        
        outlier_info = {}
        
        for col in columns:
            if col in df_processed.columns:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers_count = ((df_processed[col] < lower_bound) | 
                                (df_processed[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    # Cap outliers instead of removing them
                    df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
                    outlier_info[col] = {
                        'count': outliers_count,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    print(f"Capped {outliers_count} outliers in {col}")
        
        return df_processed, outlier_info
    
    def engineer_features(self, df):
        """Engineer new features from existing ones."""
        df_engineered = df.copy()
        
        # 1. Tenure buckets
        if 'Tenure' in df_engineered.columns:
            df_engineered['TenureBucket'] = pd.cut(
                df_engineered['Tenure'], 
                bins=self.config.TENURE_BUCKETS,
                labels=self.config.TENURE_LABELS,
                include_lowest=True
            )
        
        # 2. Customer engagement score
        engagement_features = ['HourSpendOnApp', 'OrderCount', 'DaySinceLastOrder']
        if all(col in df_engineered.columns for col in engagement_features):
            # Normalize features for engagement score
            df_engineered['EngagementScore'] = (
                (df_engineered['HourSpendOnApp'] / df_engineered['HourSpendOnApp'].max()) * 0.4 +
                (df_engineered['OrderCount'] / df_engineered['OrderCount'].max()) * 0.4 +
                (1 - df_engineered['DaySinceLastOrder'] / df_engineered['DaySinceLastOrder'].max()) * 0.2
            )
        
        # 3. Satisfaction bucket
        if 'SatisfactionScore' in df_engineered.columns:
            df_engineered['SatisfactionBucket'] = pd.cut(
                df_engineered['SatisfactionScore'],
                bins=self.config.SATISFACTION_BUCKETS,
                labels=self.config.SATISFACTION_LABELS,
                include_lowest=True
            )
        
        # 4. Customer value score
        value_features = ['OrderAmountHikeFromlastYear', 'CashbackAmount']
        if all(col in df_engineered.columns for col in value_features):
            df_engineered['CustomerValueScore'] = (
                df_engineered['OrderAmountHikeFromlastYear'] * 0.6 +
                df_engineered['CashbackAmount'] * 0.4
            )
        
        # 5. Binary features
        if 'Complain' in df_engineered.columns:
            df_engineered['HasComplaint'] = (df_engineered['Complain'] > 0).astype(int)
        
        if 'CouponUsed' in df_engineered.columns:
            df_engineered['IsCouponUser'] = (df_engineered['CouponUsed'] > 0).astype(int)
        
        # 6. Order frequency
        if all(col in df_engineered.columns for col in ['OrderCount', 'Tenure']):
            df_engineered['OrderFrequency'] = df_engineered['OrderCount'] / (df_engineered['Tenure'] + 1)
        
        # 7. Device diversity
        if 'NumberOfDeviceRegistered' in df_engineered.columns:
            df_engineered['IsMultiDevice'] = (df_engineered['NumberOfDeviceRegistered'] > 1).astype(int)
        
        # 8. Address diversity
        if 'NumberOfAddress' in df_engineered.columns:
            df_engineered['IsMultiAddress'] = (df_engineered['NumberOfAddress'] > 1).astype(int)
        
        print(f"Feature engineering completed. New shape: {df_engineered.shape}")
        return df_engineered
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features."""
        df_encoded = df.copy()
        
        categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns
        categorical_columns = [col for col in categorical_columns if col != 'CustomerID']
        
        for col in categorical_columns:
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df_encoded[col].astype(str))
                    known_values = set(self.label_encoders[col].classes_)
                    unseen_values = unique_values - known_values
                    
                    if unseen_values:
                        print(f"Warning: Unseen values in {col}: {unseen_values}")
                        # Map unseen values to most frequent class
                        most_frequent_class = self.label_encoders[col].classes_[0]
                        df_encoded[col] = df_encoded[col].astype(str).replace(
                            list(unseen_values), most_frequent_class
                        )
                    
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def prepare_features(self, df, target_column='Churn', fit=True):
        """Prepare features for modeling."""
        # Remove ID column and prepare features
        feature_columns = [col for col in df.columns if col not in ['CustomerID', target_column]]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy() if target_column in df.columns else None
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        if fit and len(numerical_columns) > 0:
            X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
            self.feature_columns = feature_columns
        elif len(numerical_columns) > 0:
            X[numerical_columns] = self.scaler.transform(X[numerical_columns])
        
        return X, y
    
    def split_data(self, X, y, test_size=None, validation_size=None, random_state=None):
        """Split data into train, validation, and test sets."""
        test_size = test_size or self.config.TEST_SIZE
        validation_size = validation_size or self.config.VALIDATION_SIZE
        random_state = random_state or self.config.RANDOM_STATE
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessing_objects(self, models_dir):
        """Save preprocessing objects."""
        models_dir = Path(models_dir)
        models_dir.mkdir(exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, models_dir / self.config.SCALER_FILE)
        
        # Save label encoders
        joblib.dump(self.label_encoders, models_dir / self.config.ENCODER_FILE)
        
        # Save feature columns
        joblib.dump(self.feature_columns, models_dir / self.config.FEATURE_COLUMNS_FILE)
        
        print("Preprocessing objects saved successfully!")
    
    def load_preprocessing_objects(self, models_dir):
        """Load preprocessing objects."""
        models_dir = Path(models_dir)
        
        # Load scaler
        self.scaler = joblib.load(models_dir / self.config.SCALER_FILE)
        
        # Load label encoders
        self.label_encoders = joblib.load(models_dir / self.config.ENCODER_FILE)
        
        # Load feature columns
        self.feature_columns = joblib.load(models_dir / self.config.FEATURE_COLUMNS_FILE)
        
        print("Preprocessing objects loaded successfully!")
    
    def process_pipeline(self, filepath, save_processed=True):
        """Complete data processing pipeline."""
        print("Starting data processing pipeline...")
        
        # Load data
        df = self.load_data(filepath)
        
        # Get summary
        summary = self.get_data_summary(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Handle outliers
        df, outlier_info = self.handle_outliers(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Save processed data
        if save_processed:
            output_path = Path(filepath).parent / self.config.PROCESSED_DATA_FILE
            df.to_csv(output_path, index=False)
            print(f"Processed data saved to: {output_path}")
        
        return df, summary, outlier_info
