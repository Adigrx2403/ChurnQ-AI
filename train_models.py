"""
Training script for customer churn prediction models.
Run this script to train and save all models.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config.config import *
from src.data_processor import DataProcessor
from src.model_developer import ChurnModelDeveloper
from src.explainability import ModelExplainer
from src.advanced_analytics import AdvancedAnalytics
from src.business_intelligence import BusinessIntelligence

def main():
    """Main training pipeline."""
    print("🚀 Starting Customer Churn Analysis Training Pipeline")
    print("=" * 60)
    
    # Create config object
    class Config:
        def __init__(self):
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
    
    config = Config()
    
    # Step 1: Data Processing
    print("\n1️⃣ Data Processing")
    print("-" * 30)
    
    processor = DataProcessor(config)
    
    # Load and process data
    raw_data_path = PROJECT_ROOT / RAW_DATA_FILE
    if not raw_data_path.exists():
        print(f"❌ Raw data file not found: {raw_data_path}")
        return
    
    df, summary, outlier_info = processor.process_pipeline(raw_data_path)
    
    print(f"✅ Data processed successfully!")
    print(f"   - Shape: {df.shape}")
    print(f"   - Churn rate: {df['Churn'].mean():.1%}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    
    # Step 2: Prepare features and split data
    print("\n2️⃣ Feature Preparation")
    print("-" * 30)
    
    X, y = processor.prepare_features(df, target_column='Churn', fit=True)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    print(f"✅ Data split completed!")
    print(f"   - Training set: {X_train.shape[0]} samples")
    print(f"   - Validation set: {X_val.shape[0]} samples")
    print(f"   - Test set: {X_test.shape[0]} samples")
    print(f"   - Features: {X_train.shape[1]}")
    
    # Save preprocessing objects
    processor.save_preprocessing_objects(MODELS_DIR)
    
    # Step 3: Model Training
    print("\n3️⃣ Model Training")
    print("-" * 30)
    
    model_developer = ChurnModelDeveloper(config)
    
    # Train all models (with hyperparameter tuning if requested)
    tune_hyperparameters = True  # Set to False for faster training
    models = model_developer.train_all_models(X_train, y_train, tune_hyperparameters)
    
    print(f"✅ Models trained successfully!")
    print(f"   - Models trained: {list(models.keys())}")
    
    # Step 4: Model Evaluation
    print("\n4️⃣ Model Evaluation")
    print("-" * 30)
    
    results_summary = model_developer.evaluate_all_models(X_test, y_test)
    print("Model Performance Summary:")
    print(results_summary.round(3).to_string(index=False))
    
    # Cross-validation
    cv_results = model_developer.cross_validate_models(X_train, y_train)
    print("\nCross-Validation Results:")
    for model_name, metrics in cv_results.items():
        print(f"  {model_name}:")
        print(f"    ROC-AUC: {metrics['roc_auc_mean']:.3f} ± {metrics['roc_auc_std']:.3f}")
        print(f"    F1-Score: {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}")
    
    # Save models
    model_developer.save_models(MODELS_DIR)
    
    # Step 5: Advanced Analytics (Optional)
    print("\n5️⃣ Advanced Analytics")
    print("-" * 30)
    
    try:
        advanced_analytics = AdvancedAnalytics()
        
        # Survival analysis
        survival_df = advanced_analytics.prepare_survival_data(df)
        if survival_df is not None:
            cox_model = advanced_analytics.fit_cox_model(survival_df)
            if cox_model is not None:
                hazard_ratios = advanced_analytics.calculate_hazard_ratios()
                print("✅ Survival analysis completed!")
                if hazard_ratios is not None:
                    print(f"   - Top risk factors identified: {len(hazard_ratios)}")
        
    except Exception as e:
        print(f"⚠️ Advanced analytics skipped: {e}")
    
    # Step 6: Business Intelligence
    print("\n6️⃣ Business Intelligence")
    print("-" * 30)
    
    business_intel = BusinessIntelligence(config)
    
    # Calculate business metrics
    best_model = model_developer.best_model
    if best_model is not None:
        test_predictions = best_model.predict(X_test)
        metrics = business_intel.calculate_business_metrics(
            df.iloc[X_test.index], test_predictions
        )
        
        print("✅ Business metrics calculated!")
        print(f"   - Total customers: {metrics['total_customers']:,}")
        print(f"   - Churn rate: {metrics['churn_rate']:.1%}")
        print(f"   - Revenue at risk: ${metrics['revenue_at_risk']:,.0f}")
        print(f"   - Model precision: {metrics.get('model_precision', 0):.1%}")
        print(f"   - Model recall: {metrics.get('model_recall', 0):.1%}")
    
    # Step 7: Create sample predictions for demo
    print("\n7️⃣ Sample Predictions")
    print("-" * 30)
    
    if best_model is not None:
        # Create sample predictions file
        sample_customers = X_test.head(100).copy()
        sample_predictions = best_model.predict(sample_customers)
        sample_probabilities = best_model.predict_proba(sample_customers)[:, 1]
        
        sample_results = pd.DataFrame({
            'customer_index': sample_customers.index,
            'churn_prediction': sample_predictions,
            'churn_probability': sample_probabilities
        })
        
        # Save sample results
        sample_file = PROJECT_ROOT / 'data' / 'sample_predictions.csv'
        sample_results.to_csv(sample_file, index=False)
        
        print(f"✅ Sample predictions created!")
        print(f"   - File saved: {sample_file}")
        print(f"   - High-risk customers: {(sample_predictions == 1).sum()}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📋 Summary:")
    print(f"   ✅ Data processed: {df.shape[0]:,} customers")
    print(f"   ✅ Models trained: {len(models)} models")
    print(f"   ✅ Best model: {model_developer.best_model_name}")
    try:
        # Try to get the best model's ROC-AUC score
        best_model_formatted = model_developer.best_model_name.replace('_', ' ').title()
        best_model_mask = results_summary['Model'].str.lower() == best_model_formatted.lower()
        if best_model_mask.any():
            best_roc = results_summary.loc[best_model_mask, 'ROC-AUC'].iloc[0]
            print(f"   ✅ Best ROC-AUC: {best_roc:.3f}")
    except Exception as e:
        print(f"   ℹ️ ROC-AUC score available in model evaluation section above")
    print(f"   ✅ Files saved in: {MODELS_DIR}")
    
    print("\n🚀 Next Steps:")
    print("   1. Run the Streamlit app: streamlit run app.py")
    print("   2. Explore the interactive dashboard")
    print("   3. Use the prediction tool for new customers")
    print("   4. Review business intelligence insights")
    
    print("\n💡 Tip: Check the README.md for detailed usage instructions!")

if __name__ == "__main__":
    main()
