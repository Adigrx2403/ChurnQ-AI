"""
Model development and evaluation module for customer churn analysis.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ChurnModelDeveloper:
    """Class for developing and evaluating churn prediction models."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def create_baseline_model(self, X_train, y_train):
        """Create a baseline model using majority class strategy."""
        baseline_model = DummyClassifier(strategy='most_frequent', random_state=self.config.RANDOM_STATE)
        baseline_model.fit(X_train, y_train)
        self.models['baseline'] = baseline_model
        return baseline_model
    
    def create_logistic_regression(self, X_train, y_train, tune_hyperparameters=True):
        """Create and train logistic regression model."""
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
            
            lr = LogisticRegression(random_state=self.config.RANDOM_STATE, class_weight='balanced')
            grid_search = GridSearchCV(
                lr, param_grid, cv=5, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_lr = grid_search.best_estimator_
            print(f"Best Logistic Regression parameters: {grid_search.best_params_}")
        else:
            best_lr = LogisticRegression(
                random_state=self.config.RANDOM_STATE, 
                class_weight='balanced',
                max_iter=1000
            )
            best_lr.fit(X_train, y_train)
        
        self.models['logistic_regression'] = best_lr
        return best_lr
    
    def create_random_forest(self, X_train, y_train, tune_hyperparameters=True):
        """Create and train random forest model."""
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            rf = RandomForestClassifier(random_state=self.config.RANDOM_STATE)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_rf = grid_search.best_estimator_
            print(f"Best Random Forest parameters: {grid_search.best_params_}")
        else:
            best_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=self.config.RANDOM_STATE,
                class_weight='balanced'
            )
            best_rf.fit(X_train, y_train)
        
        self.models['random_forest'] = best_rf
        return best_rf
    
    def create_xgboost(self, X_train, y_train, tune_hyperparameters=True):
        """Create and train XGBoost model."""
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(
                random_state=self.config.RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_xgb = grid_search.best_estimator_
            print(f"Best XGBoost parameters: {grid_search.best_params_}")
        else:
            best_xgb = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            )
            best_xgb.fit(X_train, y_train)
        
        self.models['xgboost'] = best_xgb
        return best_xgb
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model and return metrics."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.model_results[model_name] = metrics
        return metrics
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models."""
        results_summary = pd.DataFrame()
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Add to summary
            results_summary = pd.concat([
                results_summary,
                pd.DataFrame({
                    'Model': [model_name],
                    'Accuracy': [metrics['accuracy']],
                    'Precision': [metrics['precision']],
                    'Recall': [metrics['recall']],
                    'F1-Score': [metrics['f1_score']],
                    'ROC-AUC': [metrics['roc_auc']]
                })
            ], ignore_index=True)
        
        # Find best model based on ROC-AUC
        if 'ROC-AUC' in results_summary.columns:
            best_idx = results_summary['ROC-AUC'].idxmax()
            self.best_model_name = results_summary.loc[best_idx, 'Model']
            self.best_model = self.models[self.best_model_name]
            print(f"Best model: {self.best_model_name} with ROC-AUC: {results_summary.loc[best_idx, 'ROC-AUC']:.4f}")
        
        return results_summary
    
    def plot_model_comparison(self, results_summary):
        """Plot model comparison charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = ax.bar(results_summary['Model'], results_summary[metric])
            ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, X_test, y_test):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                
                plt.plot(recall, precision, label=f'{model_name}', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_confusion_matrices(self, X_test, y_test):
        """Plot confusion matrices for all models."""
        n_models = len(self.models)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            ax = axes[i] if n_models > 1 else axes[0]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide extra subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from tree-based models."""
        importance_df = pd.DataFrame({'Feature': feature_names})
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df[f'{model_name}_importance'] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For logistic regression, use absolute coefficients
                importance_df[f'{model_name}_importance'] = np.abs(model.coef_[0])
        
        return importance_df
    
    def cross_validate_models(self, X_train, y_train, cv=5):
        """Perform cross-validation on all models."""
        cv_results = {}
        
        for model_name, model in self.models.items():
            if model_name != 'baseline':  # Skip baseline for CV
                print(f"Cross-validating {model_name}...")
                
                # ROC-AUC scores
                roc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
                
                # F1 scores
                f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
                
                cv_results[model_name] = {
                    'roc_auc_mean': roc_scores.mean(),
                    'roc_auc_std': roc_scores.std(),
                    'f1_mean': f1_scores.mean(),
                    'f1_std': f1_scores.std()
                }
        
        return cv_results
    
    def save_models(self, models_dir):
        """Save all trained models."""
        models_dir = Path(models_dir)
        models_dir.mkdir(exist_ok=True)
        
        # Save individual models
        if 'logistic_regression' in self.models:
            joblib.dump(self.models['logistic_regression'], 
                       models_dir / self.config.LOGISTIC_MODEL_FILE)
        
        if 'random_forest' in self.models:
            joblib.dump(self.models['random_forest'], 
                       models_dir / self.config.RF_MODEL_FILE)
        
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], 
                       models_dir / self.config.XGBOOST_MODEL_FILE)
        
        # Save model results
        joblib.dump(self.model_results, models_dir / "model_results.joblib")
        
        print("Models saved successfully!")
    
    def load_models(self, models_dir):
        """Load saved models."""
        models_dir = Path(models_dir)
        
        # Load individual models
        if (models_dir / self.config.LOGISTIC_MODEL_FILE).exists():
            self.models['logistic_regression'] = joblib.load(
                models_dir / self.config.LOGISTIC_MODEL_FILE
            )
        
        if (models_dir / self.config.RF_MODEL_FILE).exists():
            self.models['random_forest'] = joblib.load(
                models_dir / self.config.RF_MODEL_FILE
            )
        
        if (models_dir / self.config.XGBOOST_MODEL_FILE).exists():
            self.models['xgboost'] = joblib.load(
                models_dir / self.config.XGBOOST_MODEL_FILE
            )
        
        # Load model results
        if (models_dir / "model_results.joblib").exists():
            self.model_results = joblib.load(models_dir / "model_results.joblib")
        
        print("Models loaded successfully!")
    
    def train_all_models(self, X_train, y_train, tune_hyperparameters=False):
        """Train all models in the pipeline."""
        print("Training all models...")
        
        # Create and train baseline
        self.create_baseline_model(X_train, y_train)
        
        # Create and train logistic regression
        self.create_logistic_regression(X_train, y_train, tune_hyperparameters)
        
        # Create and train random forest
        self.create_random_forest(X_train, y_train, tune_hyperparameters)
        
        # Create and train XGBoost
        self.create_xgboost(X_train, y_train, tune_hyperparameters)
        
        print("All models trained successfully!")
        return self.models
