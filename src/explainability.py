"""
Model explainability and interpretability module using SHAP and LIME.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """Class for model explainability and interpretability."""
    
    def __init__(self, model, X_train, feature_names, class_names=['No Churn', 'Churn']):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = class_names
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Initialize explainers
        self._initialize_shap()
        self._initialize_lime()
    
    def _initialize_shap(self):
        """Initialize SHAP explainer."""
        if not SHAP_AVAILABLE:
            return
            
        try:
            # Choose appropriate explainer based on model type
            model_type = type(self.model).__name__.lower()
            
            if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type:
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for other models (slower but more general)
                background_sample = shap.sample(self.X_train, min(100, len(self.X_train)))
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, background_sample
                )
            
            print("SHAP explainer initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _initialize_lime(self):
        """Initialize LIME explainer."""
        if not LIME_AVAILABLE:
            return
            
        try:
            self.lime_explainer = LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification',
                discretize_continuous=True
            )
            print("LIME explainer initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize LIME explainer: {e}")
            self.lime_explainer = None
    
    def get_shap_values(self, X_explain):
        """Get SHAP values for explanation."""
        if self.shap_explainer is None:
            return None
            
        try:
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(X_explain)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # For binary classification, take the positive class
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            
            return shap_values
            
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None
    
    def plot_shap_summary(self, X_explain, max_display=20):
        """Plot SHAP summary plot."""
        if self.shap_explainer is None:
            print("SHAP explainer not available")
            return None
            
        try:
            shap_values = self.get_shap_values(X_explain)
            if shap_values is None:
                return None
            
            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_explain, 
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Summary Plot - Feature Importance', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
            return None
    
    def plot_shap_bar(self, X_explain, max_display=20):
        """Plot SHAP bar plot showing feature importance."""
        if self.shap_explainer is None:
            print("SHAP explainer not available")
            return None
            
        try:
            shap_values = self.get_shap_values(X_explain)
            if shap_values is None:
                return None
            
            # Create bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_explain,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Bar Plot - Global Feature Importance', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Error creating SHAP bar plot: {e}")
            return None
    
    def plot_shap_waterfall(self, X_explain, instance_idx=0):
        """Plot SHAP waterfall plot for a single instance."""
        if self.shap_explainer is None:
            print("SHAP explainer not available")
            return None
            
        try:
            shap_values = self.get_shap_values(X_explain)
            if shap_values is None:
                return None
            
            # Get expected value (baseline)
            if hasattr(self.shap_explainer, 'expected_value'):
                expected_value = self.shap_explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[1]  # Positive class for binary
            else:
                expected_value = 0
            
            # Create waterfall plot
            plt.figure(figsize=(10, 8))
            
            # Manual waterfall plot if shap waterfall is not available
            instance_shap = shap_values[instance_idx]
            instance_data = X_explain.iloc[instance_idx]
            
            # Sort by absolute SHAP values
            sorted_idx = np.argsort(np.abs(instance_shap))[::-1][:15]  # Top 15 features
            
            feature_names_sorted = [self.feature_names[i] for i in sorted_idx]
            shap_values_sorted = instance_shap[sorted_idx]
            feature_values_sorted = [instance_data.iloc[i] for i in sorted_idx]
            
            # Create horizontal bar plot
            colors = ['red' if val < 0 else 'blue' for val in shap_values_sorted]
            bars = plt.barh(range(len(shap_values_sorted)), shap_values_sorted, color=colors, alpha=0.7)
            
            # Add feature names and values
            for i, (name, value, shap_val) in enumerate(zip(feature_names_sorted, feature_values_sorted, shap_values_sorted)):
                plt.text(shap_val + (0.01 if shap_val >= 0 else -0.01), i, 
                        f'{name} = {value:.2f}', 
                        va='center', ha='left' if shap_val >= 0 else 'right', fontsize=9)
            
            plt.yticks(range(len(feature_names_sorted)), feature_names_sorted)
            plt.xlabel('SHAP Value (Impact on Model Output)')
            plt.title(f'SHAP Explanation for Instance {instance_idx}', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Error creating SHAP waterfall plot: {e}")
            return None
    
    def get_lime_explanation(self, instance, num_features=10):
        """Get LIME explanation for a single instance."""
        if self.lime_explainer is None:
            print("LIME explainer not available")
            return None
            
        try:
            # Convert to numpy array if needed
            if hasattr(instance, 'values'):
                instance = instance.values
            if len(instance.shape) > 1:
                instance = instance[0]
            
            # Get explanation
            explanation = self.lime_explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=num_features,
                top_labels=1
            )
            
            return explanation
            
        except Exception as e:
            print(f"Error getting LIME explanation: {e}")
            return None
    
    def plot_lime_explanation(self, instance, num_features=10):
        """Plot LIME explanation."""
        explanation = self.get_lime_explanation(instance, num_features)
        if explanation is None:
            return None
            
        try:
            # Get the explanation for the positive class (churn)
            exp_list = explanation.as_list()
            
            if not exp_list:
                print("No explanation available")
                return None
            
            # Extract features and importance scores
            features = [item[0] for item in exp_list]
            importance_scores = [item[1] for item in exp_list]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            colors = ['red' if score < 0 else 'blue' for score in importance_scores]
            bars = plt.barh(range(len(features)), importance_scores, color=colors, alpha=0.7)
            
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance (LIME)')
            plt.title('LIME Local Explanation', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Error plotting LIME explanation: {e}")
            return None
    
    def get_global_feature_importance(self, X_explain, method='shap'):
        """Get global feature importance using SHAP or model-based methods."""
        if method == 'shap' and self.shap_explainer is not None:
            try:
                shap_values = self.get_shap_values(X_explain)
                if shap_values is not None:
                    # Calculate mean absolute SHAP values
                    importance_scores = np.abs(shap_values).mean(axis=0)
                    
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importance_scores
                    }).sort_values('importance', ascending=False)
                    
                    return importance_df
            except Exception as e:
                print(f"Error calculating SHAP-based importance: {e}")
        
        # Fallback to model-based importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        elif hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        else:
            print("No importance method available for this model")
            return None
    
    def plot_feature_importance(self, X_explain, top_n=20, method='shap'):
        """Plot global feature importance."""
        importance_df = self.get_global_feature_importance(X_explain, method)
        
        if importance_df is None:
            return None
        
        # Take top N features
        importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel(f'Feature Importance ({method.upper()})')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_explanation_dashboard(self, X_explain, instance_idx=0):
        """Create a comprehensive explanation dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Global SHAP summary
        if self.shap_explainer is not None:
            plt.sca(axes[0, 0])
            self.plot_shap_summary(X_explain, max_display=15)
            axes[0, 0].set_title('SHAP Global Summary', fontsize=14, fontweight='bold')
        
        # 2. Feature importance bar plot
        plt.sca(axes[0, 1])
        self.plot_feature_importance(X_explain, top_n=15)
        
        # 3. SHAP waterfall for specific instance
        if self.shap_explainer is not None:
            plt.sca(axes[1, 0])
            self.plot_shap_waterfall(X_explain, instance_idx)
        
        # 4. LIME explanation for specific instance
        if self.lime_explainer is not None:
            plt.sca(axes[1, 1])
            instance = X_explain.iloc[instance_idx]
            self.plot_lime_explanation(instance, num_features=15)
        
        plt.tight_layout()
        return fig
    
    def generate_business_insights(self, X_explain, top_n=10):
        """Generate business insights from feature importance."""
        importance_df = self.get_global_feature_importance(X_explain)
        
        if importance_df is None:
            return "Feature importance not available"
        
        top_features = importance_df.head(top_n)
        
        insights = []
        insights.append(f"## Key Churn Drivers (Top {top_n} Features)")
        insights.append("")
        
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            feature = row['feature']
            importance = row['importance']
            
            # Generate business-friendly insights based on feature names
            if 'tenure' in feature.lower():
                insight = f"**{i}. Customer Tenure** - Newer customers are at higher risk of churning."
            elif 'satisfaction' in feature.lower():
                insight = f"**{i}. Satisfaction Score** - Customer satisfaction is a critical factor in retention."
            elif 'complain' in feature.lower():
                insight = f"**{i}. Customer Complaints** - Complaint handling significantly impacts churn risk."
            elif 'order' in feature.lower() and 'count' in feature.lower():
                insight = f"**{i}. Order Frequency** - Purchase behavior strongly predicts churn likelihood."
            elif 'cashback' in feature.lower():
                insight = f"**{i}. Cashback Amount** - Reward programs influence customer retention."
            elif 'device' in feature.lower():
                insight = f"**{i}. Device Usage** - Multi-device engagement affects loyalty."
            elif 'payment' in feature.lower():
                insight = f"**{i}. Payment Method** - Payment preferences correlate with churn risk."
            elif 'app' in feature.lower():
                insight = f"**{i}. App Usage** - Time spent on app indicates engagement level."
            elif 'warehouse' in feature.lower():
                insight = f"**{i}. Delivery Distance** - Logistics convenience impacts satisfaction."
            else:
                insight = f"**{i}. {feature}** - Important predictor of customer churn."
            
            insights.append(insight)
            insights.append(f"   - Relative importance: {importance:.3f}")
            insights.append("")
        
        insights.append("### Business Recommendations:")
        insights.append("- Focus retention efforts on customers with low satisfaction scores")
        insights.append("- Implement proactive complaint resolution processes")
        insights.append("- Develop targeted campaigns for new customers (low tenure)")
        insights.append("- Optimize delivery logistics and payment options")
        insights.append("- Enhance mobile app experience to increase engagement")
        
        return "\n".join(insights)
