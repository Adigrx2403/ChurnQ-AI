"""
Advanced analytics module for survival analysis and causal inference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Lifelines not available. Install with: pip install lifelines")

class AdvancedAnalytics:
    """Class for advanced analytics including survival analysis and causal inference."""
    
    def __init__(self):
        self.cox_model = None
        self.km_fitter = None
        self.survival_data = None
        
    def prepare_survival_data(self, df, duration_col='Tenure', event_col='Churn'):
        """Prepare data for survival analysis."""
        if not LIFELINES_AVAILABLE:
            print("Lifelines not available for survival analysis")
            return None
            
        survival_df = df.copy()
        
        # Create duration column (time to event or censoring)
        if duration_col not in survival_df.columns:
            # If tenure is not available, create synthetic duration
            # Based on customer behavior patterns
            np.random.seed(42)
            
            # Customers who churned have observed event times
            churned_mask = survival_df[event_col] == 1
            
            # For churned customers, use a realistic time distribution
            # Assume most churn happens early in customer lifecycle
            survival_df.loc[churned_mask, 'duration'] = np.random.exponential(
                scale=12, size=churned_mask.sum()
            )
            
            # For non-churned customers, they are censored at observation time
            survival_df.loc[~churned_mask, 'duration'] = np.random.uniform(
                6, 36, size=(~churned_mask).sum()
            )
        else:
            survival_df['duration'] = survival_df[duration_col]
            
        # Ensure positive durations
        survival_df['duration'] = np.maximum(survival_df['duration'], 0.1)
        
        # Event indicator (1 = event observed, 0 = censored)
        survival_df['event'] = survival_df[event_col]
        
        self.survival_data = survival_df
        return survival_df
    
    def fit_cox_model(self, df, duration_col='duration', event_col='event', 
                     feature_cols=None):
        """Fit Cox Proportional Hazards model."""
        if not LIFELINES_AVAILABLE:
            print("Lifelines not available for Cox model")
            return None
            
        if feature_cols is None:
            # Select relevant features for survival analysis
            feature_cols = [col for col in df.columns 
                          if col not in ['CustomerID', 'Churn', duration_col, event_col]
                          and df[col].dtype in ['int64', 'float64']]
        
        # Prepare data for Cox model
        cox_data = df[[duration_col, event_col] + feature_cols].copy()
        cox_data = cox_data.dropna()
        
        # Fit Cox model
        self.cox_model = CoxPHFitter()
        try:
            self.cox_model.fit(cox_data, duration_col=duration_col, event_col=event_col)
            print("Cox Proportional Hazards model fitted successfully!")
            return self.cox_model
        except Exception as e:
            print(f"Error fitting Cox model: {e}")
            return None
    
    def plot_survival_curves(self, df, group_col=None, duration_col='duration', 
                           event_col='event'):
        """Plot Kaplan-Meier survival curves."""
        if not LIFELINES_AVAILABLE:
            print("Lifelines not available for survival curves")
            return None
            
        plt.figure(figsize=(12, 8))
        
        if group_col is None:
            # Overall survival curve
            kmf = KaplanMeierFitter()
            kmf.fit(df[duration_col], df[event_col], label='All Customers')
            kmf.plot_survival_function()
        else:
            # Grouped survival curves
            groups = df[group_col].unique()
            
            for group in groups:
                group_data = df[df[group_col] == group]
                if len(group_data) > 0:
                    kmf = KaplanMeierFitter()
                    kmf.fit(group_data[duration_col], group_data[event_col], 
                           label=f'{group_col} = {group}')
                    kmf.plot_survival_function()
        
        plt.title('Customer Survival Curves', fontsize=14, fontweight='bold')
        plt.xlabel('Time (Months)')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def calculate_hazard_ratios(self):
        """Calculate and display hazard ratios from Cox model."""
        if self.cox_model is None:
            print("Cox model not fitted")
            return None
            
        # Get hazard ratios
        hazard_ratios = np.exp(self.cox_model.params_)
        confidence_intervals = np.exp(self.cox_model.confidence_intervals_)
        p_values = self.cox_model.summary['p']
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Feature': hazard_ratios.index,
            'Hazard_Ratio': hazard_ratios.values,
            'CI_Lower': confidence_intervals.iloc[:, 0].values,
            'CI_Upper': confidence_intervals.iloc[:, 1].values,
            'P_Value': p_values.values
        })
        
        # Sort by hazard ratio
        results_df = results_df.sort_values('Hazard_Ratio', ascending=False)
        
        return results_df
    
    def plot_hazard_ratios(self, top_n=15):
        """Plot hazard ratios with confidence intervals."""
        hazard_df = self.calculate_hazard_ratios()
        if hazard_df is None:
            return None
            
        # Take top N features
        hazard_df = hazard_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Create forest plot
        y_pos = range(len(hazard_df))
        
        # Plot hazard ratios
        plt.scatter(hazard_df['Hazard_Ratio'], y_pos, color='blue', s=50, zorder=3)
        
        # Plot confidence intervals
        for i, (_, row) in enumerate(hazard_df.iterrows()):
            plt.plot([row['CI_Lower'], row['CI_Upper']], [i, i], 'b-', alpha=0.6)
        
        # Add vertical line at HR = 1
        plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='HR = 1 (No Effect)')
        
        plt.yticks(y_pos, hazard_df['Feature'])
        plt.xlabel('Hazard Ratio')
        plt.title('Cox Model - Hazard Ratios with 95% Confidence Intervals', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add significance indicators
        for i, (_, row) in enumerate(hazard_df.iterrows()):
            significance = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
            if significance:
                plt.text(row['Hazard_Ratio'] + 0.1, i, significance, 
                        fontsize=12, va='center', color='red')
        
        plt.tight_layout()
        return plt.gcf()
    
    def predict_survival_probability(self, customer_data, time_points=None):
        """Predict survival probability for new customers."""
        if self.cox_model is None:
            print("Cox model not fitted")
            return None
            
        if time_points is None:
            time_points = [1, 3, 6, 12, 24, 36]  # months
            
        try:
            # Predict survival function
            survival_func = self.cox_model.predict_survival_function(customer_data)
            
            # Get survival probabilities at specific time points
            survival_probs = {}
            for t in time_points:
                # Find closest time point in survival function
                closest_time = survival_func.index[survival_func.index <= t]
                if len(closest_time) > 0:
                    survival_probs[f'{t}M'] = survival_func.loc[closest_time[-1]].values[0]
                else:
                    survival_probs[f'{t}M'] = 1.0  # No events observed yet
            
            return survival_probs
            
        except Exception as e:
            print(f"Error predicting survival probability: {e}")
            return None
    
    def propensity_score_matching(self, df, treatment_col, outcome_col, 
                                 feature_cols=None, caliper=0.1):
        """Perform propensity score matching for causal inference."""
        if feature_cols is None:
            feature_cols = [col for col in df.columns 
                          if col not in ['CustomerID', treatment_col, outcome_col]
                          and df[col].dtype in ['int64', 'float64']]
        
        # Prepare data
        analysis_df = df[feature_cols + [treatment_col, outcome_col]].dropna()
        
        # Fit propensity score model
        X = analysis_df[feature_cols]
        treatment = analysis_df[treatment_col]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression for propensity scores
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X_scaled, treatment)
        
        # Calculate propensity scores
        propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
        analysis_df['propensity_score'] = propensity_scores
        
        # Perform matching
        treated_idx = analysis_df[analysis_df[treatment_col] == 1].index
        control_idx = analysis_df[analysis_df[treatment_col] == 0].index
        
        matched_pairs = []
        used_controls = set()
        
        for t_idx in treated_idx:
            t_ps = analysis_df.loc[t_idx, 'propensity_score']
            
            # Find closest control unit within caliper
            available_controls = [c for c in control_idx if c not in used_controls]
            if not available_controls:
                continue
                
            control_ps = analysis_df.loc[available_controls, 'propensity_score']
            distances = np.abs(control_ps - t_ps)
            
            if distances.min() <= caliper:
                best_match = available_controls[distances.argmin()]
                matched_pairs.append((t_idx, best_match))
                used_controls.add(best_match)
        
        # Create matched dataset
        matched_treated = [pair[0] for pair in matched_pairs]
        matched_control = [pair[1] for pair in matched_pairs]
        matched_indices = matched_treated + matched_control
        
        matched_df = analysis_df.loc[matched_indices].copy()
        
        # Calculate treatment effect
        treated_outcome = matched_df[matched_df[treatment_col] == 1][outcome_col].mean()
        control_outcome = matched_df[matched_df[treatment_col] == 0][outcome_col].mean()
        treatment_effect = treated_outcome - control_outcome
        
        results = {
            'matched_df': matched_df,
            'n_matched_pairs': len(matched_pairs),
            'treatment_effect': treatment_effect,
            'treated_outcome': treated_outcome,
            'control_outcome': control_outcome,
            'propensity_model': ps_model,
            'scaler': scaler
        }
        
        print(f"Matched {len(matched_pairs)} pairs")
        print(f"Treatment effect: {treatment_effect:.4f}")
        
        return results
    
    def plot_propensity_score_distribution(self, psm_results):
        """Plot propensity score distributions before and after matching."""
        matched_df = psm_results['matched_df']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Before matching (if we had the original data)
        ax1.hist(matched_df[matched_df.iloc[:, -3] == 1]['propensity_score'], 
                alpha=0.5, label='Treated', bins=20)
        ax1.hist(matched_df[matched_df.iloc[:, -3] == 0]['propensity_score'], 
                alpha=0.5, label='Control', bins=20)
        ax1.set_title('Propensity Score Distribution - After Matching')
        ax1.set_xlabel('Propensity Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Box plot comparison
        treated_ps = matched_df[matched_df.iloc[:, -3] == 1]['propensity_score']
        control_ps = matched_df[matched_df.iloc[:, -3] == 0]['propensity_score']
        
        ax2.boxplot([treated_ps, control_ps], labels=['Treated', 'Control'])
        ax2.set_title('Propensity Score Box Plot - After Matching')
        ax2.set_ylabel('Propensity Score')
        
        plt.tight_layout()
        return fig
    
    def survival_analysis_insights(self, hazard_df):
        """Generate business insights from survival analysis."""
        if hazard_df is None:
            return "Survival analysis not available"
            
        insights = []
        insights.append("## Survival Analysis Insights")
        insights.append("")
        
        # High risk factors (HR > 1.5)
        high_risk = hazard_df[hazard_df['Hazard_Ratio'] > 1.5]
        if len(high_risk) > 0:
            insights.append("### High Risk Factors (Hazard Ratio > 1.5):")
            for _, row in high_risk.iterrows():
                significance = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
                insights.append(f"- **{row['Feature']}**: HR = {row['Hazard_Ratio']:.2f} {significance}")
                
                if row['Hazard_Ratio'] > 2:
                    insights.append(f"  - *Critical factor: {((row['Hazard_Ratio']-1)*100):.0f}% increased churn risk*")
                else:
                    insights.append(f"  - *Moderate factor: {((row['Hazard_Ratio']-1)*100):.0f}% increased churn risk*")
            insights.append("")
        
        # Protective factors (HR < 0.8)
        protective = hazard_df[hazard_df['Hazard_Ratio'] < 0.8]
        if len(protective) > 0:
            insights.append("### Protective Factors (Hazard Ratio < 0.8):")
            for _, row in protective.iterrows():
                significance = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
                insights.append(f"- **{row['Feature']}**: HR = {row['Hazard_Ratio']:.2f} {significance}")
                insights.append(f"  - *Reduces churn risk by {((1-row['Hazard_Ratio'])*100):.0f}%*")
            insights.append("")
        
        insights.append("### Time-to-Churn Recommendations:")
        insights.append("- Implement early warning systems for high-risk customer segments")
        insights.append("- Focus retention efforts on protective factors identified")
        insights.append("- Monitor customers with high hazard ratio features more frequently")
        insights.append("- Design time-based intervention strategies based on survival curves")
        
        return "\n".join(insights)
    
    def create_advanced_analytics_dashboard(self, df):
        """Create comprehensive advanced analytics dashboard."""
        if not LIFELINES_AVAILABLE:
            print("Advanced analytics not available without lifelines")
            return None
            
        # Prepare survival data
        survival_df = self.prepare_survival_data(df)
        
        # Fit Cox model
        cox_model = self.fit_cox_model(survival_df)
        
        if cox_model is None:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Survival curves by satisfaction score
        plt.sca(axes[0, 0])
        if 'SatisfactionScore' in df.columns:
            # Create satisfaction groups
            survival_df['SatisfactionGroup'] = pd.cut(
                survival_df['SatisfactionScore'], 
                bins=[0, 2, 3, 4, 5], 
                labels=['Low', 'Medium-Low', 'Medium-High', 'High']
            )
            self.plot_survival_curves(survival_df, 'SatisfactionGroup')
        else:
            self.plot_survival_curves(survival_df)
        
        # 2. Hazard ratios
        plt.sca(axes[0, 1])
        self.plot_hazard_ratios(top_n=12)
        
        # 3. Survival curves by tenure
        plt.sca(axes[1, 0])
        if 'Tenure' in df.columns:
            survival_df['TenureGroup'] = pd.cut(
                survival_df['Tenure'], 
                bins=[0, 6, 12, 24, 36, float('inf')], 
                labels=['0-6M', '6-12M', '12-24M', '24-36M', '36M+']
            )
            self.plot_survival_curves(survival_df, 'TenureGroup')
        
        # 4. Cox model summary plot
        axes[1, 1].text(0.1, 0.9, "Cox Model Summary:", fontsize=14, fontweight='bold', 
                        transform=axes[1, 1].transAxes)
        
        # Add model statistics
        if hasattr(cox_model, 'summary'):
            concordance = getattr(cox_model, 'concordance_index_', 'N/A')
            log_likelihood = getattr(cox_model, 'log_likelihood_', 'N/A')
            
            summary_text = f"""
Model Performance:
- Concordance Index: {concordance:.3f if concordance != 'N/A' else 'N/A'}
- Log Likelihood: {log_likelihood:.2f if log_likelihood != 'N/A' else 'N/A'}

Top Risk Factors:
"""
            
            hazard_df = self.calculate_hazard_ratios()
            if hazard_df is not None:
                top_risks = hazard_df.head(5)
                for _, row in top_risks.iterrows():
                    summary_text += f"\n• {row['Feature']}: HR = {row['Hazard_Ratio']:.2f}"
            
            axes[1, 1].text(0.1, 0.7, summary_text, fontsize=11, 
                           transform=axes[1, 1].transAxes, verticalalignment='top')
        
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
