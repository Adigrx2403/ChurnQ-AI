"""
Business intelligence and strategy module for customer churn analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class BusinessIntelligence:
    """Class for business intelligence and strategic insights."""
    
    def __init__(self, config):
        self.config = config
        self.customer_segments = None
        self.segment_profiles = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        
    def calculate_business_metrics(self, df, predictions=None):
        """Calculate key business metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['total_customers'] = len(df)
        metrics['churned_customers'] = df['Churn'].sum()
        metrics['churn_rate'] = df['Churn'].mean()
        metrics['retention_rate'] = 1 - metrics['churn_rate']
        
        # Financial impact
        clv = self.config.CUSTOMER_LIFETIME_VALUE
        acquisition_cost = self.config.ACQUISITION_COST
        
        metrics['revenue_at_risk'] = metrics['churned_customers'] * clv
        metrics['potential_savings'] = metrics['revenue_at_risk'] * 0.3  # Assume 30% are saveable
        
        # Model impact (if predictions provided)
        if predictions is not None:
            true_positives = ((predictions == 1) & (df['Churn'] == 1)).sum()
            false_positives = ((predictions == 1) & (df['Churn'] == 0)).sum()
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / metrics['churned_customers'] if metrics['churned_customers'] > 0 else 0
            
            metrics['model_precision'] = precision
            metrics['model_recall'] = recall
            metrics['customers_correctly_identified'] = true_positives
            metrics['intervention_efficiency'] = precision  # How many identified customers actually churn
        
        # Segment analysis
        if 'Tenure' in df.columns:
            tenure_segments = pd.cut(df['Tenure'], bins=[0, 6, 12, 24, float('inf')], 
                                   labels=['New', 'Growing', 'Mature', 'Veteran'])
            metrics['churn_by_tenure'] = df.groupby(tenure_segments)['Churn'].mean().to_dict()
        
        if 'SatisfactionScore' in df.columns:
            satisfaction_segments = pd.cut(df['SatisfactionScore'], bins=[0, 2, 3, 4, 5], 
                                         labels=['Low', 'Medium', 'High', 'Very High'])
            metrics['churn_by_satisfaction'] = df.groupby(satisfaction_segments)['Churn'].mean().to_dict()
        
        return metrics
    
    def create_customer_segments(self, df, n_clusters=5, features=None):
        """Create customer segments using K-means clustering."""
        if features is None:
            # Select relevant features for segmentation
            numeric_features = df.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_features 
                       if col not in ['CustomerID', 'Churn'] and df[col].notna().sum() > len(df) * 0.8]
        
        # Prepare data
        segmentation_data = df[features].fillna(df[features].median())
        
        # Scale features
        scaled_data = self.scaler.fit_transform(segmentation_data)
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=self.config.RANDOM_STATE)
        cluster_labels = self.kmeans_model.fit_predict(scaled_data)
        
        # Add segment labels to original data
        df_segmented = df.copy()
        df_segmented['Segment'] = cluster_labels
        
        # Create segment profiles
        self.segment_profiles = self.create_segment_profiles(df_segmented, features)
        self.customer_segments = df_segmented
        
        return df_segmented
    
    def create_segment_profiles(self, df_segmented, features):
        """Create detailed profiles for each customer segment."""
        profiles = {}
        
        for segment in df_segmented['Segment'].unique():
            segment_data = df_segmented[df_segmented['Segment'] == segment]
            
            profile = {
                'size': len(segment_data),
                'churn_rate': segment_data['Churn'].mean(),
                'size_percentage': len(segment_data) / len(df_segmented) * 100,
            }
            
            # Calculate feature statistics
            for feature in features:
                if feature in segment_data.columns:
                    profile[f'{feature}_mean'] = segment_data[feature].mean()
                    profile[f'{feature}_median'] = segment_data[feature].median()
            
            # Business-friendly segment naming
            profile['segment_name'] = self.name_segment(profile, segment)
            profiles[segment] = profile
        
        return profiles
    
    def name_segment(self, profile, segment_id):
        """Generate business-friendly names for segments."""
        churn_rate = profile['churn_rate']
        
        # Simple naming based on churn rate and size
        if churn_rate > 0.4:
            return f"High Risk Customers"
        elif churn_rate > 0.2:
            return f"Medium Risk Customers"
        elif churn_rate > 0.1:
            return f"Low Risk Customers"
        else:
            return f"Loyal Customers"
    
    def plot_segment_analysis(self, df_segmented):
        """Create comprehensive segment analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Segment size and churn rate
        segment_summary = df_segmented.groupby('Segment').agg({
            'CustomerID': 'count',
            'Churn': 'mean'
        }).rename(columns={'CustomerID': 'Size', 'Churn': 'ChurnRate'})
        
        ax1 = axes[0, 0]
        bars = ax1.bar(segment_summary.index, segment_summary['Size'])
        ax1.set_title('Customer Segment Sizes', fontweight='bold')
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Number of Customers')
        
        # Add churn rate labels
        for i, (idx, row) in enumerate(segment_summary.iterrows()):
            ax1.text(i, row['Size'] + 50, f"Churn: {row['ChurnRate']:.1%}", 
                    ha='center', va='bottom', fontsize=10)
        
        # 2. Churn rate by segment
        ax2 = axes[0, 1]
        colors = ['red' if rate > 0.3 else 'orange' if rate > 0.15 else 'green' 
                 for rate in segment_summary['ChurnRate']]
        bars = ax2.bar(segment_summary.index, segment_summary['ChurnRate'], color=colors, alpha=0.7)
        ax2.set_title('Churn Rate by Segment', fontweight='bold')
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('Churn Rate')
        ax2.set_ylim(0, segment_summary['ChurnRate'].max() * 1.1)
        
        # Add value labels
        for i, rate in enumerate(segment_summary['ChurnRate']):
            ax2.text(i, rate + 0.01, f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. Feature comparison across segments
        if 'SatisfactionScore' in df_segmented.columns:
            ax3 = axes[1, 0]
            df_segmented.boxplot(column='SatisfactionScore', by='Segment', ax=ax3)
            ax3.set_title('Satisfaction Score by Segment', fontweight='bold')
            ax3.set_xlabel('Segment')
            ax3.set_ylabel('Satisfaction Score')
        
        # 4. Tenure distribution by segment
        if 'Tenure' in df_segmented.columns:
            ax4 = axes[1, 1]
            for segment in df_segmented['Segment'].unique():
                segment_data = df_segmented[df_segmented['Segment'] == segment]
                ax4.hist(segment_data['Tenure'], alpha=0.6, label=f'Segment {segment}', bins=20)
            ax4.set_title('Tenure Distribution by Segment', fontweight='bold')
            ax4.set_xlabel('Tenure (Months)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def identify_at_risk_customers(self, df, model, threshold=0.7):
        """Identify customers at high risk of churning."""
        # Get churn probabilities
        if hasattr(model, 'predict_proba'):
            churn_probabilities = model.predict_proba(df.drop(['CustomerID', 'Churn'], axis=1, errors='ignore'))[:, 1]
        else:
            churn_probabilities = model.predict(df.drop(['CustomerID', 'Churn'], axis=1, errors='ignore'))
        
        # Identify high-risk customers
        high_risk_mask = churn_probabilities >= threshold
        at_risk_customers = df[high_risk_mask].copy()
        at_risk_customers['ChurnProbability'] = churn_probabilities[high_risk_mask]
        
        # Sort by risk level
        at_risk_customers = at_risk_customers.sort_values('ChurnProbability', ascending=False)
        
        return at_risk_customers
    
    def generate_retention_strategies(self, segment_profiles, feature_importance_df):
        """Generate targeted retention strategies for each segment."""
        strategies = {}
        
        for segment_id, profile in segment_profiles.items():
            segment_name = profile['segment_name']
            churn_rate = profile['churn_rate']
            
            strategy = {
                'segment_name': segment_name,
                'priority': 'High' if churn_rate > 0.3 else 'Medium' if churn_rate > 0.15 else 'Low',
                'tactics': [],
                'expected_impact': None,
                'investment_required': None
            }
            
            # Generate tactics based on segment characteristics and feature importance
            if churn_rate > 0.3:  # High risk segment
                strategy['tactics'] = [
                    "Immediate intervention with personalized offers",
                    "Dedicated customer success manager assignment",
                    "Proactive complaint resolution",
                    "Loyalty program enrollment with premium benefits",
                    "Regular satisfaction surveys and feedback collection"
                ]
                strategy['expected_impact'] = "20-30% churn reduction"
                strategy['investment_required'] = "High"
                
            elif churn_rate > 0.15:  # Medium risk segment
                strategy['tactics'] = [
                    "Targeted email campaigns with personalized recommendations",
                    "Mobile app engagement features",
                    "Cashback and coupon incentives",
                    "Improved delivery experience",
                    "Customer education and onboarding programs"
                ]
                strategy['expected_impact'] = "10-20% churn reduction"
                strategy['investment_required'] = "Medium"
                
            else:  # Low risk segment
                strategy['tactics'] = [
                    "Referral programs to leverage loyalty",
                    "Upselling and cross-selling opportunities",
                    "Community building and engagement initiatives",
                    "Premium service offerings",
                    "Regular appreciation campaigns"
                ]
                strategy['expected_impact'] = "5-10% churn reduction"
                strategy['investment_required'] = "Low"
            
            strategies[segment_id] = strategy
        
        return strategies
    
    def calculate_roi_projections(self, strategies, segment_profiles, total_customers):
        """Calculate ROI projections for retention strategies."""
        roi_analysis = {}
        clv = self.config.CUSTOMER_LIFETIME_VALUE
        
        for segment_id, strategy in strategies.items():
            profile = segment_profiles[segment_id]
            segment_size = profile['size']
            current_churn_rate = profile['churn_rate']
            
            # Parse expected impact
            impact_range = strategy['expected_impact']
            if '20-30%' in impact_range:
                impact_factor = 0.25  # Average of 20-30%
                investment_cost_per_customer = 50
            elif '10-20%' in impact_range:
                impact_factor = 0.15  # Average of 10-20%
                investment_cost_per_customer = 30
            else:
                impact_factor = 0.075  # Average of 5-10%
                investment_cost_per_customer = 15
            
            # Calculate projections
            customers_saved = segment_size * current_churn_rate * impact_factor
            revenue_saved = customers_saved * clv
            total_investment = segment_size * investment_cost_per_customer
            net_benefit = revenue_saved - total_investment
            roi_ratio = (net_benefit / total_investment) * 100 if total_investment > 0 else 0
            
            roi_analysis[segment_id] = {
                'segment_name': strategy['segment_name'],
                'customers_saved': round(customers_saved),
                'revenue_saved': round(revenue_saved),
                'total_investment': round(total_investment),
                'net_benefit': round(net_benefit),
                'roi_percentage': round(roi_ratio, 1),
                'payback_months': round((total_investment / (revenue_saved / 12)), 1) if revenue_saved > 0 else float('inf')
            }
        
        return roi_analysis
    
    def create_business_dashboard_data(self, df, model, feature_importance_df):
        """Create comprehensive data for business dashboard."""
        dashboard_data = {}
        
        # Basic metrics
        dashboard_data['metrics'] = self.calculate_business_metrics(df)
        
        # Customer segmentation
        df_segmented = self.create_customer_segments(df)
        dashboard_data['segments'] = self.segment_profiles
        
        # At-risk customer identification
        at_risk_customers = self.identify_at_risk_customers(df_segmented, model)
        dashboard_data['at_risk_customers'] = at_risk_customers
        
        # Retention strategies
        strategies = self.generate_retention_strategies(self.segment_profiles, feature_importance_df)
        dashboard_data['strategies'] = strategies
        
        # ROI analysis
        roi_analysis = self.calculate_roi_projections(strategies, self.segment_profiles, len(df))
        dashboard_data['roi_analysis'] = roi_analysis
        
        return dashboard_data
    
    def plot_roi_analysis(self, roi_analysis):
        """Plot ROI analysis for different strategies."""
        if not roi_analysis:
            return None
            
        roi_df = pd.DataFrame(roi_analysis).T
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ROI percentage by segment
        ax1 = axes[0, 0]
        colors = ['green' if roi > 100 else 'orange' if roi > 50 else 'red' 
                 for roi in roi_df['roi_percentage']]
        bars = ax1.bar(range(len(roi_df)), roi_df['roi_percentage'], color=colors, alpha=0.7)
        ax1.set_title('ROI by Customer Segment', fontweight='bold')
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('ROI (%)')
        ax1.set_xticks(range(len(roi_df)))
        ax1.set_xticklabels([f"Seg {i}" for i in roi_df.index], rotation=45)
        
        # Add value labels
        for i, roi in enumerate(roi_df['roi_percentage']):
            ax1.text(i, roi + 5, f'{roi}%', ha='center', va='bottom')
        
        # 2. Investment vs Revenue Saved
        ax2 = axes[0, 1]
        x = np.arange(len(roi_df))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, roi_df['total_investment'], width, label='Investment', color='red', alpha=0.7)
        bars2 = ax2.bar(x + width/2, roi_df['revenue_saved'], width, label='Revenue Saved', color='green', alpha=0.7)
        
        ax2.set_title('Investment vs Revenue Saved', fontweight='bold')
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('Amount ($)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Seg {i}" for i in roi_df.index], rotation=45)
        ax2.legend()
        
        # 3. Customers Saved by Segment
        ax3 = axes[1, 0]
        bars = ax3.bar(range(len(roi_df)), roi_df['customers_saved'], color='blue', alpha=0.7)
        ax3.set_title('Customers Saved by Segment', fontweight='bold')
        ax3.set_xlabel('Segment')
        ax3.set_ylabel('Number of Customers')
        ax3.set_xticks(range(len(roi_df)))
        ax3.set_xticklabels([f"Seg {i}" for i in roi_df.index], rotation=45)
        
        # Add value labels
        for i, customers in enumerate(roi_df['customers_saved']):
            ax3.text(i, customers + 1, f'{int(customers)}', ha='center', va='bottom')
        
        # 4. Payback Period
        ax4 = axes[1, 1]
        payback_finite = roi_df['payback_months'].replace([float('inf')], [0])
        bars = ax4.bar(range(len(roi_df)), payback_finite, color='purple', alpha=0.7)
        ax4.set_title('Payback Period (Months)', fontweight='bold')
        ax4.set_xlabel('Segment')
        ax4.set_ylabel('Months')
        ax4.set_xticks(range(len(roi_df)))
        ax4.set_xticklabels([f"Seg {i}" for i in roi_df.index], rotation=45)
        
        # Add value labels
        for i, months in enumerate(roi_df['payback_months']):
            if months != float('inf'):
                ax4.text(i, months + 0.5, f'{months:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def generate_executive_summary(self, dashboard_data):
        """Generate executive summary for business stakeholders."""
        metrics = dashboard_data['metrics']
        roi_analysis = dashboard_data['roi_analysis']
        
        # Calculate totals across all segments
        total_investment = sum([analysis['total_investment'] for analysis in roi_analysis.values()])
        total_revenue_saved = sum([analysis['revenue_saved'] for analysis in roi_analysis.values()])
        total_customers_saved = sum([analysis['customers_saved'] for analysis in roi_analysis.values()])
        overall_roi = ((total_revenue_saved - total_investment) / total_investment * 100) if total_investment > 0 else 0
        
        summary = f"""
# Executive Summary: Customer Churn Analysis

## Key Business Metrics
- **Total Customers**: {metrics['total_customers']:,}
- **Current Churn Rate**: {metrics['churn_rate']:.1%}
- **Revenue at Risk**: ${metrics['revenue_at_risk']:,.0f}
- **Potential Savings**: ${metrics.get('potential_savings', 0):,.0f}

## Strategic Recommendations

### Immediate Actions Required
1. **High Priority Segments**: Focus on segments with >30% churn rate
2. **Model Deployment**: Implement predictive model with {metrics.get('model_precision', 0):.1%} precision
3. **Customer Intervention**: Target {len(dashboard_data.get('at_risk_customers', []))} high-risk customers

### Financial Impact Projection
- **Total Investment Required**: ${total_investment:,.0f}
- **Projected Revenue Saved**: ${total_revenue_saved:,.0f}
- **Net Benefit**: ${total_revenue_saved - total_investment:,.0f}
- **Overall ROI**: {overall_roi:.1f}%
- **Customers Saved**: {total_customers_saved:,.0f}

### Strategic Focus Areas
1. **Customer Satisfaction**: Primary driver of churn - invest in satisfaction improvement
2. **Early Intervention**: Target customers within first 6 months of tenure
3. **Complaint Resolution**: Implement proactive complaint handling system
4. **Engagement Programs**: Increase app usage and order frequency

### Success Metrics
- Target churn rate reduction: 15-25%
- Customer satisfaction improvement: 10-15%
- Revenue retention: ${total_revenue_saved:,.0f}
- ROI achievement: {overall_roi:.0f}% within 12 months

## Next Steps
1. **Week 1-2**: Deploy predictive model and identify at-risk customers
2. **Week 3-4**: Launch targeted retention campaigns for high-risk segments
3. **Month 2-3**: Implement systematic satisfaction improvement programs
4. **Month 4-6**: Monitor results and optimize strategies based on performance

*This analysis provides data-driven insights to reduce customer churn and maximize revenue retention.*
"""
        
        return summary
