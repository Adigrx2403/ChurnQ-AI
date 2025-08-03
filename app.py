"""
Main Streamlit application for Customer Churn Analysis Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import custom modules
from config.config import *
from src.data_processor import DataProcessor
from src.model_developer import ChurnModelDeveloper
from src.explainability import ModelExplainer
from src.advanced_analytics import AdvancedAnalytics
from src.business_intelligence import BusinessIntelligence
from src.app_config import AppConfig

# Business constants
CUSTOMER_LIFETIME_VALUE = 1200  # Average customer lifetime value in dollars
RETENTION_COST_RATIO = 0.15     # Cost of retention as percentage of CLV

# Configure Streamlit page
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .highlight {
        background-color: #ffeaa7;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process data."""
    try:
        config = AppConfig()  
        processor = DataProcessor(config)
        
        # Check if processed data exists - it's saved in the root directory
        processed_file = PROJECT_ROOT / PROCESSED_DATA_FILE
        
        if processed_file.exists():
            df = pd.read_csv(processed_file)
            # st.success("✅ Loaded processed data from cache")  # Removed to reduce UI clutter
            return df, "cached"
        else:
            # Process raw data
            raw_file = PROJECT_ROOT / RAW_DATA_FILE
            df, summary, outlier_info = processor.process_pipeline(raw_file)
            # st.success("✅ Data processed successfully")  # Removed to reduce UI clutter
            return df, "processed"
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None

def get_data_processor():
    """Get a data processor instance."""
    config = AppConfig()
    return DataProcessor(config)

@st.cache_resource
def load_models():
    """Load trained models with caching."""
    try:
        config = AppConfig()
        model_developer = ChurnModelDeveloper(config)
        model_developer.load_models(MODELS_DIR)
        return model_developer
    except Exception as e:
        st.warning(f"⚠️ Models not found. Please train models first: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables."""
    # Just initialize basic session state without complex config objects
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True

def sidebar_navigation():
    """Create sidebar navigation."""
    st.sidebar.markdown("# 🎯 Navigation")
    
    pages = {
        "🏠 Executive Summary": "executive_summary",
        "📊 Data Overview": "data_overview", 
        "🔍 Exploratory Analysis": "eda",
        "🤖 Model Performance": "model_performance",
        "💡 Model Explainability": "explainability",
        "⏱️ Advanced Analytics": "advanced_analytics",
        "💼 Business Intelligence": "business_intelligence",
        "🎯 Customer Segmentation": "segmentation",
        "📈 Retention Strategies": "retention_strategies",
        "🔮 Prediction Tool": "prediction_tool"
    }
    
    selected_page = st.sidebar.selectbox(
        "Select Page", 
        list(pages.keys()),
        index=0
    )
    
    return pages[selected_page]

def display_executive_summary(df, model_developer, business_intel):
    """Display executive summary page."""
    st.markdown("<h1 class='main-header'>📋 Executive Summary</h1>", unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    churned_customers = df['Churn'].sum()
    churn_rate = df['Churn'].mean()
    revenue_at_risk = churned_customers * CUSTOMER_LIFETIME_VALUE
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        st.metric("Churned Customers", f"{churned_customers:,}")
    
    with col3:
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    
    with col4:
        st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")
    
    # Business impact
    st.markdown("## 🎯 Business Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Current Situation
        - Our e-commerce platform has a churn rate of **{:.1%}**
        - This represents **${:,.0f}** in potential revenue loss
        - **{:,}** customers at risk of churning
        - Early intervention could save **30-40%** of at-risk customers
        """.format(churn_rate, revenue_at_risk, churned_customers))
    
    with col2:
        st.markdown("""
        ### Strategic Priorities
        1. **Immediate**: Deploy predictive model for early identification
        2. **Short-term**: Implement targeted retention campaigns
        3. **Medium-term**: Address root causes of dissatisfaction
        4. **Long-term**: Build sustainable customer loyalty programs
        """)
    
    # Model performance summary
    if model_developer and model_developer.model_results:
        st.markdown("## 🤖 Model Performance Overview")
        
        results_df = pd.DataFrame()
        for model_name, metrics in model_developer.model_results.items():
            results_df = pd.concat([
                results_df,
                pd.DataFrame({
                    'Model': [model_name.replace('_', ' ').title()],
                    'Accuracy': [metrics['accuracy']],
                    'Precision': [metrics['precision']],
                    'Recall': [metrics['recall']],
                    'F1-Score': [metrics['f1_score']],
                    'ROC-AUC': [metrics['roc_auc']]
                })
            ], ignore_index=True)
        
        st.dataframe(results_df.round(3), use_container_width=True)
    
    # ROI projection
    st.markdown("## 💰 ROI Projection")
    
    if business_intel:
        # Create sample ROI calculation
        potential_savings = revenue_at_risk * 0.3  # Assume 30% are saveable
        investment_required = total_customers * 25  # $25 per customer
        net_benefit = potential_savings - investment_required
        roi_percentage = (net_benefit / investment_required) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Potential Savings", f"${potential_savings:,.0f}")
        
        with col2:
            st.metric("Investment Required", f"${investment_required:,.0f}")
        
        with col3:
            st.metric("Projected ROI", f"{roi_percentage:.0f}%")

def display_data_overview(df, processor):
    """Display data overview page."""
    st.markdown("<h1 class='main-header'>📊 Data Overview</h1>", unsafe_allow_html=True)
    
    # Dataset information
    st.markdown("## 📋 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns  
        **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB  
        **Missing Values:** {df.isnull().sum().sum()} total
        """)
    
    with col2:
        st.markdown(f"""
        **Target Variable:** Churn (Binary)  
        **Churn Rate:** {df['Churn'].mean():.1%}  
        **Data Types:** {df.dtypes.value_counts().to_dict()}
        """)
    
    # Feature summary
    st.markdown("## 🔍 Feature Summary")
    
    # Data types breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Numerical Features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_cols].describe()
        st.dataframe(numeric_df, use_container_width=True)
    
    with col2:
        st.markdown("### Categorical Features")
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:5]:  # Show first 5
                st.write(f"**{col}:** {df[col].nunique()} unique values")
                st.write(df[col].value_counts().head(3).to_dict())
    
    # Missing values analysis
    if df.isnull().sum().sum() > 0:
        st.markdown("## ❓ Missing Values Analysis")
        
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing Percentage': (df.isnull().sum() / len(df) * 100).values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
    
    # Sample data
    st.markdown("## 👀 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

def display_eda(df):
    """Display exploratory data analysis page."""
    st.markdown("<h1 class='main-header'>🔍 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    # Churn distribution
    st.markdown("## 🎯 Churn Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn pie chart
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=['Retained', 'Churned'],
            title="Customer Churn Distribution",
            color_discrete_sequence=['#2ca02c', '#d62728']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn bar chart
        fig = px.bar(
            x=['Retained', 'Churned'],
            y=churn_counts.values,
            title="Customer Count by Churn Status",
            color=['Retained', 'Churned'],
            color_discrete_sequence=['#2ca02c', '#d62728']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.markdown("## 📈 Feature Analysis")
    
    # Select features for analysis
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['CustomerID', 'Churn']]
    
    selected_features = st.multiselect(
        "Select features for analysis:",
        numeric_features,
        default=numeric_features[:4] if len(numeric_features) >= 4 else numeric_features
    )
    
    if selected_features:
        # Feature distributions by churn
        for feature in selected_features:
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot
                fig = px.box(
                    df, x='Churn', y=feature,
                    title=f"{feature} Distribution by Churn Status",
                    color='Churn',
                    color_discrete_sequence=['#2ca02c', '#d62728']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Histogram
                fig = px.histogram(
                    df, x=feature, color='Churn',
                    title=f"{feature} Histogram by Churn Status",
                    barmode='overlay',
                    color_discrete_sequence=['#2ca02c', '#d62728']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("## 🔗 Correlation Analysis")
    
    # Calculate correlation matrix
    corr_features = [col for col in numeric_features if col in df.columns] + ['Churn']
    corr_matrix = df[corr_features].corr()
    
    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations with churn
    if 'Churn' in corr_matrix.columns:
        churn_corr = corr_matrix['Churn'].abs().sort_values(ascending=False)[1:11]  # Top 10
        
        st.markdown("### Top Features Correlated with Churn")
        
        fig = px.bar(
            x=churn_corr.values,
            y=churn_corr.index,
            orientation='h',
            title="Features Most Correlated with Churn",
            labels={'x': 'Absolute Correlation', 'y': 'Features'}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_model_performance(model_developer):
    """Display model performance page."""
    st.markdown("<h1 class='main-header'>🤖 Model Performance</h1>", unsafe_allow_html=True)
    
    if not model_developer or not model_developer.model_results:
        st.warning("⚠️ No model results available. Please train models first.")
        return
    
    # Model comparison
    st.markdown("## 📊 Model Comparison")
    
    # Create results dataframe
    results_df = pd.DataFrame()
    for model_name, metrics in model_developer.model_results.items():
        results_df = pd.concat([
            results_df,
            pd.DataFrame({
                'Model': [model_name.replace('_', ' ').title()],
                'Accuracy': [metrics['accuracy']],
                'Precision': [metrics['precision']],
                'Recall': [metrics['recall']],
                'F1-Score': [metrics['f1_score']],
                'ROC-AUC': [metrics['roc_auc']]
            })
        ], ignore_index=True)
    
    st.dataframe(results_df.round(3), use_container_width=True)
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics comparison
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model'],
                y=results_df[metric],
                text=results_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Best model highlight
        if 'ROC-AUC' in results_df.columns:
            best_model_idx = results_df['ROC-AUC'].idxmax()
            best_model = results_df.loc[best_model_idx]
            
            st.markdown("### 🏆 Best Performing Model")
            st.markdown(f"""
            **Model:** {best_model['Model']}  
            **ROC-AUC:** {best_model['ROC-AUC']:.3f}  
            **Precision:** {best_model['Precision']:.3f}  
            **Recall:** {best_model['Recall']:.3f}  
            **F1-Score:** {best_model['F1-Score']:.3f}
            """)
    
    # Confusion matrices
    st.markdown("## 📋 Confusion Matrices")
    
    model_names = list(model_developer.model_results.keys())
    selected_model = st.selectbox("Select model for detailed analysis:", model_names)
    
    if selected_model in model_developer.model_results:
        metrics = model_developer.model_results[selected_model]
        cm = metrics['confusion_matrix']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix heatmap
            fig = px.imshow(
                cm,
                text_auto=True,
                title=f"Confusion Matrix - {selected_model.replace('_', ' ').title()}",
                labels=dict(x="Predicted", y="Actual"),
                x=['No Churn', 'Churn'],
                y=['No Churn', 'Churn']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Classification report
            if 'classification_report' in metrics:
                st.markdown("### Classification Report")
                class_report = metrics['classification_report']
                
                # Convert to dataframe for better display
                report_df = pd.DataFrame(class_report).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)

def display_explainability(df, model_developer):
    """Display model explainability page."""
    st.markdown("<h1 class='main-header'>💡 Model Explainability</h1>", unsafe_allow_html=True)
    
    if not model_developer or not model_developer.models:
        st.warning("⚠️ No models available for explanation. Please train models first.")
        return
    
    st.markdown("## 🔍 Feature Importance Analysis")
    
    # Select model for explanation
    available_models = list(model_developer.models.keys())
    selected_model_name = st.selectbox("Select model for explanation:", available_models)
    
    if selected_model_name in model_developer.models:
        model = model_developer.models[selected_model_name]
        
        # Get feature names
        feature_columns = [col for col in df.columns if col not in ['CustomerID', 'Churn']]
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 15 features
                top_features = importance_df.head(15)
                
                fig = px.bar(
                    top_features,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 15 Feature Importance - {selected_model_name.replace('_', ' ').title()}"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Business Insights")
                
                # Generate insights based on top features
                top_5_features = importance_df.head(5)
                insights = []
                
                for _, row in top_5_features.iterrows():
                    feature = row['Feature']
                    importance = row['Importance']
                    
                    if 'satisfaction' in feature.lower():
                        insights.append(f"• **Customer Satisfaction** is critical - focus on improving satisfaction scores")
                    elif 'tenure' in feature.lower():
                        insights.append(f"• **Customer Tenure** matters - target retention efforts on newer customers")
                    elif 'complain' in feature.lower():
                        insights.append(f"• **Complaints** strongly predict churn - improve complaint resolution")
                    elif 'order' in feature.lower():
                        insights.append(f"• **Order Behavior** is important - encourage frequent purchases")
                    else:
                        insights.append(f"• **{feature}** significantly impacts churn prediction")
                
                for insight in insights:
                    st.markdown(insight)
        
        elif hasattr(model, 'coef_'):
            # For logistic regression
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Coefficient': model.coef_[0],
                'Abs_Coefficient': np.abs(model.coef_[0])
            }).sort_values('Abs_Coefficient', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 15 features
                top_features = importance_df.head(15)
                
                # Color code positive and negative coefficients
                colors = ['red' if coef < 0 else 'blue' for coef in top_features['Coefficient']]
                
                fig = go.Figure(go.Bar(
                    x=top_features['Coefficient'],
                    y=top_features['Feature'],
                    orientation='h',
                    marker_color=colors
                ))
                
                fig.update_layout(
                    title=f"Feature Coefficients - {selected_model_name.replace('_', ' ').title()}",
                    xaxis_title="Coefficient Value",
                    yaxis=dict(categoryorder='total ascending')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Coefficient Interpretation")
                st.markdown("""
                - **Positive coefficients** (blue): Increase churn probability
                - **Negative coefficients** (red): Decrease churn probability
                - **Larger absolute values**: More influential features
                """)

def display_prediction_tool(df, model_developer, processor):
    """Display prediction tool page."""
    st.markdown("<h1 class='main-header'>🔮 Customer Churn Prediction Tool</h1>", unsafe_allow_html=True)
    
    if not model_developer or not model_developer.models:
        st.warning("⚠️ No models available for prediction. Please train models first.")
        return
    
    st.markdown("## 🎯 Individual Customer Prediction")
    
    # Model selection
    available_models = list(model_developer.models.keys())
    selected_model_name = st.selectbox("Select model for prediction:", available_models)
    
    if selected_model_name in model_developer.models:
        model = model_developer.models[selected_model_name]
        
        # Create input form
        st.markdown("### Enter Customer Information")
        
        col1, col2 = st.columns(2)
        
        # Get feature ranges from existing data
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col not in ['CustomerID', 'Churn']]
        
        input_data = {}
        
        with col1:
            for i, feature in enumerate(numeric_features[:len(numeric_features)//2]):
                if feature in df.columns:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    
                    input_data[feature] = st.number_input(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{feature}"
                    )
        
        with col2:
            for feature in numeric_features[len(numeric_features)//2:]:
                if feature in df.columns:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    
                    input_data[feature] = st.number_input(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{feature}"
                    )
        
        # Prediction button
        if st.button("🔮 Predict Churn", type="primary"):
            try:
                # Prepare input data
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 HIGH RISK - Customer likely to churn")
                    else:
                        st.success("✅ LOW RISK - Customer likely to stay")
                
                with col2:
                    if prediction_proba is not None:
                        churn_prob = prediction_proba[1]
                        st.metric("Churn Probability", f"{churn_prob:.1%}")
                
                with col3:
                    if prediction_proba is not None:
                        retention_prob = prediction_proba[0]
                        st.metric("Retention Probability", f"{retention_prob:.1%}")
                
                # Recommendations
                if prediction == 1:
                    st.markdown("### 🎯 Recommended Actions")
                    st.markdown("""
                    1. **Immediate Contact**: Reach out to customer within 24 hours
                    2. **Personalized Offer**: Provide targeted discount or incentive
                    3. **Satisfaction Survey**: Understand specific pain points
                    4. **Account Review**: Assign dedicated customer success manager
                    5. **Follow-up**: Schedule regular check-ins over next 90 days
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    # Batch prediction
    st.markdown("## 📊 Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction:",
        type=['csv'],
        help="Upload a CSV file with customer data for batch churn prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            batch_df = pd.read_csv(uploaded_file)
            
            st.markdown("### Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🔮 Run Batch Prediction", type="primary"):
                # Make predictions
                predictions = model.predict(batch_df)
                prediction_probas = model.predict_proba(batch_df) if hasattr(model, 'predict_proba') else None
                
                # Add predictions to dataframe
                batch_df['Churn_Prediction'] = predictions
                if prediction_probas is not None:
                    batch_df['Churn_Probability'] = prediction_probas[:, 1]
                
                # Display results
                st.markdown("### Prediction Results")
                st.dataframe(batch_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Customers", len(batch_df))
                
                with col2:
                    high_risk_count = (predictions == 1).sum()
                    st.metric("High Risk Customers", high_risk_count)
                
                with col3:
                    if prediction_probas is not None:
                        avg_churn_prob = prediction_probas[:, 1].mean()
                        st.metric("Average Churn Probability", f"{avg_churn_prob:.1%}")
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing batch prediction: {str(e)}")

def display_advanced_analytics(df, model_developer):
    """Display advanced analytics page."""
    st.markdown("<h1 class='main-header'>⏱️ Advanced Analytics</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This page provides advanced statistical analysis including survival analysis, 
    time-to-churn estimation, and cohort analysis.
    """)
    
    # Survival Analysis Section
    st.markdown("## 📊 Survival Analysis")
    
    try:
        advanced_analytics = AdvancedAnalytics()
        
        # Create survival data
        df_survival = df.copy()
        df_survival['T'] = df_survival['Tenure'] + 1  # Duration (add 1 to avoid 0 values)
        df_survival['E'] = df_survival['Churn']  # Event indicator
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Kaplan-Meier Survival Curve")
            fig_km = advanced_analytics.plot_survival_curves(df_survival, duration_col='T', event_col='E')
            if fig_km:
                st.pyplot(fig_km)
        
        with col2:
            st.markdown("### Survival Statistics")
            median_survival = df_survival[df_survival['Churn'] == 1]['Tenure'].median()
            avg_survival = df_survival['Tenure'].mean()
            
            st.metric("Median Time to Churn", f"{median_survival:.1f} months")
            st.metric("Average Customer Tenure", f"{avg_survival:.1f} months")
            
            # Risk groups
            high_risk = df_survival[df_survival['Tenure'] <= 6]['Churn'].mean()
            low_risk = df_survival[df_survival['Tenure'] > 12]['Churn'].mean()
            
            st.metric("New Customer Risk (≤6 months)", f"{high_risk:.1%}")
            st.metric("Established Customer Risk (>12 months)", f"{low_risk:.1%}")
    
    except Exception as e:
        st.warning(f"⚠️ Advanced analytics temporarily unavailable: {str(e)}")
        
        # Show basic analytics as fallback
        st.markdown("### Basic Time-to-Churn Analysis")
        
        # Tenure distribution by churn
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            churned = df[df['Churn'] == 1]['Tenure']
            retained = df[df['Churn'] == 0]['Tenure']
            
            ax.hist([retained, churned], bins=20, alpha=0.7, 
                   label=['Retained', 'Churned'], color=['green', 'red'])
            ax.set_xlabel('Tenure (months)')
            ax.set_ylabel('Count')
            ax.set_title('Tenure Distribution by Churn Status')
            ax.legend()
            
            st.pyplot(fig)
        
        with col2:
            # Churn rate by tenure buckets
            df_temp = df.copy()
            df_temp['TenureBucket'] = pd.cut(df_temp['Tenure'], 
                                           bins=[0, 6, 12, 24, 36, float('inf')],
                                           labels=['0-6', '6-12', '12-24', '24-36', '36+'])
            
            churn_by_tenure = df_temp.groupby('TenureBucket')['Churn'].agg(['mean', 'count']).reset_index()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(churn_by_tenure['TenureBucket'], churn_by_tenure['mean'])
            ax.set_xlabel('Tenure Bucket (months)')
            ax.set_ylabel('Churn Rate')
            ax.set_title('Churn Rate by Tenure Bucket')
            
            # Add count labels
            for i, (rate, count) in enumerate(zip(churn_by_tenure['mean'], churn_by_tenure['count'])):
                ax.text(i, rate + 0.01, f'n={count}', ha='center', va='bottom')
            
            st.pyplot(fig)

def display_business_intelligence(df, business_intel):
    """Display business intelligence page."""
    st.markdown("<h1 class='main-header'>💼 Business Intelligence</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Strategic insights and recommendations for customer retention and business growth.
    """)
    
    # Business Metrics
    st.markdown("## 📊 Key Business Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    churned_customers = df['Churn'].sum()
    churn_rate = df['Churn'].mean()
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        st.metric("Churned Customers", f"{churned_customers:,}")
    
    with col3:
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    
    with col4:
        revenue_at_risk = churned_customers * CUSTOMER_LIFETIME_VALUE
        st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")
    
    # Customer Value Analysis
    st.markdown("## 💰 Customer Value Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Value by satisfaction score
        value_analysis = df.groupby('SatisfactionScore').agg({
            'Churn': ['count', 'mean'],
            'OrderCount': 'mean',
            'CashbackAmount': 'mean'
        }).round(2)
        
        value_analysis.columns = ['Customer Count', 'Churn Rate', 'Avg Orders', 'Avg Cashback']
        st.markdown("### Value Metrics by Satisfaction Score")
        st.dataframe(value_analysis)
    
    with col2:
        # ROI Analysis
        st.markdown("### Retention ROI Analysis")
        
        retention_cost = CUSTOMER_LIFETIME_VALUE * RETENTION_COST_RATIO
        potential_savings = churned_customers * (CUSTOMER_LIFETIME_VALUE - retention_cost)
        
        st.metric("Cost per Retention Effort", f"${retention_cost:.0f}")
        st.metric("Potential Annual Savings", f"${potential_savings:,.0f}")
        
        roi = (potential_savings / (churned_customers * retention_cost)) * 100
        st.metric("Retention Campaign ROI", f"{roi:.0f}%")
    
    # Strategic Recommendations
    st.markdown("## 🎯 Strategic Recommendations")
    
    recommendations = [
        {
            "priority": "🔥 High",
            "action": "Immediate Satisfaction Improvement",
            "description": "Focus on customers with satisfaction scores ≤ 2. Implement proactive support.",
            "impact": "Could reduce churn by 20-30%"
        },
        {
            "priority": "⚡ Medium",
            "action": "New Customer Onboarding",
            "description": "Enhanced onboarding program for customers in first 6 months.",
            "impact": "Reduce early churn by 15-25%"
        },
        {
            "priority": "📈 Low",
            "action": "Loyalty Program Enhancement",
            "description": "Improve cashback and coupon programs for high-value customers.",
            "impact": "Increase retention by 10-15%"
        }
    ]
    
    for rec in recommendations:
        with st.expander(f"{rec['priority']} Priority: {rec['action']}"):
            st.write(f"**Description:** {rec['description']}")
            st.write(f"**Expected Impact:** {rec['impact']}")

def display_customer_segmentation(df, business_intel):
    """Display customer segmentation page."""
    st.markdown("<h1 class='main-header'>🎯 Customer Segmentation</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Customer segmentation analysis based on behavior, value, and risk profiles.
    """)
    
    # Risk-based Segmentation
    st.markdown("## 🚨 Risk-Based Customer Segments")
    
    # Create risk segments based on key features
    df_segments = df.copy()
    
    # Define risk scores
    risk_factors = []
    
    # Satisfaction risk (higher weight)
    satisfaction_risk = (5 - df_segments['SatisfactionScore']) / 4 * 0.4
    risk_factors.append(satisfaction_risk)
    
    # Tenure risk (new customers are riskier)
    tenure_risk = np.where(df_segments['Tenure'] <= 6, 0.3, 
                          np.where(df_segments['Tenure'] <= 12, 0.2, 0.1))
    risk_factors.append(pd.Series(tenure_risk))
    
    # Complaint risk
    complaint_risk = df_segments['Complain'] * 0.2
    risk_factors.append(complaint_risk)
    
    # Engagement risk (low app usage)
    engagement_risk = (df_segments['HourSpendOnApp'].max() - df_segments['HourSpendOnApp']) / df_segments['HourSpendOnApp'].max() * 0.1
    risk_factors.append(engagement_risk)
    
    # Combine risk factors
    df_segments['RiskScore'] = sum(risk_factors)
    
    # Create risk segments
    df_segments['RiskSegment'] = pd.cut(df_segments['RiskScore'], 
                                       bins=[0, 0.3, 0.6, 1.0],
                                       labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    # Display segment analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Segment Distribution")
        segment_counts = df_segments['RiskSegment'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['green', 'orange', 'red']
        ax.pie(segment_counts.values, labels=segment_counts.index, 
               autopct='%1.1f%%', colors=colors)
        ax.set_title('Customer Risk Segment Distribution')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Segment Characteristics")
        segment_analysis = df_segments.groupby('RiskSegment').agg({
            'Churn': 'mean',
            'SatisfactionScore': 'mean',
            'Tenure': 'mean',
            'OrderCount': 'mean'
        }).round(2)
        
        segment_analysis.columns = ['Actual Churn Rate', 'Avg Satisfaction', 'Avg Tenure', 'Avg Orders']
        st.dataframe(segment_analysis)
    
    # Detailed segment profiles
    st.markdown("## 📋 Detailed Segment Profiles")
    
    for segment in ['High Risk', 'Medium Risk', 'Low Risk']:
        segment_data = df_segments[df_segments['RiskSegment'] == segment]
        segment_size = len(segment_data)
        churn_rate = segment_data['Churn'].mean()
        
        color = '🔴' if segment == 'High Risk' else '🟡' if segment == 'Medium Risk' else '🟢'
        
        with st.expander(f"{color} {segment} Segment ({segment_size:,} customers)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Rate", f"{churn_rate:.1%}")
                st.metric("Avg Satisfaction", f"{segment_data['SatisfactionScore'].mean():.1f}")
            
            with col2:
                st.metric("Avg Tenure", f"{segment_data['Tenure'].mean():.1f} months")
                st.metric("Complaint Rate", f"{segment_data['Complain'].mean():.1%}")
            
            with col3:
                st.metric("Avg Orders", f"{segment_data['OrderCount'].mean():.1f}")
                st.metric("Avg App Usage", f"{segment_data['HourSpendOnApp'].mean():.1f} hours")
            
            # Recommendations per segment
            if segment == 'High Risk':
                st.markdown("**Recommended Actions:**")
                st.markdown("- Immediate outreach and support")
                st.markdown("- Personalized retention offers")
                st.markdown("- Priority customer service")
            elif segment == 'Medium Risk':
                st.markdown("**Recommended Actions:**") 
                st.markdown("- Proactive engagement campaigns")
                st.markdown("- Product recommendations")
                st.markdown("- Feedback collection")
            else:
                st.markdown("**Recommended Actions:**")
                st.markdown("- Loyalty program enrollment")
                st.markdown("- Cross-selling opportunities")
                st.markdown("- Referral incentives")

def display_retention_strategies(df, business_intel):
    """Display retention strategies page."""
    st.markdown("<h1 class='main-header'>📈 Retention Strategies</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Data-driven retention strategies and campaign recommendations.
    """)
    
    # Strategy Overview
    st.markdown("## 🎯 Strategic Framework")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 Identify
        - High-risk customers
        - Churn warning signals
        - Value segments
        """)
    
    with col2:
        st.markdown("""
        ### 🎬 Engage
        - Personalized campaigns
        - Proactive support
        - Value demonstrations
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Measure
        - Campaign effectiveness
        - ROI tracking
        - Long-term impact
        """)
    
    # Campaign Recommendations
    st.markdown("## 🚀 Recommended Retention Campaigns")
    
    campaigns = [
        {
            "name": "Win-Back Satisfaction Campaign",
            "target": "Customers with satisfaction ≤ 2",
            "size": len(df[df['SatisfactionScore'] <= 2]),
            "strategy": [
                "Personal outreach from customer success team",
                "Free premium support for 3 months", 
                "Product training and optimization sessions",
                "Direct feedback collection and resolution"
            ],
            "investment": "$150 per customer",
            "expected_roi": "250-300%"
        },
        {
            "name": "New Customer Success Program",
            "target": "Customers with tenure ≤ 6 months", 
            "size": len(df[df['Tenure'] <= 6]),
            "strategy": [
                "Enhanced onboarding experience",
                "Weekly check-ins for first month",
                "Personalized product recommendations",
                "Early success milestone celebrations"
            ],
            "investment": "$75 per customer",
            "expected_roi": "180-220%"
        },
        {
            "name": "Complaint Resolution Blitz",
            "target": "Customers who have complained",
            "size": len(df[df['Complain'] == 1]),
            "strategy": [
                "Immediate escalation to senior support",
                "Compensation and goodwill gestures", 
                "Process improvement implementation",
                "Follow-up satisfaction surveys"
            ],
            "investment": "$200 per customer",
            "expected_roi": "300-400%"
        }
    ]
    
    for campaign in campaigns:
        with st.expander(f"🎯 {campaign['name']} (Target: {campaign['size']:,} customers)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Campaign Strategy:**")
                for strategy in campaign['strategy']:
                    st.markdown(f"- {strategy}")
            
            with col2:
                st.metric("Investment per Customer", campaign['investment'])
                st.metric("Expected ROI", campaign['expected_roi'])
                
                total_investment = campaign['size'] * int(campaign['investment'].replace('$', '').replace(' per customer', ''))
                st.metric("Total Campaign Investment", f"${total_investment:,}")
    
    # Implementation Timeline
    st.markdown("## 📅 Implementation Timeline")
    
    timeline = [
        {"phase": "Phase 1 (Month 1)", "focus": "High-Risk Immediate Response", "actions": "Launch complaint resolution and satisfaction campaigns"},
        {"phase": "Phase 2 (Month 2)", "focus": "New Customer Success", "actions": "Implement enhanced onboarding program"},  
        {"phase": "Phase 3 (Month 3)", "focus": "Systematic Improvement", "actions": "Process improvements and loyalty enhancements"},
        {"phase": "Phase 4 (Month 4+)", "focus": "Optimization & Scale", "actions": "Refine campaigns based on results and scale successful initiatives"}
    ]
    
    for item in timeline:
        st.markdown(f"**{item['phase']}** - {item['focus']}")
        st.markdown(f"   {item['actions']}")
        st.markdown("")
    
    # Success Metrics
    st.markdown("## 📊 Success Metrics & KPIs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Primary Metrics")
        st.markdown("- **Churn Rate Reduction**: Target 20-30% decrease")
        st.markdown("- **Customer Satisfaction**: Increase average score by 0.5 points")
        st.markdown("- **Revenue Retention**: Reduce revenue at risk by $100k+")
        st.markdown("- **Campaign ROI**: Achieve 200%+ return on investment")
    
    with col2:
        st.markdown("### Secondary Metrics")
        st.markdown("- **Engagement Increase**: 15% more app usage")
        st.markdown("- **Order Frequency**: 10% increase in order count")
        st.markdown("- **Complaint Resolution**: 50% faster resolution time")
        st.markdown("- **Net Promoter Score**: Improve NPS by 15 points")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Load data and models
    df, _ = load_and_process_data()
    processor = get_data_processor()
    model_developer = load_models()
    
    if df is None:
        st.error("❌ Unable to load data. Please check your data files.")
        return
    
    # Initialize business intelligence
    config = AppConfig()
    business_intel = BusinessIntelligence(config) if df is not None else None
    
    # Sidebar navigation
    selected_page = sidebar_navigation()
    
    # Display selected page
    if selected_page == "executive_summary":
        display_executive_summary(df, model_developer, business_intel)
    
    elif selected_page == "data_overview":
        display_data_overview(df, processor)
    
    elif selected_page == "eda":
        display_eda(df)
    
    elif selected_page == "model_performance":
        display_model_performance(model_developer)
    
    elif selected_page == "explainability":
        display_explainability(df, model_developer)
    
    elif selected_page == "advanced_analytics":
        display_advanced_analytics(df, model_developer)
    
    elif selected_page == "business_intelligence":
        display_business_intelligence(df, business_intel)
    
    elif selected_page == "segmentation":
        display_customer_segmentation(df, business_intel)
    
    elif selected_page == "retention_strategies":
        display_retention_strategies(df, business_intel)
    
    elif selected_page == "prediction_tool":
        display_prediction_tool(df, model_developer, processor)
    
    else:
        st.info(f"Page '{selected_page}' is under development. Coming soon!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🔬 Customer Churn Analysis Dashboard | Built with Streamlit | 
        📊 Data-Driven Business Intelligence
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
