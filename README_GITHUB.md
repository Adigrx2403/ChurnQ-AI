# ChurnQ-AI 🎯
*Advanced Customer Churn Prediction System with AI-Powered Analytics*

## 🚀 Project Overview

ChurnQ-AI is a comprehensive customer churn analysis and prediction system built with Python and Streamlit. It combines machine learning, advanced analytics, and business intelligence to help organizations predict and prevent customer churn.

## ✨ Key Features

- **🤖 Machine Learning Models**: XGBoost, Random Forest, and Logistic Regression
- **📊 Interactive Dashboard**: 10-page Streamlit application with real-time analytics
- **🎯 Churn Prediction**: Individual customer risk assessment
- **📈 Business Intelligence**: Revenue impact analysis and ROI calculations
- **🔍 Advanced Analytics**: Customer segmentation and retention strategies
- **📱 Model Explainability**: SHAP values for prediction interpretation
- **🧪 Comprehensive Testing**: Full test suite for production readiness

## 🏗️ Architecture

```
ChurnQ-AI/
├── app.py                 # Main Streamlit application
├── src/                   # Core application modules
├── config/               # Configuration management
├── tests/                # Comprehensive test suite
├── models/               # Trained ML models (local only)
├── notebooks/            # Development notebooks
└── requirements.txt      # Dependencies
```

## 🛠️ Technology Stack

- **Backend**: Python 3.13+
- **Frontend**: Streamlit
- **ML Libraries**: XGBoost, Scikit-learn, Pandas
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Testing**: Unittest, Pytest-compatible
- **Deployment**: Streamlit Cloud ready

## 📋 Requirements

```bash
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
plotly>=5.15.0
seaborn>=0.12.0
joblib>=1.3.0
shap>=0.42.0
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/abhi03sagar/ChurnQ-AI.git
   cd ChurnQ-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard**
   Open your browser to `http://localhost:8501`

## 📊 Dashboard Pages

1. **🏠 Home** - Project overview and key metrics
2. **📈 Data Overview** - Dataset exploration and statistics
3. **🤖 Model Performance** - ML model comparison and metrics
4. **🎯 Churn Prediction** - Individual customer predictions
5. **🔍 Feature Importance** - Model explainability analysis
6. **📊 Advanced Analytics** - Deep-dive statistical analysis
7. **💼 Business Intelligence** - Revenue and ROI insights
8. **👥 Customer Segmentation** - Risk-based customer groups
9. **🔄 Retention Strategies** - Actionable business recommendations
10. **📖 Documentation** - Comprehensive project guide

## 🧪 Testing

Run the complete test suite:
```bash
python run_tests.py
```

Test categories:
- **System Health**: Infrastructure validation
- **Data Processing**: Pipeline integrity
- **Model Performance**: ML model validation
- **Integration**: End-to-end testing

## 📈 Model Performance

- **XGBoost**: 98.4% ROC-AUC
- **Random Forest**: 97.8% ROC-AUC  
- **Logistic Regression**: 89.2% ROC-AUC

## 🤝 Contributors

- **Abhinav** - [abhinav.techy.codes@gmail.com](mailto:abhinav.techy.codes@gmail.com)
- **Aditya Singh** - [as6717@srmist.edu.in](mailto:as6717@srmist.edu.in)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- E-commerce dataset for model training
- Open-source ML libraries and frameworks
- Streamlit community for dashboard capabilities

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the contributors directly
- Check the documentation page in the app

---

*Built with ❤️ for better customer retention strategies*
