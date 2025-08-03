# Data Dictionary - E-Commerce Customer Churn Dataset

## Overview
This document provides detailed descriptions of all features in the E-Commerce Customer Churn dataset, including their business significance, data types, and analytical importance.

## Target Variable

| Variable | Description | Type | Values | Business Impact |
|----------|-------------|------|--------|-----------------|
| **Churn** | Customer churn flag | Binary | 0 = Retained, 1 = Churned | Primary business outcome - indicates customer leaving the platform |

## Customer Demographics

| Variable | Description | Type | Values | Business Significance |
|----------|-------------|------|--------|----------------------|
| **CustomerID** | Unique customer identifier | Integer | 50001-55630 | Primary key for customer tracking and analysis |
| **Gender** | Customer gender | Categorical | Male, Female | Segmentation variable for targeted marketing |
| **MaritalStatus** | Marital status | Categorical | Single, Married, Divorced | Lifestyle indicator affecting purchase behavior |
| **CityTier** | City classification tier | Integer | 1, 2, 3 | Market segment indicator (1=Metro, 2=Tier-2, 3=Tier-3) |

## Customer Behavior & Engagement

| Variable | Description | Type | Range | Business Impact |
|----------|-------------|------|-------|-----------------|
| **Tenure** | Customer lifetime in months | Float | 0-61 months | Loyalty indicator - longer tenure typically means lower churn risk |
| **HourSpendOnApp** | Monthly app usage hours | Float | 0-5 hours | Digital engagement measure - higher usage indicates platform stickiness |
| **NumberOfDeviceRegistered** | Registered devices count | Integer | 1-6 devices | Multi-channel engagement indicator |
| **OrderCount** | Total orders placed | Float | 1-16 orders | Purchase frequency - key engagement metric |
| **DaySinceLastOrder** | Days since last purchase | Float | 0-46 days | Recency indicator - longer gaps suggest disengagement |

## Customer Satisfaction & Service

| Variable | Description | Type | Range | Critical Impact |
|----------|-------------|------|-------|-----------------|
| **SatisfactionScore** | Customer satisfaction rating | Integer | 1-5 scale | **Primary churn predictor** - satisfaction drives retention |
| **Complain** | Complaint history flag | Binary | 0 = No, 1 = Yes | **High-risk indicator** - complaints strongly correlate with churn |

## Purchase & Financial Behavior

| Variable | Description | Type | Range | Revenue Impact |
|----------|-------------|------|-------|----------------|
| **OrderAmountHikeFromlastYear** | Year-over-year order value change | Float | 11-26% | Growth indicator - positive growth suggests loyalty |
| **CashbackAmount** | Cashback earned ($) | Float | $0-324 | Reward engagement - higher cashback may increase retention |
| **CouponUsed** | Coupons utilized count | Float | 0-16 coupons | Price sensitivity indicator |

## Preferences & Service Usage

| Variable | Description | Type | Values | Operational Insights |
|----------|-------------|------|--------|---------------------|
| **PreferredLoginDevice** | Primary login device | Categorical | Mobile Phone, Computer | Channel preference for targeted UX |
| **PreferredPaymentMode** | Payment method preference | Categorical | Debit Card, Credit Card, UPI, COD, E wallet | Payment friction indicator |
| **PreferedOrderCat** | Preferred product category | Categorical | Laptop & Accessory, Mobile Phone, Fashion, Grocery, Others | Purchase pattern for cross-selling |
| **WarehouseToHome** | Distance to warehouse (km) | Float | 5-127 km | Logistics satisfaction factor |
| **NumberOfAddress** | Saved addresses count | Integer | 1-22 addresses | Usage depth indicator |

## Engineered Features (Created during processing)

| Variable | Description | Type | Business Value |
|----------|-------------|------|----------------|
| **TenureBucket** | Tenure categories | Categorical | New (0-6M), Growing (6-12M), Mature (12-24M), Veteran (24M+) |
| **EngagementScore** | Composite engagement metric | Float | Combines app usage, order frequency, recency |
| **CustomerValueScore** | Financial value indicator | Float | Order growth + cashback composite |
| **SatisfactionBucket** | Satisfaction categories | Categorical | Low, Medium, High, Very High |
| **HasComplaint** | Binary complaint indicator | Binary | Simplified complaint flag |
| **IsCouponUser** | Coupon usage flag | Binary | Price-conscious customer indicator |
| **OrderFrequency** | Orders per month ratio | Float | Purchase velocity metric |
| **IsMultiDevice** | Multi-device user flag | Binary | Cross-platform engagement |
| **IsMultiAddress** | Multiple address flag | Binary | Usage diversity indicator |

## Data Quality Summary

### Missing Values
- **Tenure**: ~15% missing (imputed with median)
- **WarehouseToHome**: Minimal missing values
- **HourSpendOnApp**: Complete data available
- **OrderCount**: Complete data available
- **DaySinceLastOrder**: Complete data available

### Data Distribution
- **Churn Rate**: 16.8% (948 churned out of 5,630 customers)
- **Gender**: Fairly balanced (Male: 51%, Female: 49%)
- **City Tier**: Tier 1: 34%, Tier 2: 31%, Tier 3: 35%
- **Satisfaction**: Average 3.1/5 (concerning for retention)

### Outliers Treatment
- **Numerical features**: Capped using IQR method (1.5 * IQR)
- **Extreme values**: Clipped to reasonable business ranges
- **Zero values**: Retained where business-appropriate

## Feature Importance Rankings

Based on model analysis, features ranked by churn prediction importance:

1. **SatisfactionScore** (Critical) - Primary predictor
2. **Tenure** (High) - Loyalty indicator  
3. **Complain** (High) - Risk amplifier
4. **OrderCount** (Medium-High) - Engagement measure
5. **HourSpendOnApp** (Medium-High) - Digital engagement
6. **DaySinceLastOrder** (Medium) - Recency factor
7. **CashbackAmount** (Medium) - Value perception
8. **NumberOfDeviceRegistered** (Medium) - Platform adoption
9. **WarehouseToHome** (Low-Medium) - Service convenience
10. **OrderAmountHikeFromlastYear** (Low-Medium) - Growth indicator

## Business Interpretation Guide

### High-Risk Indicators
- **Satisfaction Score ≤ 2**: Immediate intervention required
- **Complaints = 1**: Proactive resolution needed
- **Tenure < 6 months**: New customer risk
- **DaySinceLastOrder > 30**: Re-engagement campaign
- **OrderCount ≤ 2**: Low engagement pattern

### Protective Factors
- **Satisfaction Score ≥ 4**: Loyal customer base
- **Tenure > 24 months**: Established relationship
- **High app usage**: Strong platform engagement
- **Multiple devices**: Deep integration
- **Regular orders**: Consistent usage pattern

### Segmentation Insights
- **New Users (0-6M)**: Higher churn risk, need onboarding support
- **Growing Users (6-12M)**: Critical retention period
- **Mature Users (12-24M)**: Stable but need value demonstration
- **Veteran Users (24M+)**: Focus on loyalty programs and upgrades

## Data Collection Recommendations

### Missing Features (Future Enhancement)
1. **Customer Support Interactions**: Ticket count, resolution time
2. **Marketing Touchpoints**: Email opens, click rates
3. **Social Engagement**: Reviews, ratings, social shares
4. **Product Returns**: Return rate, return reasons
5. **Competitive Analysis**: Price comparison behavior
6. **Seasonal Patterns**: Purchase seasonality data

### Data Quality Improvements
1. **Real-time Data**: Reduce batch processing delays
2. **Behavioral Tracking**: More granular app usage metrics
3. **Satisfaction Surveys**: Regular NPS tracking
4. **External Data**: Economic indicators, competitor actions
5. **Predictive Features**: Leading indicators vs lagging

## Usage Guidelines

### For Data Scientists
- **Feature Selection**: Start with top 10 importance-ranked features
- **Encoding**: Use provided preprocessing pipeline
- **Validation**: Maintain temporal split for time-series nature
- **Interpretation**: Focus on business-actionable insights

### For Business Users
- **Risk Scoring**: Use satisfaction + tenure for quick assessment
- **Intervention Triggers**: Complaints + low satisfaction = immediate action
- **Segmentation**: Tenure buckets for targeted campaigns
- **Success Metrics**: Track satisfaction improvement impact

### For Product Teams
- **App Engagement**: Focus on increasing HourSpendOnApp
- **User Experience**: Reduce friction in high-importance features
- **Feature Development**: Prioritize satisfaction-driving features
- **Retention Tools**: Build engagement-increasing capabilities

---

*This data dictionary is living document and should be updated as new features are added or business understanding evolves.*

**Last Updated**: August 2025  
**Version**: 1.0  
**Maintained By**: Data Science Team
