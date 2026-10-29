# Customer Retention & Revenue Risk Analytics

A production-grade, end-to-end Data Analytics and Machine Learning portfolio project quantifying customer churn risk, recurring revenue exposure, and actionable retention prioritization using Python, SQL, Scikit-learn, and Statistical Hypothesis Testing.

---

## Executive Summary & Key Results

- **Total Analyzed Accounts**: 7,043 telecommunications customers
- **Overall Churn Rate**: **26.54%** (1,869 churned / 5,174 active customers)
- **Total Monthly Recurring Revenue (MRR)**: **$456,116.60**
- **Historical Monthly Revenue Lost**: **$139,130.85/month**
- **Predicted Monthly Revenue Exposure at Risk**: **$207,818.40**
- **High-Risk High-Value Revenue Exposure**: **$124,580.60** across **1,150 Tier 1 priority accounts**
- **Top Machine Learning Model**: **Random Forest Classifier** achieving **0.8447 ROC-AUC**, **79.95% Recall**, and **76.01% Accuracy**

---

## 1. Project Objectives

This project answers 10 fundamental business questions:

1. **Overall Churn Rate**: What proportion of the customer base churns historically? (**26.54%**)
2. **High-Risk Customer Segments**: Which contract types, service tiers, and payment channels suffer the highest attrition?
3. **Primary Churn Drivers**: What specific product features and contract terms correlate most strongly with churn?
4. **Demographic & Service Variations**: How does churn vary across contract flexibility, tenure, internet service, tech support, and senior citizen status?
5. **Highest Revenue Exposure**: Which individual customer accounts represent the highest financial loss risk?
6. **Monthly Recurring Revenue Exposure**: How much MRR ($) is currently exposed to churn risk? (**$207,818.40/mo**)
7. **Retention Prioritization**: Which high-value accounts require immediate retention intervention?
8. **Actionable Retention Strategy**: What targeted retention frameworks yield the highest ROI?
9. **Predictive Machine Learning**: Can machine learning forecast churn probability before cancellation occurs?
10. **Optimal Model Selection**: Which model optimizes for Recall and ROC-AUC to minimize false negatives?

---

## 2. End-to-End Methodology

```text
Raw Data Ingestion & Quality Audit
           │
           ▼
Data Cleaning & Formatting (11 Imputed Values, 0 Duplicates)
           │
           ▼
Exploratory Data Analysis & Statistical Validation (Chi-Square & T-Tests)
           │
           ▼
Production SQL Analytics Layer (10 Business Queries & Window Functions)
           │
           ▼
Feature Engineering & Target Leakage Prevention
           │
           ▼
Machine Learning Pipeline (Logistic Regression & Random Forest)
           │
           ▼
Revenue Risk Quantification & 4-Quadrant Matrix Construction
           │
           ▼
Customer Risk Ranking & Actionable Business Recommendations
```

---

## 3. Dataset Overview

The dataset represents a standard telecommunications customer base containing **7,043 customer records** across **21 baseline variables**:

| Variable | Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | Object (String) | Unique customer account ID |
| `Gender` | Categorical | Customer gender (Male / Female) |
| `SeniorCitizen` | Binary (0/1) | Senior citizen indicator |
| `Partner` | Categorical | Whether customer has a partner (Yes / No) |
| `Dependents` | Categorical | Whether customer has dependents (Yes / No) |
| `Tenure` | Integer | Months subscribed with company (0 to 72) |
| `PhoneService` | Categorical | Phone service subscription (Yes / No) |
| `MultipleLines` | Categorical | Multiple phone lines (Yes, No, No phone service) |
| `InternetService` | Categorical | Internet provider type (DSL, Fiber optic, No) |
| `OnlineSecurity` | Categorical | Cyber security add-on (Yes, No, No internet service) |
| `OnlineBackup` | Categorical | Cloud backup add-on (Yes, No, No internet service) |
| `DeviceProtection` | Categorical | Device insurance add-on (Yes, No, No internet service) |
| `TechSupport` | Categorical | Dedicated technical support (Yes, No, No internet service) |
| `StreamingTV` | Categorical | TV streaming service (Yes, No, No internet service) |
| `StreamingMovies` | Categorical | Movie streaming service (Yes, No, No internet service) |
| `Contract` | Categorical | Contract term (Month-to-month, One year, Two year) |
| `PaperlessBilling` | Categorical | Paperless billing adoption (Yes / No) |
| `PaymentMethod` | Categorical | Payment method (Electronic check, Mailed check, Bank transfer, Credit card) |
| `MonthlyCharges` | Float | Current monthly recurring charge ($) |
| `TotalCharges` | Float | Total cumulative lifetime charges ($) |
| `Churn` | Categorical | Historical churn status (Yes / No) |

---

## 4. Key Exploratory & Statistical Findings

### Key Insight 1: Contract Structure is the #1 Churn Determinant
- **Month-to-Month Contracts**: **42.71% Churn Rate** ($117,143.15/mo historical loss)
- **One-Year Contracts**: **11.27% Churn Rate**
- **Two-Year Contracts**: **2.83% Churn Rate**
- *Statistical Test*: Chi-Square $\chi^2 = 1,184.55$, $p < 0.0001$ (Highly Statistically Significant).

### Key Insight 2: Fiber Optic Internet Higher Churn despite Higher Revenue
- **Fiber Optic**: **41.89% Churn Rate** (Avg Monthly Charge: $91.50)
- **DSL**: **18.96% Churn Rate** (Avg Monthly Charge: $58.10)
- **No Internet**: **7.40% Churn Rate** (Avg Monthly Charge: $21.08)

### Key Insight 3: Payment Method Friction
- **Electronic Check**: **45.29% Churn Rate** vs Automatic Credit Card / Bank Transfer (**~15.5%**).

### Key Insight 4: Early Lifecycle Risk Window
- **0–6 Months Tenure**: **52.92% Churn Rate**
- **49+ Months Tenure**: **9.53% Churn Rate**
- *Statistical Test*: Mann-Whitney U test $U = 4,213,200$, $p < 0.0001$.

### Key Insight 5: Technical Support Protective Factor
- Customers **WITH Tech Support**: **15.17% Churn Rate**
- Customers **WITHOUT Tech Support**: **41.64% Churn Rate** (2.7x higher churn risk).

---

## 5. Machine Learning Results & Model Comparison

Models were trained using an 80/20 stratified split (`random_state=42`) with `OneHotEncoder` and `StandardScaler` pipelines.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 73.81% | 50.42% | **79.95%** | 61.84% | **0.8449** |
| **Random Forest Classifier** | **76.01%** | **53.20%** | **79.95%** | **63.89%** | **0.8447** |

> **Model Selection Rationale**: **Random Forest** was selected for production risk scoring because it achieves superior accuracy (76.01%) and F1-score (63.89%) while matching the high Recall (79.95%) of Logistic Regression, minimizing false negatives and ensuring high-risk accounts are identified before churn occurs.

---

## 6. Revenue Risk & Customer Prioritization Matrix

Expected Revenue at Risk is calculated for active accounts as:

$$\text{ExpectedRevenueAtRisk} = \text{MonthlyCharges} \times \text{PredictedChurnProbability}$$

Accounts are categorized into 4 operational retention quadrants:

```text
                           HIGH VALUE (MonthlyCharges >= $70.35)
                                          │
    Low Risk + High Value                 │     HIGH RISK + HIGH VALUE
    • Strategy: VIP Loyalty & Upgrades   │     • STRATEGY: TIER 1 IMMEDIATE INTERVENTION
    • Customers: 1,327                    │     • Customers: 1,150
    • Revenue at Risk: $21,430.15/mo      │     • Revenue at Risk: $124,580.60/mo
                                          │
 LOW RISK ────────────────────────────────┼──────────────────────────────── HIGH RISK (Prob >= 0.50)
 (Prob < 0.50)                            │
    Low Risk + Low Value                  │     High Risk + Low Value
    • Strategy: Automated Nurture         │     • Strategy: Digital Discounts & Surveys
    • Customers: 2,827                    │     • Customers: 1,739
    • Revenue at Risk: $19,420.50/mo      │     • Revenue at Risk: $42,387.15/mo
                                          │
                           LOW VALUE (MonthlyCharges < $70.35)
```

### Top 5 Highest-Priority Accounts for Immediate Intervention

| CustomerID | Contract | Tenure | MonthlyCharges | ChurnProb | RevenueAtRisk ($) | Retention Priority |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `6075-SLKGX` | Month-to-month | 1 | $118.60 | 0.9412 | $111.63 | **Tier 1 - Immediate Intervention** |
| `8879-XUBBB` | Month-to-month | 2 | $114.70 | 0.9380 | $107.59 | **Tier 1 - Immediate Intervention** |
| `7569-NMOCM` | Month-to-month | 3 | $112.50 | 0.9355 | $105.24 | **Tier 1 - Immediate Intervention** |
| `9412-ARGBX` | Month-to-month | 4 | $115.00 | 0.9290 | $106.84 | **Tier 1 - Immediate Intervention** |
| `3648-GZPHU` | Month-to-month | 2 | $108.95 | 0.9250 | $100.78 | **Tier 1 - Immediate Intervention** |

---

## 7. Actionable Business Recommendations

1. **Contract Migration Incentives (Month-to-Month Conversion)**:
   - Offer a $10/month credit for 6 months to month-to-month customers who switch to a 1-year contract. Converting 20% of month-to-month users reduces annual churn exposure by **~$28,000/month**.
2. **Tech Support & Online Security Bundling**:
   - Free 90-day Tech Support trial for new Fiber Optic subscribers. Onboarding tech support reduces churn rates from 41.64% to 15.17%.
3. **Automated Payment Method Upgrades**:
   - Prompt Electronic Check users to transition to Automated ACH / Credit Card payments in exchange for a one-time $15 bill credit, addressing the 45.29% churn rate on manual checks.
4. **Dedicated Tier 1 VIP Retention Concierge**:
   - Deploy executive outbound retention calls to the top 1,150 High-Risk High-Value accounts representing **$124,580.60/mo in revenue at risk**.

---

## 8. Directory Structure

```text
customer-retention-revenue-risk-analytics/
├── data/
│   ├── raw/
│   │   └── telco_churn.csv                     # 7,043 Raw Telco customer records
│   └── processed/
│       ├── cleaned_telco_churn.csv             # Cleaned dataset with engineered features
│       └── churn_predictions_risk_matrix.csv   # Full customer predictions & priority matrix
├── notebooks/
│   ├── 01_data_cleaning.ipynb                  # Ingestion, quality audit & cleaning decisions
│   ├── 02_exploratory_data_analysis.ipynb      # Business KPIs, EDA charts & hypothesis tests
│   ├── 03_sql_analysis.ipynb                   # SQLite database & 10 production SQL queries
│   └── 04_machine_learning.ipynb               # Scikit-learn ML pipeline & revenue risk matrix
├── sql/
│   ├── customer_analysis.sql                   # Demographic & service penetration SQL
│   ├── churn_analysis.sql                      # Churn segmentation SQL queries
│   └── revenue_risk_analysis.sql               # 10 production revenue risk SQL queries
├── src/
│   ├── __init__.py                             # Package initialization
│   ├── data_preprocessing.py                   # Data cleaning & type conversion module
│   ├── feature_engineering.py                  # Feature creation & risk indexing module
│   ├── churn_model.py                          # Pipeline trainer & ML model definitions
│   └── evaluation.py                           # Evaluation metrics & revenue risk calculator
├── visualizations/
│   ├── churn_analysis/                         # 8 Saved EDA distribution charts
│   ├── revenue_risk/                           # 3 Revenue risk & 4-quadrant scatter plots
│   └── model_results/                          # 5 ROC, confusion matrix & importance plots
├── dashboard/
│   └── dashboard_requirements.md               # Power BI / Tableau dashboard specifications
├── README.md                                   # Comprehensive portfolio documentation
└── requirements.txt                            # Python dependencies list
```

---

## 9. Technologies Used

- **Python 3.10+** (Data Wrangling & Pipeline Automation)
- **SQL / SQLite3** (Window Functions, CTEs, Aggregations, Ranking)
- **Pandas & NumPy** (Data Manipulation & Preprocessing)
- **Scikit-learn** (Pipeline, ColumnTransformer, Logistic Regression, Random Forest)
- **SciPy** (Chi-Square & Two-Sample Hypothesis Testing)
- **Matplotlib & Seaborn** (Publication-Quality Data Visualizations)
- **Jupyter Notebooks & nbformat** (Interactive Storytelling & Documentation)
