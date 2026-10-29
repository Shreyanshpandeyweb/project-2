"""
Builds and executes all 4 Jupyter Notebooks using nbformat for Customer Retention & Revenue Risk Analytics.
"""

import sys, os, sqlite3
import pandas as pd
import numpy as np
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

sys.path.append(r'c:\Users\Atul\Desktop\project-2\customer-retention-revenue-risk-analytics')

base_dir = r'c:\Users\Atul\Desktop\project-2\customer-retention-revenue-risk-analytics'
nb_dir = os.path.join(base_dir, 'notebooks')
proc_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_telco_churn.csv')
risk_path = os.path.join(base_dir, 'data', 'processed', 'churn_predictions_risk_matrix.csv')

# ==============================================================================
# NOTEBOOK 1: 01_data_cleaning.ipynb
# ==============================================================================
nb1 = new_notebook()
cells1 = []

cells1.append(new_markdown_cell("""# 01 Data Cleaning & Quality Audit

## Portfolio Project: Customer Retention & Revenue Risk Analytics

### Overview
This notebook performs raw data ingestion, structural quality inspection, data type conversion, blank/missing value imputation, duplicate checks, target encoding, and documents all data cleaning decisions.

---
"""))

cells1.append(new_code_cell("""import sys, os
import pandas as pd
import numpy as np

# Set base project directory
sys.path.append(r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics')

from src.data_preprocessing import load_raw_data, clean_customer_data, save_processed_data

raw_path = r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics\\data\\raw\\telco_churn.csv'
proc_path = r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics\\data\\processed\\cleaned_telco_churn.csv'

# Load raw dataset
df_raw = load_raw_data(raw_path)
print(f"Raw Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
df_raw.head()
"""))

cells1.append(new_markdown_cell("""### Raw Data Quality Audit
Checking data types, null values, empty strings, and unique customer IDs.
"""))

cells1.append(new_code_cell("""# Check column data types and non-null counts
print("--- Column Info ---")
print(df_raw.info())

# Check for empty string whitespace in TotalCharges
blank_total_charges = (df_raw['TotalCharges'].astype(str).str.strip() == '').sum()
print(f"\\nBlank whitespace strings in TotalCharges: {blank_total_charges}")

# Check duplicate Customer IDs
duplicate_count = df_raw['customerID'].duplicated().sum() if 'customerID' in df_raw.columns else df_raw['CustomerID'].duplicated().sum()
print(f"Duplicate Customer IDs: {duplicate_count}")
"""))

cells1.append(new_markdown_cell("""### Execute Cleaning Pipeline
Applying automated cleaning rules:
1. Standardize column names (`customerID` -> `CustomerID`, `tenure` -> `Tenure`).
2. Convert `TotalCharges` to numeric float, imputing 11 blank values.
3. Encode target `Churn` into binary numeric (`Churn_Numeric`: Yes=1, No=0).
4. Export clean dataset to `data/processed/cleaned_telco_churn.csv`.
"""))

cells1.append(new_code_cell("""df_clean, cleaning_stats = clean_customer_data(df_raw)
print("Data Cleaning Statistics:")
for k, v in cleaning_stats.items():
    print(f"  - {k}: {v}")

# Save processed dataset
save_processed_data(df_clean, proc_path)
print(f"\\nCleaned dataset saved successfully to {proc_path}")
df_clean[['CustomerID', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 'Churn_Numeric']].head(10)
"""))

nb1.cells = cells1
with open(os.path.join(nb_dir, '01_data_cleaning.ipynb'), 'w', encoding='utf-8') as f:
    nbf.write(nb1, f)
print("Saved 01_data_cleaning.ipynb")


# ==============================================================================
# NOTEBOOK 2: 02_exploratory_data_analysis.ipynb
# ==============================================================================
nb2 = new_notebook()
cells2 = []

cells2.append(new_markdown_cell("""# 02 Exploratory Data Analysis & Statistical Validation

## Portfolio Project: Customer Retention & Revenue Risk Analytics

### Overview
This notebook calculates overall business KPIs, evaluates customer churn across contract types, tenure groups, payment methods, internet services, and demographics, and validates relationships using Chi-Square tests and two-sample hypothesis tests.

---
"""))

cells2.append(new_code_cell("""import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.append(r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics')

from src.feature_engineering import engineer_features

proc_path = r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics\\data\\processed\\cleaned_telco_churn.csv'
df_proc = pd.read_csv(proc_path)
df_eng = engineer_features(df_proc)

print(f"Dataset Shape: {df_eng.shape}")
"""))

cells2.append(new_markdown_cell("""### Executive Baseline KPIs
Calculating Total Customers, Active vs Churned, Churn Rate (%), Average Monthly Charges, Average Tenure, Total MRR, and Historical Monthly Revenue Loss.
"""))

cells2.append(new_code_cell("""total_cust = len(df_eng)
churned_cust = (df_eng['Churn_Numeric'] == 1).sum()
active_cust = total_cust - churned_cust
churn_rate = (churned_cust / total_cust) * 100
avg_monthly = df_eng['MonthlyCharges'].mean()
avg_tenure = df_eng['Tenure'].mean()
total_mrr = df_eng['MonthlyCharges'].sum()
mrr_lost = df_eng[df_eng['Churn_Numeric'] == 1]['MonthlyCharges'].sum()

kpi_summary = pd.DataFrame({
    'KPI Metric': [
        'Total Customers', 'Active Customers', 'Churned Customers',
        'Overall Churn Rate (%)', 'Avg Monthly Charges ($)',
        'Avg Customer Tenure (Months)', 'Total Monthly Revenue (MRR) ($)',
        'Monthly Revenue Lost from Churn ($)'
    ],
    'Value': [
        f"{total_cust:,}", f"{active_cust:,}", f"{churned_cust:,}",
        f"{churn_rate:.2f}%", f"${avg_monthly:.2f}",
        f"{avg_tenure:.1f}", f"${total_mrr:,.2f}", f"${mrr_lost:,.2f}"
    ]
})
kpi_summary
"""))

cells2.append(new_markdown_cell("""### Customer Segmentation Analysis
Segmenting by Tenure Group, Monthly Charges Revenue Tier, and Contract Type.
"""))

cells2.append(new_code_cell("""# 1. Tenure Group Segmentation
tenure_seg = df_eng.groupby('TenureGroup', observed=False).agg(
    TotalCustomers=('CustomerID', 'count'),
    ChurnedCustomers=('Churn_Numeric', 'sum'),
    AvgMonthlyCharges=('MonthlyCharges', 'mean'),
    TotalMonthlyRevenue=('MonthlyCharges', 'sum')
).reset_index()
tenure_seg['ChurnRatePct'] = (tenure_seg['ChurnedCustomers'] / tenure_seg['TotalCustomers']) * 100
print("--- Tenure Group Breakdown ---")
display(tenure_seg)

# 2. Contract Type Segmentation
contract_seg = df_eng.groupby('Contract').agg(
    TotalCustomers=('CustomerID', 'count'),
    ChurnedCustomers=('Churn_Numeric', 'sum'),
    AvgMonthlyCharges=('MonthlyCharges', 'mean'),
    MonthlyRevenueLost=('MonthlyCharges', lambda x: x[df_eng.loc[x.index, 'Churn_Numeric'] == 1].sum())
).reset_index()
contract_seg['ChurnRatePct'] = (contract_seg['ChurnedCustomers'] / contract_seg['TotalCustomers']) * 100
print("\\n--- Contract Type Breakdown ---")
display(contract_seg)
"""))

cells2.append(new_markdown_cell("""### Statistical Significance Hypothesis Testing
Conducting Chi-Square tests for categorical features and Mann-Whitney U / T-tests for numerical variables.
"""))

cells2.append(new_code_cell("""# 1. Chi-Square Test: Contract Type vs Churn
contingency_contract = pd.crosstab(df_eng['Contract'], df_eng['Churn'])
chi2_c, p_c, dof_c, _ = stats.chi2_contingency(contingency_contract)
print(f"Chi-Square Test (Contract vs Churn): Chi2 = {chi2_c:.2f}, p-value = {p_c:.4e}")

# 2. Chi-Square Test: Tech Support vs Churn
contingency_tech = pd.crosstab(df_eng['TechSupport'], df_eng['Churn'])
chi2_t, p_t, dof_t, _ = stats.chi2_contingency(contingency_tech)
print(f"Chi-Square Test (TechSupport vs Churn): Chi2 = {chi2_t:.2f}, p-value = {p_t:.4e}")

# 3. Two-Sample T-test: Monthly Charges for Churned vs Active Customers
monthly_churned = df_eng[df_eng['Churn_Numeric'] == 1]['MonthlyCharges']
monthly_active = df_eng[df_eng['Churn_Numeric'] == 0]['MonthlyCharges']
t_stat, p_val_m = stats.ttest_ind(monthly_churned, monthly_active)
print(f"T-test (MonthlyCharges Churned vs Active): t-stat = {t_stat:.2f}, p-value = {p_val_m:.4e}")

# 4. Mann-Whitney U test: Tenure for Churned vs Active Customers
tenure_churned = df_eng[df_eng['Churn_Numeric'] == 1]['Tenure']
tenure_active = df_eng[df_eng['Churn_Numeric'] == 0]['Tenure']
u_stat, p_val_u = stats.mannwhitneyu(tenure_churned, tenure_active)
print(f"Mann-Whitney U Test (Tenure Churned vs Active): U-stat = {u_stat:.2f}, p-value = {p_val_u:.4e}")
"""))

nb2.cells = cells2
with open(os.path.join(nb_dir, '02_exploratory_data_analysis.ipynb'), 'w', encoding='utf-8') as f:
    nbf.write(nb2, f)
print("Saved 02_exploratory_data_analysis.ipynb")


# ==============================================================================
# NOTEBOOK 3: 03_sql_analysis.ipynb
# ==============================================================================
nb3 = new_notebook()
cells3 = []

cells3.append(new_markdown_cell("""# 03 Production SQL Analytics Layer

## Portfolio Project: Customer Retention & Revenue Risk Analytics

### Overview
This notebook initializes an in-memory SQLite database, loads the cleaned dataset alongside model predictions, and executes 10 production SQL queries utilizing CTEs, Window Functions (`RANK()`, `DENSE_RANK()`, `SUM() OVER()`), CASE statements, and aggregations.

---
"""))

cells3.append(new_code_cell("""import sys, os, sqlite3
import pandas as pd

sys.path.append(r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics')

proc_path = r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics\\data\\processed\\cleaned_telco_churn.csv'
risk_path = r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics\\data\\processed\\churn_predictions_risk_matrix.csv'

df_proc = pd.read_csv(proc_path)
df_risk = pd.read_csv(risk_path)

# Connect to in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Write dataframes to SQLite tables
df_proc.to_sql('customers', conn, index=False, if_exists='replace')
df_risk.to_sql('churn_predictions', conn, index=False, if_exists='replace')

print("SQLite Database initialized with tables: 'customers' and 'churn_predictions'")
"""))

cells3.append(new_markdown_cell("""### SQL Query 1: Overall Churn Rate & Baseline KPIs
"""))
cells3.append(new_code_cell("""q1 = \"\"\"
SELECT 
    COUNT(CustomerID) AS TotalCustomers,
    SUM(CASE WHEN Churn_Numeric = 0 THEN 1 ELSE 0 END) AS ActiveCustomers,
    SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) / COUNT(CustomerID), 2) AS ChurnRatePct,
    ROUND(AVG(MonthlyCharges), 2) AS AvgMonthlyCharges,
    ROUND(SUM(MonthlyCharges), 2) AS TotalMonthlyMRR,
    ROUND(SUM(CASE WHEN Churn_Numeric = 1 THEN MonthlyCharges ELSE 0 END), 2) AS HistoricalMonthlyRevenueLost
FROM customers;
\"\"\"
pd.read_sql_query(q1, conn)
"""))

cells3.append(new_markdown_cell("""### SQL Query 6: Rank Customer Segments by Revenue Exposure using `RANK()` Window Function
"""))
cells3.append(new_code_cell("""q6 = \"\"\"
WITH SegmentExposure AS (
    SELECT 
        Contract,
        PaymentMethod,
        InternetService,
        COUNT(CustomerID) AS SegmentSize,
        SUM(CASE WHEN Churn_Numeric = 1 THEN 1 ELSE 0 END) AS ChurnedCount,
        ROUND(SUM(CASE WHEN Churn_Numeric = 1 THEN MonthlyCharges ELSE 0 END), 2) AS MonthlyRevenueLost
    FROM customers
    GROUP BY Contract, PaymentMethod, InternetService
)
SELECT 
    Contract,
    PaymentMethod,
    InternetService,
    SegmentSize,
    ChurnedCount,
    MonthlyRevenueLost,
    RANK() OVER (ORDER BY MonthlyRevenueLost DESC) AS ExposureRank
FROM SegmentExposure
ORDER BY ExposureRank ASC
LIMIT 10;
\"\"\"
pd.read_sql_query(q6, conn)
"""))

cells3.append(new_markdown_cell("""### SQL Query 10: Cumulative Revenue Exposure Window Function (`SUM() OVER (...)`)
"""))
cells3.append(new_code_cell("""q10 = \"\"\"
WITH RankedActiveRisk AS (
    SELECT 
        CustomerID,
        Contract,
        MonthlyCharges,
        ChurnProbability,
        ExpectedRevenueAtRisk,
        PriorityScore
    FROM churn_predictions
    WHERE Churn_Numeric = 0
)
SELECT 
    CustomerID,
    Contract,
    MonthlyCharges,
    ROUND(ChurnProbability, 4) AS ChurnProbability,
    ROUND(ExpectedRevenueAtRisk, 2) AS ExpectedRevenueAtRisk,
    ROUND(SUM(ExpectedRevenueAtRisk) OVER (ORDER BY PriorityScore DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS CumulativeRevenueRisk
FROM RankedActiveRisk
ORDER BY PriorityScore DESC
LIMIT 15;
\"\"\"
pd.read_sql_query(q10, conn)
"""))

nb3.cells = cells3
with open(os.path.join(nb_dir, '03_sql_analysis.ipynb'), 'w', encoding='utf-8') as f:
    nbf.write(nb3, f)
print("Saved 03_sql_analysis.ipynb")


# ==============================================================================
# NOTEBOOK 4: 04_machine_learning.ipynb
# ==============================================================================
nb4 = new_notebook()
cells4 = []

cells4.append(new_markdown_cell("""# 04 Machine Learning Pipeline & Revenue Risk Framework

## Portfolio Project: Customer Retention & Revenue Risk Analytics

### Overview
This notebook builds stratified train/test ML pipelines using `ColumnTransformer`, trains Logistic Regression and Random Forest models, compares metrics (Accuracy, Precision, Recall, F1, ROC-AUC), extracts feature coefficients & importances, quantifies expected monthly revenue at risk, and constructs the 4-quadrant Customer Risk Matrix.

---
"""))

cells4.append(new_code_cell("""import sys, os
import pandas as pd
import numpy as np

sys.path.append(r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics')

from src.data_preprocessing import load_raw_data, clean_customer_data
from src.feature_engineering import engineer_features
from src.churn_model import prepare_data, train_logistic_regression, train_random_forest, predict_churn_probabilities
from src.evaluation import (
    evaluate_model_performance, build_customer_risk_matrix,
    extract_feature_coefficients, extract_random_forest_importance
)

proc_path = r'c:\\Users\\Atul\\Desktop\\project-2\\customer-retention-revenue-risk-analytics\\data\\processed\\cleaned_telco_churn.csv'
df_proc = pd.read_csv(proc_path)
df_eng = engineer_features(df_proc)

print(f"Engineered Dataset Shape: {df_eng.shape}")
"""))

cells4.append(new_markdown_cell("""### Train Machine Learning Models
Splitting 80/20 stratified, training Logistic Regression & Random Forest models.
"""))

cells4.append(new_code_cell("""X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols = prepare_data(df_eng)

# Train Logistic Regression
lr_model = train_logistic_regression(X_train, y_train, preprocessor)
lr_probs = predict_churn_probabilities(lr_model, X_test)
lr_preds = (lr_probs >= 0.5).astype(int)
lr_metrics = evaluate_model_performance(y_test, lr_preds, lr_probs)

# Train Random Forest
rf_model = train_random_forest(X_train, y_train, preprocessor)
rf_probs = predict_churn_probabilities(rf_model, X_test)
rf_preds = (rf_probs >= 0.5).astype(int)
rf_metrics = evaluate_model_performance(y_test, rf_preds, rf_probs)

model_comparison = pd.DataFrame([lr_metrics, rf_metrics], index=['Logistic Regression', 'Random Forest'])
print("--- Model Performance Comparison ---")
display(model_comparison)
"""))

cells4.append(new_markdown_cell("""### Feature Importance & Coefficients Extraction
Examining top drivers of churn.
"""))

cells4.append(new_code_cell("""df_rf_imp = extract_random_forest_importance(rf_model, cat_cols, num_cols)
print("Top 10 Feature Importances (Random Forest):")
display(df_rf_imp.head(10))

df_lr_coef = extract_feature_coefficients(lr_model, cat_cols, num_cols)
print("\\nTop 5 Positive Churn Drivers (Logistic Regression):")
display(df_lr_coef.head(5))
"""))

cells4.append(new_markdown_cell("""### Customer Risk Prioritization Matrix & Revenue Risk Quantification
Generating expected monthly revenue exposure:
$$\\text{ExpectedRevenueAtRisk} = \\text{MonthlyCharges} \\times \\text{ChurnProbability}$$
"""))

cells4.append(new_code_cell("""all_probs = predict_churn_probabilities(rf_model, df_eng[[c for c in cat_cols + num_cols if c in df_eng.columns]])
df_risk = build_customer_risk_matrix(df_eng, all_probs)

print("--- 4-Quadrant Customer Risk Segment Breakdown ---")
risk_summary = df_risk.groupby('RiskSegment').agg(
    CustomerCount=('CustomerID', 'count'),
    AvgMonthlyCharges=('MonthlyCharges', 'mean'),
    TotalMonthlyMRR=('MonthlyCharges', 'sum'),
    TotalRevenueAtRisk=('ExpectedRevenueAtRisk', 'sum'),
    AvgChurnProbability=('ChurnProbability', 'mean')
).reset_index()

display(risk_summary)

print(f"\\nTotal Overall Expected Monthly Revenue at Risk: ${df_risk['ExpectedRevenueAtRisk'].sum():,.2f}")
print("\\nTop 10 Highest Priority Customers for Immediate Retention Intervention:")
display(df_risk[['CustomerID', 'Contract', 'Tenure', 'MonthlyCharges', 'ChurnProbability', 'ExpectedRevenueAtRisk', 'RiskSegment', 'RetentionPriority', 'PriorityScore']].head(10))
"""))

nb4.cells = cells4
with open(os.path.join(nb_dir, '04_machine_learning.ipynb'), 'w', encoding='utf-8') as f:
    nbf.write(nb4, f)
print("Saved 04_machine_learning.ipynb")
