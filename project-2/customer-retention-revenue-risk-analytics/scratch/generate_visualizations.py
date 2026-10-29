"""
Script to generate publication-quality visualizations for Customer Retention & Revenue Risk Analytics.
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.append(r'c:\Users\Atul\Desktop\project-2\customer-retention-revenue-risk-analytics')

from src.data_preprocessing import load_raw_data, clean_customer_data
from src.feature_engineering import engineer_features
from src.churn_model import prepare_data, train_logistic_regression, train_random_forest, predict_churn_probabilities
from src.evaluation import build_customer_risk_matrix, extract_random_forest_importance

# Plot styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

PRIMARY_COLOR = '#1E3A8A'  # Deep Blue
SECONDARY_COLOR = '#EF4444' # Vibrant Red
ACCENT_COLOR = '#10B981'   # Emerald Green
NEUTRAL_COLOR = '#64748B'  # Slate Gray

base_dir = r'c:\Users\Atul\Desktop\project-2\customer-retention-revenue-risk-analytics'
proc_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_telco_churn.csv')
risk_path = os.path.join(base_dir, 'data', 'processed', 'churn_predictions_risk_matrix.csv')

df_proc = pd.read_csv(proc_path)
df_risk = pd.read_csv(risk_path)

churn_dir = os.path.join(base_dir, 'visualizations', 'churn_analysis')
rev_dir = os.path.join(base_dir, 'visualizations', 'revenue_risk')
ml_dir = os.path.join(base_dir, 'visualizations', 'model_results')

# 1. Churn Distribution
fig, ax = plt.subplots(figsize=(7, 5))
churn_counts = df_proc['Churn'].value_counts()
bars = ax.bar(churn_counts.index, churn_counts.values, color=['#3B82F6', '#EF4444'], width=0.5)
ax.set_title('Overall Customer Churn Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Churn Status', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df_proc)) * 100
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(churn_dir, 'churn_distribution.png'), dpi=300)
plt.close()

# 2. Churn by Contract Type
fig, ax = plt.subplots(figsize=(8, 5))
contract_churn = df_proc.groupby('Contract')['Churn_Numeric'].mean().reset_index()
contract_churn['ChurnRatePct'] = contract_churn['Churn_Numeric'] * 100
bars = ax.bar(contract_churn['Contract'], contract_churn['ChurnRatePct'], color=['#EF4444', '#F59E0B', '#10B981'], width=0.5)
ax.set_title('Churn Rate by Contract Type (%)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Churn Rate (%)', fontsize=12)
ax.set_ylim(0, 50)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(churn_dir, 'churn_by_contract.png'), dpi=300)
plt.close()

# 3. Churn by Internet Service
fig, ax = plt.subplots(figsize=(8, 5))
net_churn = df_proc.groupby('InternetService')['Churn_Numeric'].mean().reset_index()
net_churn['ChurnRatePct'] = net_churn['Churn_Numeric'] * 100
bars = ax.bar(net_churn['InternetService'], net_churn['ChurnRatePct'], color=['#3B82F6', '#EF4444', '#10B981'], width=0.5)
ax.set_title('Churn Rate by Internet Service Type (%)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Churn Rate (%)', fontsize=12)
ax.set_ylim(0, 50)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(churn_dir, 'churn_by_internet.png'), dpi=300)
plt.close()

# 4. Churn by Payment Method
fig, ax = plt.subplots(figsize=(10, 5))
pay_churn = df_proc.groupby('PaymentMethod')['Churn_Numeric'].mean().reset_index()
pay_churn['ChurnRatePct'] = pay_churn['Churn_Numeric'] * 100
pay_churn.sort_values(by='ChurnRatePct', ascending=False, inplace=True)
bars = ax.barh(pay_churn['PaymentMethod'], pay_churn['ChurnRatePct'], color=['#EF4444', '#F59E0B', '#3B82F6', '#10B981'], height=0.5)
ax.set_title('Churn Rate by Payment Method (%)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Churn Rate (%)', fontsize=12)
ax.set_xlim(0, 50)
for bar in bars:
    xval = bar.get_width()
    ax.text(xval + 1, bar.get_y() + bar.get_height()/2.0, f'{xval:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(churn_dir, 'churn_by_payment.png'), dpi=300)
plt.close()

# 5. Churn by Tenure Group
fig, ax = plt.subplots(figsize=(9, 5))
tenure_order = ['0-6 months', '7-12 months', '13-24 months', '25-48 months', '49+ months']
tenure_churn = df_proc.groupby('TenureGroup', observed=False)['Churn_Numeric'].mean().reindex(tenure_order).reset_index()
tenure_churn['ChurnRatePct'] = tenure_churn['Churn_Numeric'] * 100
bars = ax.bar(tenure_churn['TenureGroup'], tenure_churn['ChurnRatePct'], color='#1E3A8A', width=0.5)
ax.set_title('Churn Rate across Customer Lifecycle Groups', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Churn Rate (%)', fontsize=12)
ax.set_ylim(0, 60)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(churn_dir, 'churn_by_tenure.png'), dpi=300)
plt.close()

# 6. Churn by Monthly Charges Distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(data=df_proc, x='MonthlyCharges', hue='Churn', palette={'No': '#10B981', 'Yes': '#EF4444'}, fill=True, common_norm=False, alpha=0.4, ax=ax)
ax.set_title('Monthly Charges Density by Churn Status', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Monthly Charges ($)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(churn_dir, 'churn_by_monthly_charges.png'), dpi=300)
plt.close()

# 7. Churn by Demographics (Senior Citizen, Partner, Dependents)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
demo_cols = [('SeniorCitizen', 'Senior Citizen Status'), ('Partner', 'Has Partner'), ('Dependents', 'Has Dependents')]
for idx, (col, label) in enumerate(demo_cols):
    res = df_proc.groupby(col)['Churn_Numeric'].mean() * 100
    res.index = ['No', 'Yes'] if col == 'SeniorCitizen' else res.index
    axes[idx].bar(res.index, res.values, color=['#3B82F6', '#EF4444'], width=0.4)
    axes[idx].set_title(f'Churn by {label}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Churn Rate (%)')
    axes[idx].set_ylim(0, 50)
    for bar in axes[idx].patches:
        yval = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(churn_dir, 'churn_by_demographics.png'), dpi=300)
plt.close()

# 8. Churn by Support Services
fig, ax = plt.subplots(figsize=(9, 5))
supp_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
supp_rates = []
for c in supp_cols:
    rate_yes = df_proc[df_proc[c] == 'Yes']['Churn_Numeric'].mean() * 100
    rate_no = df_proc[df_proc[c] == 'No']['Churn_Numeric'].mean() * 100
    supp_rates.append({'Service': c, 'Subscribed (Yes)': rate_yes, 'Not Subscribed (No)': rate_no})
df_supp_plot = pd.DataFrame(supp_rates).set_index('Service')
df_supp_plot.plot(kind='bar', color=['#10B981', '#EF4444'], ax=ax, width=0.6)
ax.set_title('Churn Rate Impact of Tech Support Services (%)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Churn Rate (%)', fontsize=12)
ax.set_ylim(0, 50)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(churn_dir, 'churn_by_support_services.png'), dpi=300)
plt.close()

# --- REVENUE RISK VISUALIZATIONS ---

# 9. Revenue at Risk by Contract Segment
fig, ax = plt.subplots(figsize=(8, 5))
rev_segment = df_risk.groupby('Contract')['ExpectedRevenueAtRisk'].sum().reset_index()
bars = ax.bar(rev_segment['Contract'], rev_segment['ExpectedRevenueAtRisk'], color=['#EF4444', '#F59E0B', '#10B981'], width=0.5)
ax.set_title('Total Expected Monthly Revenue at Risk by Contract ($)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Monthly Revenue at Risk ($)', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2000, f'${yval:,.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(rev_dir, 'revenue_at_risk_by_segment.png'), dpi=300)
plt.close()

# 10. Customer Risk Matrix 4-Quadrant Scatter Plot
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(
    data=df_risk, x='MonthlyCharges', y='ChurnProbability', hue='RiskSegment',
    palette={'High Risk + High Value': '#EF4444', 'High Risk + Low Value': '#F59E0B',
             'Low Risk + High Value': '#3B82F6', 'Low Risk + Low Value': '#10B981'},
    alpha=0.6, s=30, ax=ax
)
med_val = df_risk['MonthlyCharges'].median()
ax.axhline(0.50, color='black', linestyle='--', linewidth=1.2, label='Churn Threshold (0.50)')
ax.axvline(med_val, color='black', linestyle=':', linewidth=1.2, label=f'Median Monthly Charge (${med_val:.1f})')
ax.set_title('Customer Risk Matrix (Churn Probability vs Monthly Revenue)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Monthly Charges ($)', fontsize=12)
ax.set_ylabel('Predicted Churn Probability', fontsize=12)
ax.legend(title='Risk Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(rev_dir, 'customer_risk_matrix.png'), dpi=300)
plt.close()

# 11. Top High Value at Risk Customers Distribution
fig, ax = plt.subplots(figsize=(9, 5))
segment_rev = df_risk.groupby('RiskSegment')['ExpectedRevenueAtRisk'].sum().reset_index()
segment_rev.sort_values(by='ExpectedRevenueAtRisk', ascending=False, inplace=True)
bars = ax.barh(segment_rev['RiskSegment'], segment_rev['ExpectedRevenueAtRisk'], color=['#EF4444', '#F59E0B', '#3B82F6', '#10B981'], height=0.5)
ax.set_title('Total Expected Revenue Exposure by Risk Segment ($)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Expected Monthly Revenue at Risk ($)', fontsize=12)
for bar in bars:
    xval = bar.get_width()
    ax.text(xval + 1000, bar.get_y() + bar.get_height()/2.0, f'${xval:,.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(rev_dir, 'top_high_value_at_risk_customers.png'), dpi=300)
plt.close()

# --- MODEL RESULTS VISUALIZATIONS ---

# Train models to fetch ROC and Feature Importances
X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols = prepare_data(df_proc)
lr_model = train_logistic_regression(X_train, y_train, preprocessor)
rf_model = train_random_forest(X_train, y_train, preprocessor)

lr_probs = lr_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# 12. ROC Curve Comparison
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_lr, tpr_lr, color='#3B82F6', lw=2, label=f'Logistic Regression (AUC = {auc_lr:.4f})')
ax.plot(fpr_rf, tpr_rf, color='#10B981', lw=2, label=f'Random Forest (AUC = {auc_rf:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
ax.set_title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(ml_dir, 'roc_curve_comparison.png'), dpi=300)
plt.close()

# 13. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
lr_preds = (lr_probs >= 0.5).astype(int)
rf_preds = (rf_probs >= 0.5).astype(int)

cm_lr = confusion_matrix(y_test, lr_preds)
cm_rf = confusion_matrix(y_test, rf_preds)

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_xticklabels(['No Churn (0)', 'Churn (1)'])
axes[0].set_yticklabels(['No Churn (0)', 'Churn (1)'])

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Random Forest Confusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
axes[1].set_xticklabels(['No Churn (0)', 'Churn (1)'])
axes[1].set_yticklabels(['No Churn (0)', 'Churn (1)'])

plt.tight_layout()
plt.savefig(os.path.join(ml_dir, 'confusion_matrices.png'), dpi=300)
plt.close()

# 14. Feature Importances (Random Forest)
df_imp = extract_random_forest_importance(rf_model, cat_cols, num_cols).head(15)
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(df_imp['Feature'], df_imp['Importance'], color='#1E3A8A', height=0.6)
ax.invert_yaxis()
ax.set_title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Feature Importance Score', fontsize=12)
for bar in bars:
    xval = bar.get_width()
    ax.text(xval + 0.002, bar.get_y() + bar.get_height()/2.0, f'{xval:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(ml_dir, 'feature_importance.png'), dpi=300)
plt.close()

# 15. Churn Probability Distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df_risk['ChurnProbability'], bins=30, kde=True, color='#1E3A8A', ax=ax)
ax.axvline(0.30, color='#10B981', linestyle='--', linewidth=1.5, label='Low/Medium Threshold (0.30)')
ax.axvline(0.60, color='#EF4444', linestyle='--', linewidth=1.5, label='Medium/High Threshold (0.60)')
ax.set_title('Predicted Churn Probability Distribution Across All Customers', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Churn Probability', fontsize=12)
ax.set_ylabel('Customer Count', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(ml_dir, 'churn_probability_distribution.png'), dpi=300)
plt.close()

# 16. Model Performance Comparison
fig, ax = plt.subplots(figsize=(9, 5))
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Logistic Regression': [0.7381, 0.5042, 0.7995, 0.6184, 0.8449],
    'Random Forest': [0.7601, 0.5320, 0.7995, 0.6389, 0.8447]
}).set_index('Metric')

metrics_df.plot(kind='bar', color=['#3B82F6', '#10B981'], ax=ax, width=0.5)
ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
ax.set_ylim(0, 1.0)
plt.xticks(rotation=0)
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h / 2.0), ha='center', va='center', color='white', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(ml_dir, 'model_performance_comparison.png'), dpi=300)
plt.close()

print('ALL 16 VISUALIZATIONS GENERATED SUCCESSFULLY!')
