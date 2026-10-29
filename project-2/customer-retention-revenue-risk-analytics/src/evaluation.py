"""
Evaluation and Revenue Risk Prioritization Module for Customer Retention & Revenue Risk Analytics.
Calculates ML metrics, extracts model feature importances/coefficients, computes expected monthly
revenue exposure, and builds the 4-quadrant Customer Risk Matrix & Retention Priority Table.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def evaluate_model_performance(y_true, y_pred, y_prob) -> dict:
    """
    Computes core ML evaluation metrics:
    Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
    """
    metrics = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1 Score': float(f1_score(y_true, y_pred, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(y_true, y_prob))
    }
    return metrics

def get_confusion_matrix_summary(y_true, y_pred) -> dict:
    """Returns confusion matrix breakdown (TN, FP, FN, TP)."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'True Negatives': int(tn),
        'False Positives': int(fp),
        'False Negatives': int(fn),
        'True Positives': int(tp)
    }

def extract_feature_coefficients(model_pipeline, feature_names_cat: list, feature_names_num: list) -> pd.DataFrame:
    """
    Extracts feature coefficients for Logistic Regression pipeline.
    """
    preprocessor = model_pipeline.named_steps['preprocessor']
    classifier = model_pipeline.named_steps['classifier']
    
    # Extract OHE column names
    ohe = preprocessor.named_transformers_['cat']
    encoded_cat_features = ohe.get_feature_names_out(feature_names_cat).tolist()
    
    all_feature_names = encoded_cat_features + feature_names_num
    coefficients = classifier.coef_[0]
    
    df_coef = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefficients,
        'OddsRatio': np.exp(coefficients)
    }).sort_values(by='Coefficient', ascending=False).reset_index(drop=True)
    
    return df_coef

def extract_random_forest_importance(model_pipeline, feature_names_cat: list, feature_names_num: list) -> pd.DataFrame:
    """
    Extracts feature importances for Random Forest pipeline.
    """
    preprocessor = model_pipeline.named_steps['preprocessor']
    classifier = model_pipeline.named_steps['classifier']
    
    ohe = preprocessor.named_transformers_['cat']
    encoded_cat_features = ohe.get_feature_names_out(feature_names_cat).tolist()
    
    all_feature_names = encoded_cat_features + feature_names_num
    importances = classifier.feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return df_imp

def assign_risk_category(prob: float) -> str:
    """Categorizes churn probability into risk buckets."""
    if prob < 0.30:
        return 'Low Risk'
    elif prob <= 0.60:
        return 'Medium Risk'
    else:
        return 'High Risk'

def build_customer_risk_matrix(df: pd.DataFrame, churn_probs: np.ndarray) -> pd.DataFrame:
    """
    Generates Revenue Risk Exposure and Risk Prioritization Matrix.
    
    Calculations:
    - Expected Revenue at Risk = MonthlyCharges * ChurnProbability
    - Risk Segment (4 Quadrants):
        1. High Risk + High Value (ChurnProb > 0.5 & MonthlyCharges > median)
        2. High Risk + Low Value (ChurnProb > 0.5 & MonthlyCharges <= median)
        3. Low Risk + High Value (ChurnProb <= 0.5 & MonthlyCharges > median)
        4. Low Risk + Low Value (ChurnProb <= 0.5 & MonthlyCharges <= median)
    - PriorityScore = ChurnProbability * (MonthlyCharges / max_MonthlyCharges)
    """
    df_matrix = df.copy()
    df_matrix['ChurnProbability'] = churn_probs.round(4)
    df_matrix['RiskCategory'] = df_matrix['ChurnProbability'].apply(assign_risk_category)
    df_matrix['ExpectedRevenueAtRisk'] = (df_matrix['MonthlyCharges'] * df_matrix['ChurnProbability']).round(2)
    
    median_charges = df_matrix['MonthlyCharges'].median()
    max_charges = df_matrix['MonthlyCharges'].max()
    
    df_matrix['PriorityScore'] = (df_matrix['ChurnProbability'] * (df_matrix['MonthlyCharges'] / max_charges)).round(4)
    
    def classify_segment(row):
        prob = row['ChurnProbability']
        val = row['MonthlyCharges']
        is_high_risk = prob >= 0.50
        is_high_value = val >= median_charges
        
        if is_high_risk and is_high_value:
            return 'High Risk + High Value'
        elif is_high_risk and not is_high_value:
            return 'High Risk + Low Value'
        elif not is_high_risk and is_high_value:
            return 'Low Risk + High Value'
        else:
            return 'Low Risk + Low Value'
            
    def assign_retention_priority(segment):
        priority_map = {
            'High Risk + High Value': 'Tier 1 - Immediate Intervention',
            'High Risk + Low Value': 'Tier 2 - Automated Campaign',
            'Low Risk + High Value': 'Tier 3 - Loyalty & VIP Lock-in',
            'Low Risk + Low Value': 'Tier 4 - Standard Engagement'
        }
        return priority_map.get(segment, 'Tier 4 - Standard Engagement')
        
    df_matrix['RiskSegment'] = df_matrix.apply(classify_segment, axis=1)
    df_matrix['RetentionPriority'] = df_matrix['RiskSegment'].apply(assign_retention_priority)
    
    # Sort from highest priority score to lowest
    df_matrix.sort_values(by='PriorityScore', ascending=False, inplace=True)
    df_matrix.reset_index(drop=True, inplace=True)
    
    return df_matrix
