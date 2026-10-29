"""
Feature Engineering Module for Customer Retention & Revenue Risk Analytics.
Creates domain-specific features for exploratory analysis, statistical testing,
and machine learning modeling without causing target leakage.
"""

import pandas as pd
import numpy as np

def create_tenure_group(df: pd.DataFrame) -> pd.Series:
    """
    Categorizes tenure into customer lifecycle groups:
    - 0-6 months
    - 7-12 months
    - 13-24 months
    - 25-48 months
    - 49+ months
    """
    bins = [-1, 6, 12, 24, 48, 120]
    labels = ['0-6 months', '7-12 months', '13-24 months', '25-48 months', '49+ months']
    return pd.cut(df['Tenure'], bins=bins, labels=labels)

def create_monthly_charge_group(df: pd.DataFrame) -> pd.Series:
    """
    Categorizes MonthlyCharges into Revenue Groups:
    - Low Revenue: < $35
    - Medium Revenue: $35 - $75
    - High Revenue: > $75
    """
    bins = [-1, 35, 75, 500]
    labels = ['Low Revenue', 'Medium Revenue', 'High Revenue']
    return pd.cut(df['MonthlyCharges'], bins=bins, labels=labels)

def calculate_service_count(df: pd.DataFrame) -> pd.Series:
    """
    Counts total active tech/add-on services subscribed by each customer.
    Services considered: PhoneService, MultipleLines, InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
    """
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    count = pd.Series(0, index=df.index)
    
    for col in service_cols:
        if col in df.columns:
            # Count positive subscription instances
            count += df[col].apply(lambda val: 1 if str(val).strip().lower() in ['yes', 'dsl', 'fiber optic'] else 0)
            
    return count

def calculate_support_service_count(df: pd.DataFrame) -> pd.Series:
    """
    Counts specialized technical support add-ons subscribed:
    - OnlineSecurity
    - OnlineBackup
    - DeviceProtection
    - TechSupport
    """
    support_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    count = pd.Series(0, index=df.index)
    
    for col in support_cols:
        if col in df.columns:
            count += df[col].apply(lambda val: 1 if str(val).strip().lower() == 'yes' else 0)
            
    return count

def calculate_customer_value_score(df: pd.DataFrame) -> pd.Series:
    """
    Generates a normalized Customer Value Score (0.0 to 1.0)
    combining MonthlyCharges (50% weight) and Tenure (50% weight).
    """
    m_min, m_max = df['MonthlyCharges'].min(), df['MonthlyCharges'].max()
    t_min, t_max = df['Tenure'].min(), df['Tenure'].max()
    
    norm_monthly = (df['MonthlyCharges'] - m_min) / (m_max - m_min + 1e-6)
    norm_tenure = (df['Tenure'] - t_min) / (t_max - t_min + 1e-6)
    
    value_score = 0.5 * norm_monthly + 0.5 * norm_tenure
    return value_score.round(4)

def calculate_contract_risk_index(df: pd.DataFrame) -> pd.Series:
    """
    Maps contract flexibility to relative churn risk index:
    - Month-to-month: 1.0 (Highest risk)
    - One year: 0.5 (Moderate risk)
    - Two year: 0.1 (Lowest risk)
    """
    contract_map = {
        'Month-to-month': 1.0,
        'One year': 0.5,
        'Two year': 0.1
    }
    return df['Contract'].map(contract_map).fillna(1.0)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering functions to the dataset.
    Returns DataFrame populated with engineered columns.
    """
    df_engineered = df.copy()
    df_engineered['TenureGroup'] = create_tenure_group(df_engineered)
    df_engineered['MonthlyChargeGroup'] = create_monthly_charge_group(df_engineered)
    df_engineered['ServiceCount'] = calculate_service_count(df_engineered)
    df_engineered['SupportServiceCount'] = calculate_support_service_count(df_engineered)
    df_engineered['CustomerValueScore'] = calculate_customer_value_score(df_engineered)
    df_engineered['ContractRiskIndex'] = calculate_contract_risk_index(df_engineered)
    
    return df_engineered
