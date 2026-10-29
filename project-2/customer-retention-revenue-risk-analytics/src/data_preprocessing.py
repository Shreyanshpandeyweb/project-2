"""
Data Preprocessing Module for Customer Retention & Revenue Risk Analytics.
Provides robust functions for loading raw customer data, handling missing/blank values,
converting data types, checking duplicates, and saving cleaned dataset.
"""

import pandas as pd
import numpy as np

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df

def clean_customer_data(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Cleans customer churn data:
    1. Standardizes column names (customerID -> CustomerID, tenure -> Tenure).
    2. Converts TotalCharges to numeric, coercing blank spaces ' ' to NaN.
    3. Imputes missing TotalCharges (for tenure=0, TotalCharges = 0; otherwise tenure * MonthlyCharges).
    4. Encodes Churn target ('Yes': 1, 'No': 0).
    5. Verifies data types and duplicate records.
    
    Returns cleaned DataFrame and cleaning statistics dictionary.
    """
    df_clean = df.copy()
    
    # 1. Standardize customerID column name if needed
    if 'customerID' in df_clean.columns:
        df_clean.rename(columns={'customerID': 'CustomerID'}, inplace=True)
        
    initial_rows = len(df_clean)
    initial_cols = len(df_clean.columns)
    
    # 2. Check duplicates
    duplicate_ids = df_clean['CustomerID'].duplicated().sum()
    if duplicate_ids > 0:
        df_clean.drop_duplicates(subset=['CustomerID'], keep='first', inplace=True)
        
    # 3. Clean TotalCharges
    # Replace whitespace strings with NaN and convert to float
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = df_clean['TotalCharges'].astype(str).str.strip()
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Identify missing TotalCharges count
        missing_total_charges = df_clean['TotalCharges'].isna().sum()
        
        # Impute TotalCharges: if tenure is 0, set to 0; else set to tenure * MonthlyCharges
        mask_missing = df_clean['TotalCharges'].isna()
        if 'tenure' in df_clean.columns:
            tenure_col = 'tenure'
        else:
            tenure_col = 'Tenure'
            
        df_clean.loc[mask_missing & (df_clean[tenure_col] == 0), 'TotalCharges'] = 0.0
        df_clean.loc[mask_missing & (df_clean[tenure_col] > 0), 'TotalCharges'] = (
            df_clean.loc[mask_missing & (df_clean[tenure_col] > 0), tenure_col] * 
            df_clean.loc[mask_missing & (df_clean[tenure_col] > 0), 'MonthlyCharges']
        )
    else:
        missing_total_charges = 0

    # 4. Standardize tenure column name if needed (tenure -> Tenure)
    if 'tenure' in df_clean.columns:
        df_clean.rename(columns={'tenure': 'Tenure'}, inplace=True)
        
    # 5. Target Variable Encoding
    if 'Churn' in df_clean.columns:
        df_clean['Churn_Numeric'] = df_clean['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    cleaning_stats = {
        'initial_rows': initial_rows,
        'initial_cols': initial_cols,
        'final_rows': len(df_clean),
        'duplicates_removed': duplicate_ids,
        'missing_total_charges_imputed': missing_total_charges,
        'total_churned': int((df_clean['Churn_Numeric'] == 1).sum()) if 'Churn_Numeric' in df_clean.columns else 0,
        'overall_churn_rate': float((df_clean['Churn_Numeric'] == 1).mean()) if 'Churn_Numeric' in df_clean.columns else 0.0
    }
    
    return df_clean, cleaning_stats

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned DataFrame to CSV."""
    df.to_csv(output_path, index=False)
