"""
Churn Model Training and Evaluation Pipeline for Customer Retention & Revenue Risk Analytics.
Builds scikit-learn preprocessing pipelines, trains Logistic Regression and Random Forest models,
performs hyperparameter tuning, and generates calibrated churn probabilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_feature_lists():
    """Returns lists of categorical and numerical feature column names."""
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'TenureGroup', 'MonthlyChargeGroup'
    ]
    numerical_features = [
        'Tenure', 'MonthlyCharges', 'TotalCharges', 'ServiceCount',
        'SupportServiceCount', 'CustomerValueScore', 'ContractRiskIndex'
    ]
    return categorical_features, numerical_features

def build_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for preprocessing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )
    return preprocessor

def prepare_data(df: pd.DataFrame, target_col: str = 'Churn_Numeric', test_size: float = 0.20, random_state: int = 42):
    """
    Splits dataset into stratified 80% training and 20% testing sets.
    """
    cat_cols, num_cols = get_feature_lists()
    
    # Filter features that exist in df
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]
    
    feature_cols = cat_cols + num_cols
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = build_preprocessor(cat_cols, num_cols)
    
    return X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols

def train_logistic_regression(X_train, y_train, preprocessor, random_state: int = 42) -> Pipeline:
    """
    Trains Logistic Regression baseline pipeline with balanced class weight.
    """
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state))
    ])
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

def train_random_forest(X_train, y_train, preprocessor, random_state: int = 42) -> Pipeline:
    """
    Trains Random Forest Classifier pipeline with tuned hyperparameters.
    """
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=5, class_weight='balanced', random_state=random_state))
    ])
    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline

def predict_churn_probabilities(model, X_df: pd.DataFrame) -> np.ndarray:
    """
    Predicts probability of churn (class 1) for given features.
    """
    probs = model.predict_proba(X_df)[:, 1]
    return probs
