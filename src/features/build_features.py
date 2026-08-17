"""
Feature engineering and preprocessing pipeline module for Telco Customer Churn.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers tenure buckets, interaction terms, and binary flag features.
    """
    df = df.copy()
    
    # 1. Tenure Buckets
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 48, 100],
        labels=['0-12m', '13-24m', '25-48m', '49+m']
    ).astype(str)
    
    # 2. Charges per tenure ratio (monthly intensity)
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1.0)
    
    # 3. High Monthly Charges Flag (greater than 75th percentile ~ $80)
    df['high_monthly_charges'] = (df['MonthlyCharges'] > 80.0).astype(int)
    
    # 4. Total Add-on Services Count
    addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_addons'] = 0
    for col in addon_cols:
        if col in df.columns:
            df['total_addons'] += (df[col] == 'Yes').astype(int)
            
    return df


def get_feature_columns():
    """
    Returns numeric, categorical, and target feature column names.
    """
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_tenure', 'total_addons']
    cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure_group', 'high_monthly_charges'
    ]
    return num_cols, cat_cols


def create_preprocessor(num_cols, cat_cols):
    """
    Creates a scikit-learn ColumnTransformer for scaling numeric features
    and One-Hot Encoding categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def prepare_features_and_target(df: pd.DataFrame, preprocessor=None, is_train: bool = True):
    """
    Applies feature engineering and preprocessor transformation to input DataFrame.
    """
    df_engineered = engineer_features(df)
    num_cols, cat_cols = get_feature_columns()
    
    y = None
    if 'Churn' in df_engineered.columns:
        y = (df_engineered['Churn'] == 'Yes').astype(int).values
        
    X_raw = df_engineered[num_cols + cat_cols]
    
    if is_train:
        preprocessor = create_preprocessor(num_cols, cat_cols)
        X_trans = preprocessor.fit_transform(X_raw)
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor must be provided for evaluation/test transform.")
        X_trans = preprocessor.transform(X_raw)
        
    # Get feature names after OneHotEncoding
    ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(ohe_cols)
    
    return X_trans, y, preprocessor, feature_names
