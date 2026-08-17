"""
Unit tests for model training and evaluation module.
"""

import pandas as pd
import numpy as np
import pytest
from src.data.make_dataset import generate_telco_churn_dataset, clean_dataset
from src.features.build_features import prepare_features_and_target
from src.models.train_model import train_and_compare_models, evaluate_model


def test_train_and_compare_models():
    df_train = clean_dataset(generate_telco_churn_dataset(n_samples=200, seed=42))
    df_val = clean_dataset(generate_telco_churn_dataset(n_samples=50, seed=100))
    
    X_train, y_train, preprocessor, feature_names = prepare_features_and_target(df_train, is_train=True)
    X_val, y_val, _, _ = prepare_features_and_target(df_val, preprocessor=preprocessor, is_train=False)
    
    models, summary_df, results = train_and_compare_models(X_train, y_train, X_val, y_val)
    
    assert "XGBoost" in models
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert len(summary_df) == 4
    assert "Recall" in summary_df.columns
    assert "ROC-AUC" in summary_df.columns
