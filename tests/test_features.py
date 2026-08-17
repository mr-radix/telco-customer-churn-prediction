"""
Unit tests for feature engineering and preprocessor module.
"""

import pandas as pd
import numpy as np
import pytest
from src.data.make_dataset import generate_telco_churn_dataset, clean_dataset
from src.features.build_features import engineer_features, prepare_features_and_target


def test_engineer_features():
    df = clean_dataset(generate_telco_churn_dataset(n_samples=50, seed=42))
    df_eng = engineer_features(df)
    assert "tenure_group" in df_eng.columns
    assert "charges_per_tenure" in df_eng.columns
    assert "high_monthly_charges" in df_eng.columns
    assert "total_addons" in df_eng.columns


def test_prepare_features_and_target():
    df = clean_dataset(generate_telco_churn_dataset(n_samples=50, seed=42))
    X_trans, y, preprocessor, feature_names = prepare_features_and_target(df, is_train=True)
    assert X_trans.shape[0] == 50
    assert len(y) == 50
    assert X_trans.shape[1] == len(feature_names)
    assert not np.isnan(X_trans).any()
