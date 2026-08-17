"""
Unit tests for data acquisition and cleaning module.
"""

import pandas as pd
import numpy as np
import pytest
from src.data.make_dataset import generate_telco_churn_dataset, clean_dataset


def test_generate_telco_churn_dataset():
    df = generate_telco_churn_dataset(n_samples=100, seed=42)
    assert len(df) == 100
    assert "Churn" in df.columns
    assert "TotalCharges" in df.columns
    assert set(df["Churn"].unique()).issubset({"Yes", "No"})


def test_clean_dataset():
    df = generate_telco_churn_dataset(n_samples=50, seed=42)
    df_cleaned = clean_dataset(df)
    assert df_cleaned["TotalCharges"].isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(df_cleaned["TotalCharges"])
