# Telco Customer Churn Dataset Documentation

## Overview
This directory contains the dataset used for customer churn prediction and business insight synthesis.

- **Primary Source**: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Record Count**: 7,032 accounts (after cleaning missing TotalCharges)
- **Feature Count**: 21 raw columns (demographics, services, contract, billing, churn target)
- **Target Label**: `Churn` (Binary: `Yes` or `No`)

## Directory Layout
- `raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`: Immutable raw dataset.
- `processed/train.csv`: Training split (70%, 4,922 records).
- `processed/val.csv`: Validation split (15%, 1,055 records).
- `processed/test.csv`: Holdout test split (15%, 1,055 records).

## Provenance & Data Cleaning Rules
1. `TotalCharges` contains blank space strings (`' '`) for zero-tenure accounts. These are coerced to float and imputed using `MonthlyCharges * tenure`.
2. All feature transformers (StandardScaler, OneHotEncoder) are fit strictly on `train.csv` to eliminate data leakage.
