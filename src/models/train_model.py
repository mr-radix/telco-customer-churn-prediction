"""
Model training, evaluation, and serialization module for Telco Customer Churn.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGBOOST = False


def evaluate_model(model, X, y, model_name: str = "Model") -> dict:
    """
    Evaluates a model's classification metrics.
    """
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = preds

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y, preds)
    
    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Confusion_Matrix": cm
    }


def train_and_compare_models(X_train, y_train, X_val, y_val) -> tuple[dict, pd.DataFrame]:
    """
    Trains Majority Baseline, Logistic Regression, Random Forest, and XGBoost models.
    Returns trained models dictionary and comparison summary DataFrame.
    """
    models = {}
    results = []
    
    # 1. Baseline Majority Predictor
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    models["Baseline (Majority)"] = dummy
    results.append(evaluate_model(dummy, X_val, y_val, "Baseline (Majority)"))
    
    # 2. Logistic Regression Baseline
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr
    results.append(evaluate_model(lr, X_val, y_val, "Logistic Regression"))
    
    # 3. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    results.append(evaluate_model(rf, X_val, y_val, "Random Forest"))
    
    # 4. Gradient Boosting Classifier (XGBoost / HistGradientBoosting)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    if HAS_XGBOOST:
        gb = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss"
        )
        gb_name = "XGBoost"
    else:
        gb = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42
        )
        gb_name = "HistGradientBoosting"
        
    gb.fit(X_train, y_train)
    models["XGBoost"] = gb
    results.append(evaluate_model(gb, X_val, y_val, gb_name))
    
    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k != "Confusion_Matrix"} for r in results
    ])
    
    return models, df_results, results


import joblib

def save_artifacts(model, preprocessor, feature_names, models_dir: str = "models"):
    """
    Serializes trained model, preprocessor, and feature names to models directory.
    """
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    joblib.dump(model, os.path.join(models_dir, "best_model.joblib"))
    with open(os.path.join(models_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(models_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    print(f"Artifacts successfully saved to '{models_dir}/'")


if __name__ == "__main__":
    from src.data.make_dataset import load_and_prepare_data
    from src.features.build_features import prepare_features_and_target
    
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    proc_dir = "data/processed"
    train_df, val_df, test_df = load_and_prepare_data(raw_path, proc_dir)
    
    X_train, y_train, preprocessor, feature_names = prepare_features_and_target(train_df, is_train=True)
    X_val, y_val, _, _ = prepare_features_and_target(val_df, preprocessor=preprocessor, is_train=False)
    
    models_dict, summary_df, results = train_and_compare_models(X_train, y_train, X_val, y_val)
    print("\nModel Training Complete! Validation Set Comparison:")
    print(summary_df)
    
    best_model = models_dict["XGBoost"]
    save_artifacts(best_model, preprocessor, feature_names, models_dir="models")
