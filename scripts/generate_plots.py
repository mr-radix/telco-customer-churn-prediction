"""
Script to generate and save publication-quality visualization figures for README.md.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.make_dataset import load_and_prepare_data
from src.features.build_features import prepare_features_and_target


def generate_all_figures():
    os.makedirs("reports/figures", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Load dataset
    df = load_and_prepare_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", "data/processed")[0]

    
    # 1. Churn Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Churn", palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Customer Churn Target Distribution (26.5% Churn Rate)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Churn Status", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/01_churn_distribution.png", dpi=300)
    plt.close()
    
    # 2. Tenure KDE Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, common_norm=False, palette=["#2ecc71", "#e74c3c"], alpha=0.5, ax=ax)
    ax.set_title("Customer Tenure Distribution by Churn Status", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Tenure (Months)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    plt.tight_layout()
    plt.savefig("reports/figures/02_tenure_vs_churn.png", dpi=300)
    plt.close()
    
    # 3. Contract Type Bar Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    contract_churn.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Contract Type", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Churn", loc="upper right")
    plt.tight_layout()
    plt.savefig("reports/figures/03_contract_type_churn.png", dpi=300)
    plt.close()

    # 4. Feature Importance & Confusion Matrix
    import joblib, pickle
    model_path = "models/best_model.pkl"
    prep_path = "models/preprocessor.pkl"
    feat_path = "models/feature_names.pkl"
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        try:
            preprocessor = joblib.load(prep_path)
        except Exception:
            with open(prep_path, "rb") as f:
                preprocessor = pickle.load(f)
        with open(feat_path, "rb") as f:
            feature_names = pickle.load(f)
            
        df_test = pd.read_csv("data/processed/test.csv")
        X_test, y_test, _, _ = prepare_features_and_target(df_test, preprocessor=preprocessor, is_train=False)

        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        ax.set_title("Gradient Boosting Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        plt.tight_layout()
        plt.savefig("reports/figures/05_confusion_matrix.png", dpi=300)
        plt.close()

        # Permutation Feature Importance
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        perm_sorted_idx = result.importances_mean.argsort()[-10:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(perm_sorted_idx)), result.importances_mean[perm_sorted_idx], color="#3498db")
        ax.set_yticks(range(len(perm_sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in perm_sorted_idx])
        ax.set_title("Top 10 Permutation Feature Importances", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Mean Importance Decrease", fontsize=12)
        plt.tight_layout()
        plt.savefig("reports/figures/04_feature_importance.png", dpi=300)
        plt.close()

    print("All 5 publication figures successfully generated in 'reports/figures/'!")

if __name__ == "__main__":
    generate_all_figures()
