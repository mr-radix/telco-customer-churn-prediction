"""
Visualization utility module for EDA, model evaluation, and SHAP charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def set_plot_style():
    """
    Sets clean publication-quality aesthetic for plots.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16
    })


def plot_churn_distribution(df, target_col: str = "Churn", save_path: str = None):
    """
    Plots target variable distribution and percentage breakdown.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[target_col].value_counts()
    pcts = df[target_col].value_counts(normalize=True) * 100
    
    bars = ax.bar(counts.index, counts.values, color=["#2b5c8f", "#d9534f"])
    ax.set_title("Customer Churn Class Distribution", fontweight="bold")
    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Count")
    
    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (height * 0.01),
            f"{int(height)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_feature_vs_churn(df, feature: str, target_col: str = "Churn", title: str = None, save_path: str = None):
    """
    Plots stacked or grouped bar chart of a categorical feature against churn.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ct = pd.crosstab(df[feature], df[target_col], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, color=["#2b5c8f", "#d9534f"], ax=ax)
    
    ax.set_title(title or f"Churn Rate by {feature}", fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Churn", labels=["No (Retained)", "Yes (Churned)"])
    plt.xticks(rotation=0 if len(ct) <= 4 else 30, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_confusion_matrix(cm, model_name: str, save_path: str = None):
    """
    Plots confusion matrix heatmap.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"])
    ax.set_title(f"Confusion Matrix: {model_name}", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
