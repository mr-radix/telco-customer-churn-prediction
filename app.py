"""
Streamlit Web Application: Customer Churn Scoring & Technical Insights Dashboard.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_resource
def load_model_artifacts():
    models_dir = "models"
    model_path = os.path.join(models_dir, "best_model.pkl")
    prep_path = os.path.join(models_dir, "preprocessor.pkl")
    feat_path = os.path.join(models_dir, "feature_names.pkl")
    
    if not os.path.exists(model_path):
        return None, None, None
        
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception:
        model = joblib.load(model_path)

    try:
        with open(prep_path, "rb") as f:
            preprocessor = pickle.load(f)
    except Exception:
        preprocessor = joblib.load(prep_path)

    try:
        with open(feat_path, "rb") as f:
            feature_names = pickle.load(f)
    except Exception:
        feature_names = joblib.load(feat_path)
        
    return model, preprocessor, feature_names


def main():
    st.set_page_config(
        page_title="Telco Churn Risk Scoring Dashboard",
        page_icon="🔮",
        layout="wide"
    )
    
    st.title("🔮 Telco Customer Churn Risk Scoring & Insights Dashboard")
    st.markdown("""
    Real-time customer churn probability scoring, risk tier classification, automated recommendations, and technical model design explanations.
    *Master Notebook*: [`notebooks/master_churn_prediction_lifecycle.ipynb`](file:///home/mrradix/Projects/Ai-ML/notebooks/master_churn_prediction_lifecycle.ipynb)
    """)
    
    model, preprocessor, feature_names = load_model_artifacts()
    
    if model is None:
        st.warning("⚠️ Model artifacts not found. Please train models first using `python -m src.models.train_model` or run `scripts/build_master_notebook.py`.")
        return

    st.sidebar.header("📋 Customer Attribute Inputs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Demographics & Account")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=6)
        
    with col2:
        st.subheader("2. Services & Subscriptions")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
    with col3:
        st.subheader("3. Contract & Financials")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(monthly_charges * tenure), step=10.0)

    if st.button("🚀 Calculate Churn Risk Score", type="primary"):
        input_data = pd.DataFrame([{
            "customerID": "PROD-DASH-001",
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": "No"
        }])

        from src.features.build_features import prepare_features_and_target
        X_trans, _, _, _ = prepare_features_and_target(input_data, preprocessor=preprocessor, is_train=False)

        churn_prob = model.predict_proba(X_trans)[0, 1]
        churn_pct = churn_prob * 100

        st.markdown("---")
        st.subheader("📊 Real-Time Churn Risk Assessment")
        
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            st.metric(label="Predicted Churn Probability", value=f"{churn_pct:.1f}%")
            if churn_prob < 0.30:
                st.success("🟢 LOW CHURN RISK (< 30% Probability)")
                st.write("Customer exhibits stable subscription patterns.")
            elif churn_prob < 0.60:
                st.warning("🟡 MEDIUM CHURN RISK (30% - 60% Probability)")
                st.write("Customer exhibits price or contract friction.")
            else:
                st.error("🔴 HIGH CHURN RISK (> 60% Probability)")
                st.write("Immediate automated retention outreach required!")
                
        with m_col2:
            st.subheader("💡 Strategic Action Recommendations")
            if contract == "Month-to-month":
                st.info("📌 **Contract Upgrade**: Offer a 15% discount on an annual contract upgrade.")
            if tech_support == "No" and internet_service == "Fiber optic":
                st.info("📌 **Tech Support Bundle**: Add 6 months free TechSupport & OnlineSecurity.")
            if payment_method == "Electronic check":
                st.info("📌 **Auto-Pay Credit**: Offer a $5 bill credit for switching to recurring Auto-Pay.")
            if tenure <= 12:
                st.info("📌 **Onboarding Outreach**: Schedule a proactive Customer Success check-in call.")

    st.markdown("---")
    with st.expander("🔍 Technical Architecture & Decision Justifications (Why X vs Why Not Y)"):
        st.markdown("""
        | Choice Category | Selected Approach (X) | Alternative Evaluated (Y) | Technical Justification |
        |---|---|---|---|
        | **Ecosystem** | **Python & Scikit-Learn** | R / Julia / PySpark | Python provides standard ML serving tools (FastAPI, Streamlit). ~7k dataset size does not need PySpark cluster overhead. |
        | **Data Splitting** | **Stratified 70/15/15 Split** | Simple Random Split | Guarantees exact 26.5% target churn ratio across train, val, and test splits without distribution shift. |
        | **Missing Values** | **Contextual Imputation** | Row Deletion | Imputing `TotalCharges = MonthlyCharges * tenure` preserves new account records, eliminating early tenure bias. |
        | **Categorical Encoding** | **One-Hot Encoding** | Label / Ordinal Encoding | Prevents models from assuming artificial numerical rank across nominal categories like `PaymentMethod`. |
        | **Scaling** | **StandardScaler** | MinMaxScaler | Normalizes features to mean=0, std=1 while preserving outlier variance and distance relationships. |
        | **Model Selection** | **Gradient Boosting** | Deep Learning / NN | Tree ensembles consistently beat neural networks on small-to-medium tabular datasets without GPU overhead. |
        | **Primary Metric** | **Recall & ROC-AUC** | Accuracy | Accuracy ignores false negatives. Missing a churner costs $780/yr revenue; a false positive costs only $5 credit. |
        | **Serialization** | **Joblib / Pickle** | ONNX / PMML | Native Python binary formats save model weights and preprocessor state with zero format conversion overhead. |
        """)


if __name__ == "__main__":
    main()
