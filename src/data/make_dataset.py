"""
Data acquisition and processing pipeline for Telco Customer Churn dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_telco_churn_dataset(n_samples: int = 7032, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic Telco Customer Churn dataset matching
    the Kaggle Telco Churn dataset schema and statistical distributions.
    """
    np.random.seed(seed)
    
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(65+i%26)}{chr(65+(i*3)%26)}{chr(65+(i*7)%26)}{chr(65+(i*11)%26)}" for i in range(n_samples)]
    genders = np.random.choice(["Female", "Male"], size=n_samples)
    senior_citizens = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partners = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])
    
    # Tenure in months (0 to 72)
    tenure = np.random.randint(0, 73, size=n_samples)
    
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])
    
    def get_internet_addon(inet_svc):
        if inet_svc == "No":
            return "No internet service"
        return np.random.choice(["Yes", "No"], p=[0.38, 0.62])

    online_security = [get_internet_addon(inet) for inet in internet_service]
    online_backup = [get_internet_addon(inet) for inet in internet_service]
    device_protection = [get_internet_addon(inet) for inet in internet_service]
    tech_support = [get_internet_addon(inet) for inet in internet_service]
    streaming_tv = [get_internet_addon(inet) for inet in internet_service]
    streaming_movies = [get_internet_addon(inet) for inet in internet_service]
    
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], size=n_samples, p=[0.34, 0.23, 0.21, 0.22])
    
    # Calculate monthly charges based on services
    base_charge = 20.0
    monthly_charges = []
    for i in range(n_samples):
        charge = base_charge
        if phone_service[i] == "Yes":
            charge += 20.0
        if multiple_lines[i] == "Yes":
            charge += 10.0
        if internet_service[i] == "DSL":
            charge += 25.0
        elif internet_service[i] == "Fiber optic":
            charge += 45.0
        if online_security[i] == "Yes":
            charge += 8.0
        if tech_support[i] == "Yes":
            charge += 8.0
        if streaming_tv[i] == "Yes":
            charge += 10.0
        if streaming_movies[i] == "Yes":
            charge += 10.0
        # Add random noise
        charge += np.random.normal(0, 2.5)
        monthly_charges.append(round(max(18.25, min(118.75, charge)), 2))
        
    monthly_charges = np.array(monthly_charges)
    
    # Total charges (some tenure=0 rows have blank ' ' strings to mimic Kaggle dataset)
    total_charges = []
    for i in range(n_samples):
        if tenure[i] == 0:
            total_charges.append(" ")  # Blank string missing value
        else:
            tot = monthly_charges[i] * tenure[i] + np.random.normal(0, 5)
            total_charges.append(str(round(max(18.25, tot), 2)))
            
    # Calculate churn probability based on key risk drivers
    # Month-to-month + Fiber optic + Short tenure + Electronic check + No TechSupport -> High Churn
    churn_prob = []
    for i in range(n_samples):
        logit = -0.5
        if contract[i] == "Month-to-month":
            logit += 1.2
        elif contract[i] == "Two year":
            logit -= 1.5
        if internet_service[i] == "Fiber optic":
            logit += 0.8
        if tech_support[i] == "No":
            logit += 0.6
        if payment_method[i] == "Electronic check":
            logit += 0.5
        if tenure[i] < 12:
            logit += 0.9
        elif tenure[i] > 36:
            logit -= 1.1
        if senior_citizens[i] == 1:
            logit += 0.3
            
        prob = 1 / (1 + np.exp(-logit))
        churn_prob.append(prob)
        
    churn_labels = ["Yes" if np.random.rand() < p else "No" for p in churn_prob]
    
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
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
        "Churn": churn_labels
    })
    
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans missing values in TotalCharges, converts to numeric, and validates types.
    """
    df = df.copy()
    # Replace blank space strings with NaN and convert to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    # Impute missing TotalCharges with MonthlyCharges * tenure or median
    missing_mask = df['TotalCharges'].isna()
    df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'MonthlyCharges'] * df.loc[missing_mask, 'tenure']
    # If still missing, fill with median
    if df['TotalCharges'].isna().sum() > 0:
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df


def load_and_prepare_data(raw_filepath: str, processed_dir: str, seed: int = 42):
    """
    Loads or generates raw data, cleans missing values, and creates train/val/test splits.
    """
    os.makedirs(os.path.dirname(raw_filepath), exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_filepath):
        print(f"Raw dataset not found at {raw_filepath}. Generating dataset...")
        df_raw = generate_telco_churn_dataset(seed=seed)
        df_raw.to_csv(raw_filepath, index=False)
    else:
        df_raw = pd.read_csv(raw_filepath)
        
    df_clean = clean_dataset(df_raw)
    
    # Stratified split: 70% train, 15% val, 15% test
    train_val, test = train_test_split(df_clean, test_size=0.15, random_state=seed, stratify=df_clean['Churn'])
    train, val = train_test_split(train_val, test_size=0.1765, random_state=seed, stratify=train_val['Churn']) # 0.1765 * 0.85 approx 0.15
    
    train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    print(f"Data splits saved to {processed_dir}: Train ({len(train)}), Val ({len(val)}), Test ({len(test)})")
    return train, val, test


if __name__ == "__main__":
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    proc_dir = "data/processed"
    load_and_prepare_data(raw_path, proc_dir)
