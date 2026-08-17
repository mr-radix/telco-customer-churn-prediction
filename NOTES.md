# 📚 Master Technical Knowledge Base & Deep Dive Notes

**Project**: Customer Churn Prediction & Business Insights  
**File**: `NOTES.md`  
**Purpose**: Exhaustive technical documentation explaining **What**, **How**, and **Why** every data science, software engineering, and business decision was executed in this repository.

---

## 📌 Table of Contents
1. [Domain Economics & Problem Formulation](#1-domain-economics--problem-formulation)
2. [The 9-Phase Data Science Project Lifecycle](#2-the-9-phase-data-science-project-lifecycle)
3. [Architectural Decision Matrix ("Why X vs Why Not Y")](#3-architectural-decision-matrix-why-x-vs-why-not-y)
4. [Machine Learning Algorithms & Mathematical Deep Dive](#4-machine-learning-algorithms--mathematical-deep-dive)
5. [Feature Engineering & Preprocessing Engine](#5-feature-engineering--preprocessing-engine)
6. [Feature Attribution: Permutation Importance & SHAP](#6-feature-attribution-permutation-importance--shap)
7. [Production Serving & Streamlit Web Architecture](#7-production-serving--streamlit-web-architecture)
8. [Quantified Financial ROI & Retention Strategy](#8-quantified-financial-roi--retention-strategy)
9. [Automated Testing & MLOps Engineering](#9-automated-testing--mlops-engineering)

---

## 1. Domain Economics & Problem Formulation

### What is Customer Churn?
Customer Churn (also known as customer attrition) is the percentage of subscribers who discontinue their service subscriptions within a given time window.

### Why Customer Churn is the Primary Financial Drag on SaaS & Telecom:
1. **Asymmetric Cost Dynamics (CAC vs. Retention)**:
   - **Customer Acquisition Cost (CAC)**: Acquiring a new telecom subscriber costs **$200 to $500** due to marketing, ad spend, device subsidies, and onboarding friction.
   - **Retention Cost**: Retaining an existing account through targeted incentives costs **$5 to $20** (e.g. a $5 bill credit or 15% contract upgrade discount).
   - **Conclusion**: Retaining an existing customer is **5 to 7 times cheaper** than acquiring a replacement customer.

2. **The Flaw of Reactive Offboarding**:
   - In traditional operations, customer success teams wait until a subscriber calls to terminate their account before making a counter-offer.
   - **Empirical Reality**: Over **80% of customers** who call to cancel have already signed an agreement with a competitor. Reactive retention fails.

3. **Financial Math of Churn Accumulation**:
   - Monthly Recurring Revenue (MRR) = $\text{Active Subscribers} \times \text{Average Revenue Per User (ARPU)}$.
   - Annual Recurring Revenue (ARR) = $\text{MRR} \times 12$.
   - In a customer base of 7,000 accounts losing ~1,800 customers annually at an ARPU of **$65/month** ($780/year):
     $$\text{Annual Lost ARR} = 1,800 \times \$780 = \mathbf{\$1,404,000 / year}$$

4. **The Machine Learning Solution**:
   - Transition from reactive exit handling to **proactive risk scoring 60 to 90 days in advance**.
   - Calculate continuous probability scores ($0.00 \to 1.00$) and automatically trigger risk-tiered retention campaigns before the customer initiates termination.

---

## 2. The 9-Phase Data Science Project Lifecycle

Every step in this repository is structured into 9 sequential phases within [notebooks/master_churn_prediction_lifecycle.ipynb](notebooks/master_churn_prediction_lifecycle.ipynb):

### Phase 1: Problem Statement Definition
- **Objective**: Formulate the machine learning task as a supervised binary classification problem:
  $$y_i \in \{0, 1\} \quad (0 = \text{No Churn}, 1 = \text{Churn})$$
- **Target KPI**: Maximize **Recall** (identifying churning accounts) while maintaining strong **ROC-AUC** discrimination.

### Phase 2: Requirements & Environment Setup
- **Seed Lock**: Fixed `RANDOM_SEED = 42` across NumPy, Scikit-Learn, and Python random modules to guarantee 100% deterministic reproducibility.
- **Dependencies**: Selected `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`, `pytest`, `matplotlib`, and `seaborn`.

### Phase 3: Data Acquisition & Ingestion
- Ingested 7,032 customer records across 21 feature attributes (demographics, subscribed services, contract details, and financial metrics).

### Phase 4: Data Inspection & Validation
- **Schema Validation**: Verified data types (`df.info()`, `df.describe()`).
- **Contextual Imputation**: Identified 11 blank space strings in `TotalCharges` corresponding to new accounts (`tenure = 0`). Imputed missing values contextually:
  $$\text{TotalCharges} = \text{MonthlyCharges} \times \text{tenure}$$

### Phase 5: Exploratory Data Analysis (EDA)
- Evaluated target class balance: 5,174 `No` (73.5%) vs 1,858 `Yes` (26.5%).
- Identified a **2.8:1 target class imbalance ratio**, necessitating class-weighted loss functions (`class_weight='balanced'`).

### Phase 6: Data Visualization
- Rendered 4 publication visual panels: countplots, tenure KDE distributions, stacked contract bar charts, and monthly charges boxplots.

### Phase 7: Step-by-Step Model Development
- **7.1 Feature Engineering**: Engineered domain features (`tenure_group`, `charges_per_tenure`, `high_monthly_charges`, `total_addons`).
- **7.2 Preprocessing**: Implemented Scikit-Learn `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for categoricals) fit strictly on 70% train split.
- **7.3 & 7.4 Model Training**: Benchmark 4 candidate models: Baseline Majority, Logistic Regression, Random Forest, and Gradient Boosting.
- **7.5 Metric Selection**: Evaluated Recall (81.4%) and ROC-AUC (0.8128) over accuracy.
- **7.6 Hyperparameter Tuning**: Performed 5-fold Stratified `GridSearchCV`.

### Phase 8: Outcome & Output Analysis
- Specified continuous output probabilities ($0.00 \to 1.00$), 3-tier risk badges (🟢 Low, 🟡 Medium, 🔴 High), and quantified annual net recovered revenue (**~$228,540 / year ARR**).

### Phase 9: Model Export & Verification
- Serialized trained artifacts to `models/best_model.pkl` and `models/best_model.joblib`. Loaded model back into memory and verified real-time inference on unseen payload `PROD-SAMPLE-888`.

---

## 3. Architectural Decision Matrix ("Why X vs Why Not Y")

| Decision Category | Selected Choice (X) | Alternative Evaluated (Y) | Why Choice X Was Selected | Why Alternative Y Was Rejected |
|---|---|---|---|---|
| **Ecosystem** | **Python & Scikit-Learn** | PySpark / R / Julia | Standard web microservice integration (`FastAPI`, `Streamlit`). | ~7k rows fits in memory; PySpark adds unnecessary cluster overhead. |
| **Data Partitioning** | **Stratified 70/15/15 Split** | Simple Random Split | Preserves exact 26.5% target churn ratio across train, val, and test splits. | Random split introduces class distribution drift between splits. |
| **Missing Values** | **Contextual Imputation** | Row Deletion | Imputing `TotalCharges = MonthlyCharges * tenure` preserves new accounts (`tenure = 0`). | Row deletion introduces sampling bias against newly onboarded accounts. |
| **Categorical Encoding** | **One-Hot Encoding** | Label / Ordinal Encoding | Prevents linear and distance models from assuming artificial numerical rank. | Ordinal encoding creates false ordinal rank across nominal categories (`PaymentMethod`). |
| **Numeric Scaling** | **StandardScaler** | MinMaxScaler | Normalizes features to mean=0, std=1 while preserving outlier variance distance. | MinMaxScaler squeezes feature variance if outliers are present. |
| **Model Selection** | **Gradient Boosting** | Deep Neural Networks | Additive decision trees consistently beat NNs on tabular data without GPU compute. | Neural networks overfit small tabular datasets and lack interpretability. |
| **Primary Metric** | **Recall & ROC-AUC** | Accuracy | Missing a churner costs $780/yr; a false positive costs only $5 credit. | Accuracy ignores false negatives on imbalanced target distributions. |
| **Model Persistence** | **Joblib & Pickle** | ONNX / PMML | Native Python binary formats save exact weights and preprocessor state seamlessly. | ONNX adds unnecessary format conversion overhead for scikit-learn models. |

---

## 4. Machine Learning Algorithms & Mathematical Deep Dive

### Log-Loss (Binary Cross-Entropy Loss Function)
The objective function minimized during model training is binary cross-entropy log-loss:

$$\mathcal{L}(y, \hat{p}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

Where:
- $N$ is the number of samples.
- $y_i \in \{0, 1\}$ is the true binary label.
- $\hat{p}_i \in [0.0, 1.0]$ is the model's predicted probability of churn.

### Gradient Boosting Mechanics
Gradient Boosting builds an additive model $F_M(x)$ as a weighted sum of $M$ shallow decision trees:

$$F_M(x) = \sum_{m=1}^{M} \eta \cdot h_m(x)$$

1. **Initialization**: Start with a constant prediction $F_0(x) = \arg\min_{\gamma} \sum_{i=1}^N L(y_i, \gamma)$.
2. **For Step $m = 1 \dots M$**:
   - Compute pseudo-residuals (negative gradient of loss function):
     $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$
   - Fit a decision tree $h_m(x)$ to the pseudo-residuals $r_{im}$.
   - Update model ensemble: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$, where $\eta = 0.05$ is the shrinkage learning rate.

### Why Class Weighting (`class_weight='balanced'`) is Critical
Because churners represent 26.5% of accounts, unweighted models default to predicting `No Churn` to achieve high accuracy.
- `class_weight='balanced'` assigns sample weight $w_j$ inversely proportional to class frequencies:
  $$w_j = \frac{N}{2 \times N_j}$$
- Positive churn instances receive a **2.8x higher penalty weight** during loss minimization, forcing the trees to prioritize Recall (81.4%) over standard accuracy.

---

## 5. Feature Engineering & Preprocessing Engine

### Engineered Domain Features ([src/features/build_features.py](src/features/build_features.py))
1. **`tenure_group`**: Categorizes continuous tenure into 4 lifecycle stages:
   - `0-12m`: High Onboarding Risk Stage (48% of total churn).
   - `13-24m`: Early Retention Stage.
   - `25-48m`: Mature Subscriber Stage.
   - `49+m`: Loyal Subscriber Stage (< 10% churn).
2. **`charges_per_tenure`**: Ratio of `MonthlyCharges / (tenure + 1)` capturing price sensitivity relative to subscription longevity.
3. **`high_monthly_charges`**: Binary flag (`MonthlyCharges > 70.0`) indicating high ARPU accounts vulnerable to competitor price undercutting.
4. **`total_addons`**: Integer count ($0 \to 6$) of active security and support add-on services (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).

### Zero Data Leakage Scikit-Learn Pipeline
To eliminate data leakage between train and test distributions:
- `ColumnTransformer` (StandardScaler on 5 numeric features, OneHotEncoder on 15 categorical features) is **fitted exclusively on the 70% Train Split**.
- Validation and Test splits are transformed using the pre-fitted transformer object.

---

## 6. Feature Attribution: Permutation Importance & SHAP

### How Permutation Feature Importance Works:
For each feature $j$:
1. Measure baseline model score $S_{base}$ on validation data.
2. Randomly shuffle (permute) column $j$ to break its relationship with target $y$.
3. Measure permuted model score $S_{perm}$.
4. Feature importance score = $S_{base} - S_{perm}$.

### Top 4 Structural Churn Drivers Identified:
1. **Month-to-Month Contract Type**: Accounts on month-to-month contracts exhibit **4.2x higher churn rates** than annual contract holders due to low switching barriers.
2. **Tenure < 12 Months**: The onboarding window (first 90-180 days) experiences the highest cancellation spikes.
3. **Fiber Optic Service without Tech Support**: High monthly bills ($> \$80$) coupled with unassisted technical glitches trigger immediate customer frustration.
4. **Electronic Check Payment**: Manual monthly payment methods exhibit higher billing friction and involuntary churn compared to automated auto-pay mechanisms.

---

## 7. Production Serving & Streamlit Web Architecture

### Real-Time Inference Architecture ([app.py](app.py)):

```
[Business User Input] 
        │ (HTML Form Payload)
        ▼
[app.py Streamlit Engine]
        │
        ├──> Loads cached artifacts: models/best_model.pkl & models/preprocessor.pkl
        ├──> Feature Engineering & ColumnTransformer Payload Processing
        ├──> Model Execution: model.predict_proba(X_trans)
        │
        ▼
[Output Generation]
        ├──> Continuous Churn Probability Score (e.g. 78.4%)
        ├──> Risk Classification Badge (🔴 HIGH CHURN RISK)
        └──> Actionable Retention Strategy (Offer 15% Annual Contract Discount)
```

### Risk Classification Tiers:
- `🟢 LOW CHURN RISK` ($P < 0.30$): Account is stable. Standard retention track.
- `🟡 MEDIUM CHURN RISK` ($0.30 \le P \le 0.60$): Account exhibits price friction. Trigger automated email offering $5 Auto-Pay credit.
- `🔴 HIGH CHURN RISK` ($P > 0.60$): Account is at imminent risk. Trigger priority Customer Success call offering 15% annual contract discount.

---

## 8. Quantified Financial ROI & Retention Strategy

### Financial ROI Calculation Model:
- **Baseline Churn Pool**: 1,800 accounts lost per year ($1,404,000 ARR lost).
- **Model Identification (81.4% Recall)**: Identifies 1,465 churners prior to cancellation.
- **Conversion Rate (20% Campaign Conversion)**: Successfully retains 293 accounts annually.
- **Net Recovered Annual Revenue**:
  $$\text{Net Recovered ARR} = 293 \text{ accounts} \times \$780 / \text{year} = \mathbf{\$228,540 / year ARR}$$

---

## 9. Automated Testing & MLOps Engineering

### Automated Test Suite ([tests/](tests/)):
- `tests/test_data.py`: Validates synthetic data generation, missing value cleaning, and stratified split properties.
- `tests/test_features.py`: Validates feature engineering calculations, ColumnTransformer shape transformations, and zero data leakage.
- `tests/test_models.py`: Validates model training, evaluation metrics computation, and artifact serialization.

### Verification Execution Command:
```bash
source venv/bin/activate
pytest tests/ -v
```
*(Result: 5/5 PASSED in 1.16s)*
