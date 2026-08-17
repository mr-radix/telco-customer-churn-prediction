# 📚 Master Technical Knowledge Base: From Basics to Advanced MLOps

**Project**: Customer Churn Prediction & Business Insights  
**File**: `NOTES.md`  
**Target Audience**: Beginners learning Data Science, Intermediate ML Practitioners, and Senior Engineers/Leadership.

---

## 📌 Table of Contents
1. [🟢 LEVEL 1: Absolute Beginner Basics (What is AI/ML & Churn?)](#1--level-1-absolute-beginner-basics)
2. [🟡 LEVEL 2: Data Preprocessing & Feature Engineering](#2--level-2-data-preprocessing--feature-engineering)
3. [🔴 LEVEL 3: Machine Learning Algorithms, Math & Metrics](#3--level-3-machine-learning-algorithms-math--metrics)
4. [🟣 LEVEL 4: Senior MLOps, Serving & Financial ROI](#4--level-4-senior-mlops-serving--financial-roi)

---

<a id="1--level-1-absolute-beginner-basics"></a>
## 🟢 LEVEL 1: Absolute Beginner Basics

### 1.1 What is Machine Learning?
Machine Learning (ML) is a branch of Artificial Intelligence (AI) where computers learn patterns from historical data to make predictions on new, unseen data—without being explicitly programmed with fixed hardcoded rules.

```
TRADITIONAL PROGRAMMING:
Data + Hardcoded Rules ─────────► Output

MACHINE LEARNING:
Historical Data + Desired Outputs ──► Computer Learns Rules (Model)
New Data + Learned Model ─────────► Prediction Output
```

#### The 3 Main Types of Machine Learning:
1. **Supervised Learning (Used in this project)**: The algorithm is given historical inputs ($X$) along with the correct target labels ($y$). The model learns a mathematical function $f(X) \approx y$.
   - *Example*: Predicting whether a customer will churn ($y=1$) or stay ($y=0$).
2. **Unsupervised Learning**: The algorithm finds hidden patterns or clusters in unlabeled data without target answers ($y$).
   - *Example*: Grouping customers into demographic segments.
3. **Reinforcement Learning**: An agent learns through trial-and-error rewards and penalties.
   - *Example*: Game-playing AI (e.g. Chess or AlphaGo).

---

### 1.2 What is Customer Churn?
Customer Churn (customer attrition) occurs when a subscriber cancels their subscription or stops doing business with a company.

#### The Leaky Bucket Analogy:
Imagine your customer base as a bucket filled with water. 
- **Customer Acquisition** is pouring new water into the bucket.
- **Customer Churn** is a hole at the bottom of the bucket leaking water out.

```
       [ Pouring New Customers ($200-$500 CAC) ]
                          │
                          ▼
            ┌──────────────────────────┐
            │                          │
            │      CUSTOMER BASE       │
            │   (Monthly Subscribers)  │
            │                          │
            └────────────┬─────────────┘
                         │
                         ▼  ◄─── [ LEAK: Customer Churn ($1.4M ARR Lost) ]
```

If you focus only on pouring new water in without fixing the leak, the bucket will eventually run dry—or become prohibitively expensive to maintain.

#### Key Subscription Metrics:
- **ARPU (Average Revenue Per User)**: Average monthly bill per customer ($65/month in our dataset).
- **MRR (Monthly Recurring Revenue)**: $\text{Active Subscribers} \times \text{ARPU}$.
- **ARR (Annual Recurring Revenue)**: $\text{MRR} \times 12$.
- **CAC (Customer Acquisition Cost)**: Total marketing and sales cost spent to acquire 1 new customer ($200–$500 per account).
- **Retention Cost**: The cost of keeping an existing customer through proactive offers ($5–$20 per account).

---

### 1.3 Python Data Science Ecosystem (The Core Tools)
- **Python**: The standard programming language for data science due to its clean syntax and massive scientific library ecosystem.
- **Pandas**: A library for data manipulation. It structures data into **DataFrames** (which look and act like digital Excel spreadsheets with rows and columns).
- **NumPy**: A fast numerical computing library for matrix vector math.
- **Scikit-Learn**: The premier Python machine learning library containing tools for data splitting, feature scaling, model training, evaluation, and hyperparameter tuning.
- **Streamlit**: A framework for turning Python scripts into interactive web dashboards.

---

<a id="2--level-2-data-preprocessing--feature-engineering"></a>
## 🟡 LEVEL 2: Data Preprocessing & Feature Engineering

### 2.1 Missing Value Resolution: Contextual Imputation
In raw datasets, missing data occurs due to system logging errors or missing user entries.

#### What Happened in Our Dataset?
Column `TotalCharges` contained 11 blank space strings (`" "`) instead of numerical values.

```
Row Index | tenure | MonthlyCharges | TotalCharges (Raw) | Cleaned TotalCharges
──────────┼────────┼────────────────┼────────────────────┼──────────────────────
488       | 0      | 52.55          | " "                | 0.00 (52.55 * 0)
753       | 0      | 20.25          | " "                | 0.00 (20.25 * 0)
```

#### Why Contextual Imputation is Better Than Dropping Rows:
1. **Row Deletion (Bad)**: Deleting these rows would discard newly onboarded accounts (`tenure = 0`), creating an early-tenure sampling bias.
2. **Mean/Median Imputation (Naive)**: Filling with the median ($1,397) would assign high historical charges to a customer who joined today!
3. **Contextual Imputation (Selected)**: Since `TotalCharges` represents cumulative historical billing, for new accounts where `tenure = 0`, the correct mathematical value is:
   $$\text{TotalCharges} = \text{MonthlyCharges} \times \text{tenure} = \$52.55 \times 0 = \mathbf{\$0.00}$$

---

### 2.2 Feature Engineering
Feature Engineering is the process of creating new mathematical input variables (features) from raw data to help the model learn faster and make better predictions.

#### Features Engineered in `src/features/build_features.py`:
1. **`tenure_group`**: Converts continuous month counts into 4 distinct customer lifecycle stages:
   - `0-12m`: High-risk onboarding phase (over 48% of total churn occurs here).
   - `13-24m`: Early stability phase.
   - `25-48m`: Established subscriber phase.
   - `49+m`: Highly loyal subscriber phase (< 10% churn).
2. **`charges_per_tenure`**: $\frac{\text{MonthlyCharges}}{\text{tenure} + 1}$ — measures price sensitivity relative to subscription longevity.
3. **`high_monthly_charges`**: Binary flag ($1$ if $\text{MonthlyCharges} > \$70.0$, else $0$) marking high-value accounts vulnerable to price undercutting.
4. **`total_addons`**: Sum of active security and utility add-ons ($0 \to 6$) including `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies`.

---

### 2.3 Feature Encoding & Scaling

#### 1. One-Hot Encoding (Categorical Data)
Machine learning models cannot process raw text strings like `"Fiber optic"` or `"Month-to-month"`. They require numbers.

- **Naive Approach (Ordinal / Label Encoding)**: Assigning `DSL=1, Fiber optic=2, No=3` causes linear models to assume `No` is 3x greater than `DSL`!
- **One-Hot Encoding (Selected)**: Creates binary dummy columns ($0$ or $1$) for each category:

```
Raw Category      ──►   Internet_DSL   Internet_Fiber   Internet_No
"Fiber optic"     ──►        0               1               0
"DSL"             ──►        1               0               0
"No"              ──►        0               0               1
```

#### 2. StandardScaler Normalization (Numerical Data)
Features operate on drastically different scales: `tenure` ranges from $0 \to 72$, while `MonthlyCharges` ranges from $\$18 \to \$120$. Without scaling, distance-based and gradient descent models will treat `MonthlyCharges` as 100x more important than `SeniorCitizen` ($0$ or $1$).

- **StandardScaler Formula**:
  $$z = \frac{x - \mu}{\sigma}$$
  Where $\mu$ is the feature mean and $\sigma$ is the standard deviation.
- **Result**: Rescales all numeric features to have mean $\mu = 0$ and variance $\sigma^2 = 1$.

---

### 2.4 Data Partitioning & Zero Data Leakage
To evaluate how well a model will perform in the real world, we split our data into 3 distinct sets:

```
FULL DATASET (7,032 Accounts)
├── Train Set (70% = 4,922 Rows) ──► Used to fit model weights & ColumnTransformer
├── Validation Set (15% = 1,055 Rows) ──► Used to evaluate candidate models & tune hyperparameters
└── Test Set (15% = 1,055 Rows) ───────► Locked holdout set used ONLY for final score reporting
```

> [!CRITICAL]
> **Data Leakage Rule**: `ColumnTransformer` (StandardScaler and OneHotEncoder) MUST be fitted **EXCLUSIVELY on the Training Set**. Fitting transformers on the full dataset before splitting causes information from the test set to "leak" into training, producing artificially inflated test scores that fail in production!

---

<a id="3--level-3-machine-learning-algorithms-math--metrics"></a>
## 🔴 LEVEL 3: Machine Learning Algorithms, Math & Metrics

### 3.1 Evaluation Metrics: Why Recall Dominates Accuracy

#### The Confusion Matrix:
When a model makes predictions on 1,055 test accounts, 4 outcomes occur:

```
                       ACTUAL VALUE (Ground Truth)
                       Churn = 1 (Yes)    Churn = 0 (No)
PREDICTED  Churn = 1   True Positive (TP) False Positive (FP)
VALUE      Churn = 0   False Negative (FN) True Negative (TN)
```

- **True Positive (TP)**: Model predicted Churn; customer actually churned. (Success!)
- **True Negative (TN)**: Model predicted No Churn; customer stayed. (Success!)
- **False Positive (FP)**: Model predicted Churn; customer stayed. (Cost: $5 bill credit).
- **False Negative (FN)**: Model predicted No Churn; customer churned! (Cost: **$780 lost revenue**).

#### Comparing Metrics:
1. **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
   - *Why Accuracy Fails*: In a dataset where 73.5% of customers stay, a dummy model that predicts `No Churn` for everyone gets **73.5% Accuracy**—yet catches **0% of churners**!
2. **Recall (Primary Metric)**: $\frac{TP}{TP + FN}$
   - Measures what percentage of actual churners the model correctly detected.
   - Our Gradient Boosting model achieved **81.4% Recall** (catching 4 out of 5 churners).
3. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**:
   - Measures the model's ability to discriminate between churners and non-churners across all probability thresholds ($0.0 \to 1.0$). Our model achieved **0.8128 ROC-AUC**.

---

### 3.2 Machine Learning Algorithms Benchmarked

We benchmarked 4 distinct model families under identical zero-leakage conditions:

```
Model Architecture Comparison (Holdout Test Set):

Baseline Majority ──► Recall: 0.0%    | ROC-AUC: 0.5000 (Floor)
Logistic Reg.     ──► Recall: 75.1%   | ROC-AUC: 0.8151
Random Forest     ──► Recall: 75.4%   | ROC-AUC: 0.8093
Gradient Boosting ──► Recall: 81.4% ⭐ | ROC-AUC: 0.8128 (SELECTED)
```

#### 1. Baseline Majority Classifier
- Predicts `Churn = No` for all inputs.
- **Recall**: 0.0%. Serves as the performance floor.

#### 2. Logistic Regression (Linear Model)
- Passes a linear combination of features through the Sigmoid function to output a probability $\hat{p}$:
  $$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i X_i)}}$$
- **Recall**: 75.1%. Fast linear baseline, but struggled with non-linear feature thresholding (e.g. sharp churn dropoff at Month 12).

#### 3. Random Forest Classifier (Bagging Ensemble)
- Trains an ensemble of 100 deep decision trees in parallel using **Bootstrap Aggregation (Bagging)** and feature subsampling.
- **Recall**: 75.4%. Good stability, but exhibited slight overfitting compared to boosting.

#### 4. Gradient Boosting Classifier (Selected Production Model)
- An **Additive Boosting Ensemble** that trains decision trees sequentially. Each new tree fits to the errors (pseudo-residuals) of the previous trees:
  $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
  Where $\eta = 0.05$ is the learning rate shrinkage factor.
- **Recall**: **81.4%**. Achieved the highest churn detection rate (+81.4% gain over baseline).

---

### 3.3 Class Balancing (`class_weight='balanced'`)
Because churners represent only 26.5% of dataset accounts, unweighted models default to favoring the majority class.
- Setting `class_weight='balanced'` calculates loss weights $w_j$ inversely proportional to class frequencies:
  $$w_j = \frac{N}{2 \times N_j}$$
- Positive churn instances receive a **2.8x higher penalty weight** during log-loss minimization, forcing the gradient boosting trees to prioritize catching True Positives (Recall).

---

### 3.4 Hyperparameter Tuning via 5-Fold GridSearchCV
We performed 5-fold Stratified Cross-Validation tuning over the parameter space:

```python
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],      # Controls step size during gradient descent
    'max_depth': [3, 5, 7],                  # Controls maximum tree depth to prevent overfitting
    'min_samples_leaf': [10, 20, 50],        # Minimum samples per leaf node for generalization
    'class_weight': ['balanced'],            # Penalizes False Negatives 2.8x more heavily
}
```

- **Optimal Hyperparameters Selected**: `learning_rate = 0.05`, `max_depth = 5`, `min_samples_leaf = 20`, `class_weight = 'balanced'`.

---

<a id="4--level-4-senior-mlops-serving--financial-roi"></a>
## 🟣 LEVEL 4: Senior MLOps, Serving & Financial ROI

### 4.1 Permutation Importance & SHAP Feature Attribution

#### Permutation Importance Calculation:
1. Measure baseline Recall score $S_{\text{base}}$ on validation data.
2. Shuffle (permute) the values of feature column $j$ to break its relationship with target $y$.
3. Measure permuted Recall score $S_{\text{perm}}$.
4. Feature Importance Score = $S_{\text{base}} - S_{\text{perm}}$.

```
Top 4 Structural Churn Drivers (Feature Attribution):

1. Contract_Month-to-month ──► +42% Churn Risk Impact (4.2x higher churn vs annual)
2. tenure (< 12 Months)    ──► 48% of total churn occurs in first 12 months
3. Fiber Optic (No Support)──► Price friction ($80+/mo) without technical support
4. Electronic Check        ──► Friction in manual monthly billing methods
```

---

### 4.2 Production Web Architecture (`app.py`)

The production web interface is built with **Streamlit** and serves real-time inferences:

```
[User Input Payload] 
        │
        ▼
[app.py Microservice Engine]
        │
        ├──> Load Serialized Models: models/best_model.pkl & models/preprocessor.pkl
        ├──> Feature Engineering & ColumnTransformer Scaling (X_trans)
        ├──> Execute Probabilities: model.predict_proba(X_trans)
        │
        ▼
[Real-Time Risk Scoring Output]
        ├──> Continuous Churn Probability Score (e.g. 0.784 / 78.4%)
        ├──> Risk Classification Badge (🔴 HIGH CHURN RISK)
        └──> Actionable Business Strategy (Offer 15% Annual Contract Discount)
```

#### Risk Classification Tiers:
- `🟢 LOW RISK` ($P < 0.30$): Account is healthy. Standard retention track.
- `🟡 MEDIUM RISK` ($0.30 \le P \le 0.60$): Account exhibits price friction. Trigger automated email offering $5 Auto-Pay credit.
- `🔴 HIGH RISK` ($P > 0.60$): Account at imminent risk. Trigger priority Customer Success call offering 15% annual contract discount.

---

### 4.3 Quantified Financial ROI Model

```
+-----------------------------------------------------------------------------------------------+
| METRIC                               | VALUE                                                  |
+--------------------------------------+--------------------------------------------------------+
| Annual At-Risk Churn Pool            | 1,800 accounts ($1,404,000 ARR at $65/month ARPU)      |
| Model Detection (81.4% Recall)       | 1,465 churners identified prior to cancellation        |
| Retention Conversion (20% success)   | 293 accounts retained annually                         |
| Net Recovered Annual Revenue         | ~$228,540 / year ARR                                   |
+-----------------------------------------------------------------------------------------------+
```

---

### 4.4 Automated Testing & MLOps Suite ([tests/](tests/))
We maintain automated Pytest unit tests to enforce software quality:
- `tests/test_data.py`: Tests synthetic dataset generation, missing value imputation, and 70/15/15 stratified split ratios.
- `tests/test_features.py`: Tests engineered feature calculations, ColumnTransformer shape consistency, and zero data leakage.
- `tests/test_models.py`: Tests model training, evaluation metric calculations, and binary serialization/deserialization.

```bash
source venv/bin/activate
pytest tests/ -v
# Result: 5/5 PASSED in 1.16s
```
