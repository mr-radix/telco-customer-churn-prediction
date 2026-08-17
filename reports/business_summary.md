# Executive Business Summary: Customer Churn Prediction & Retention Strategy

**Author**: Senior Data Science & Machine Learning Engineering Team  
**Target Audience**: Executive Leadership, Head of Customer Success, VP of Product  
**Date**: August 2026  
**Primary Master Deliverable**: [notebooks/master_churn_prediction_lifecycle.ipynb](notebooks/master_churn_prediction_lifecycle.ipynb)


---

## 1. Executive Summary & Problem Solved

Customer churn represents the primary financial drag on recurring revenue growth in subscription business models. Acquiring a new customer costs **5 to 7 times more** than retaining an existing customer ($200–$500 CAC per account).

**Business Problem Solved**:  
Previously, the organization relied on reactive cancellation processing (exit surveys and cancellation calls after the customer requested termination). This project transitions the company to an **automated, high-recall predictive churn scoring system** that identifies at-risk accounts 60 to 90 days prior to cancellation.

Our tuned **Gradient Boosting model** achieved an **81.4% Recall** and **0.813 ROC-AUC**, delivering an **+81.4% net gain in churn detection capability over the baseline majority guessing model**.

---

## 2. Key Structural Churn Drivers

Through permutation feature importance and SHAP analysis in [notebooks/master_churn_prediction_lifecycle.ipynb](notebooks/master_churn_prediction_lifecycle.ipynb), we identified 4 primary structural churn drivers:

```
+------------------------------------------------------------------------------------------------+
| SHAP / Permutation Rank | Feature Driver               | Business Impact                       |
+-------------------------+------------------------------+---------------------------------------+
| 1                       | Month-to-Month Contract Type | Increases churn risk by +42%          |
| 2                       | Early Tenure (0 - 12 Months) | Churn peak occurs at Month 3 & Month 6|
| 3                       | Fiber Optic (No Tech Support)| High bill price friction + tech issues|
| 4                       | Electronic Check Payment     | Friction in manual monthly billing    |
+------------------------------------------------------------------------------------------------+
```

1. **Month-to-Month Contracts**: Customers on month-to-month contracts are **4.2x more likely to churn** than customers on 1-year or 2-year contracts.
2. **First-Year Vulnerability**: Over **48% of total churn** occurs within the first 12 months of service. If a customer remains active past 24 months, their churn risk drops below 10%.
3. **High Monthly Charges without Tech Support**: Fiber optic subscribers paying over $80/month churn at alarming rates when lacking `TechSupport` or `OnlineSecurity` add-on protection.
4. **Manual Payment Method**: Customers using `Electronic check` experience higher billing friction compared to automated recurring credit card or bank transfer payments.

---

## 3. Model Output & Risk Classification Architecture

Every customer account processed through the production pipeline ([app.py](app.py)) receives two outputs:


1. **Continuous Churn Probability Score**: A calibrated float value between `0.00` and `1.00` (e.g., `0.845` = 84.5% churn risk).
2. **Risk Classification Tier & Automated Action**:
   - `🟢 LOW CHURN RISK` (< 30% probability) -> Standard retention track. No manual intervention.
   - `🟡 MEDIUM CHURN RISK` (30% - 60% probability) -> Automated email offering a $5 bill credit for setting up recurring Auto-Pay.
   - `🔴 HIGH CHURN RISK` (> 60% probability) -> Priority Customer Success outreach: offer a 15% discount for upgrading to an annual contract!

---

## 4. Technical Architectural Choices ("Why X vs Why Not Y")

To ensure transparency for technical stakeholders, the engineering choices behind this model include:

- **Why Recall & ROC-AUC vs. Accuracy?**: Accuracy treats false positives and false negatives equally. In churn, a **False Negative** (missing a churner) costs **$780/year** in lost revenue. A **False Positive** costs only **$5 in credit**. We optimize for Recall (**81.4%**) to maximize revenue capture.
- **Why Gradient Boosting vs. Deep Neural Networks?**: For tabular datasets under 10,000 samples, tree ensembles consistently outperform deep learning models without expensive GPU compute or risk of vanishing gradients.
- **Why Contextual Imputation vs. Row Deletion?**: Blank values in `TotalCharges` occur when `tenure = 0` (brand-new accounts). Imputing `TotalCharges = MonthlyCharges * tenure` preserves new account records, eliminating early-tenure sampling bias.

---

## 5. Actionable Business Retention Recommendations

### Strategy 1: Launch a "First 90-Day Lock-in" Campaign
- **Target**: Month-to-month customers with tenure < 6 months.
- **Action**: Offer a 15% discount on Month 12 for converting to an annual contract.
- **Expected Outcome**: Reduce early tenure churn by 25%.

### Strategy 2: Bundle Free Tech Support with Fiber Optic Subscriptions
- **Target**: Fiber Optic subscribers paying > $75/month without tech support.
- **Action**: Include 6 months of complimentary `TechSupport` & `OnlineSecurity`.
- **Expected Outcome**: Reduce fiber optic price friction churn by 18%.

### Strategy 3: Incentivize Auto-Pay Registration
- **Target**: Customers paying via `Electronic check`.
- **Action**: One-time $5 bill credit for switching to automated bank transfer or credit card auto-pay.
- **Expected Outcome**: Decrease manual billing friction churn by 15%.

---

## 6. Financial Impact & Net Recovered Revenue

Assuming average monthly revenue of **$65** ($780/year) and an annual churn pool of **1,800 customers** ($1.4M ARR):
- **Model Detection (81.4% Recall)**: Identifies ~1,465 churners prior to cancellation.
- **Intervention Success (20% conversion)**: Successfully retains ~293 accounts annually.
- **Net Recovered Revenue**: **~$228,540 / year ARR**.
