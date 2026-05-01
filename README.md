# Credit Risk Modelling

An end-to-end credit risk analysis and modelling project built to demonstrate industry-relevant skills for credit risk analyst roles in banking and financial services.

---

## Overview

This project uses a real-world style lending dataset to build, evaluate, and explain credit risk models — covering the full workflow from raw data to deployment-ready scorecard.

**Target variable:** `loan_status` (0 = No Default, 1 = Default)  
**Dataset size:** 32,581 rows × 12 columns

---

## Project Roadmap

| Stage | Description |
|-------|-------------|
| 1. Exploratory Data Analysis | Distributions, outliers, class imbalance, correlations |
| 2. Preprocessing | Missing value imputation, encoding, feature engineering |
| 3. Class Imbalance Handling | SMOTE, class weighting strategies |
| 4. Machine Learning Models | Logistic Regression, Random Forest, XGBoost |
| 5. Banking Model Evaluation | AUC-ROC, KS Statistic, Gini Coefficient |
| 6. Model Explainability | SHAP values, feature importance |
| 7. Credit Scorecard | Weight of Evidence (WoE) encoding, scorecard generation |

---

## Key Findings

**Data Quality**
- Missing values in `loan_int_rate` and `person_emp_length` were imputed with 0
- Age outlier detected (value of 144) — flagged during EDA
- Class imbalance of ~78% no default / ~22% default addressed using SMOTE

**Feature Engineering**
- Created `LTI` (Loan-to-Income ratio) as an additional predictor
- Counter-intuitively, defaulted loans clustered at lower LTI ratios — suggesting income level alone is not a reliable default predictor in this dataset

**Model Performance**

| Model | AUC-ROC | Gini | KS Stat |
|-------|---------|------|---------|
| Logistic Regression | 0.8614 | 0.7229 | 0.5755 |
| Random Forest | 0.9336 | 0.8672 | 0.7374 |
| XGBoost | 0.9510 | 0.9021 | 0.7644 |

- XGBoost is the best performing model across all three banking metrics
- SMOTE improved recall for default detection but reduced precision; XGBoost without SMOTE still outperformed its SMOTE equivalent on AUC-ROC
- All models predicted non-defaults more reliably than defaults, consistent with the class imbalance

**Credit Scorecard**
- Built using `optbinning` with WoE binning across 10 features
- Scaled using PDO-odds method (PDO=20, base score=600 at 1:1 odds)
- Generates an individual credit score per borrower, with lower scores indicating higher default risk

---

## Tech Stack

- **Language:** Python 3.x (Anaconda)
- **Notebook:** Jupyter
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, XGBoost, SHAP, optbinning

---

## Dataset

The dataset contains borrower and loan attributes including:

| Feature | Description |
|---------|-------------|
| `person_age` | Borrower age |
| `person_income` | Annual income |
| `person_emp_length` | Employment length (years) |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Interest rate |
| `loan_percent_income` | Loan-to-income ratio |
| `loan_status` | Target: 0 = No Default, 1 = Default |
| `cb_person_default_on_file` | Historical default on file (Y/N) |

---

## How to Run

```bash
git clone https://github.com/nguyenhaducnam/credit_risk_modelling.git
cd credit_risk_modelling
pip install -r requirements.txt
jupyter notebook
```

---

## About

Built as a portfolio project to demonstrate practical credit risk modelling skills, including data analysis, machine learning, and banking-specific model validation techniques.
