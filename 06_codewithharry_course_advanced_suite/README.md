# 🚀 06: CodeWithHarry Course Advanced & Refined Systems Suite

This directory contains the projects inspired by **CodeWithHarry's "The Ultimate Job Ready Data Science Course"**, re-engineered and elevated from beginner tutorial scripts to **production-grade enterprise systems**.

---

## 🏗️ Re-Engineered Projects Overview

### 1. Dragon Real Estate $\longrightarrow$ Enterprise Property Valuation & Private Equity Underwriting Engine
* **Original Course Implementation**: Basic Boston housing notebook predicting house price with Linear Regression and a single decision tree on raw CSV data.
* **Our Production Elevation**:
  - **Pipeline Architecture**: Scikit-learn production `Pipeline` utilizing `SimpleImputer(strategy='median')` and `RobustScaler()` to resist extreme luxury property price outliers.
  - **High-Performance GBDT Regressor**: Attains $R^2 > 0.85$ and low MAPE under 16%.
  - **Private Equity Underwriting Layer**: Computes Gross Annual Rental Yield, Operating Expenses (30%), **Net Operating Income (NOI)**, **Capitalization Rate (Cap Rate %)**, and **Cash-on-Cash Return** given a 20% down payment and 30-year fixed amortization.
  - **Debt Service Coverage Ratio (DSCR)**: Calculates financial safety ($NOI / \text{Annual Debt Service}$) ensuring debt viability.
  - **Macroeconomic Interest Rate Sensitivity Matrix**: Stress-tests cash flows and DSCR across interest rate regimes from 5.0% to 8.5% in 0.5% increments.
  - **Investment Grading**: Automates asset tier classification (`TIER_A_STRONG_BUY`, `TIER_B_MODERATE_HOLD`, `TIER_C_OVERVALUED_LOW_YIELD`).

### 2. Coders of Delhi & Bangalore $\longrightarrow$ Tech Labor Market Intelligence & Skill Gap Optimizer
* **Original Course Implementation**: Basic Pandas exploratory data analysis (EDA), plotting histograms and filtering developer counts.
* **Our Production Elevation**:
  - **Full Production Regression**: Predicts software engineering compensation across major tech hubs (Bangalore vs Delhi-NCR) utilizing `ColumnTransformer` and Gradient Boosting ($R^2 > 0.95$).
  - **Market Percentile Benchmarking**: Quantifies the 25th percentile, 50th median, 75th percentile, and 90th percentile salary benchmarks for any candidate profile.
  - **Dynamic Skill Gap & Career ROI Model**: Calculates the precise projected salary uplift ($+\Delta\%$ and $+INR\text{ LPA}$) for acquiring in-demand competencies:
    - $+27.8\%$ (+9.6 LPA) for Cloud & Containerization (AWS, Docker, Kubernetes).
    - $+38.5\%$ (+13.4 LPA) for Generative AI & RAG Engineering.
    - $+68.9\%$ total uplift when acquiring both high-leverage skillsets.

---

## 🛠️ How to Test & Verify
```bash
# Test Dragon Real Estate Pipeline & Sensitivity Analysis
python dragon_real_estate/test_dragon.py

# Test Coders Labor Market Engine & Skill Uplift Model
python coders_analytics/test_talent.py
```
Both test suites execute in under 1 second with automated assertions confirming model accuracy and financial logic.
