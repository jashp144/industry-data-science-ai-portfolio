# 🚀 06: CodeWithHarry Course Advanced & Refined Systems Suite

This directory contains the projects inspired by **CodeWithHarry's "The Ultimate Job Ready Data Science Course"**, re-engineered and elevated to **production-grade enterprise standards**.

---

## 🏗️ Re-Engineered Projects Overview

### 1. Dragon Real Estate $\longrightarrow$ Enterprise Property Valuation & Investment ROI Engine
* **Original Course Implementation**: Basic Boston housing notebook predicting house price with Linear Regression & Decision Tree using default scikit-learn settings.
* **Our Production Elevation**:
  - Implements a scikit-learn `Pipeline` utilizing `SimpleImputer` and `RobustScaler` to handle extreme real estate price outliers.
  - Multi-model evaluation with regularized Gradient Boosting.
  - **Financial Investment Layer**: Computes Monthly Rental Yield, Net Operating Income (NOI), **Capitalization Rate (Cap Rate %)**, and **Cash-on-Cash Return** given financing and down payment.
  - Automated investment grading (`TIER_A_STRONG_BUY`, `TIER_B_MODERATE_HOLD`, `TIER_C_OVERVALUED`).

### 2. Coders of Delhi & Bangalore $\longrightarrow$ Tech Labor Market Intelligence & Skill Gap Optimizer
* **Original Course Implementation**: Basic Pandas exploratory data analysis (EDA), plotting histograms and filtering developer counts.
* **Our Production Elevation**:
  - Full machine learning regression model predicting compensation across tech hubs (Bangalore vs Delhi-NCR vs Remote).
  - Categorical `OneHotEncoder` and `ColumnTransformer` pipeline.
  - **Strategic Skill Gap & Salary Uplift Engine**: Dynamically calculates the projected salary uplift ($+\Delta\%$) when a developer learns in-demand technologies (Cloud/Docker/Kubernetes and Generative AI/RAG).

---

## 🛠️ How to Test
```bash
# Test Dragon Real Estate
python dragon_real_estate/test_dragon.py

# Test Coders Labor Market Engine
python coders_analytics/test_talent.py
```
