# 🚀 Production-Grade Data Science & AI Portfolio (2025–2026)

Welcome to the **End-to-End Industry-Grade Data Science & AI Repository**. 
This repository is engineered to transcend cookie-cutter tutorial projects (such as Titanic or Iris) by demonstrating **production-ready MLOps, high-impact business problem solving, real-time streaming, GenAI RAG architectures, advanced SQL analytics, and re-engineered CodeWithHarry course systems**.

---

## 📂 Projects & Systems Overview

| # | System / Project Name | Domain | Core Tech Stack | Key Industry Differentiator |
|---|---|---|---|---|
| **01** | [**E-Commerce Churn & LTV MLOps Pipeline**](./01_ecommerce_churn_ltv_pipeline) | E-Commerce / SaaS | Scikit-Learn, FastAPI, KS-Test, PSI | Real-time statistical **data drift monitoring** & dual-target prediction (Churn % + LTV $) |
| **02** | [**Enterprise Production RAG Knowledge Engine**](./02_enterprise_rag_knowledge_engine) | GenAI / NLP | Vector Search, Cosine Sim, Guardrails, FastAPI | **Anti-hallucination guardrail** with threshold calibration & exact source citations |
| **03** | [**Real-Time Financial Fraud Detection System**](./03_financial_fraud_detection_system) | Fintech / Banking | Balanced Ensemble, PR-AUC, FastAPI | **Business cost matrix optimization** ($400 false negative vs $15 false positive) |
| **04** | [**AI Retail Demand & Safety Stock Optimizer**](./04_retail_demand_forecasting_engine) | Retail / Supply Chain | Time Series, GBDT, Operations Research | Multi-store lag & rolling feature engineering + **Safety Stock & Reorder Point math** |
| **05** | [**Automated SQL Analytics & Executive Insights Agent**](./05_automated_sql_analytics_agent) | Analytics / Business Intelligence | SQLite, Window Functions, CTEs, AST Guardrails | Translates English queries into complex SQL (`LAG()`, `DENSE_RANK()`) + **Executive Briefs** |
| **06A**| [**CWH Re-Engineered: Dragon Real Estate**](./06_codewithharry_course_advanced_suite/dragon_real_estate) | PropTech / Valuation | Scikit-Learn, RobustScaler, GBDT | House price regression elevated with **Rental Yield, Cap Rate % & Cash-on-Cash ROI** |
| **06B**| [**CWH Re-Engineered: Coders Labor Market**](./06_codewithharry_course_advanced_suite/coders_analytics) | HR Tech / Analytics | Target Encoding, GBDT, Labor Economics | Salary prediction across Bangalore/Delhi-NCR + **Dynamic Skill Gap & Salary Uplift Recommender** |

---

## ⚡ Quick Start: Verify All 7 Systems in 1 Command

Run the master test runner to execute unit test suites across all 7 systems:

```bash
python run_all_tests.py
```

Expected output:
```text
==========================================================
  PORTFOLIO VERIFICATION SUMMARY
==========================================================
01 E-Commerce Churn & LTV Pipeline........... PASSED
02 Enterprise RAG Knowledge Engine........... PASSED
03 Financial Fraud Detection System.......... PASSED
04 Retail Demand Forecasting Engine.......... PASSED
05 Automated SQL Analytics Agent............. PASSED
06A CWH Advanced Dragon Real Estate.......... PASSED
06B CWH Advanced Coders Labor Market......... PASSED

ALL 7 SYSTEMS VERIFIED AND 100% OPERATIONAL!
```

---

## 🖥️ Launch the Unified Web Dashboard

Test all systems visually in an interactive dark-mode dashboard:
```bash
python unified_dashboard.py
```
Open **`http://localhost:8080`** in your browser.

---

## 🐙 Pushing this Entire Portfolio to Your GitHub

1. Ensure you have created the empty repository on GitHub:
   - Go to **[https://github.com/new](https://github.com/new)**
   - Name the repository: `industry-data-science-ai-portfolio`
   - Keep it **Public** and do not add README/license (we already have them here)
   - Click **Create repository**
2. In your terminal, run:
   ```bash
   git push -u origin main
   ```
*(Git Credential Manager will use your saved GitHub credentials and push automatically!)*
