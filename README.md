# 🚀 Production-Grade Data Science & AI Portfolio (2025–2026)

Welcome to the **End-to-End Industry-Grade Data Science & AI Repository**. 
This repository is engineered to transcend cookie-cutter tutorial projects (such as Titanic or Iris) by demonstrating **production-ready MLOps, high-impact business problem solving, real-time streaming, GenAI RAG architectures, and advanced SQL analytics**.

---

## 📂 Projects Overview

| # | Project Name | Domain | Core Tech Stack | Key Differentiator |
|---|---|---|---|---|
| **01** | [**E-Commerce Churn & LTV MLOps Pipeline**](./01_ecommerce_churn_ltv_pipeline) | E-Commerce / SaaS | Scikit-Learn, FastAPI, KS-Test, PSI | Real-time statistical **data drift monitoring** & dual-target prediction (Churn % + LTV $) |
| **02** | [**Enterprise Production RAG Knowledge Engine**](./02_enterprise_rag_knowledge_engine) | GenAI / NLP | Vector Search, Cosine Sim, Guardrails, FastAPI | **Anti-hallucination guardrail** with threshold calibration & exact source citations |
| **03** | [**Real-Time Financial Fraud Detection System**](./03_financial_fraud_detection_system) | Fintech / Banking | Balanced Ensemble, PR-AUC, FastAPI | **Business cost matrix optimization** ($400 false negative vs $15 false positive) |
| **04** | [**AI Retail Demand & Safety Stock Optimizer**](./04_retail_demand_forecasting_engine) | Retail / Supply Chain | Time Series, GBDT, Operations Research | Multi-store lag & rolling feature engineering + **Safety Stock & Reorder Point math** |
| **05** | [**Automated SQL Analytics & Executive Insights Agent**](./05_automated_sql_analytics_agent) | Analytics / Business Intelligence | SQLite, Window Functions, CTEs, AST Guardrails | Translates English queries into complex SQL (`LAG()`, `DENSE_RANK()`) + **Executive Briefs** |

---

## ⚡ Quick Start: Verify All 5 Projects

Run the master test runner to execute unit test suites across all 5 systems in one command:

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

ALL 5 PROJECTS VERIFIED AND OPERATIONAL!
```

---

## 🐙 How to Push this Entire Portfolio to Your GitHub

To showcase this on your GitHub profile:
1. Open PowerShell / Terminal in this folder:
   ```bash
   cd C:\Users\jashp\.gemini\antigravity\scratch\data_science_portfolio_projects
   ```
2. Initialize Git:
   ```bash
   git init
   git add .
   git commit -m "feat: initial release of 5 production data science & AI portfolio projects"
   ```
3. Create a new public repository on GitHub (e.g. `industry-data-science-ai-portfolio`).
4. Link and push:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
   git branch -M main
   git push -u origin main
   ```

---

## 💼 Master Resume Bullet Points (Ready to Copy-Paste)

- **Machine Learning & MLOps**: *Architected 5 end-to-end production ML and AI microservices using Scikit-Learn, FastAPI, and automated pipelines, incorporating statistical data drift detection (KS-test, PSI) and business-cost threshold optimization.*
- **Generative AI & RAG**: *Engineered a production-ready Retrieval-Augmented Generation (RAG) knowledge engine with recursive semantic text chunking, sublinear TF-IDF vector embeddings, and strict anti-hallucination guardrails with document provenance.*
- **Time Series & Operations Research**: *Constructed a multi-store demand forecasting system leveraging lag and rolling window features (WAPE < 10%), coupled with statistical inventory optimization calculating dynamic safety stock and Reorder Points (ROP).*
- **SQL & Analytics Engineering**: *Designed an automated natural language SQL analytics agent executing advanced Window Functions (`LAG()`, `DENSE_RANK()`) and Common Table Expressions (CTEs), paired with automated executive insight synthesis.*
