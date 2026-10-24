# Production Data Science & Machine Learning Systems Suite

A collection of 7 end-to-end machine learning, operations research, and retrieval systems built with modular Python architectures, statistical monitoring, and automated test coverage.

[![CI/CD Pipeline](https://github.com/jashp144/industry-data-science-ai-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/jashp144/industry-data-science-ai-portfolio/actions)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-F7931E.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)

---

## Systems Architecture & Directory Structure

```text
industry-data-science-ai-portfolio/
├── .github/workflows/ci.yml           # Automated CI/CD test suite on Python 3.11 & 3.12
├── 01_ecommerce_churn_ltv_pipeline/   # Dual-target retention & LTV modeling with PSI drift monitoring
├── 02_enterprise_rag_knowledge_engine/# Dense vector search, anti-hallucination thresholds & citations
├── 03_financial_fraud_detection_system/# Imbalanced streaming classification & cost-sensitive cutoff optimization
├── 04_retail_demand_forecasting_engine/# Time-series lag feature engineering & inventory replenishment (Safety Stock/ROP)
├── 05_automated_sql_analytics_agent/  # Text-to-SQL translation with Window Functions, CTEs & AST guardrails
├── 06_codewithharry_course_advanced_suite/
│   ├── dragon_real_estate/            # Robust property valuation with Cap Rate & Cash-on-Cash ROI math
│   └── coders_analytics/              # Tech labor market regression & dynamic skill gap uplift model
├── Dockerfile                         # Production container definition
├── docker-compose.yml                 # Multi-service composition
├── requirements.txt                   # Locked dependencies
├── run_all_tests.py                   # Master test runner across all 7 systems
└── unified_dashboard.py               # Central interactive web management console (FastAPI + Bootstrap)
```

---

## Systems Overview

| # | System | Domain | Tech Stack | Production Highlights |
|:---:|---|---|---|---|
| **01** | **E-Commerce Churn & LTV Pipeline** | SaaS / Retail | Scikit-Learn, FastAPI, SciPy | Predicts churn probability and lifetime value ($). Implements **Population Stability Index (PSI)** and **Kolmogorov-Smirnov tests** to flag feature drift before model performance degrades. |
| **02** | **Enterprise RAG Knowledge Engine** | NLP / Information Retrieval | Vector Search, Cosine Sim, FastAPI | Recursive semantic chunking (300 chars, 50 overlap) with strict **anti-hallucination guardrail thresholds** refusing ungrounded queries with exact chunk provenance. |
| **03** | **Financial Fraud Detection Engine** | Fintech / Payments | Balanced Ensembles, PR-AUC | Solves extreme class imbalance (2% fraud) and computes a **business loss matrix** ($400 false negative vs $15 false positive) to identify the optimal profit-preserving cutoff. |
| **04** | **Retail Demand & Inventory Optimizer** | Supply Chain / Ops | GBDT, Lag/Rolling Features | Multi-store lag ($t-1, t-7, t-14$) demand forecasting (WAPE < 10%) linked to **Safety Stock** ($SS = Z \cdot \sigma_d \cdot \sqrt{L}$), **Reorder Points (ROP)**, and **EOQ**. |
| **05** | **Automated SQL Analytics Agent** | Business Intelligence | SQLite, Window Functions, CTEs | Generates production SQL with `LAG()` and `DENSE_RANK()`, validates AST read-only safety, and outputs structured Executive Briefs with business findings. |
| **06A**| **Dragon Real Estate Valuation & ROI** | PropTech / Valuation | RobustScaler, GBDT, Financial Math | Property regression resistant to luxury price outliers, paired with an investment analysis layer computing **Gross Rent Yield**, **Cap Rate %**, and **Cash-on-Cash Return**. |
| **06B**| **Coders Labor Market Intelligence** | HR Tech / Talent Analytics | ColumnTransformer, GBDT | Tech compensation regression across Delhi-NCR, Bangalore, and Remote roles with **dynamic skill gap uplift modeling** (+$\Delta$% for learning Cloud vs GenAI). |

---

## Quick Start & Verification

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/jashp144/industry-data-science-ai-portfolio.git
cd industry-data-science-ai-portfolio
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Test Suites

Verify all 7 systems simultaneously via the master test runner:

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

## Interactive Control Center

To launch the centralized management dashboard:

```bash
python unified_dashboard.py
```

Navigate to `http://localhost:8080` to interactively run inferences, test data drift scenarios, query the RAG knowledge base, score financial transactions, and inspect generated SQL queries.

---

## Container Deployment

Run the complete portfolio in an isolated Docker container:

```bash
docker compose up --build
```

Access the service at `http://localhost:8080`.

---

## Author

**Jash Patel**  
GitHub: [@jashp144](https://github.com/jashp144)  
License: MIT
