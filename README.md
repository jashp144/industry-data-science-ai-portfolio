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
├── 01_ecommerce_churn_ltv_pipeline/   # Dual-target retention & LTV modeling with PSI drift & explainability
├── 02_enterprise_rag_knowledge_engine/# Hybrid Dense-Sparse RRF vector search, anti-hallucination guardrails
├── 03_financial_fraud_detection_system/# Balanced RF + Isolation Forest ensemble with cost-sensitive cutoff
├── 04_retail_demand_forecasting_engine/# GBDT time-series regression & Monte Carlo inventory stockout simulation
├── 05_automated_sql_analytics_agent/  # Text-to-SQL translation with Window Functions, RFM NTILE(4) & AST safety
├── 06_codewithharry_course_advanced_suite/
│   ├── dragon_real_estate/            # Outlier-resistant valuation with Cap Rate, DSCR & Rate Sensitivity Matrix
│   └── coders_analytics/              # Tech labor market regression & dynamic skill gap uplift model
├── Dockerfile                         # Production container definition
├── docker-compose.yml                 # Multi-service composition
├── requirements.txt                   # Locked dependencies
├── run_all_tests.py                   # Master test runner across all 8 suites (7 systems + dashboard)
├── test_unified_dashboard.py          # Full HTTP integration test for all API endpoints
└── unified_dashboard.py               # Central interactive web management console (FastAPI + Bootstrap)
```

---

## Systems Overview

| # | System | Domain | Tech Stack | Production Highlights |
|:---:|---|---|---|---|
| **01** | **E-Commerce Churn & LTV Pipeline** | SaaS / Retail | Scikit-Learn, FastAPI, SciPy | Dual-target retention & LTV modeling with **Population Stability Index (PSI)** and **Kolmogorov-Smirnov tests** for drift detection, plus **Explainable AI feature attribution** ranking top churn drivers per user. |
| **02** | **Enterprise Hybrid RAG Engine** | NLP / Information Retrieval | BM25, TF-IDF Cosine, FastAPI | **Hybrid Dense-Sparse retrieval** powered by **Reciprocal Rank Fusion (RRF, k=60)** and strict **anti-hallucination guardrail thresholds** refusing ungrounded queries with chunk citations. |
| **03** | **Financial Fraud Detection Engine** | Fintech / Payments | Balanced RF, Isolation Forest | Solves extreme class imbalance (2% fraud) using a **hybrid supervised + unsupervised anomaly composite score** ($0.70 \times \text{BalancedRF} + 0.30 \times \text{IsoForest}$) optimizing a business loss matrix ($400 false negative vs $15 false positive). |
| **04** | **Retail Demand & Inventory Optimizer** | Supply Chain / Ops | GBDT, Monte Carlo Simulation | Multi-store lag ($t-1, t-7, t-14$) demand forecasting linked to **Safety Stock**, **Reorder Points (ROP)**, and **1,000-trial empirical Monte Carlo stockout simulations** under stochastic lead times. |
| **05** | **Automated SQL Analytics Agent** | Business Intelligence | SQLite, Window Functions, CTEs | Generates production SQL with `LAG()`, `DENSE_RANK()`, and **`NTILE(4)` RFM customer segmentation**, validates AST read-only safety, and outputs structured Executive Briefs with business findings. |
| **06A**| **Dragon Real Estate Valuation & ROI** | PropTech / Valuation | RobustScaler, GBDT, Financial Math | Property regression resistant to luxury price outliers, paired with an investment underwriting layer computing **Gross Rent Yield**, **Cap Rate %**, **DSCR**, and a **Mortgage Interest Rate Sensitivity Analysis Matrix** (5.0% - 8.5%). |
| **06B**| **Coders Labor Market Intelligence** | HR Tech / Talent Analytics | ColumnTransformer, GBDT | Tech compensation regression across Delhi-NCR and Bangalore with **dynamic skill gap uplift modeling** (+$\Delta$% for acquiring Cloud vs Generative AI competencies). |

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

Verify all systems simultaneously via the master test runner:

```bash
python run_all_tests.py
```

Expected output:
```text
==========================================================
  PORTFOLIO VERIFICATION SUMMARY
==========================================================
00 Unified Production API & Dashboard........ PASSED
01 E-Commerce Churn & LTV Pipeline........... PASSED
02 Enterprise RAG Knowledge Engine........... PASSED
03 Financial Fraud Detection System.......... PASSED
04 Retail Demand Forecasting Engine.......... PASSED
05 Automated SQL Analytics Agent............. PASSED
06A CWH Advanced Dragon Real Estate.......... PASSED
06B CWH Advanced Coders Labor Market......... PASSED

ALL 8 SUITES (7 SYSTEMS + DASHBOARD) 100% OPERATIONAL!
```

---

## Interactive Control Center

To launch the centralized management dashboard:

```bash
python unified_dashboard.py
```

Navigate to `http://localhost:8080` to interactively run inferences across all 7 systems: test explainable churn drivers, execute hybrid RRF knowledge search, score transactions with anomaly detection, simulate stockout risks, run RFM SQL segmentation, and stress-test property debt against fluctuating interest rates.

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
