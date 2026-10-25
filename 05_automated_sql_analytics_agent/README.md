# 📊 Project 5: Automated SQL Analytics & Executive Insights Agent (Text-to-SQL + RFM Segmentation)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![SQL](https://img.shields.io/badge/SQL-Window%20Functions%20%26%20CTEs-orange.svg)
![RFM](https://img.shields.io/badge/Analytics-NTILE(4)%20RFM%20Segmentation-purple.svg)
![Security](https://img.shields.io/badge/Security-AST%20Read--Only%20Guardrails-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Analytics%20Service-teal.svg)

## 📌 Executive Summary & Business Rationale
Data science candidates often know Python but stumble in technical rounds when asked about advanced SQL Window Functions or translating raw query results into strategic business decisions. 

This project implements an automated business intelligence and natural-language-to-SQL analytics agent:
1. **Advanced SQL Generation**: Automatically crafts complex analytical SQL with **Window Functions** (`NTILE(4)`, `LAG()`, `DENSE_RANK()`) and multi-stage **Common Table Expressions (CTEs)**.
2. **Customer RFM Segmentation**: Computes Recency, Frequency, and Monetary scores using `NTILE(4)` windowing to categorize enterprise accounts into actionable personas (*Champion / VIP*, *Promising / New*, *At Risk / High Value*, *Hibernating*).
3. **Custom SQL Execution Engine**: Accepts both natural language questions and direct raw SQL queries with microsecond execution latency profiling.
4. **Strict AST-Level Security Guardrails**: Sanitizes inputs and rejects all mutating operations (`DROP`, `DELETE`, `TRUNCATE`, `UPDATE`, `INSERT`, `ALTER`), enforcing read-only execution.
5. **Interactive Schema Explorer**: Queries database catalog metadata (`sqlite_master` and `PRAGMA table_info`) to provide real-time schema inspection.
6. **Automated Executive Brief Synthesis**: Translates result sets into a prioritized executive brief with key business findings and revenue metrics.

---

## 🏛️ System Architecture
```mermaid
flowchart TD
    UserQuery[Business Question or Custom SQL] --> Agent[SQL Analytics Agent]
    Agent --> Guardrail{AST Safety & Injection Guardrail}
    Guardrail -- Fails (Mutating DDL/DML Keyword) --> Reject[Reject with Security Error]
    Guardrail -- Passes (Read-Only SELECT/WITH) --> DB[(Enterprise Relational DB)]
    
    DB --> QueryExec[Execute Window Function & CTE Query]
    QueryExec --> Latency[Execution Latency Profiling < 5ms]
    QueryExec --> ResultDF[Structured Result DataFrame]
    
    ResultDF --> Synthesizer[Executive Insights Generator]
    Synthesizer --> Brief[Executive Brief: Key Findings + Personas + Revenue Growth]
```

---

## 📋 Engineering & Performance Highlights
- *Engineered an automated natural language analytics agent translating complex business queries into optimized SQL with Common Table Expressions (CTEs) and Window Functions (`NTILE(4)`, `LAG()`, `DENSE_RANK()`).*
- *Implemented RFM customer segmentation identifying VIP versus at-risk client accounts from 500+ relational enterprise transactions.*
- *Implemented AST-based SQL security guardrails sanitizing inputs and enforcing read-only safety with sub-5ms execution latency.*
- *Built an automated executive insights synthesis module translating analytical dataframes into prioritized strategic business recommendations.*

---

## 🛠️ How to Run & Verify
```bash
python test_sql_agent.py
```
To run the interactive standalone API:
```bash
python app.py
```
Open `http://localhost:8004/docs` in your browser.
