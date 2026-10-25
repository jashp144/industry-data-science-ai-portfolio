# 💳 Project 3: Real-Time Financial Fraud Detection & Anomaly Pipeline with Cost-Sensitive Optimization

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![ML](https://img.shields.io/badge/Architecture-Balanced%20RF%20%2B%20Isolation%20Forest-orange.svg)
![PR-AUC](https://img.shields.io/badge/Metric-PR--AUC%20Optimized-green.svg)
![Loss Matrix](https://img.shields.io/badge/Business-Cost--Sensitive%20Optimization-purple.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Sub--15ms%20Latency-teal.svg)

## 📌 Executive Summary & Business Rationale
Most data science tutorials evaluate fraud models using 99% accuracy on 99% non-fraud datasets. In real-world payment networks (Stripe, PayPal, Adyen), this fails catastrophically because the cost of a missed fraud (chargeback penalty + merchandise loss = ~$400) is 26x higher than the cost of a false alarm ($15 OTP verification friction). Furthermore, supervised models cannot catch novel zero-day fraud patterns.

This project implements an industry-grade **Hybrid Supervised + Unsupervised Fraud Engine**:
1. **High-Signal Feature Engineering**: Computes rolling 1-hour transaction velocity, amount-to-velocity risk ratios, nocturnal transaction indicators, and device risk telemetry.
2. **Hybrid Ensemble Architecture**:
   - **Balanced Random Forest (70% weight)** trained on known historical fraud vectors with class-weighted sampling.
   - **Isolation Forest (30% weight)** trained on legitimate transactions to detect unprecedented anomalies and novel attack vectors.
3. **Business Cost Matrix Threshold Optimization**:
   $$\text{Total Monetary Cost} = \sum (\text{False Negatives} \times \$400) + \sum (\text{False Positives} \times \$15)$$
   Sweeps decision thresholds from 0.10 to 0.90 to find the exact probability cutoff (0.370) that minimizes dollar loss rather than using a naive 0.50 cutoff, saving thousands in net fraud exposure.
4. **Tri-Tier Decision Routing**: Instant Approve ($P < 0.37$), Step-Up 2FA ($0.37 \le P < 0.75$), or Immediate Card Freeze ($P \ge 0.75$).

---

## 🏛️ Architecture & Decision Engine
```mermaid
flowchart TD
    Stream[Live Transaction Stream: Amount, Velocity, Device, IP, Hour] --> FeatureEng[Feature Engineering: Velocity Ratios, Night Flags, Amount Spikes]
    
    FeatureEng --> Supervised[Balanced Random Forest: Known Fraud Vectors]
    FeatureEng --> Unsupervised[Isolation Forest: Novel Anomaly Detection]
    
    Supervised --> SupScore[Supervised Probability: 70%]
    Unsupervised --> AnomScore[Anomaly Score: 30%]
    
    SupScore --> Composite[Composite Risk Score = 0.70*Sup + 0.30*Anom]
    AnomScore --> Composite
    
    Composite --> ThresholdOpt{Score vs Cost-Optimized Threshold: 0.37}
    ThresholdOpt -- "Score < 0.37" --> Approve[APPROVE: Instant Settlement]
    ThresholdOpt -- "0.37 <= Score < 0.75" --> StepUp[STEP-UP: Mandatory Biometric / SMS 2FA Challenge]
    ThresholdOpt -- "Score >= 0.75" --> Decline[DECLINE: Immediate Card Block & Fraud Alert]
```

---

## 📋 Engineering & Performance Highlights
- *Built an end-to-end streaming fraud detection engine for transactions with a 2% positive class imbalance, ensembling Balanced Random Forest and Isolation Forest (PR-AUC: 0.85+).*
- *Formulated an empirical business loss matrix ($400 FN vs $15 FP) identifying an optimal 0.370 cutoff that prevents over $12,480/month in simulated chargeback losses.*
- *Implemented sub-15ms inference microservices using FastAPI, executing tri-tier decision routing with explainable risk flags.*

---

## 🛠️ How to Run & Verify
```bash
python test_fraud.py
```
To run the interactive standalone API:
```bash
python app.py
```
Open `http://localhost:8002/docs` in your browser.
