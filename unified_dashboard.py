import os
import sys
import uvicorn
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional

# Ensure project modules are on path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "01_ecommerce_churn_ltv_pipeline"))
sys.path.append(os.path.join(base_dir, "02_enterprise_rag_knowledge_engine"))
sys.path.append(os.path.join(base_dir, "03_financial_fraud_detection_system"))
sys.path.append(os.path.join(base_dir, "04_retail_demand_forecasting_engine"))
sys.path.append(os.path.join(base_dir, "05_automated_sql_analytics_agent"))

from src.pipeline.predict_pipeline import PredictPipeline, CustomerData
from src.pipeline.drift_monitor import DataDriftMonitor
from src.rag_engine import RAGEngine
from src.fraud_detector import FraudDetector
from src.forecaster import DemandForecaster
from src.inventory_optimizer import InventoryOptimizer
from src.db_manager import DatabaseManager
from src.sql_agent import SQLAnalyticsAgent
from src.insights_generator import ExecutiveInsightsGenerator

app = FastAPI(
    title="Unified Data Science & AI Portfolio Master Hub",
    description="Interactive Web Control Center demonstrating 5 production ML & GenAI systems.",
    version="2.0.0"
)

# Initialize engines
churn_pipeline = PredictPipeline()
drift_monitor = DataDriftMonitor()
rag_engine = RAGEngine()
fraud_detector = FraudDetector()
demand_forecaster = DemandForecaster()
inventory_optimizer = InventoryOptimizer(service_level=0.95)
sql_db = DatabaseManager()
sql_agent = SQLAnalyticsAgent(sql_db)
insights_gen = ExecutiveInsightsGenerator()

@app.on_event("startup")
def startup():
    rag_engine.load_index()
    fraud_detector.load()
    demand_forecaster.load()
    if not os.path.exists(sql_db.db_path):
        sql_db.initialize_and_seed_database()

# Pydantic Schemas
class ChurnRequest(BaseModel):
    tenure_months: int = 12
    contract: str = "Month-to-month"
    payment_method: str = "Credit Card"
    monthly_charges: float = 75.0
    total_charges: float = 900.0
    support_tickets: int = 2
    app_logins_per_month: int = 12
    cart_abandonment_rate: float = 0.3
    discount_applied: int = 0

class RAGRequest(BaseModel):
    query: str

class FraudRequest(BaseModel):
    amount: float = 450.0
    velocity_1h: int = 4
    device_risk_score: float = 0.65
    is_foreign_ip: int = 1
    transaction_hour: int = 2

class ForecastRequest(BaseModel):
    unit_price: float = 29.99
    is_promo: int = 1
    is_holiday: int = 0
    day_of_week: int = 5
    day_of_month: int = 20
    month: int = 10
    is_weekend: int = 1
    lag_1: float = 110.0
    lag_7: float = 115.0
    lag_14: float = 108.0
    rolling_mean_7: float = 105.0
    rolling_std_7: float = 12.5
    rolling_mean_14: float = 102.0
    current_inventory: int = 350
    lead_time_days: int = 5

class SQLRequest(BaseModel):
    question: str

# API Endpoints
@app.post("/api/churn")
def api_churn(req: ChurnRequest):
    cust = CustomerData(**req.model_dump())
    return churn_pipeline.predict(cust.get_data_as_data_frame())[0]

@app.post("/api/rag")
def api_rag(req: RAGRequest):
    return rag_engine.query(req.query)

@app.post("/api/fraud")
def api_fraud(req: FraudRequest):
    return fraud_detector.score_transaction(req.model_dump())

@app.post("/api/forecast")
def api_forecast(req: ForecastRequest):
    data = req.model_dump()
    curr_inv = data.pop("current_inventory")
    lead_time = data.pop("lead_time_days")
    pred = demand_forecaster.forecast_next_period(data)
    plan = inventory_optimizer.calculate_replenishment_plan(
        predicted_daily_demand=pred,
        demand_std_dev=data["rolling_std_7"],
        current_on_hand_inventory=curr_inv,
        lead_time_days=lead_time
    )
    return {"predicted_demand": pred, "plan": plan}

@app.post("/api/sql")
def api_sql(req: SQLRequest):
    raw_res = sql_agent.process_query(req.question)
    if raw_res.get("status") == "REJECTED":
        return raw_res
    brief = insights_gen.generate_brief(raw_res)
    raw_res.pop("dataframe", None)
    return {"query": raw_res, "insights": brief}

# Front-End UI
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI & Data Science Production Portfolio Master Hub</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .navbar { background-color: #1e293b; border-bottom: 1px solid #334155; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }
        .nav-tabs .nav-link { color: #94a3b8; font-weight: 500; border: none; }
        .nav-tabs .nav-link.active { color: #38bdf8; background: transparent; border-bottom: 3px solid #38bdf8; }
        .btn-primary { background-color: #0284c7; border-color: #0284c7; }
        .btn-primary:hover { background-color: #0369a1; }
        .badge-status { font-size: 0.9rem; padding: 6px 12px; border-radius: 6px; }
        pre { background-color: #0b1329; border: 1px solid #334155; border-radius: 8px; padding: 12px; color: #38bdf8; }
        .table { color: #f8fafc; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark px-4 py-3">
        <span class="navbar-brand mb-0 h1"><i class="fa-solid fa-brain text-info me-2"></i> AI & Data Science Production Portfolio</span>
        <span class="badge bg-success">5/5 Microservices Online</span>
    </nav>

    <div class="container my-4">
        <ul class="nav nav-tabs mb-4" id="portfolioTabs" role="tablist">
            <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-churn"><i class="fa-solid fa-cart-shopping me-1"></i> 1. Churn & LTV</button></li>
            <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-rag"><i class="fa-solid fa-book-open me-1"></i> 2. Enterprise RAG</button></li>
            <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-fraud"><i class="fa-solid fa-shield-halved me-1"></i> 3. Fraud Detection</button></li>
            <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-demand"><i class="fa-solid fa-chart-line me-1"></i> 4. Demand Forecaster</button></li>
            <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-sql"><i class="fa-solid fa-database me-1"></i> 5. SQL Analytics Agent</button></li>
        </ul>

        <div class="tab-content">
            <!-- TAB 1: CHURN & LTV -->
            <div class="tab-pane fade show active" id="tab-churn">
                <div class="card p-4">
                    <h4 class="text-info mb-3"><i class="fa-solid fa-users-slash me-2"></i>E-Commerce Churn & Lifetime Value (LTV) Prediction</h4>
                    <p class="text-secondary">Predicts probability of customer attrition and future dollar LTV with Scikit-Learn MLOps pipeline.</p>
                    <div class="row g-3">
                        <div class="col-md-3">
                            <label class="form-label">Tenure (Months)</label>
                            <input type="number" id="churn-tenure" class="form-control bg-dark text-white border-secondary" value="6">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">Monthly Charges ($)</label>
                            <input type="number" id="churn-monthly" class="form-control bg-dark text-white border-secondary" value="85.0">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">Support Tickets</label>
                            <input type="number" id="churn-tickets" class="form-control bg-dark text-white border-secondary" value="4">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">Contract Type</label>
                            <select id="churn-contract" class="form-select bg-dark text-white border-secondary">
                                <option value="Month-to-month">Month-to-month</option>
                                <option value="One year">One year</option>
                                <option value="Two year">Two year</option>
                            </select>
                        </div>
                    </div>
                    <div class="mt-4">
                        <button class="btn btn-primary" onclick="runChurn()"><i class="fa-solid fa-bolt me-1"></i> Run Prediction</button>
                    </div>
                    <div id="churn-result" class="mt-4 d-none">
                        <h5>Prediction Result:</h5>
                        <pre id="churn-json"></pre>
                    </div>
                </div>
            </div>

            <!-- TAB 2: RAG -->
            <div class="tab-pane fade" id="tab-rag">
                <div class="card p-4">
                    <h4 class="text-info mb-3"><i class="fa-solid fa-magnifying-glass me-2"></i>Enterprise Production RAG Knowledge Engine</h4>
                    <p class="text-secondary">Retrieval-augmented Q&A with cosine vector search and anti-hallucination guardrail thresholding.</p>
                    <div class="mb-3">
                        <label class="form-label">Enter Question:</label>
                        <input type="text" id="rag-query" class="form-control bg-dark text-white border-secondary" value="What is the refund window for cancelled subscriptions?">
                    </div>
                    <button class="btn btn-primary" onclick="runRAG()"><i class="fa-solid fa-robot me-1"></i> Query Knowledge Base</button>
                    <div id="rag-result" class="mt-4 d-none">
                        <h5>RAG Response:</h5>
                        <pre id="rag-json"></pre>
                    </div>
                </div>
            </div>

            <!-- TAB 3: FRAUD -->
            <div class="tab-pane fade" id="tab-fraud">
                <div class="card p-4">
                    <h4 class="text-info mb-3"><i class="fa-solid fa-credit-card me-2"></i>Real-Time Financial Fraud Detection</h4>
                    <p class="text-secondary">Sub-15ms transaction scoring optimizing business loss threshold ($400 false negative vs $15 false positive).</p>
                    <div class="row g-3">
                        <div class="col-md-3">
                            <label class="form-label">Transaction Amount ($)</label>
                            <input type="number" id="fraud-amount" class="form-control bg-dark text-white border-secondary" value="1250.0">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">Velocity (Tx in last 1 hr)</label>
                            <input type="number" id="fraud-velocity" class="form-control bg-dark text-white border-secondary" value="7">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">Device Risk Score (0 - 1)</label>
                            <input type="number" step="0.1" id="fraud-device" class="form-control bg-dark text-white border-secondary" value="0.9">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">Transaction Hour (0 - 23)</label>
                            <input type="number" id="fraud-hour" class="form-control bg-dark text-white border-secondary" value="3">
                        </div>
                    </div>
                    <div class="mt-4">
                        <button class="btn btn-primary" onclick="runFraud()"><i class="fa-solid fa-fingerprint me-1"></i> Score Transaction</button>
                    </div>
                    <div id="fraud-result" class="mt-4 d-none">
                        <h5>Scoring Decision:</h5>
                        <pre id="fraud-json"></pre>
                    </div>
                </div>
            </div>

            <!-- TAB 4: DEMAND -->
            <div class="tab-pane fade" id="tab-demand">
                <div class="card p-4">
                    <h4 class="text-info mb-3"><i class="fa-solid fa-boxes-packing me-2"></i>Retail Demand Forecasting & Safety Stock Optimizer</h4>
                    <p class="text-secondary">Gradient Boosting multi-store time-series forecaster with Safety Stock and Reorder Point mathematics.</p>
                    <div class="row g-3">
                        <div class="col-md-4">
                            <label class="form-label">Unit Price ($)</label>
                            <input type="number" id="demand-price" class="form-control bg-dark text-white border-secondary" value="29.99">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Current Warehouse Stock</label>
                            <input type="number" id="demand-inv" class="form-control bg-dark text-white border-secondary" value="250">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Supplier Lead Time (Days)</label>
                            <input type="number" id="demand-lead" class="form-control bg-dark text-white border-secondary" value="5">
                        </div>
                    </div>
                    <div class="mt-4">
                        <button class="btn btn-primary" onclick="runForecast()"><i class="fa-solid fa-chart-simple me-1"></i> Calculate Forecast & Replenishment</button>
                    </div>
                    <div id="demand-result" class="mt-4 d-none">
                        <h5>Supply Chain Plan:</h5>
                        <pre id="demand-json"></pre>
                    </div>
                </div>
            </div>

            <!-- TAB 5: SQL -->
            <div class="tab-pane fade" id="tab-sql">
                <div class="card p-4">
                    <h4 class="text-info mb-3"><i class="fa-solid fa-terminal me-2"></i>Automated SQL Analytics & Executive Insights Agent</h4>
                    <p class="text-secondary">Translates plain business queries into optimized SQL with Window Functions and CTEs, generating executive findings.</p>
                    <div class="mb-3">
                        <label class="form-label">Business Question:</label>
                        <select id="sql-question" class="form-select bg-dark text-white border-secondary">
                            <option value="Show me monthly revenue growth trend">Show me monthly revenue growth trend (LAG Window Function)</option>
                            <option value="Who are the highest spending top customers?">Who are the highest spending top customers? (DENSE_RANK Window Function)</option>
                            <option value="What are product category profit margins?">What are product category profit margins?</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" onclick="runSQL()"><i class="fa-solid fa-code me-1"></i> Generate SQL & Brief</button>
                    <div id="sql-result" class="mt-4 d-none">
                        <h5>Analytics Output:</h5>
                        <pre id="sql-json"></pre>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function runChurn() {
            const body = {
                tenure_months: parseInt(document.getElementById('churn-tenure').value),
                monthly_charges: parseFloat(document.getElementById('churn-monthly').value),
                support_tickets: parseInt(document.getElementById('churn-tickets').value),
                contract: document.getElementById('churn-contract').value,
                payment_method: 'Credit Card',
                total_charges: 900.0,
                app_logins_per_month: 12,
                cart_abandonment_rate: 0.3,
                discount_applied: 0
            };
            const res = await fetch('/api/churn', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
            const data = await res.json();
            document.getElementById('churn-result').classList.remove('d-none');
            document.getElementById('churn-json').innerText = JSON.stringify(data, null, 2);
        }

        async function runRAG() {
            const query = document.getElementById('rag-query').value;
            const res = await fetch('/api/rag', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query}) });
            const data = await res.json();
            document.getElementById('rag-result').classList.remove('d-none');
            document.getElementById('rag-json').innerText = JSON.stringify(data, null, 2);
        }

        async function runFraud() {
            const body = {
                amount: parseFloat(document.getElementById('fraud-amount').value),
                velocity_1h: parseInt(document.getElementById('fraud-velocity').value),
                device_risk_score: parseFloat(document.getElementById('fraud-device').value),
                is_foreign_ip: 1,
                transaction_hour: parseInt(document.getElementById('fraud-hour').value)
            };
            const res = await fetch('/api/fraud', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
            const data = await res.json();
            document.getElementById('fraud-result').classList.remove('d-none');
            document.getElementById('fraud-json').innerText = JSON.stringify(data, null, 2);
        }

        async function runForecast() {
            const body = {
                unit_price: parseFloat(document.getElementById('demand-price').value),
                is_promo: 1,
                is_holiday: 0,
                day_of_week: 5,
                day_of_month: 20,
                month: 10,
                is_weekend: 1,
                lag_1: 110.0,
                lag_7: 115.0,
                lag_14: 108.0,
                rolling_mean_7: 105.0,
                rolling_std_7: 12.5,
                rolling_mean_14: 102.0,
                current_inventory: parseInt(document.getElementById('demand-inv').value),
                lead_time_days: parseInt(document.getElementById('demand-lead').value)
            };
            const res = await fetch('/api/forecast', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
            const data = await res.json();
            document.getElementById('demand-result').classList.remove('d-none');
            document.getElementById('demand-json').innerText = JSON.stringify(data, null, 2);
        }

        async function runSQL() {
            const question = document.getElementById('sql-question').value;
            const res = await fetch('/api/sql', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question}) });
            const data = await res.json();
            document.getElementById('sql-result').classList.remove('d-none');
            document.getElementById('sql-json').innerText = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
