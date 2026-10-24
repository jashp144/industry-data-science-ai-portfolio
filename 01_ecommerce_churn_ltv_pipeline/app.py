import os
import sys
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.predict_pipeline import PredictPipeline, CustomerData
from src.pipeline.drift_monitor import DataDriftMonitor

app = FastAPI(
    title="E-Commerce Churn & LTV MLOps Service",
    description="Production-grade ML API providing dual customer retention scoring, Lifetime Value (LTV) estimation, and statistical data drift detection.",
    version="1.0.0"
)

pipeline = PredictPipeline()
drift_monitor = DataDriftMonitor()

class CustomerInput(BaseModel):
    tenure_months: int = Field(..., ge=1, le=120, example=12)
    contract: str = Field(..., example="Month-to-month")
    payment_method: str = Field(..., example="Credit Card")
    monthly_charges: float = Field(..., ge=0, example=75.50)
    total_charges: float = Field(..., ge=0, example=906.00)
    support_tickets: int = Field(..., ge=0, example=2)
    app_logins_per_month: int = Field(..., ge=0, example=15)
    cart_abandonment_rate: float = Field(..., ge=0.0, le=1.0, example=0.25)
    discount_applied: int = Field(..., ge=0, le=1, example=0)

class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    churn_risk_tier: str
    estimated_ltv_usd: float
    recommended_action: str

@app.get("/")
def read_root():
    return {
        "service": "E-Commerce Churn & Lifetime Value MLOps API",
        "status": "online",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "check_drift": "/check_drift"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerInput):
    try:
        cust_obj = CustomerData(**customer.model_dump())
        df = cust_obj.get_data_as_data_frame()
        result = pipeline.predict(df)[0]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=List[PredictionResponse])
def predict_batch(customers: List[CustomerInput]):
    try:
        records = [c.model_dump() for c in customers]
        df = pd.DataFrame(records)
        results = pipeline.predict(df)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_drift")
def check_drift(batch: List[CustomerInput]):
    try:
        records = [c.model_dump() for c in batch]
        df = pd.DataFrame(records)
        drift_report = drift_monitor.evaluate_drift(df)
        return drift_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
