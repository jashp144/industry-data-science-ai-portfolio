import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.fraud_detector import FraudDetector

app = FastAPI(
    title="Financial Fraud & Anomaly Scoring API",
    description="Real-time transaction fraud scoring microservice with business cost-optimized thresholding and step-up auth triggers.",
    version="1.0.0"
)

detector = FraudDetector()

class TransactionRequest(BaseModel):
    amount: float = Field(..., ge=0.01, example=850.00)
    velocity_1h: int = Field(..., ge=0, example=6)
    device_risk_score: float = Field(..., ge=0.0, le=1.0, example=0.82)
    is_foreign_ip: int = Field(..., ge=0, le=1, example=1)
    transaction_hour: int = Field(..., ge=0, le=23, example=3)

class ScoringResponse(BaseModel):
    fraud_probability: float
    optimal_threshold: float
    decision: str
    recommended_action: str
    risk_factors: dict

@app.on_event("startup")
def startup_event():
    detector.load()

@app.get("/")
def root():
    return {
        "service": "Financial Fraud & Anomaly Detection API",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/metrics")
def get_metrics():
    detector.load()
    return detector.metrics

@app.post("/score", response_model=ScoringResponse)
def score_transaction(req: TransactionRequest):
    try:
        res = detector.score_transaction(req.model_dump())
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
