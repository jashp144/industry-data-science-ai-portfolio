import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.forecaster import DemandForecaster
from src.inventory_optimizer import InventoryOptimizer

app = FastAPI(
    title="AI Retail Demand & Safety Stock Optimization API",
    description="Multi-store time-series demand forecasting and automated replenishment planning engine.",
    version="1.0.0"
)

forecaster = DemandForecaster()
optimizer = InventoryOptimizer(service_level=0.95)

class ForecastPlanRequest(BaseModel):
    unit_price: float = Field(..., example=29.99)
    is_promo: int = Field(..., ge=0, le=1, example=1)
    is_holiday: int = Field(..., ge=0, le=1, example=0)
    day_of_week: int = Field(..., ge=0, le=6, example=5)
    day_of_month: int = Field(..., ge=1, le=31, example=20)
    month: int = Field(..., ge=1, le=12, example=10)
    is_weekend: int = Field(..., ge=0, le=1, example=1)
    lag_1: float = Field(..., example=110.0)
    lag_7: float = Field(..., example=115.0)
    lag_14: float = Field(..., example=108.0)
    rolling_mean_7: float = Field(..., example=105.0)
    rolling_std_7: float = Field(..., example=12.5)
    rolling_mean_14: float = Field(..., example=102.0)
    current_on_hand_inventory: int = Field(..., example=250)
    lead_time_days: int = Field(default=5, example=5)

@app.on_event("startup")
def startup():
    forecaster.load()

@app.get("/")
def root():
    return {
        "service": "AI Retail Demand & Inventory Optimization API",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/metrics")
def get_metrics():
    forecaster.load()
    return forecaster.metrics

@app.post("/forecast_and_plan")
def forecast_and_plan(req: ForecastPlanRequest):
    try:
        data_dict = req.model_dump()
        curr_inv = data_dict.pop("current_on_hand_inventory")
        lead_time = data_dict.pop("lead_time_days")

        predicted_demand = forecaster.forecast_next_period(data_dict)
        plan = optimizer.calculate_replenishment_plan(
            predicted_daily_demand=predicted_demand,
            demand_std_dev=data_dict["rolling_std_7"],
            current_on_hand_inventory=curr_inv,
            lead_time_days=lead_time
        )

        return {
            "forecast": {
                "predicted_daily_units": predicted_demand
            },
            "inventory_replenishment_plan": plan
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
