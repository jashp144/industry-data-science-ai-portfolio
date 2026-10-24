import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.db_manager import DatabaseManager
from src.sql_agent import SQLAnalyticsAgent
from src.insights_generator import ExecutiveInsightsGenerator

app = FastAPI(
    title="Automated SQL Analytics & Executive Insights Agent API",
    description="Natural language to SQL agent with Window functions, CTEs, security guardrails, and executive insights synthesis.",
    version="1.0.0"
)

db = DatabaseManager()
sql_agent = SQLAnalyticsAgent(db)
insights_gen = ExecutiveInsightsGenerator()

class AnalyticsQueryRequest(BaseModel):
    question: str = Field(..., example="Show me top customer rankings and spend breakdown.")

@app.on_event("startup")
def startup():
    if not os.path.exists(db.db_path):
        db.initialize_and_seed_database()

@app.get("/")
def root():
    return {
        "service": "Automated SQL Analytics & Insights API",
        "status": "online",
        "docs": "/docs"
    }

@app.post("/ask")
def ask_analytics(req: AnalyticsQueryRequest):
    try:
        raw_res = sql_agent.process_query(req.question)
        if raw_res.get("status") == "REJECTED":
            return raw_res

        insights = insights_gen.generate_brief(raw_res)
        # Drop raw dataframe before returning JSON
        raw_res.pop("dataframe", None)
        return {
            "query_metadata": raw_res,
            "executive_insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
