import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_engine import RAGEngine

app = FastAPI(
    title="Enterprise Production RAG Knowledge Engine API",
    description="Vector search & retrieval-augmented generation engine with source provenance and anti-hallucination guardrails.",
    version="1.0.0"
)

rag = RAGEngine()

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the refund window for cancelled subscriptions?")
    top_k: Optional[int] = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence_score: float
    guardrail_status: str
    sources: List[str]
    retrieved_contexts: Optional[List[str]] = None

@app.on_event("startup")
def startup_event():
    rag.load_index()

@app.get("/")
def root():
    return {
        "service": "Enterprise Production RAG Knowledge Engine",
        "status": "online",
        "docs": "/docs"
    }

@app.post("/query", response_model=QueryResponse)
def execute_query(req: QueryRequest):
    try:
        res = rag.query(req.query, top_k=req.top_k)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
def reindex_knowledge_base():
    try:
        n_docs, n_chunks = rag.initialize_knowledge_base()
        return {
            "status": "success",
            "message": f"Successfully reindexed {n_docs} documents into {n_chunks} vector chunks."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
