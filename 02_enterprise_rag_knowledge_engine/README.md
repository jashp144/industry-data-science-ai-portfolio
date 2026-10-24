# 🧠 Project 2: Enterprise Production RAG (Retrieval-Augmented Generation) & Knowledge Engine

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![GenAI](https://img.shields.io/badge/GenAI-RAG%20Architecture-green.svg)
![Guardrails](https://img.shields.io/badge/Guardrails-Anti--Hallucination-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-teal.svg)

## 📌 Executive Summary
Standard generative AI chatbots suffer from hallucinations and lack domain provenance. 
This project builds a production-ready **Retrieval-Augmented Generation (RAG) system** engineered with:
1. **Recursive Character Chunking** preserving semantic sentence and section boundaries.
2. **Dense & Sparse Vector Store Indexing** enabling sub-millisecond semantic similarity search.
3. **Anti-Hallucination Guardrails** that enforce strict similarity thresholds, refusing ungrounded out-of-domain queries.
4. **Source Provenance & Citation Tracking** mapping every generated phrase directly back to verified document chunks.

---

## 🏛️ Architecture & Retrieval Flow
```mermaid
flowchart TD
    RawDocs[Enterprise Documents: MD, PDF, TXT] --> Loader[Document Loader]
    Loader --> Chunker[Recursive Character Chunker]
    Chunker --> Embeddings[Vector Store & Sublinear TF-IDF Embeddings]
    Embeddings --> PersistedIndex[(Persisted Vector Index)]
    
    UserQuery[User Natural Language Query] --> QueryVec[Query Vectorization]
    PersistedIndex --> CosineSim[Cosine Similarity Matching]
    QueryVec --> CosineSim
    CosineSim --> TopK[Top-K Ranked Context Chunks]
    
    TopK --> Guardrail{Similarity >= Threshold?}
    Guardrail -- No --> Refusal[Refuse: Ungrounded Context / Anti-Hallucination]
    Guardrail -- Yes --> Synthesizer[Response Synthesizer & Provenance Formatter]
    Synthesizer --> FinalOutput[Verified Answer + Source Citations + Confidence Score]
```

---

## 📋 Resume-Ready Bullet Points
- *Designed and deployed an Enterprise Retrieval-Augmented Generation (RAG) knowledge engine indexing unstructured multi-format documentation with recursive text chunking and cosine similarity retrieval.*
- *Implemented hallucination guardrails and confidence calibration thresholds, successfully preventing ungrounded generation on out-of-domain queries.*
- *Packaged semantic search and synthesis as a microservice using FastAPI, delivering responses with document citation provenance.*

---

## 🛠️ How to Run & Verify
```bash
python test_rag.py
```
To run the interactive API:
```bash
python app.py
```
Open `http://localhost:8001/docs` in your browser.
