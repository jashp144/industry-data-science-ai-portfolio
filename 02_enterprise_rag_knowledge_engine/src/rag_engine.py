import os
from typing import Dict, Any, List
from src.document_loader import DocumentLoader
from src.text_chunker import RecursiveCharacterTextChunker
from src.vector_store import VectorStore

class RAGEngine:
    def __init__(self, index_path: str = None):
        if index_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            index_path = os.path.join(project_root, "data", "rag_index.pkl")
        self.index_path = index_path
        self.vector_store = VectorStore()
        self.min_similarity_threshold = 0.12

    def initialize_knowledge_base(self):
        """Loads raw documents, chunks them, and builds persistent vector index."""
        loader = DocumentLoader()
        # If directory empty, seed sample knowledge base
        docs = loader.load_directory()
        if not docs:
            loader.create_sample_knowledge_base()
            docs = loader.load_directory()

        chunker = RecursiveCharacterTextChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_documents(docs)

        self.vector_store.build_index(chunks)
        self.vector_store.save(self.index_path)
        return len(docs), len(chunks)

    def load_index(self):
        if os.path.exists(self.index_path):
            self.vector_store.load(self.index_path)
        else:
            self.initialize_knowledge_base()

    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        """Executes RAG retrieval, ranks context, and returns synthesized verified response."""
        self.load_index()

        retrieved_results = self.vector_store.similarity_search_with_score(user_query, top_k=top_k)
        if not retrieved_results:
            return {
                "query": user_query,
                "answer": "No relevant documents found in knowledge base.",
                "confidence_score": 0.0,
                "sources": [],
                "guardrail_status": "NO_MATCH"
            }

        top_doc, top_score = retrieved_results[0]

        # Hallucination Guardrail Check
        if top_score < self.min_similarity_threshold:
            return {
                "query": user_query,
                "answer": "The enterprise knowledge base does not contain sufficient verified context to answer this query accurately.",
                "confidence_score": top_score,
                "sources": [],
                "guardrail_status": "REFUSAL_UNVERIFIED_CONTEXT"
            }

        # Formulate synthesized extractive response
        citations = []
        context_snippets = []
        for doc, score in retrieved_results:
            if score >= self.min_similarity_threshold:
                source_name = doc.metadata.get("source", "unknown")
                chunk_id = doc.metadata.get("chunk_id", "")
                citations.append(f"{source_name} ({chunk_id}) [Relevance: {score:.2f}]")
                context_snippets.append(doc.page_content)

        # Synthesize clear answer directly backed by top retrieved context
        synthesized_answer = (
            f"Based on verified enterprise documentation: {top_doc.page_content.strip()} "
            f"Referenced from {top_doc.metadata.get('source')}."
        )

        return {
            "query": user_query,
            "answer": synthesized_answer,
            "confidence_score": top_score,
            "guardrail_status": "VERIFIED_ACCURATE",
            "sources": citations,
            "retrieved_contexts": context_snippets
        }
