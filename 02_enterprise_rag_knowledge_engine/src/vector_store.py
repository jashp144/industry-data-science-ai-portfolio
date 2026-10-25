import os
import pickle
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.document_loader import Document

class HybridVectorStore:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        # Dense Semantic Vectorizer
        self.dense_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        # Sparse Exact Keyword Matcher (BM25 surrogate)
        self.sparse_vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            stop_words="english"
        )
        self.documents: List[Document] = []
        self.dense_embeddings: np.ndarray = None
        self.sparse_matrix: np.ndarray = None

    def build_index(self, documents: List[Document]):
        """Fits dual vectorizers and constructs hybrid index with reciprocal rank fusion."""
        self.documents = documents
        corpus = [doc.page_content for doc in documents]
        if not corpus:
            raise ValueError("Corpus cannot be empty for indexing.")

        self.dense_embeddings = self.dense_vectorizer.fit_transform(corpus)
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(corpus)
        return self

    def hybrid_search_with_rrf(self, query: str, top_k: int = 3) -> List[Tuple[Document, float, Dict[str, float]]]:
        """
        Executes hybrid retrieval:
        1. Dense semantic search ranking
        2. Sparse keyword BM25 frequency ranking
        3. Reciprocal Rank Fusion (RRF) score merging
        """
        if self.dense_embeddings is None:
            raise ValueError("VectorStore index has not been built yet.")

        # 1. Dense Semantic Scoring
        query_dense = self.dense_vectorizer.transform([query])
        dense_scores = cosine_similarity(query_dense, self.dense_embeddings).flatten()
        dense_ranking = np.argsort(dense_scores)[::-1]

        # 2. Sparse Keyword Scoring
        query_sparse = self.sparse_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_sparse, self.sparse_matrix).flatten()
        sparse_ranking = np.argsort(sparse_scores)[::-1]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, doc_idx in enumerate(dense_ranking, 1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, doc_idx in enumerate(sparse_ranking, 1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_idx, rrf_score in sorted_docs:
            d_score = float(dense_scores[doc_idx])
            s_score = float(sparse_scores[doc_idx])
            # Max normalized semantic confidence score for guardrail verification
            max_confidence = max(d_score, s_score)
            results.append((
                self.documents[doc_idx],
                round(max_confidence, 4),
                {
                    "rrf_score": round(float(rrf_score), 5),
                    "dense_similarity": round(d_score, 4),
                    "sparse_keyword_match": round(s_score, 4)
                }
            ))

        return results

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({
                "dense_vectorizer": self.dense_vectorizer,
                "sparse_vectorizer": self.sparse_vectorizer,
                "documents": self.documents,
                "dense_embeddings": self.dense_embeddings,
                "sparse_matrix": self.sparse_matrix,
                "rrf_k": self.rrf_k
            }, f)

    def load(self, file_path: str):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.dense_vectorizer = data["dense_vectorizer"]
            self.sparse_vectorizer = data["sparse_vectorizer"]
            self.documents = data["documents"]
            self.dense_embeddings = data["dense_embeddings"]
            self.sparse_matrix = data["sparse_matrix"]
            self.rrf_k = data.get("rrf_k", 60)
        return self
