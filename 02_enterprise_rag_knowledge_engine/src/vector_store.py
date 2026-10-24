import os
import pickle
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.document_loader import Document

class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        self.documents: List[Document] = []
        self.doc_embeddings: np.ndarray = None

    def build_index(self, documents: List[Document]):
        """Fits vectorizer and builds vector index across all chunked documents."""
        self.documents = documents
        corpus = [doc.page_content for doc in documents]
        if not corpus:
            raise ValueError("Corpus cannot be empty for indexing.")
        
        self.doc_embeddings = self.vectorizer.fit_transform(corpus)
        return self

    def similarity_search_with_score(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Queries the vector index and returns top-k documents with cosine similarity scores."""
        if self.doc_embeddings is None:
            raise ValueError("VectorStore index has not been built yet.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_embeddings).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((self.documents[idx], round(score, 4)))

        return results

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "documents": self.documents,
                "embeddings": self.doc_embeddings
            }, f)

    def load(self, file_path: str):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.documents = data["documents"]
            self.doc_embeddings = data["embeddings"]
        return self
