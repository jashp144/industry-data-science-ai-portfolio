import re
from typing import List
from src.document_loader import Document

class RecursiveCharacterTextChunker:
    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # Split primarily by double newlines, single newlines, or sentence punctuation
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_len + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Retain overlap from end of previous chunk if possible
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text, para] if len(overlap_text) < self.chunk_overlap else [para]
                current_length = sum(len(p) for p in current_chunk) + len(current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for idx, chunk in enumerate(chunks):
                meta = doc.metadata.copy()
                meta["chunk_id"] = f"{meta.get('source', 'doc')}_chunk_{idx}"
                chunked_docs.append(Document(page_content=chunk, metadata=meta))
        return chunked_docs
