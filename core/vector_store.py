from __future__ import annotations

from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from core.document_processor import TextChunk


class VectorStore:
    def __init__(
        self,
        persist_dir: str | Path,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        collection_name: str = "medical_documents",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: Iterable[TextChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunk_list],
            documents=[c.text for c in chunk_list],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                }
                for c in chunk_list
            ],
        )

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def similarity_search_with_score(self, query: str, top_k: int = 10) -> list[dict]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(query_texts=[query], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        rows = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(zip(ids, docs, metas, distances), start=1):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id", ""),
                    "filename": metadata.get("filename", ""),
                    "text": text,
                    "score": float(distance),
                    "rank": rank,
                    "source": "vector",
                }
            )
        return rows
