from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class BM25Chunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str


class BM25Store:
    """Small persistent BM25 store.

    rank_bm25 has no incremental update API, so this class rebuilds the index
    after every add/delete. For local medical document sets this is usually
    simpler and sufficiently fast.
    """

    def __init__(self, persist_path: str | Path) -> None:
        self.persist_path = Path(persist_path)
        self.chunks: list[BM25Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self.load()

    def load(self) -> None:
        if not self.persist_path.exists():
            self._rebuild()
            return
        with self.persist_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks = [BM25Chunk(**item) for item in raw.get("chunks", [])]
        self._rebuild()

    def persist(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("w", encoding="utf-8") as f:
            json.dump({"chunks": [asdict(c) for c in self.chunks]}, f, ensure_ascii=False, indent=2)

    def add_chunks(self, chunks: list[BM25Chunk]) -> None:
        existing = {c.chunk_id for c in self.chunks}
        self.chunks.extend([c for c in chunks if c.chunk_id not in existing])
        self._rebuild()
        self.persist()

    def delete_document(self, document_id: str) -> None:
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        self._rebuild()
        self.persist()

    def list_documents(self) -> list[dict[str, str | int]]:
        docs: dict[str, dict[str, str | int]] = {}
        for chunk in self.chunks:
            row = docs.setdefault(
                chunk.document_id,
                {"document_id": chunk.document_id, "filename": chunk.filename, "chunks": 0},
            )
            row["chunks"] = int(row["chunks"]) + 1
        return sorted(docs.values(), key=lambda x: str(x["filename"]).lower())

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self._bm25 or not self.chunks:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(ranked, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "text": chunk.text,
                    "score": score,
                    "rank": rank,
                    "source": "bm25",
                }
            )
        return results

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [tok.strip().lower() for tok in jieba.lcut(text) if tok.strip()]

    def _rebuild(self) -> None:
        self._tokenized_corpus = [self.tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
