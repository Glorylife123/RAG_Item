from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.bm25_store import BM25Store
    from core.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    fused_score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None


class HybridRetriever:
    def __init__(
        self,
        vector_store: "VectorStore",
        bm25_store: "BM25Store",
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

    def search(self, query: str) -> list[RetrievedChunk]:
        vector_hits = self.vector_store.similarity_search_with_score(query, self.vector_top_k)
        bm25_hits = self.bm25_store.search(query, self.bm25_top_k)
        return reciprocal_rank_fusion(vector_hits, bm25_hits, self.rrf_k, self.final_top_k)


def reciprocal_rank_fusion(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Fuse vector and BM25 rankings with RRF.

    RRF ignores raw score scale differences and rewards documents that appear
    high in either list, especially those appearing in both.
    """

    by_id: dict[str, dict] = {}

    def add_hit(hit: dict, channel: str) -> None:
        chunk_id = hit["chunk_id"]
        row = by_id.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "document_id": hit.get("document_id", ""),
                "filename": hit.get("filename", ""),
                "text": hit.get("text", ""),
                "fused_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
                "vector_score": None,
                "bm25_score": None,
            },
        )
        rank = int(hit["rank"])
        row["fused_score"] += 1.0 / (rrf_k + rank)
        row[f"{channel}_rank"] = rank
        row[f"{channel}_score"] = float(hit.get("score", 0.0))

    for hit in vector_hits:
        add_hit(hit, "vector")
    for hit in bm25_hits:
        add_hit(hit, "bm25")

    ranked = sorted(by_id.values(), key=lambda x: x["fused_score"], reverse=True)
    return [RetrievedChunk(**row) for row in ranked[:top_k]]
