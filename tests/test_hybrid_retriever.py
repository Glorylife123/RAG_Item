from core.hybrid_retriever import reciprocal_rank_fusion


def test_rrf_rewards_hits_from_both_rankers():
    vector_hits = [
        {"chunk_id": "a", "document_id": "d", "filename": "f.md", "text": "A", "rank": 1, "score": 0.1},
        {"chunk_id": "b", "document_id": "d", "filename": "f.md", "text": "B", "rank": 2, "score": 0.2},
    ]
    bm25_hits = [
        {"chunk_id": "b", "document_id": "d", "filename": "f.md", "text": "B", "rank": 1, "score": 3.0},
        {"chunk_id": "c", "document_id": "d", "filename": "f.md", "text": "C", "rank": 2, "score": 2.0},
    ]

    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, rrf_k=60, top_k=3)

    assert [hit.chunk_id for hit in fused] == ["b", "a", "c"]
    assert fused[0].vector_rank == 2
    assert fused[0].bm25_rank == 1
