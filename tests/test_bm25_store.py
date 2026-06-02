from core.bm25_store import BM25Chunk, BM25Store


def test_bm25_add_search_delete(tmp_path):
    store = BM25Store(tmp_path / "bm25.json")
    store.add_chunks(
        [
            BM25Chunk("c1", "d1", "a.md", "华法林 阿司匹林 出血 风险"),
            BM25Chunk("c2", "d2", "b.md", "高血压 糖尿病 肾功能"),
        ]
    )

    hits = store.search("华法林 阿司匹林", top_k=2)
    assert hits
    assert hits[0]["chunk_id"] == "c1"

    docs = store.list_documents()
    assert {doc["document_id"] for doc in docs} == {"d1", "d2"}

    store.delete_document("d1")
    assert all(chunk.document_id != "d1" for chunk in store.chunks)
    assert not store.search("华法林 阿司匹林", top_k=2)
