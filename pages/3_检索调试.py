from __future__ import annotations

import html

import streamlit as st

from core.app_state import get_retriever, init_state
from core.hybrid_retriever import reciprocal_rank_fusion
from core.settings import render_llm_settings
from core.ui import apply_theme, hero, mini_stats


def render_result(title: str, meta: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="rag-source">
          <div class="rag-source-title">{html.escape(title)}</div>
          <div class="rag-source-meta">{html.escape(meta)}</div>
          <div class="rag-source-text">{html.escape(text[:700])}{"..." if len(text) > 700 else ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="检索调试", page_icon="🔎", layout="wide")
init_state()
apply_theme()

docs = st.session_state.bm25_store.list_documents()
chunk_count = len(st.session_state.bm25_store.chunks)

hero(
    "RETRIEVAL DEBUGGER",
    "检索调试",
    "分别观察向量检索、BM25 检索和 RRF 融合结果，快速判断召回质量、关键词命中和融合参数是否合理。",
)

mini_stats(
    [
        ("文档", str(len(docs))),
        ("文本块", str(chunk_count)),
        ("默认向量 top_k", str(st.session_state.config["app"].get("vector_top_k", 10))),
        ("默认 BM25 top_k", str(st.session_state.config["app"].get("bm25_top_k", 10))),
    ]
)

with st.sidebar:
    st.header("调试参数")
    vector_top_k = st.slider("向量 top_k", 1, 20, int(st.session_state.config["app"].get("vector_top_k", 10)))
    bm25_top_k = st.slider("BM25 top_k", 1, 20, int(st.session_state.config["app"].get("bm25_top_k", 10)))
    final_top_k = st.slider("融合 top_k", 1, 10, int(st.session_state.config["app"].get("final_top_k", 5)))
    rrf_k = st.slider("RRF k", 10, 120, int(st.session_state.config["app"].get("rrf_k", 60)))
    st.divider()
    render_llm_settings()

query = st.text_input("调试查询", value="warfarin 和 aspirin 能一起用吗？", placeholder="输入一个要检索的问题")
run = st.button("运行检索", type="primary", use_container_width=True)

if not query:
    st.info("请输入查询内容。")
elif not chunk_count:
    st.warning("当前知识库为空，请先上传或索引文档。")
elif run:
    retriever = get_retriever()
    vector_hits = retriever.vector_store.similarity_search_with_score(query, vector_top_k)
    bm25_hits = retriever.bm25_store.search(query, bm25_top_k)
    fused_hits = reciprocal_rank_fusion(vector_hits, bm25_hits, rrf_k=rrf_k, top_k=final_top_k)

    tab_fused, tab_vec, tab_bm25 = st.tabs(["RRF 融合", "向量检索", "BM25 检索"])

    with tab_fused:
        if not fused_hits:
            st.info("没有融合结果。")
        for idx, hit in enumerate(fused_hits, start=1):
            render_result(
                title=f"[{idx}] {hit.filename}",
                meta=f"chunk_id={hit.chunk_id} · RRF={hit.fused_score:.4f} · vec_rank={hit.vector_rank} · bm25_rank={hit.bm25_rank}",
                text=hit.text,
            )

    with tab_vec:
        if not vector_hits:
            st.info("没有向量检索结果。")
        for hit in vector_hits:
            render_result(
                title=f"#{hit['rank']} {hit['filename']}",
                meta=f"chunk_id={hit['chunk_id']} · distance={hit['score']:.4f}",
                text=hit["text"],
            )

    with tab_bm25:
        if not bm25_hits:
            st.info("没有 BM25 结果。")
        for hit in bm25_hits:
            render_result(
                title=f"#{hit['rank']} {hit['filename']}",
                meta=f"chunk_id={hit['chunk_id']} · bm25={hit['score']:.4f}",
                text=hit["text"],
            )
