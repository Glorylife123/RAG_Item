from __future__ import annotations

import streamlit as st

from core.app_state import delete_document, index_uploaded_file, init_state, list_documents
from core.ui import apply_theme, hero, mini_stats


st.set_page_config(page_title="上传文档", page_icon="📄", layout="wide")
init_state()
apply_theme()

docs = list_documents()
chunk_count = sum(int(doc["chunks"]) for doc in docs)

hero(
    "DOCUMENT INGESTION",
    "上传文档",
    "将 PDF、TXT 或 Markdown 转换为可检索文本块，并同步写入 Chroma 向量库与 BM25 索引。适合小型本地医疗知识库快速构建。",
)

mini_stats(
    [
        ("知识库文档", str(len(docs))),
        ("BM25 文本块", str(chunk_count)),
        ("分块大小", str(st.session_state.config["app"].get("chunk_size", 500))),
        ("重叠字符", str(st.session_state.config["app"].get("chunk_overlap", 50))),
    ]
)

st.write("")

left, right = st.columns([1.05, 1])
with left:
    st.subheader("新增文档")
    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        help="支持一次上传多个文件。索引时会为每个 chunk 生成稳定 ID。",
    )

    if uploaded_files:
        names = ", ".join(file.name for file in uploaded_files)
        st.caption(f"待索引：{names}")
        if st.button("开始索引", type="primary", use_container_width=True):
            for uploaded in uploaded_files:
                try:
                    document_id, indexed_chunks = index_uploaded_file(uploaded)
                    st.success(f"{uploaded.name} 索引完成：{indexed_chunks} 个文本块。ID: {document_id}")
                except Exception as exc:
                    st.error(f"{uploaded.name} 索引失败：{exc}")
            st.rerun()

with right:
    st.subheader("索引说明")
    st.markdown(
        """
        - 文本块 ID 会同时用于 Chroma 与 BM25。
        - BM25 每次增删后重建索引，适合本地中小规模文档集。
        - 删除文档会同步清理向量库和 BM25 缓存。
        - 首次索引会加载本地 embedding 模型，可能需要等待模型下载。
        """
    )

st.divider()
st.subheader("已上传文档")

if not docs:
    st.info("还没有索引任何文档。上传一个 Markdown 或 TXT 文件即可开始。")
else:
    for doc in docs:
        with st.container(border=True):
            cols = st.columns([4, 1, 1])
            cols[0].markdown(f"**{doc['filename']}**")
            cols[0].caption(f"document_id: {doc['document_id']}")
            cols[1].metric("Chunks", doc["chunks"])
            if cols[2].button("删除", key=f"delete_{doc['document_id']}", use_container_width=True):
                delete_document(str(doc["document_id"]))
                st.success(f"已删除 {doc['filename']}")
                st.rerun()
