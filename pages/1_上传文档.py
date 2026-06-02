from __future__ import annotations

import streamlit as st

from core.app_state import delete_document, index_uploaded_file, init_state, list_documents


st.set_page_config(page_title="上传文档", page_icon="📄", layout="wide")
init_state()

st.title("上传文档")
st.caption("支持 PDF、TXT、Markdown。上传后会同时写入 Chroma 和 BM25，删除时两侧同步移除。")

uploaded_files = st.file_uploader(
    "选择文件",
    type=["pdf", "txt", "md", "markdown"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("开始索引", type="primary"):
        for uploaded in uploaded_files:
            try:
                document_id, chunk_count = index_uploaded_file(uploaded)
                st.success(f"{uploaded.name} 索引完成：{chunk_count} 个文本块。ID: {document_id}")
            except Exception as exc:
                st.error(f"{uploaded.name} 索引失败：{exc}")
        st.rerun()

st.divider()
st.subheader("已上传文档")

docs = list_documents()
if not docs:
    st.info("还没有索引任何文档。")
else:
    for doc in docs:
        cols = st.columns([4, 1, 2])
        cols[0].write(f"**{doc['filename']}**")
        cols[1].write(f"{doc['chunks']} chunks")
        if cols[2].button("删除", key=f"delete_{doc['document_id']}"):
            delete_document(str(doc["document_id"]))
            st.success(f"已删除 {doc['filename']}")
            st.rerun()
