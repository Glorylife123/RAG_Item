from __future__ import annotations

import streamlit as st

from core.app_state import init_state


st.set_page_config(
    page_title="本地医疗 RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

st.title("本地医疗 RAG 问答系统")
st.caption("混合检索：Chroma 向量检索 + BM25 + RRF；支持对话记忆压缩和本地 MCP 风格工具调用。")

st.info("请从左侧页面进入“上传文档”或“对话”。对话页也支持直接上传并自动索引。")

with st.sidebar:
    st.header("运行状态")
    docs = st.session_state.bm25_store.list_documents()
    st.metric("已索引文档", len(docs))
    st.metric("文本块", len(st.session_state.bm25_store.chunks))
    st.write("LLM Provider:", st.session_state.config["llm"].get("provider", "fallback"))
