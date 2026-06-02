from __future__ import annotations

import streamlit as st

from core.app_state import init_state
from core.settings import effective_config, render_llm_settings
from core.ui import apply_theme, card, hero, mini_stats, pills


st.set_page_config(
    page_title="本地医疗 RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
apply_theme()

docs = st.session_state.bm25_store.list_documents()
chunk_count = len(st.session_state.bm25_store.chunks)
provider = effective_config()["llm"].get("provider", "fallback")

hero(
    "LOCAL MEDICAL RAG",
    "本地医疗 RAG 问答系统",
    "面向医疗文档的本地知识库问答工作台，集成 Chroma 向量检索、BM25 关键词检索、RRF 融合排序、会话记忆压缩和 MCP 风格工具调用。",
)

mini_stats(
    [
        ("已索引文档", str(len(docs))),
        ("文本块", str(chunk_count)),
        ("LLM Provider", provider),
        ("融合策略", "RRF 融合"),
    ]
)

st.write("")
pills(["PDF / TXT / Markdown", "本地 Embedding", "Chroma PersistentClient", "BM25 中文分词", "引用来源追踪"])

left, right = st.columns([1.05, 1])
with left:
    card(
        "文档索引",
        "在上传页面解析医疗指南、说明书、病例笔记或 Markdown 知识库。每个文本块会同时写入 Chroma 和 BM25，删除文档时两侧同步清理。",
    )
    card(
        "混合检索",
        "用户提问后并行执行向量检索和 BM25 检索，再使用 Reciprocal Rank Fusion 输出前 5 个最相关文本块作为上下文。",
    )
with right:
    card(
        "对话记忆",
        "系统会在会话变长后摘要压缩早期对话，保留最近 2 轮原文，避免 Prompt 过长且维持多轮上下文。",
    )
    card(
        "工具调用",
        "LLM 可通过 <tool_call> 标签请求调用本地 MCP 风格工具，目前内置药物相互作用示例，便于继续扩展外部医疗 API。",
    )

st.warning("本项目仅用于技术演示和学习研究，不能替代医生诊断、处方、药师审方或急诊处理。")

with st.sidebar:
    st.header("运行状态")
    st.metric("已索引文档", len(docs))
    st.metric("文本块", chunk_count)
    st.write("LLM Provider:", provider)
    st.divider()
    render_llm_settings()
