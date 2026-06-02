from __future__ import annotations

import html

import streamlit as st

from core.app_state import get_retriever, index_uploaded_file, init_state
from core.settings import render_llm_settings
from core.ui import apply_theme, hero, mini_stats


SYSTEM_PROMPT = """你是一个谨慎的医疗 RAG 助手。
必须遵守：
1. 只基于检索上下文、工具结果和一般医学常识回答，不编造文献来源。
2. 明确提示用户：回答不能替代医生诊断、处方或急诊处理。
3. 若信息不足，说明缺口并给出建议用户应补充的问题。
4. 如需调用工具，可输出 <tool_call>tool_name|{"drug_a":"warfarin","drug_b":"aspirin"}</tool_call>。
"""


def format_context(chunks) -> str:
    if not chunks:
        return "未检索到相关文档块。"
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] 来源: {chunk.filename}, chunk_id={chunk.chunk_id}, "
            f"RRF={chunk.fused_score:.4f}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def format_tools(tools: list[dict[str, str]], enabled: list[str]) -> str:
    enabled_set = set(enabled)
    lines = ["【可用工具】"]
    for tool in tools:
        status = "启用" if tool["name"] in enabled_set else "禁用"
        lines.append(f"- {tool['name']} ({status}): {tool['description']}")
    return "\n".join(lines)


st.set_page_config(page_title="对话", page_icon="💬", layout="wide")
init_state()
apply_theme()

docs = st.session_state.bm25_store.list_documents()
token_est = st.session_state.memory_compressor.estimate_tokens(st.session_state.messages)

hero(
    "HYBRID RETRIEVAL CHAT",
    "医疗知识库对话",
    "输入问题后，系统会融合向量语义召回与 BM25 关键词召回，结合压缩后的对话记忆构造 Prompt，并在回答后展示引用来源。",
)

mini_stats(
    [
        ("可检索文档", str(len(docs))),
        ("文本块", str(len(st.session_state.bm25_store.chunks))),
        ("会话 tokens", str(token_est)),
        ("最终上下文", f"Top {st.session_state.config['app'].get('final_top_k', 5)}"),
    ]
)

st.write("")

with st.sidebar:
    st.header("外部工具")
    tools = st.session_state.mcp_client.list_tools()
    enabled_tools = []
    for tool in tools:
        if st.checkbox(tool["name"], value=True, help=tool["description"]):
            enabled_tools.append(tool["name"])
    st.session_state.enabled_tools = enabled_tools

    st.divider()
    st.header("会话记忆")
    st.metric("估算 tokens", token_est)
    st.metric("压缩阈值", st.session_state.config["memory"].get("max_tokens", 2000))
    if st.button("清空对话"):
        st.session_state.messages = []
        st.session_state.memory_summary = ""
        st.rerun()

    st.divider()
    render_llm_settings()

upload_col, guide_col = st.columns([1.1, 1])
with upload_col:
    with st.expander("上传文件并自动索引", expanded=False):
        direct_files = st.file_uploader(
            "选择要加入当前知识库的文件",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key="chat_upload",
        )
        if direct_files and st.button("上传并索引", type="primary", use_container_width=True):
            for uploaded in direct_files:
                try:
                    document_id, chunk_count = index_uploaded_file(uploaded)
                    st.success(f"{uploaded.name} 已索引：{chunk_count} 个文本块。ID: {document_id}")
                except Exception as exc:
                    st.error(f"{uploaded.name} 索引失败：{exc}")
with guide_col:
    st.info("示例问题：warfarin 和 aspirin 能一起用吗？或：这份指南里高血压合并糖尿病如何处理？")

if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("**对话尚未开始**")
        st.caption("上传文档后提问，或直接测试内置药物相互作用工具。回答下方会显示 RRF 融合后的引用来源。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("请输入医疗问题，例如：这份指南里如何处理高血压合并糖尿病？")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    summary, recent, compressed = st.session_state.memory_compressor.maybe_compress(
        st.session_state.messages,
        st.session_state.memory_summary,
    )
    st.session_state.memory_summary = summary
    st.session_state.messages = recent

    retriever = get_retriever()
    retrieved = retriever.search(prompt)
    context_text = format_context(retrieved)
    memory_text = st.session_state.memory_compressor.build_memory_text(
        st.session_state.memory_summary,
        st.session_state.messages[-4:],
    )
    tool_text = format_tools(st.session_state.mcp_client.list_tools(), st.session_state.enabled_tools)

    messages_for_llm = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tool_text},
        {
            "role": "user",
            "content": f"{memory_text}\n\n【检索上下文】\n{context_text}\n\n【当前问题】\n{prompt}",
        },
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        for delta in st.session_state.llm_client.stream_chat(messages_for_llm):
            answer += delta
            placeholder.markdown(answer + "▌")
        placeholder.markdown(answer)

        tool_results = st.session_state.mcp_client.execute_tool_calls(answer, st.session_state.enabled_tools)
        if tool_results:
            tool_context = "\n\n".join(f"工具 {r['tool']} 返回：\n{r['result']}" for r in tool_results)
            followup_messages = messages_for_llm + [
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": f"【工具结果】\n{tool_context}\n\n请结合工具结果给出最终回答，不要再输出 tool_call 标签。",
                },
            ]
            answer = ""
            for delta in st.session_state.llm_client.stream_chat(followup_messages):
                answer += delta
                placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)

        if retrieved:
            with st.expander("引用来源", expanded=True):
                for idx, chunk in enumerate(retrieved, start=1):
                    st.markdown(
                        f"""
                        <div class="rag-source">
                          <div class="rag-source-title">[{idx}] {html.escape(chunk.filename)}</div>
                          <div class="rag-source-meta">RRF={chunk.fused_score:.4f} · vec_rank={chunk.vector_rank} · bm25_rank={chunk.bm25_rank}</div>
                          <div class="rag-source-text">{html.escape(chunk.text[:420])}{"..." if len(chunk.text) > 420 else ""}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if compressed:
            st.caption("已触发会话记忆压缩：早期对话已写入摘要，最近 2 轮原文保留。")

    st.session_state.messages.append({"role": "assistant", "content": answer})
