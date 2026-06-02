from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from core.bm25_store import BM25Chunk, BM25Store
from core.config import load_config
from core.document_processor import DocumentProcessor
from core.hybrid_retriever import HybridRetriever
from core.llm_client import LLMClient
from core.mcp_client import MCPClient
from core.memory_compressor import MemoryCompressor
from core.vector_store import VectorStore


def init_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    config = st.session_state.config

    if "processor" not in st.session_state:
        app_cfg = config["app"]
        st.session_state.processor = DocumentProcessor(
            chunk_size=int(app_cfg.get("chunk_size", 500)),
            chunk_overlap=int(app_cfg.get("chunk_overlap", 50)),
        )

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(
            persist_dir=config["app"]["chroma_dir"],
            embedding_model=config["embedding"].get("model_name", "BAAI/bge-small-zh-v1.5"),
        )

    if "bm25_store" not in st.session_state:
        st.session_state.bm25_store = BM25Store(config["app"]["bm25_path"])

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = LLMClient(config["llm"])

    if "memory_compressor" not in st.session_state:
        mem_cfg = config.get("memory", {})
        st.session_state.memory_compressor = MemoryCompressor(
            st.session_state.llm_client,
            max_tokens=int(mem_cfg.get("max_tokens", 2000)),
            keep_recent_turns=int(mem_cfg.get("keep_recent_turns", 2)),
        )

    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = MCPClient(config)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory_summary" not in st.session_state:
        st.session_state.memory_summary = ""


def get_retriever() -> HybridRetriever:
    cfg = st.session_state.config["app"]
    return HybridRetriever(
        vector_store=st.session_state.vector_store,
        bm25_store=st.session_state.bm25_store,
        vector_top_k=int(cfg.get("vector_top_k", 10)),
        bm25_top_k=int(cfg.get("bm25_top_k", 10)),
        final_top_k=int(cfg.get("final_top_k", 5)),
        rrf_k=int(cfg.get("rrf_k", 60)),
    )


def index_uploaded_file(uploaded_file) -> tuple[str, int]:
    content = uploaded_file.getvalue()
    filename = uploaded_file.name
    document_id = make_document_id(filename, content)
    processor: DocumentProcessor = st.session_state.processor
    text = processor.parse_uploaded_file(filename, content)
    chunks = processor.split_text(text, filename, document_id)
    if not chunks:
        raise ValueError("文件没有解析出可索引文本。")

    # Same chunk IDs are written into Chroma and BM25, which keeps RRF fusion
    # deterministic and makes deletion a single document_id operation.
    st.session_state.vector_store.add_chunks(chunks)
    st.session_state.bm25_store.add_chunks(
        [
            BM25Chunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                text=c.text,
            )
            for c in chunks
        ]
    )
    return document_id, len(chunks)


def delete_document(document_id: str) -> None:
    st.session_state.vector_store.delete_document(document_id)
    st.session_state.bm25_store.delete_document(document_id)


def list_documents() -> list[dict]:
    return st.session_state.bm25_store.list_documents()


def make_document_id(filename: str, content: bytes) -> str:
    stem = Path(filename).stem.replace(" ", "_")
    digest = hashlib.sha1(content).hexdigest()[:12]
    return f"{stem}_{digest}"
