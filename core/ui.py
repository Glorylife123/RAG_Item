from __future__ import annotations

import html

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --rag-bg: #f6f8fb;
          --rag-panel: #ffffff;
          --rag-border: #dde3ea;
          --rag-text: #172033;
          --rag-muted: #65758b;
          --rag-blue: #2454d6;
          --rag-teal: #0f766e;
          --rag-red: #b42318;
        }
        .stApp {
          background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 42%, #f7f8fb 100%);
          color: var(--rag-text);
        }
        [data-testid="stSidebar"] {
          background: #eef3f8;
          border-right: 1px solid var(--rag-border);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
          color: #1f2a44;
        }
        .block-container {
          padding-top: 2.2rem;
          padding-bottom: 4.5rem;
          max-width: 1220px;
        }
        h1, h2, h3 {
          letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
          background: #ffffff;
          border: 1px solid var(--rag-border);
          border-radius: 8px;
          padding: 16px 18px;
          box-shadow: 0 10px 28px rgba(31, 42, 68, 0.06);
        }
        div[data-testid="stMetric"] label {
          color: var(--rag-muted);
        }
        div[data-testid="stFileUploader"] {
          border: 1px dashed #a8b7c7;
          border-radius: 8px;
          padding: 12px;
          background: #fbfdff;
        }
        div.stButton > button {
          border-radius: 8px;
          font-weight: 650;
          border: 1px solid #c8d3df;
        }
        div.stButton > button[kind="primary"] {
          background: #2454d6;
          border-color: #2454d6;
        }
        .rag-hero {
          background: #ffffff;
          border: 1px solid var(--rag-border);
          border-radius: 8px;
          padding: 28px 30px;
          box-shadow: 0 18px 45px rgba(31, 42, 68, 0.08);
          margin-bottom: 22px;
        }
        .rag-kicker {
          color: var(--rag-teal);
          font-size: 0.82rem;
          font-weight: 750;
          text-transform: uppercase;
          letter-spacing: .08em;
          margin-bottom: 8px;
        }
        .rag-title {
          color: var(--rag-text);
          font-size: 2.15rem;
          line-height: 1.16;
          font-weight: 760;
          margin: 0 0 10px 0;
          letter-spacing: 0;
        }
        .rag-subtitle {
          color: var(--rag-muted);
          font-size: 1.02rem;
          line-height: 1.72;
          max-width: 860px;
          margin: 0;
        }
        .rag-card {
          background: #ffffff;
          border: 1px solid var(--rag-border);
          border-radius: 8px;
          padding: 20px 22px;
          box-shadow: 0 12px 30px rgba(31, 42, 68, 0.055);
          margin-bottom: 16px;
        }
        .rag-card-title {
          color: var(--rag-text);
          font-size: 1.05rem;
          font-weight: 740;
          margin-bottom: 7px;
        }
        .rag-card-text {
          color: var(--rag-muted);
          line-height: 1.65;
          font-size: .96rem;
        }
        .rag-pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 6px 10px;
          border-radius: 999px;
          background: #eef5ff;
          color: #2148bd;
          border: 1px solid #cfe0ff;
          font-size: .82rem;
          font-weight: 650;
          margin: 0 6px 6px 0;
        }
        .rag-row {
          display: flex;
          gap: 14px;
          align-items: stretch;
          flex-wrap: wrap;
        }
        .rag-mini {
          flex: 1 1 190px;
          min-width: 160px;
          background: #ffffff;
          border: 1px solid var(--rag-border);
          border-radius: 8px;
          padding: 16px;
        }
        .rag-mini-label {
          color: var(--rag-muted);
          font-size: .82rem;
          margin-bottom: 6px;
        }
        .rag-mini-value {
          color: var(--rag-text);
          font-size: 1.55rem;
          font-weight: 760;
        }
        .rag-source {
          border-left: 3px solid #2454d6;
          background: #f8fbff;
          border-radius: 6px;
          padding: 12px 14px;
          margin-bottom: 10px;
        }
        .rag-source-title {
          color: var(--rag-text);
          font-weight: 720;
          margin-bottom: 4px;
        }
        .rag-source-meta {
          color: var(--rag-muted);
          font-size: .82rem;
          margin-bottom: 7px;
        }
        .rag-source-text {
          color: #344256;
          line-height: 1.55;
          font-size: .9rem;
        }
        .rag-danger {
          color: var(--rag-red);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <section class="rag-hero">
          <div class="rag-kicker">{html.escape(kicker)}</div>
          <h1 class="rag-title">{html.escape(title)}</h1>
          <p class="rag-subtitle">{html.escape(subtitle)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="rag-card">
          <div class="rag-card-title">{html.escape(title)}</div>
          <div class="rag-card-text">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pills(labels: list[str]) -> None:
    body = "".join(f'<span class="rag-pill">{html.escape(label)}</span>' for label in labels)
    st.markdown(body, unsafe_allow_html=True)


def mini_stats(items: list[tuple[str, str]]) -> None:
    for start in range(0, len(items), 2):
        columns = st.columns(2)
        for col, (label, value) in zip(columns, items[start : start + 2]):
            col.metric(label, value)
