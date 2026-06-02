from __future__ import annotations

from copy import deepcopy

import requests
import streamlit as st

from core.llm_client import LLMClient


PROVIDERS = ["fallback", "ollama", "openai", "azure_openai"]


def effective_config() -> dict:
    config = deepcopy(st.session_state.config)
    override = st.session_state.get("llm_override")
    if override:
        config["llm"].update(override)
    return config


def ensure_runtime_clients() -> None:
    config = effective_config()
    current = st.session_state.get("llm_runtime_config")
    if current != config["llm"]:
        st.session_state.llm_client = LLMClient(config["llm"])
        st.session_state.memory_compressor.llm_client = st.session_state.llm_client
        st.session_state.llm_runtime_config = deepcopy(config["llm"])


def render_llm_settings() -> None:
    base = effective_config()["llm"]
    st.sidebar.header("LLM 配置")

    provider = st.sidebar.selectbox(
        "Provider",
        PROVIDERS,
        index=PROVIDERS.index(base.get("provider", "fallback")) if base.get("provider") in PROVIDERS else 0,
    )
    model = st.sidebar.text_input("Model", value=base.get("model", ""))
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, float(base.get("temperature", 0.2)), 0.05)
    max_tokens = st.sidebar.number_input("Max tokens", min_value=128, max_value=8192, value=int(base.get("max_tokens", 1200)), step=128)

    override = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "ollama_base_url": base.get("ollama_base_url", "http://localhost:11434"),
        "openai_base_url": base.get("openai_base_url", ""),
        "openai_api_key": base.get("openai_api_key", ""),
        "azure_endpoint": base.get("azure_endpoint", ""),
        "azure_api_key": base.get("azure_api_key", ""),
        "azure_api_version": base.get("azure_api_version", "2024-06-01"),
        "azure_deployment": base.get("azure_deployment", ""),
    }

    if provider == "ollama":
        override["ollama_base_url"] = st.sidebar.text_input("Ollama URL", value=override["ollama_base_url"])
    elif provider == "openai":
        override["openai_base_url"] = st.sidebar.text_input("OpenAI Base URL", value=override["openai_base_url"])
        override["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", value=override["openai_api_key"], type="password")
    elif provider == "azure_openai":
        override["azure_endpoint"] = st.sidebar.text_input("Azure endpoint", value=override["azure_endpoint"])
        override["azure_deployment"] = st.sidebar.text_input("Azure deployment", value=base.get("azure_deployment", ""))
        override["azure_api_key"] = st.sidebar.text_input("Azure API Key", value=override["azure_api_key"], type="password")
        override["azure_api_version"] = st.sidebar.text_input("Azure API version", value=override["azure_api_version"])

    st.session_state.llm_override = override
    ensure_runtime_clients()

    cols = st.sidebar.columns(2)
    if cols[0].button("测试连接", use_container_width=True):
        ok, message = test_llm_connection(effective_config()["llm"])
        if ok:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
    if cols[1].button("恢复默认", use_container_width=True):
        st.session_state.pop("llm_override", None)
        st.session_state.pop("llm_runtime_config", None)
        ensure_runtime_clients()
        st.rerun()


def test_llm_connection(llm_config: dict) -> tuple[bool, str]:
    provider = llm_config.get("provider", "fallback")
    if provider == "fallback":
        return True, "fallback 可用：不会调用真实模型。"
    if provider == "ollama":
        url = llm_config.get("ollama_base_url", "http://localhost:11434").rstrip("/") + "/api/tags"
        try:
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            return True, "Ollama 连接成功。"
        except Exception as exc:
            return False, f"Ollama 连接失败：{exc}"
    if provider == "openai":
        if not llm_config.get("openai_api_key"):
            return False, "OpenAI API Key 为空。"
        return True, "OpenAI 配置已填写；发送消息时会验证 API。"
    if provider == "azure_openai":
        required = ["azure_endpoint", "azure_api_key", "azure_deployment"]
        missing = [key for key in required if not llm_config.get(key)]
        if missing:
            return False, "Azure 配置缺失：" + ", ".join(missing)
        return True, "Azure OpenAI 配置已填写；发送消息时会验证 API。"
    return False, f"未知 provider：{provider}"
