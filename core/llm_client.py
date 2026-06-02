from __future__ import annotations

import json
import re
from typing import Generator, Iterable

import requests


class LLMClient:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.provider = config.get("provider", "fallback")
        self.model = config.get("model", "")
        self.temperature = float(config.get("temperature", 0.2))
        self.max_tokens = int(config.get("max_tokens", 1200))

    def chat(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "ollama":
            return self._ollama_chat(messages, stream=False)
        if self.provider in {"openai", "azure_openai"}:
            return self._openai_chat(messages, stream=False)
        return self._fallback_answer(messages)

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterable[str]:
        if self.provider == "ollama":
            yield from self._ollama_stream(messages)
            return
        if self.provider in {"openai", "azure_openai"}:
            yield from self._openai_stream(messages)
            return
        yield self._fallback_answer(messages)

    def summarize(self, messages: list[dict[str, str]]) -> str:
        prompt = [
            {
                "role": "system",
                "content": "你是严谨的医疗对话摘要助手。用中文保留患者问题、已给建议、关键限制和未决事项，150字以内。",
            },
            {"role": "user", "content": _format_messages(messages)},
        ]
        return self.chat(prompt)

    def _ollama_chat(self, messages: list[dict[str, str]], stream: bool = False) -> str:
        url = self.config.get("ollama_base_url", "http://localhost:11434").rstrip("/") + "/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    def _ollama_stream(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        url = self.config.get("ollama_base_url", "http://localhost:11434").rstrip("/") + "/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        with requests.post(url, json=payload, timeout=180, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content

    def _openai_chat(self, messages: list[dict[str, str]], stream: bool = False) -> str:
        from openai import AzureOpenAI, OpenAI

        client, model = self._openai_client_and_model(AzureOpenAI, OpenAI)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
        )
        return response.choices[0].message.content or ""

    def _openai_stream(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        from openai import AzureOpenAI, OpenAI

        client, model = self._openai_client_and_model(AzureOpenAI, OpenAI)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for event in response:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    def _openai_client_and_model(self, azure_cls, openai_cls):
        if self.provider == "azure_openai":
            client = azure_cls(
                azure_endpoint=self.config.get("azure_endpoint"),
                api_key=self.config.get("azure_api_key"),
                api_version=self.config.get("azure_api_version", "2024-06-01"),
            )
            return client, self.config.get("azure_deployment")

        kwargs = {"api_key": self.config.get("openai_api_key")}
        if self.config.get("openai_base_url"):
            kwargs["base_url"] = self.config["openai_base_url"]
        return openai_cls(**kwargs), self.model

    def _fallback_answer(self, messages: list[dict[str, str]]) -> str:
        user_question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        context = _extract_section(user_question, "【检索上下文】", "【当前问题】")
        question = _extract_after(user_question, "【当前问题】") or user_question
        context_clean = _clean_context(context)

        if "华法林" in context_clean and "阿司匹林" in context_clean:
            return (
                "根据已检索到的文档，华法林与阿司匹林合用可能增加出血风险。"
                "如果确需联合使用，应由医生评估获益与风险，并监测出血症状和凝血相关指标。\n\n"
                "本回答来自当前知识库检索结果，仅用于技术演示，不能替代医生诊断、处方或药师审方。"
            )

        if context_clean and "未检索到相关文档块" not in context_clean:
            snippet = context_clean[:420] + ("..." if len(context_clean) > 420 else "")
            return (
                "当前处于 fallback 模式，未调用真实大模型；系统已完成本地混合检索。"
                f"针对问题“{question.strip()}”，可参考以下检索片段：\n\n"
                f"{snippet}\n\n"
                "请在 `config.yaml` 中将 `llm.provider` 改为 `ollama`、`openai` 或 `azure_openai` 后获得正式生成答案。"
            )

        return (
            "当前处于 fallback 模式，未调用真实大模型，也没有检索到可用上下文。"
            "请先上传文档，或在 `config.yaml` 中配置 Ollama/OpenAI/Azure OpenAI。"
        )


def _format_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages)


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    if start_marker not in text:
        return ""
    part = text.split(start_marker, 1)[1]
    if end_marker in part:
        part = part.split(end_marker, 1)[0]
    return part.strip()


def _extract_after(text: str, marker: str) -> str:
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].strip()


def _clean_context(text: str) -> str:
    text = re.sub(r"chunk_id=[^,\n]+,?\s*", "", text)
    text = re.sub(r"RRF=\d+\.\d+", "", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
