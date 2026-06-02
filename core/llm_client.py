from __future__ import annotations

import json
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
        context = "\n".join(m["content"] for m in messages if "【检索上下文】" in m["content"])
        return (
            "当前 LLM provider 为 fallback，未调用外部或本地大模型。\n\n"
            "我已完成检索与 Prompt 构造，下面是可用于模型回答的问题与上下文摘要：\n\n"
            f"问题：{user_question}\n\n"
            f"{context[:1200]}\n\n"
            "请在 config.yaml 中将 llm.provider 改为 ollama、openai 或 azure_openai 后获得正式生成答案。"
        )


def _format_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages)
