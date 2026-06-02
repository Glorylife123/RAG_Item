from __future__ import annotations


class MemoryCompressor:
    def __init__(self, llm_client, max_tokens: int = 2000, keep_recent_turns: int = 2) -> None:
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        self.keep_recent_turns = keep_recent_turns

    def maybe_compress(self, messages: list[dict[str, str]], summary: str = "") -> tuple[str, list[dict[str, str]], bool]:
        if self.estimate_tokens(messages) <= self.max_tokens:
            return summary, messages, False

        keep_count = self.keep_recent_turns * 2
        old_messages = messages[:-keep_count] if keep_count else messages
        recent = messages[-keep_count:] if keep_count else []
        if not old_messages:
            return summary, messages, False

        summary_input = []
        if summary:
            summary_input.append({"role": "system", "content": f"已有历史摘要：{summary}"})
        summary_input.extend(old_messages)
        new_summary = self.llm_client.summarize(summary_input)
        return new_summary, recent, True

    @staticmethod
    def estimate_tokens(messages: list[dict[str, str]]) -> int:
        # A practical mixed Chinese/English approximation. It intentionally
        # errs high enough to trigger compression before prompts become huge.
        chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, chars // 2)

    def build_memory_text(self, summary: str, recent_messages: list[dict[str, str]]) -> str:
        parts = []
        if summary:
            parts.append(f"【历史摘要】\n{summary}")
        if recent_messages:
            formatted = "\n".join(f"{m['role']}: {m['content']}" for m in recent_messages)
            parts.append(f"【最近对话】\n{formatted}")
        return "\n\n".join(parts)
