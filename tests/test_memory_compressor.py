from core.memory_compressor import MemoryCompressor


class FakeLLM:
    def summarize(self, messages):
        return "摘要"


def test_memory_compressor_keeps_recent_turns():
    compressor = MemoryCompressor(FakeLLM(), max_tokens=10, keep_recent_turns=1)
    messages = [
        {"role": "user", "content": "很长的问题" * 10},
        {"role": "assistant", "content": "很长的回答" * 10},
        {"role": "user", "content": "最近问题"},
        {"role": "assistant", "content": "最近回答"},
    ]

    summary, recent, compressed = compressor.maybe_compress(messages)

    assert compressed is True
    assert summary == "摘要"
    assert recent == messages[-2:]
