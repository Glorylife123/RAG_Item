from __future__ import annotations


class WeatherTool:
    name = "get_weather"
    description = "示例天气工具。参数: city。用于展示如何扩展 MCP 工具。"

    def run(self, arguments: dict) -> str:
        city = arguments.get("city", "未知城市")
        return f"{city} 的天气工具为示例实现：请接入真实天气 API 后返回实时结果。"
