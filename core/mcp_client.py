from __future__ import annotations

import json
import re
from typing import Any

from tools.drug_interaction import DrugInteractionTool
from tools.weather import WeatherTool


TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class MCPClient:
    """Simplified local MCP-style tool registry.

    The interface mirrors the two operations this app needs: discover tools
    and call a selected tool. Replacing this with JSON-RPC MCP transport later
    only requires preserving list_tools() and call_tool().
    """

    def __init__(self, config: dict) -> None:
        self.tools: dict[str, Any] = {}
        drug_cfg = config.get("tools", {}).get("drug_interaction", {})
        if drug_cfg.get("enabled", True):
            self.register(DrugInteractionTool(drug_cfg.get("local_csv", "./data/drug_interactions.csv")))
        self.register(WeatherTool())

    def register(self, tool: Any) -> None:
        self.tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": name, "description": tool.description} for name, tool in self.tools.items()]

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            return f"工具不存在或未启用：{tool_name}"
        return tool.run(arguments)

    def execute_tool_calls(self, text: str, enabled_tools: list[str] | None = None) -> list[dict[str, str]]:
        enabled = set(enabled_tools or self.tools.keys())
        results = []
        for match in TOOL_CALL_PATTERN.findall(text):
            tool_name, args = parse_tool_call(match)
            if tool_name not in enabled:
                results.append({"tool": tool_name, "result": f"工具未启用：{tool_name}"})
                continue
            results.append({"tool": tool_name, "result": self.call_tool(tool_name, args)})
        return results


def parse_tool_call(raw: str) -> tuple[str, dict]:
    raw = raw.strip()
    if "|" not in raw:
        return raw, {}
    tool_name, arg_text = raw.split("|", 1)
    arg_text = arg_text.strip()
    try:
        args = json.loads(arg_text)
    except json.JSONDecodeError:
        # Friendly fallback for examples like drug_a=warfarin,drug_b=aspirin.
        args = {}
        for item in arg_text.split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                args[key.strip()] = value.strip()
    return tool_name.strip(), args
