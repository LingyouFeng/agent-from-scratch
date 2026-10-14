# tools/registry.py

import json

from src.ai.type import Tool

from .base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")

        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        """获取所有工具的 Function Calling Schema。"""
        return [tool.schema() for tool in self._tools.values()]

    def get_all_tools(self) -> list[Tool]:
        """获取所有注册的工具实例。"""
        return [
            Tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters.model_json_schema(),
            )
            for tool in self._tools.values()
        ]

    def execute(self, tool_name: str, arguments: str | dict) -> str:
        """根据工具名称执行对应工具。"""
        tool = self._tools.get(tool_name)

        if tool is None:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if not isinstance(arguments, dict):
                return "Error: arguments must be a JSON object"

            return tool.execute(**arguments)

        except (json.JSONDecodeError, TypeError) as e:
            return f"Error: Invalid tool arguments: {e}"