# tools/registry.py

import json

from src.ai.types import Tool
from .base import BaseTool
from .builtin.bash import BashTool
from .builtin.edit import EditFileTool
from .builtin.glob import GlobTool
from .builtin.ls import LsTool
from .builtin.read import ReadFileTool
from .builtin.write import WriteFileTool
from .builtin.todo import TodoTool


class ToolRegistry:
    def __init__(self):
        self._builtin_tools: list[BaseTool] = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            GlobTool(),
            BashTool(),
            LsTool(),
            TodoTool(),
        ]
        self._tools: dict[str, BaseTool] = {}
        self.register_all(self._builtin_tools)

    def get(self, tool_name: str) -> BaseTool | None:
        return self._tools.get(tool_name)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")

        self._tools[tool.name] = tool

    def register_all(self, tools: list[BaseTool]) -> None:
        [self.register(tool) for tool in tools]

    def get_all_tools(self) -> list[Tool]:
        """获取所有注册的工具实例。"""
        return [
            Tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.schema(),
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
