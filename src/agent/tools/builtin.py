"""Built-in tools."""
from .bash import BashTool
from .edit import EditFileTool
from .glob import GlobTool
from .read import ReadFileTool
from .register import ToolRegistry
from .write import WriteFileTool


def get_builtin_tools() -> ToolRegistry:
    """获取并注册所有内置工具。"""
    registry = ToolRegistry()

    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(GlobTool())
    registry.register(BashTool())

    return registry
