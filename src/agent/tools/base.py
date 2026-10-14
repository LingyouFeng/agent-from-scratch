"""Base class for all tools."""

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

ArgsT = TypeVar("ArgsT", bound=BaseModel)

class BaseTool(ABC, Generic[ArgsT]):
    """Minimal tool interface. Subclass this to add new capabilities."""
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: type[ArgsT]

    def execute(self, **kwargs) -> str:
        """统一执行入口：校验参数后调用子类的业务逻辑。"""
        try:
            args = self.parameters.model_validate(kwargs)
            return self._execute(args)
        except ValidationError as e:
            return f"无效参数: {e}"

    @abstractmethod
    def _execute(self, args: ArgsT) -> str:
        """由子类实现具体业务逻辑。"""
        ...

    @classmethod
    def schema(cls) -> dict:
        """根据 Pydantic 参数模型自动生成工具 Schema。"""
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.parameters.model_json_schema(),
            },
        }