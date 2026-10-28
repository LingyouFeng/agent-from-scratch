"""Base class for all tools."""

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class BaseTool(ABC, Generic[ArgsT]):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: type[ArgsT]
    concurrent_safety: ClassVar[bool] = False

    def validate_args(self, args: dict) -> ArgsT:
        return self.parameters.model_validate(args)

    @abstractmethod
    def execute(self, args: ArgsT) -> str:
        """由子类实现具体业务逻辑。统一执行入口：校验参数后调用子类的业务逻辑。"""
        ...

    @classmethod
    def schema(cls) -> dict:
        """根据 Pydantic 参数模型自动生成工具 Schema。
        """
        return cls.parameters.model_json_schema()
