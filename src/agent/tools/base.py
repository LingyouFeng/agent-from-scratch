"""Base class for all tools."""

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from src.agent.constant import Constants

ACTION_NOTE_KEY = "action_note"
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

        注入参数 `action_note`：由模型在每次调用时填写本次执行描述
        （如 'Listing files'），用于 UI 展示；执行前会被剥离，不会传给工具。
        """
        schema = cls.parameters.model_json_schema()
        props = schema.setdefault("properties", {})
        if ACTION_NOTE_KEY not in props:
            props[ACTION_NOTE_KEY] = {
                "type": "string",
                "description": (
                    "Describe what you are doing in this step (e.g., 'Listing files'). " +
                    f"Keep it short and user-friendly. Use {Constants.LANGUAGE}."
                ),
            }
        return schema