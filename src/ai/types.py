from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


@dataclass
class TextContent:
    type: Literal["text"]
    text: str


@dataclass
class ImageContent:
    type: Literal["image"]
    data: str  # base64 encoded image data
    mimeType: str  # e.g., "image/jpeg", "image/png"


@dataclass
class ThinkingContent:
    type: Literal["thinking"]
    thinking: str


@dataclass
class ToolCall:
    type: Literal["toolCall"]
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class UserMessage:
    role: Literal["user"]
    content: str | list[TextContent | ImageContent]
    timestamp: int  # Unix毫秒时间戳


@dataclass
class AssistantMessage:
    role: Literal["assistant"]
    content: list[TextContent | ThinkingContent | ToolCall]
    timestamp: int


@dataclass
class ToolResultMessage:
    role: Literal["toolResult"]
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    timestamp: int


type Message = UserMessage | AssistantMessage | ToolResultMessage


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the function args


@dataclass
class Context:
    systemPrompt: str
    messages: list[Message]
    tools: list[Tool]


class AssistantMessageEventType(str, Enum):
    START = "start"
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"
    THINK_START = "think_start"
    THINK_DELTA = "think_delta"
    THINK_END = "think_end"
    TOOLCALL_START = "toolcall_start"
    TOOLCALL_DELTA = "toolcall_delta"
    TOOLCALL_END = "toolcall_end"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class AssistantMessageEvent:
    type: AssistantMessageEventType
    content_index: int
    partial: AssistantMessage
