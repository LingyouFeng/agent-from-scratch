from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


@dataclass
class TextContent:
    type: Literal["text"] = field(default="text", init=False)
    text: str


@dataclass
class ImageContent:
    type: Literal["image"] = field(default="image", init=False)
    data: str  # base64 encoded image data
    mimeType: str  # e.g., "image/jpeg", "image/png"


@dataclass
class ThinkingContent:
    type: Literal["thinking"] = field(default="thinking", init=False)
    thinking: str


@dataclass
class ToolCall:
    type: Literal["toolCall"] = field(default="toolCall", init=False)
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class UserMessage:
    role: Literal["user"] = field(default="user", init=False)
    content: str | list[TextContent | ImageContent]
    timestamp: int  # Unix毫秒时间戳


@dataclass
class AssistantMessage:
    role: Literal["assistant"] = field(default="assistant", init=False)
    content: list[TextContent | ThinkingContent | ToolCall]
    timestamp: int


@dataclass
class ToolResultMessage:
    role: Literal["toolResult"] = field(default="toolResult", init=False)
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
class AssistantMessageStartEvent:
    type: Literal[AssistantMessageEventType.START] = field(
        default=AssistantMessageEventType.START, init=False
    )


@dataclass(frozen=True)
class AssistantMessageTextStartEvent:
    type: Literal[AssistantMessageEventType.TEXT_START] = field(
        default=AssistantMessageEventType.TEXT_START, init=False
    )
    content_index: int
    partial: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageTextDeltaEvent:
    type: Literal[AssistantMessageEventType.TEXT_DELTA] = field(
        default=AssistantMessageEventType.TEXT_DELTA, init=False
    )
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageTextEndEvent:
    type: Literal[AssistantMessageEventType.TEXT_END] = field(
        default=AssistantMessageEventType.TEXT_END, init=False
    )
    content_index: int
    content: str
    partial: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageThinkStartEvent:
    type: Literal[AssistantMessageEventType.THINK_START] = field(
        default=AssistantMessageEventType.THINK_START, init=False
    )
    content_index: int
    partial: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageThinkDeltaEvent:
    type: Literal[AssistantMessageEventType.THINK_DELTA] = field(
        default=AssistantMessageEventType.THINK_DELTA, init=False
    )
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageThinkEndEvent:
    type: Literal[AssistantMessageEventType.THINK_END] = field(
        default=AssistantMessageEventType.THINK_END, init=False
    )
    content_index: int
    content: str
    partial: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageToolCallStartEvent:
    type: Literal[AssistantMessageEventType.TOOLCALL_START] = field(
        default=AssistantMessageEventType.TOOLCALL_START, init=False
    )
    content_index: int
    partial: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageToolCallDeltaEvent:
    type: Literal[AssistantMessageEventType.TOOLCALL_DELTA] = field(
        default=AssistantMessageEventType.TOOLCALL_DELTA, init=False
    )
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageToolCallEndEvent:
    type: Literal[AssistantMessageEventType.TOOLCALL_END] = field(
        default=AssistantMessageEventType.TOOLCALL_END, init=False
    )
    content_index: int
    toolcall: ToolCall
    partial: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageDoneEvent:
    type: Literal[AssistantMessageEventType.DONE] = field(
        default=AssistantMessageEventType.DONE, init=False
    )
    reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    message: AssistantMessage


@dataclass(frozen=True)
class AssistantMessageErrorEvent:
    type: Literal[AssistantMessageEventType.ERROR] = field(
        default=AssistantMessageEventType.ERROR, init=False
    )
    reason: Literal["aborted", "error"]
    error: AssistantMessage


type AssistantMessageEvent = (
        AssistantMessageStartEvent
        | AssistantMessageTextStartEvent
        | AssistantMessageTextDeltaEvent
        | AssistantMessageTextEndEvent
        | AssistantMessageThinkStartEvent
        | AssistantMessageThinkDeltaEvent
        | AssistantMessageThinkEndEvent
        | AssistantMessageToolCallStartEvent
        | AssistantMessageToolCallDeltaEvent
        | AssistantMessageToolCallEndEvent
        | AssistantMessageDoneEvent
        | AssistantMessageErrorEvent
)
