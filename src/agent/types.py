from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from src.ai.types import Message, AssistantMessageEvent

type AgentMessage = Message


class AgentEventType(str, Enum):
    # 第1层：Agent 生命周期（整个运行）
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"

    # 第2层：Turn 生命周期（一轮模型调用 + 工具执行）
    TURN_START = "turn_start"
    TURN_END = "turn_end"

    # 第3层：Message 生命周期（一条消息）
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"

    # 第4层：Tool Execution 生命周期（一次工具执行）
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"

    # 其他
    STATUS_BAR_UPDATE = "status_bar_update"


@dataclass(frozen=True)
class AgentStartEvent:
    type: Literal[AgentEventType.AGENT_START] = field(
        default=AgentEventType.AGENT_START, init=False
    )


@dataclass(frozen=True)
class AgentEndEvent:
    type: Literal[AgentEventType.AGENT_END] = field(
        default=AgentEventType.AGENT_END, init=False
    )


@dataclass(frozen=True)
class AgentTurnStartEvent:
    type: Literal[AgentEventType.TURN_START] = field(
        default=AgentEventType.TURN_START, init=False
    )


@dataclass(frozen=True)
class AgentTurnEndEvent:
    type: Literal[AgentEventType.TURN_END] = field(
        default=AgentEventType.TURN_END, init=False
    )


@dataclass(frozen=True)
class AgentMessageStartEvent:
    type: Literal[AgentEventType.MESSAGE_START] = field(
        default=AgentEventType.MESSAGE_START, init=False
    )


@dataclass(frozen=True)
class AgentMessageUpdateEvent:
    type: Literal[AgentEventType.MESSAGE_UPDATE] = field(
        default=AgentEventType.MESSAGE_UPDATE, init=False
    )
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent


@dataclass(frozen=True)
class AgentMessageEndEvent:
    type: Literal[AgentEventType.MESSAGE_END] = field(
        default=AgentEventType.MESSAGE_END, init=False
    )


@dataclass(frozen=True)
class AgentToolExecutionStartEvent:
    type: Literal[AgentEventType.TOOL_EXECUTION_START] = field(
        default=AgentEventType.TOOL_EXECUTION_START, init=False
    )
    tc_id: str
    tool_name: str
    arguments: str


@dataclass(frozen=True)
class AgentToolExecutionUpdateEvent:
    type: Literal[AgentEventType.TOOL_EXECUTION_UPDATE] = field(
        default=AgentEventType.TOOL_EXECUTION_UPDATE, init=False
    )
    tc_id: str
    tool_name: str
    arguments: str
    tool_result: str


@dataclass(frozen=True)
class AgentToolExecutionEndEvent:
    type: Literal[AgentEventType.TOOL_EXECUTION_END] = field(
        default=AgentEventType.TOOL_EXECUTION_END, init=False
    )
    tc_id: str
    tool_name: str
    arguments: str
    tool_result: str
    is_success: bool


@dataclass(frozen=True)
class AgentStatusBarUpdateEvent:
    type: Literal[AgentEventType.STATUS_BAR_UPDATE] = field(
        default=AgentEventType.STATUS_BAR_UPDATE, init=False
    )
    status_bar: str


type AgentEvent = (
        AgentStartEvent
        | AgentEndEvent
        | AgentTurnStartEvent
        | AgentTurnEndEvent
        | AgentMessageStartEvent
        | AgentMessageUpdateEvent
        | AgentMessageEndEvent
        | AgentToolExecutionStartEvent
        | AgentToolExecutionUpdateEvent
        | AgentToolExecutionEndEvent
        | AgentStatusBarUpdateEvent
)
