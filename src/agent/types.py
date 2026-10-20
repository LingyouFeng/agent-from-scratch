from dataclasses import dataclass
from enum import Enum
from typing import Literal

from src.ai.type import Message

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
    type: Literal[AgentEventType.AGENT_START] = AgentEventType.AGENT_START


@dataclass(frozen=True)
class AgentEndEvent:
    type: Literal[AgentEventType.AGENT_END] = AgentEventType.AGENT_END


@dataclass(frozen=True)
class AgentTurnStartEvent:
    type: Literal[AgentEventType.TURN_START] = AgentEventType.TURN_START


@dataclass(frozen=True)
class AgentTurnEndEvent:
    type: Literal[AgentEventType.TURN_END] = AgentEventType.TURN_END


@dataclass(frozen=True)
class AgentMessageStartEvent:
    type: Literal[AgentEventType.MESSAGE_START] = AgentEventType.MESSAGE_START


@dataclass(frozen=True)
class AgentMessageUpdateEvent:
    message: AgentMessage
    type: Literal[AgentEventType.MESSAGE_UPDATE] = AgentEventType.MESSAGE_UPDATE


@dataclass(frozen=True)
class AgentMessageEndEvent:
    type: Literal[AgentEventType.MESSAGE_END] = AgentEventType.MESSAGE_END


@dataclass(frozen=True)
class AgentToolExecutionStartEvent:
    tc_id: str
    tool_name: str
    arguments: str
    type: Literal[AgentEventType.TOOL_EXECUTION_START] = AgentEventType.TOOL_EXECUTION_START


@dataclass(frozen=True)
class AgentToolExecutionUpdateEvent:
    tc_id: str
    tool_name: str
    arguments: str
    tool_result: str
    type: Literal[AgentEventType.TOOL_EXECUTION_UPDATE] = AgentEventType.TOOL_EXECUTION_UPDATE


@dataclass(frozen=True)
class AgentToolExecutionEndEvent:
    tc_id: str
    tool_name: str
    arguments: str
    tool_result: str
    is_success: bool
    type: Literal[AgentEventType.TOOL_EXECUTION_END] = AgentEventType.TOOL_EXECUTION_END
    

@dataclass(frozen=True)
class AgentStatusBarUpdateEvent:
    status_bar: str
    type: Literal[AgentEventType.STATUS_BAR_UPDATE] = AgentEventType.STATUS_BAR_UPDATE


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