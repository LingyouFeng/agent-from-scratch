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


@dataclass(frozen=True)
class AgentEvent:
    type: AgentEventType
    message: AgentMessage
