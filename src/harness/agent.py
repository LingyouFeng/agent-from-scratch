import asyncio

from src.agent.constant import Constants
from src.agent.hooks.manager import HookManager
from src.agent.runtime.agent_loop import AgentLoop
from src.agent.runtime.tool_executor import ToolExecutor
from src.agent.tools.permission import PermissionPolicy, WorkspaceScope
from src.agent.tools.registry import ToolRegistry
from src.agent.types import AgentEvent
from src.agent.types import ModelRequestOptions
from src.ai.types import Context, Message, TextContent, UserMessage
from src.harness.event_bus import EventBus
from src.util.time_util import get_timestamp_ms

WD = Constants.WORKDIR
SYSTEM_PROMPT = Constants.SYSTEM_PROMPT


class Agent:
    def __init__(
        self, bus: EventBus, messages: list[Message], options: ModelRequestOptions
    ) -> None:
        self.bus = bus
        self._messages = messages
        self.busy = False
        self._tool_registry = ToolRegistry()
        self._tool_executor = ToolExecutor(
            self._tool_registry,
            PermissionPolicy(WorkspaceScope(WD)),
            HookManager(),
        )
        self._model = options.model
        self._stream_option = options.stream_options
        self._agent_loop = AgentLoop(self._tool_executor, self.emit)

        # 当前运行的停止信号
        self._stop_event: asyncio.Event | None = None

    def set_model_request_options(self, options: ModelRequestOptions) -> None:
        """切换模型请求选项（TUI 模型选择器调用，空闲时生效于下一轮对话）。"""
        self._model = options.model
        self._stream_option = options.stream_options

    async def emit(self, ev: AgentEvent) -> None:
        if self.bus:
            await self.bus.publish(ev)

    async def run(self, user_input: str) -> None:
        if self.busy:
            return
        self.busy = True
        stop_event = asyncio.Event()
        self._stop_event = stop_event

        try:
            user_message = UserMessage(
                content=[TextContent(user_input)],
                timestamp=get_timestamp_ms(),
            )
            self._messages.append(user_message)
            context = Context(
                SYSTEM_PROMPT, self._messages, self._tool_registry.get_all_tools()
            )
            await self._agent_loop.run_loop(
                self._model, user_message, context, self._stream_option, stop_event
            )

        finally:
            self.busy = False
            # 避免清理掉其他运行周期的信号
            if self._stop_event is stop_event:
                self._stop_event = None

    def stop(self) -> None:
        """请求停止当前运行，不强行取消运行任务。"""
        if self._stop_event is not None:
            self._stop_event.set()
