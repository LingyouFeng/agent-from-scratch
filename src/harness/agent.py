import asyncio
import platform

from agent.constant import Constants
from agent.hooks.manager import HookManager
from agent.runtime.agent_loop import AgentLoop
from agent.types import AgentEvent
from agent.types import ModelRequestOptions
from ai.types import Context, Message, TextContent, UserMessage
from harness.event_bus import EventBus
from harness.skills import Skills
from harness.tools.executor import ToolExecutor
from harness.tools.permission import PermissionPolicy, WorkspaceScope
from harness.tools.registry import ToolRegistry
from util.time_util import get_timestamp_ms

WORKDIR = Constants.WORKDIR
SKILLS = Skills(WORKDIR).load_skills()

SYSTEM_PROMPT = (
    f"You are a general agent at {WORKDIR}. System: {platform.uname().system}. "
    f"Complete the task you were given, then return a concise summary in {Constants.LANGUAGE}."
    f"Skills available:\n{SKILLS}\n"
)


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
            PermissionPolicy(WorkspaceScope(WORKDIR)),
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
