import asyncio

from src.agent.constant import Constants
from src.agent.hooks.manager import HookManager
from src.agent.runtime.agent_loop import AgentLoop
from src.agent.runtime.tool_executor import ToolExecutor
from src.agent.tools.permission import PermissionPolicy, WorkspaceScope
from src.agent.tools.registry import ToolRegistry
from src.agent.types import AgentEvent
from src.ai.model import StreamOptions
from src.ai.provider.all import get_model_by_id
from src.ai.types import Context, Message, TextContent, UserMessage
from src.util.time_util import get_timestamp_ms

WD = Constants.WORKDIR
SYSTEM_PROMPT = Constants.SYSTEM_PROMPT


class Agent:
    def __init__(self, out: asyncio.Queue[AgentEvent]) -> None:
        self.out = out
        self.busy = False
        self._tool_registry = ToolRegistry()
        self._tool_executor = ToolExecutor(
            self._tool_registry,
            PermissionPolicy(WorkspaceScope(WD)),
            HookManager(),
        )
        self._stream_option = StreamOptions()
        self._agent_loop = AgentLoop(self._tool_executor, self.emit)
        self._messages: list[Message] = []
        self._model = get_model_by_id("cmhi/deepseek-v4.1-flash")

        # 当前运行的停止信号
        self._stop_event: asyncio.Event | None = None

    async def emit(self, ev: AgentEvent) -> None:
        if self.out:
            await self.out.put(ev)

    async def run(self, user_input: str) -> None:
        if self.busy:
            return
        self.busy = True
        stop_event = asyncio.Event()
        self._stop_event = stop_event

        try:
            self._messages.append(UserMessage(
                role="user",
                content=[TextContent("text", user_input)],
                timestamp=get_timestamp_ms(),
            ))
            context = Context(SYSTEM_PROMPT, self._messages, self._tool_registry.get_all_tools())
            await self._agent_loop.run_loop(self._model, context, self._stream_option, stop_event)

        finally:
            self.busy = False
            # 避免清理掉其他运行周期的信号
            if self._stop_event is stop_event:
                self._stop_event = None

    def stop(self) -> None:
        """请求停止当前运行，不强行取消运行任务。"""
        if self._stop_event is not None:
            self._stop_event.set()
