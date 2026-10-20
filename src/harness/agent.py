import asyncio
from pathlib import Path
import platform

from src.agent.hooks.manager import HookManager
from src.agent.runtime.agent_loop import AgentLoop
from src.agent.runtime.tool_executor import ToolExecutor
from src.agent.tools.permission import PermissionPolicy, WorkspaceScope
from src.agent.tools.registry import ToolRegistry
from src.agent.types import AgentEvent
from src.ai.model import StreamOptions
from src.ai.provider.all import get_model_by_id
from src.ai.type import Context, Message, TextContent, UserMessage
from src.util.time_util import get_timestamp_ms

WORKDIR = Path.cwd().resolve()
SYSTEM_PROMPT = (f"You are a cli coding agent at {WORKDIR}. System: {platform.uname().system}. "
                 f"Use tools to solve tasks. Act, don't explain.")


class Agent:
    def __init__(self, out: asyncio.Queue[AgentEvent]) -> None:
        self.out = out
        self.busy = False
        self._tool_registry = ToolRegistry()
        self._tool_executor = ToolExecutor(
            self._tool_registry,
            PermissionPolicy(WorkspaceScope(WORKDIR)),
            HookManager(),
        )
        self._stream_option = StreamOptions()
        self._agent_loop = AgentLoop(self._tool_executor, self.emit)
        self._messages: list[Message] = []
        self._model = get_model_by_id("cmhi/deepseek-v4.1-flash")

    async def emit(self, ev: AgentEvent) -> None:
        if self.out:
            await self.out.put(ev)

    async def run(self, user_input: str) -> None:
        if self.busy:
            return
        self.busy = True

        self._messages.append(UserMessage(
            role="user",
            content=[TextContent("text", user_input)],
            timestamp=get_timestamp_ms(),
        ))
        context = Context(SYSTEM_PROMPT, self._messages, self._tool_registry.get_all_tools())
        await self._agent_loop.run_loop(self._model, context, self._stream_option)
        self.busy = False
