import asyncio
import platform
from pathlib import Path

from src.agent.tools.registry import ToolRegistry
from src.agent.runtime.agent_loop import AgentLoop
from src.ai.model import StreamOptions
from src.ai.provider.all import get_model_by_id
from src.ai.type import Context, TextContent, UserMessage
from src.harness.type import AgentEvent
from src.harness.type import AgentEventType
from src.util.time_util import get_timestamp_ms

WORKDIR = Path.cwd().resolve()
SYSTEM_PROMPT = (f"You are a cli coding agent at {WORKDIR}. System: {platform.uname().system}. "
                 f"Use tools to solve tasks. Act, don't explain.")


class Agent:
    def __init__(self, out: asyncio.Queue[AgentEvent]) -> None:
        self.out = out
        self.busy = False

        self._tool_registry = ToolRegistry()
        self._stream_option = StreamOptions()
        self._agent_loop = AgentLoop(WORKDIR, self._tool_registry)
        self._model = get_model_by_id("cmhi/deepseek-v4.1-flash")

    async def emit(self, ev: AgentEvent) -> None:
        if self.out:
            await self.out.put(ev)

    async def run(self, user_input: str) -> None:
        if self.busy:
            await self.emit(AgentEvent(AgentEventType.ERROR, text="Agent 正忙，请稍候"))
            return
        self.busy = True

        messages = [UserMessage(
            role="user",
            content=[TextContent("text", user_input)],
            timestamp=get_timestamp_ms(),
        )]
        context = Context(SYSTEM_PROMPT, messages, self._tool_registry.get_all_tools())

        await self.emit(AgentEvent(AgentEventType.AGENT_START, None))
        for message in self._agent_loop.run_loop(self._model, context, self._stream_option):
            event = AgentEvent(AgentEventType.MESSAGE_UPDATE, message)
            # print(event)
            await self.emit(event)
        await self.emit(AgentEvent(AgentEventType.AGENT_END, None))

        self.busy = False
