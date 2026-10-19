from src.agent.runtime.agent_loop import AgentLoop

class Agent:
    def __init__(self, agent_loop: AgentLoop, out: asyncio.Queue[AgentEvent]) -> None:
        self._agent_loop = agent_loop
        self._out = out

    async def emit(self, ev: AgentEvent) -> None:
        await self._out.put(ev)

    async def run(self, model: Model, context: Context, stream_option: StreamOptions) -> None:
        await self._agent_loop.run_loop(model, context, stream_option)