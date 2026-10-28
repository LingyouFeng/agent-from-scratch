# src/agent/event_bus.py
import asyncio
from typing import Awaitable, Callable

from agent.types import AgentEvent

Subscriber = Callable[[AgentEvent], Awaitable[None]]


class EventBus:
    """发布-广播总线。

    publish 时先同步 await 所有订阅者（保证副作用先完成），
    再 put 进队列给 UI 等异步消费者。
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    def subscribe(self, handler: Subscriber) -> None:
        self._subscribers.append(handler)

    async def publish(self, ev: AgentEvent) -> None:
        for sub in self._subscribers:
            await sub(ev)  # 同步扇出：Session 在这里 append
        await self._queue.put(ev)  # UI 异步消费

    async def get(self) -> AgentEvent:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()
