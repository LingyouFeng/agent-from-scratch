"""会话运行时：每个会话一份独立的事件总线 + Agent + 持久化。

因为每个会话的 Agent 只往自己的总线里发事件，
多个会话可以同时流式输出而互不干扰。

对外只暴露窄接口（TUI 不再直接触碰 Agent / EventBus / SessionManager）：
    命令：send(prompt) / stop() / set_model(model_id)
    查询：is_running / model_id / meta / history()
    事件：events() 返回本会话事件流的异步迭代器

历史消息由 SessionService 负责加载（open_session 在线程里读盘），
runtime 只接收干净的历史列表；会话模型以 SessionMeta.model 为唯一事实源：
构造时从 meta 恢复，set_model 单点同时更新 Agent 选项并落盘。
"""

from __future__ import annotations

from typing import AsyncIterator

from agent.constant import Constants
from agent.types import AgentEvent, ModelRequestOptions
from ai.model import Model, StreamOptions
from ai.provider.registry import default_registry
from ai.types import Message
from harness.agent import Agent
from harness.event_bus import EventBus
from harness.session.session import Session
from harness.session.session_manager import SessionManager, SessionMeta


def _resolve_model(model_id: str) -> Model | None:
    """按 id 解析模型，找不到时回退到默认模型。"""
    model = default_registry.get_model_by_id(model_id)
    if model is None:
        model = default_registry.get_model_by_id(Constants.DEFAULT_MODEL_ID)
    return model


class SessionRuntime:
    """一个会话的完整运行时（对外的唯一门面）。"""

    def __init__(
        self,
        session_id: str,
        session_manager: SessionManager,
        messages: list[Message],
    ) -> None:
        self.session_id = session_id
        self._session_manager = session_manager

        # 历史消息：Agent 初始上下文与视图重建共用
        # （由 SessionService 加载并过滤状态注入消息，新会话传 []）
        self._history = messages

        self._bus = EventBus()
        self._agent = Agent(
            self._bus,
            list(self._history),
            ModelRequestOptions(
                model=self._load_model(), stream_options=StreamOptions()
            ),
        )
        # Session 订阅总线负责落盘；总线持有其 handler 引用，不会被 GC
        Session(self._bus, session_id, session_manager)

    # ---------------------------- 命令 ---------------------------- #
    async def send(self, prompt: str) -> None:
        """发送一条用户消息并运行 Agent（busy 时由 Agent 内部拒绝）。"""
        await self._agent.run(prompt)

    def stop(self) -> None:
        """请求停止当前运行，不强行取消运行任务。"""
        self._agent.stop()

    def set_model(self, model_id: str) -> Model | None:
        """切换会话模型：单点更新 Agent 请求选项并落盘 meta，返回生效的模型。"""
        model = default_registry.get_model_by_id(model_id)
        if model is None:
            return None  # 未知模型 id：忽略，保持现状
        self._agent.set_model_request_options(
            ModelRequestOptions(model=model, stream_options=StreamOptions())
        )
        self._session_manager.update_model(self.session_id, model_id)
        return model

    # ---------------------------- 查询 ---------------------------- #
    @property
    def is_running(self) -> bool:
        return self._agent.busy

    @property
    def meta(self) -> SessionMeta | None:
        return self._session_manager.get_index(self.session_id)

    @property
    def model_id(self) -> str:
        m = self.meta
        return m.model if m is not None else Constants.DEFAULT_MODEL_ID

    def history(self) -> list[Message]:
        """构造时载入的历史消息（视图重建消费一次，此后由事件驱动）。"""
        return self._history

    # ---------------------------- 事件 ---------------------------- #
    async def events(self) -> AsyncIterator[AgentEvent]:
        """本会话事件流；内部负责队列的 get/task_done。"""
        while True:
            ev = await self._bus.get()
            try:
                yield ev
            finally:
                self._bus.task_done()

    # ---------------------------- 内部 ---------------------------- #
    def _load_model(self) -> Model | None:
        """从会话元数据恢复模型（meta.model 是唯一事实源）。"""
        return _resolve_model(self.model_id)
