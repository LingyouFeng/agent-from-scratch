"""会话服务：会话簿记 + 运行时注册表，是 TUI 与核心层的唯一装配点。

TUI 通过它创建/打开/删除会话、获取会话列表，
不直接触碰 SessionManager 或自行构建 SessionRuntime。

历史加载在这里发生（open_session 在线程里读盘 + 过滤状态注入消息），
runtime 只接收干净的历史列表。
"""

from __future__ import annotations

import asyncio

from ai.types import Message, TextContent, UserMessage
from harness.session.runtime import SessionRuntime
from harness.session.session_manager import SessionItem, SessionManager


def _is_status_message(msg: Message) -> bool:
    """判断是否为运行时注入的状态栏消息（不应进入上下文/渲染）。"""
    if not isinstance(msg, UserMessage):
        return False
    if isinstance(msg.content, str):
        return msg.content.startswith("<status>")
    return any(
        isinstance(c, TextContent) and c.text.startswith("<status>")
        for c in msg.content
    )


def _load_history(session_manager: SessionManager, session_id: str) -> list[Message]:
    """加载历史消息，过滤掉注入的状态栏消息（过滤只在这里发生一次）。"""
    return [
        m
        for m in session_manager.load_messages(session_id)
        if not _is_status_message(m)
    ]


class SessionService:
    """会话门面：create / open / delete / list + 活跃 runtime 注册表。"""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager
        # 已打开会话的运行时注册表：open_session 幂等，delete_session 据此停止运行
        self._runtimes: dict[str, SessionRuntime] = {}

    def create_session(self, model_id: str) -> SessionRuntime:
        """新建会话并落盘初始模型，返回其运行时（新会话没有历史）。"""
        session_id = self._session_manager.new_session()
        # 先写 meta 再建 runtime：构造时从 meta 恢复模型（唯一事实源）
        self._session_manager.update_model(session_id, model_id)
        return self._register(session_id, [])

    async def open_session(self, session_id: str) -> SessionRuntime:
        """打开会话：已注册直接返回（不重载历史）；
        否则在线程里加载历史再构建运行时（避免大文件解析阻塞事件循环）。"""
        runtime = self._runtimes.get(session_id)
        if runtime is None:
            messages = await asyncio.to_thread(
                _load_history, self._session_manager, session_id
            )
            runtime = self._register(session_id, messages)
        return runtime

    def delete_session(self, session_id: str) -> None:
        """删除会话：运行中先请求停止，再删除运行时与会话数据。找不到时抛 RuntimeError。"""
        runtime = self._runtimes.pop(session_id, None)
        if runtime is not None and runtime.is_running:
            runtime.stop()
        self._session_manager.delete_sessions(session_id)

    def list_sessions(self) -> list[SessionItem]:
        """所有会话的展示数据（标题/预览/时间）。"""
        return self._session_manager.get_session_list()

    def get_session(self, session_id: str) -> SessionRuntime | None:
        """已打开会话的运行时；未打开返回 None（不构建）。"""
        return self._runtimes.get(session_id)

    def _register(self, session_id: str, messages: list[Message]) -> SessionRuntime:
        runtime = SessionRuntime(session_id, self._session_manager, messages)
        self._runtimes[session_id] = runtime
        return runtime
