"""会话运行时：每个会话一份独立的事件总线 + Agent + 持久化。

因为每个会话的 Agent 只往自己的总线里发事件，
多个会话可以同时流式输出而互不干扰。
"""

from __future__ import annotations

from agent.types import ModelRequestOptions
from ai.types import Message, TextContent, UserMessage
from harness.agent import Agent
from harness.event_bus import EventBus
from harness.session.session import Session
from harness.session.session_manager import SessionManager


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


def load_context_messages(
    session_manager: SessionManager, session_id: str
) -> list[Message]:
    """加载历史消息作为 Agent 上下文，过滤掉注入的状态栏消息。"""
    return [
        m
        for m in session_manager.load_messages(session_id)
        if not _is_status_message(m)
    ]


class SessionRuntime:
    """一个会话的完整运行时。"""

    def __init__(
        self,
        session_id: str,
        session_manager: SessionManager,
        request_options: ModelRequestOptions,
        messages: list[Message] | None = None,
    ) -> None:
        self.session_id = session_id
        self.bus = EventBus()
        # 复用旧会话时，把历史消息作为初始上下文
        self.agent = Agent(
            self.bus, list(messages) if messages is not None else [], request_options
        )
        # Session 订阅总线负责落盘；总线持有其 handler 引用，不会被 GC
        Session(self.bus, session_id, session_manager)
