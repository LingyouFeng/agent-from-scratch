#!/usr/bin/env python3
"""左侧边栏组件：新建对话按钮 + 历史会话列表。

点击某个会话后，通过 SessionSelected 消息通知上层（AgentTUI）
去激活该会话（加载历史 / 显示对应流式视图）。
"""

from __future__ import annotations

from textual.message import Message
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, ListItem, ListView, Static

from src.harness.session.session_manager import SessionManager


class Sidebar(Vertical):
    """会话列表侧边栏（样式见 sidebar.tcss，由 AgentTUI 统一加载）。"""

    class SessionSelected(Message):
        """用户点击了某个历史会话。"""

        def __init__(self, session_id: str, title: str) -> None:
            super().__init__()
            self.session_id = session_id
            self.title = title

    class SessionDeleteRequested(Message):
        """用户点击了某个会话的删除按钮。"""

        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    def __init__(self, session_manager: SessionManager, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session_manager = session_manager

    # ---------------------------- 布局 ---------------------------- #
    def compose(self):
        yield Button("[Ctrl+N] 新建对话", id="new-chat-btn", variant="primary")
        yield ListView(id="chat-list")

    def on_mount(self) -> None:
        self.refresh_sessions()

    # ---------------------------- 会话列表 ---------------------------- #
    def _make_item(self, session_id: str, title: str, preview: str = "") -> ListItem:
        # 左侧：标题 + 预览
        content = Vertical(
            Static(title, classes="chat-title"),
            Static(preview, classes="chat-preview"),
            classes="chat-item-content",
        )
        # 右侧：删除按钮（选中/悬浮时才显示，见 sidebar.tcss）
        delete_btn = Button("✕", variant="error", classes="chat-delete-btn")
        delete_btn.chat_id = session_id
        item = ListItem(Horizontal(content, delete_btn, classes="chat-item-row"))
        item.chat_id = session_id
        item.chat_title = title
        return item

    def refresh_sessions(self) -> None:
        """从 SessionManager 重新加载会话列表（新建会话/标题更新后调用）。"""
        chat_list = self.query_one("#chat-list", ListView)
        chat_list.clear()
        for chat in self.session_manager.get_session_list():
            chat_list.append(self._make_item(chat.id, chat.title, chat.preview))

    def highlight_session(self, session_id: str) -> None:
        """高亮当前激活的会话项。"""
        chat_list = self.query_one("#chat-list", ListView)
        for i, item in enumerate(chat_list.children):
            if getattr(item, "chat_id", None) == session_id:
                chat_list.index = i
                return

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """当用户点击左侧对话列表时，转发为 SessionSelected 消息。"""
        event.stop()  # 阻止继续冒泡到 App

        chat_id = getattr(event.item, "chat_id", "unknown")
        chat_title = getattr(event.item, "chat_title", "未知对话")

        self.post_message(self.SessionSelected(chat_id, chat_title))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """删除按钮：转发为 SessionDeleteRequested 消息，不触发会话切换。"""
        if "chat-delete-btn" not in event.button.classes:
            return
        event.stop()  # 阻止继续冒泡到 App 的按钮处理
        session_id = getattr(event.button, "chat_id", None)
        if session_id:
            self.post_message(self.SessionDeleteRequested(session_id))
