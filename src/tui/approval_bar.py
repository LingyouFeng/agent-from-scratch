#!/usr/bin/env python3
"""工具审批操作条：hint + 允许/拒绝按钮。

挂载在待审批工具卡片的参数下方（START 事件 pending_approval=True 时），
用户点击后冒泡 Responded 消息给 AgentTUI，转交对应会话的 runtime。
样式见 approval_bar.tcss（由 AgentTUI 统一加载）。
"""

from __future__ import annotations

from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Static


class ApprovalBar(Horizontal):
    """工具审批操作条：允许 / 拒绝按钮，选择结果冒泡给 AgentTUI 转交 runtime。"""

    class Responded(Message):
        """用户对某个工具审批做出了选择。"""

        def __init__(self, session_id: str, tc_id: str, allowed: bool) -> None:
            super().__init__()
            self.session_id = session_id
            self.tc_id = tc_id
            self.allowed = allowed

    def __init__(self, session_id: str, tc_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session_id = session_id
        self._tc_id = tc_id

    def compose(self):
        yield Static("该操作需要你的确认", classes="approval-hint")
        yield Button("允许", variant="success", classes="approval-allow-btn")
        yield Button("拒绝", variant="error", classes="approval-deny-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()  # 不触发 App 层其他按钮逻辑
        allowed = "approval-allow-btn" in event.button.classes
        self.post_message(self.Responded(self._session_id, self._tc_id, allowed))
