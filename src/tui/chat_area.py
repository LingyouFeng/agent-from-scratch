#!/usr/bin/env python3
"""右侧聊天区域组件。

ChatArea：视图容器 + 共享输入行。
ConversationView：一个会话一份，承载全部消息渲染逻辑（流式 + 历史重建），
    事件始终路由到对应会话的 view，切换会话只是显示/隐藏，
    因此多个会话可以同时流式输出且互不干扰。

流式与历史重建共用同一套渲染函数：
    流式：handle_message_event / handle_tool_execution_*（delta 增量）
    重建：render_history -> 把存储的 AgentMessage 完整内容当作一次 delta 重放
"""

from __future__ import annotations

import datetime

from rich.markup import escape
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Markdown, Static

from src.agent.types import (
    AgentMessage,
    AgentMessageUpdateEvent,
    AgentToolExecutionEndEvent,
    AgentToolExecutionStartEvent,
)
from src.ai.types import (
    AssistantMessage,
    AssistantMessageEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from src.util.text_util import _is_non_text


class ConversationView(VerticalScroll):
    """单个会话的消息视图：流式渲染与历史重建共用一套逻辑。

    样式见 chat_area.tcss（由 AgentTUI 统一加载）。
    """

    def __init__(self, session_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session_id = session_id

        # 流式正文状态
        self._stream_md: Markdown | None = None
        self._stream_buf: list[str] = []
        self._stream_dirty = False

        # 流式 thinking 状态
        self._thinking_static: Static | None = None
        self._thinking_collapsible: Collapsible | None = None
        self._thinking_buf: list[str] = []
        self._thinking_dirty = False

        # 用于追踪正在运行的工具体，key 为工具调用 ID 或工具名
        self._running_tools: dict[str, Collapsible] = {}

    def on_mount(self) -> None:
        # 20fps 节流刷新 Markdown / Static，避免每个 token 都重排
        self.set_interval(1 / 20, self._flush_stream)

    # ------------------------ 流式事件入口 ------------------------ #
    async def handle_message_event(self, ev: AgentMessageUpdateEvent) -> None:
        """流式：分发 assistant 消息事件。"""
        match ev.assistant_message_event.type:
            case AssistantMessageEventType.THINK_START:
                await self._on_thinking_start()
            case AssistantMessageEventType.THINK_DELTA:
                await self._on_thinking_delta(ev.assistant_message_event.delta)
            case AssistantMessageEventType.THINK_END:
                await self._on_thinking_end()
            case AssistantMessageEventType.TEXT_START:
                await self._on_text_start()
            case AssistantMessageEventType.TEXT_DELTA:
                await self._on_text_delta(ev.assistant_message_event.delta)
            case AssistantMessageEventType.TEXT_END:
                await self._on_text_end()
            case AssistantMessageEventType.TOOLCALL_START:
                pass
            case AssistantMessageEventType.TOOLCALL_DELTA:
                pass
            case AssistantMessageEventType.TOOLCALL_END:
                pass

    async def handle_tool_execution_start(self, ev: AgentToolExecutionStartEvent) -> None:
        await self._on_tool_execution_start(ev)

    async def handle_tool_execution_end(self, ev: AgentToolExecutionEndEvent) -> None:
        await self._on_tool_execution_end(ev)

    # ------------------------ 历史重建入口 ------------------------ #
    async def render_history(self, messages: list[AgentMessage]) -> None:
        """根据存储的消息重建对话记录。

        核心思路：把每条消息的完整内容当作一次 delta，
        重放流式渲染函数，保证与流式输出最终效果完全一致。
        """
        await self.clear()

        for msg in messages:
            if isinstance(msg, UserMessage):
                text = self._extract_user_text(msg)
                # 跳过注入的状态栏消息
                if text.startswith("<status>"):
                    continue
                await self.mount_user_message(text, timestamp_ms=msg.timestamp)

            elif isinstance(msg, AssistantMessage):
                for item in msg.content:
                    if isinstance(item, ThinkingContent):
                        # 完整 thinking 当作一次 delta 重放
                        await self._on_thinking_delta(item.thinking)
                        await self._on_thinking_end()
                    elif isinstance(item, TextContent):
                        await self._on_text_delta(item.text)
                        await self._on_text_end()
                    elif isinstance(item, ToolCall):
                        # 等后续 ToolResultMessage 到来时补充结果并收尾
                        await self._on_tool_execution_start(
                            AgentToolExecutionStartEvent(
                                tc_id=item.id,
                                tool_name=item.name,
                                arguments=str(item.arguments),
                            )
                        )

            elif isinstance(msg, ToolResultMessage):
                result = "".join(
                    c.text for c in msg.content if isinstance(c, TextContent)
                )
                await self._on_tool_execution_end(
                    AgentToolExecutionEndEvent(
                        tc_id=msg.tool_call_id,
                        tool_name=msg.tool_name,
                        arguments="",
                        tool_result=result,
                        is_success=None,  # 历史记录里没有成败标记，用中性样式
                    )
                )

        self.scroll_end(animate=False)

    @staticmethod
    def _extract_user_text(msg: UserMessage) -> str:
        if isinstance(msg.content, str):
            return msg.content
        return "".join(c.text for c in msg.content if isinstance(c, TextContent))

    # ---------------------------- 用户消息 ---------------------------- #
    async def mount_user_message(self, text: str, timestamp_ms: int | None = None) -> None:
        """挂载用户消息气泡（流式提交与历史重建共用）。"""
        if timestamp_ms is not None:
            time_str = datetime.datetime.fromtimestamp(timestamp_ms / 1000).strftime("%H:%M")
        else:
            time_str = datetime.datetime.now().strftime("%H:%M")

        bubble = Static(escape(text), classes="user-msg-bubble")
        meta = Static(f"You · {time_str}", classes="user-msg-meta")
        wrapper = Vertical(bubble, meta, classes="user-msg-wrapper")
        row = Horizontal(wrapper, classes="user-msg-row")

        await self.mount(row)
        self.scroll_end(animate=False)

    # ---------------------------- 流式 thinking ---------------------------- #
    async def _on_thinking_start(self) -> None:
        # 兜底：上一段 thinking 若没正常结束先收尾
        await self._finalize_thinking()

    async def _on_thinking_delta(self, text: str) -> None:
        if self._thinking_static is None:
            self._thinking_static = Static("", classes="thinking")
            # 默认展开、标题「思考中」
            self._thinking_collapsible = Collapsible(
                self._thinking_static,
                title="[dim]正在思考[/dim]",
                collapsed=False,
                classes="thinking-collapsible",
            )
            await self.mount(self._thinking_collapsible)
        self._thinking_buf.append(text)
        self._thinking_dirty = True

    async def _on_thinking_end(self) -> None:
        await self._finalize_thinking()

    async def _finalize_thinking(self) -> None:
        # 把最后一批 token 刷出去
        if self._thinking_static is not None and self._thinking_buf:
            self._thinking_static.update(escape("".join(self._thinking_buf)))
        # 标题改成「思考完成」并自动折叠
        if self._thinking_collapsible is not None:
            self._thinking_collapsible.title = "[dim]思考完成[/dim]"
            self._thinking_collapsible.collapsed = True
        # 重置：下一次 THINK_DELTA 会创建新的 Collapsible 区域
        self._thinking_static = None
        self._thinking_collapsible = None
        self._thinking_buf.clear()
        self._thinking_dirty = False

    # ---------------------------- 流式正文 ---------------------------- #
    async def _on_text_start(self) -> None:
        await self._finalize_text()

    async def _on_text_delta(self, text: str) -> None:
        self._stream_buf.append(text)
        if _is_non_text(text):
            return  # 纯 \n\r\t / 零宽 / 控制字符：不 mount、不刷新
        if self._stream_md is None:
            self._stream_md = Markdown("", classes="agent-msg")
            await self.mount(self._stream_md)
        self._stream_dirty = True

    async def _on_text_end(self) -> None:
        await self._finalize_text()

    async def _finalize_text(self) -> None:
        # 把最后一批 token 刷出去
        if self._stream_md is not None and self._stream_buf:
            await self._stream_md.update("".join(self._stream_buf))
        # 重置：下一次 TEXT_DELTA 会创建新的 Markdown 区域
        self._stream_md = None
        self._stream_buf.clear()
        self._stream_dirty = False

    # ---------------------------- 节流刷新 ---------------------------- #
    def _flush_stream(self) -> None:
        """interval 回调：同时刷新正文和 thinking。"""
        if self._stream_dirty and self._stream_md is not None:
            self._stream_dirty = False
            self._stream_md.update("".join(self._stream_buf))

        if self._thinking_dirty and self._thinking_static is not None:
            self._thinking_dirty = False
            self._thinking_static.update(escape("".join(self._thinking_buf)))

    def reset_stream(self) -> None:
        # 正文
        self._stream_md = None
        self._stream_buf.clear()
        self._stream_dirty = False

        # thinking
        self._thinking_static = None
        self._thinking_collapsible = None
        self._thinking_buf.clear()
        self._thinking_dirty = False

    async def finalize_stream(self) -> None:
        """一轮对话结束后：刷出残留 token 并重置流式状态。"""
        self._flush_stream()
        self.reset_stream()

    # ---------------------------- 轮次 ---------------------------- #
    async def on_turn_start(self, turn: int) -> None:
        await self.mount(
            Static(
                f"[dim]── 第 {turn} 轮 ──[/dim]",
                classes="turn-divider",
            )
        )
        self.scroll_end(animate=False)

    async def on_turn_end(self, turn: int) -> None:
        await self.mount(
            Static(
                f"[dim]── 第 {turn} 轮结束 ──[/dim]",
                classes="turn-divider",
            )
        )
        self.scroll_end(animate=False)

    # ---------------------------- 工具执行 ---------------------------- #
    async def _on_tool_execution_start(self, ev: AgentToolExecutionStartEvent) -> None:
        title = f"[yellow]🔧[/] [bold cyan]{escape(ev.tool_name)}[/]"
        if ev.arguments:
            title += f"  [dim]{escape(ev.arguments)}[/dim]"

        args_text = escape(ev.arguments) if ev.arguments else "[dim]（无参数）[/dim]"

        # 使用 Vertical 容器，上方放参数，下方留给结果
        body_container = Vertical(
            Static(args_text, classes="tool-args-body"),
            classes="tool-body-container",
        )

        collapsible = Collapsible(
            body_container,
            title=title,
            collapsed=False,  # 运行中默认展开，方便用户看参数
            classes="tool-exec tool-exec-running",
        )

        call_id = getattr(ev, "tc_id", ev.tool_name)
        self._running_tools[call_id] = collapsible

        await self.mount(collapsible)
        self.scroll_end(animate=False)

    async def _on_tool_execution_end(self, ev: AgentToolExecutionEndEvent) -> None:
        call_id = getattr(ev, "tc_id", ev.tool_name)
        collapsible = self._running_tools.pop(call_id, None)

        # 历史重建时没有成败标记（is_success=None），用中性样式
        if ev.is_success is None:
            icon, color, cls = "🔧", "cyan", "tool-exec"
        else:
            icon = "✔" if ev.is_success else "✘"
            color = "green" if ev.is_success else "red"
            cls = "tool-exec-ok" if ev.is_success else "tool-exec-fail"

        if collapsible is not None:
            # 1. 更新标题
            collapsible.title = f"[bold {color}]{icon}[/] [bold]{escape(ev.tool_name)}[/]"

            # 2. 更新外框样式（移除运行中，加上成功/失败）
            collapsible.remove_class("tool-exec-running")
            collapsible.add_class(cls)

            # 3. 将输出结果塞进刚才预留的容器里
            body_text = escape(ev.tool_result) if ev.tool_result else "[dim]（无输出）[/dim]"
            result_static = Static(body_text, classes="tool-result-body")

            container = collapsible.query_one(".tool-body-container", Vertical)
            await container.mount(result_static)

            # 4. 执行完后自动折叠
            collapsible.collapsed = True
        else:
            # 兜底逻辑：如果没有找到对应的 start 事件（比如历史记录加载）
            body_text = escape(ev.tool_result) if ev.tool_result else "[dim]（无输出）[/dim]"
            body = Static(body_text, classes="tool-result-body")
            title = f"[bold {color}]{icon}[/] [bold]{escape(ev.tool_name)}[/]"
            await self.mount(
                Collapsible(
                    body,
                    title=title,
                    collapsed=True,
                    classes=f"tool-exec {cls}",
                )
            )

        self.scroll_end(animate=False)

    # ---------------------------- 清空 ---------------------------- #
    async def clear(self) -> None:
        """清空消息并重置所有渲染状态。"""
        self._running_tools.clear()
        self.reset_stream()
        await self.remove_children()


class ChatArea(Vertical):
    """聊天区域：多个会话视图容器 + 共享输入行。

    样式见 chat_area.tcss（由 AgentTUI 统一加载）。
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._views: dict[str, ConversationView] = {}

    # ---------------------------- 布局 ---------------------------- #
    def compose(self):
        yield Vertical(id="views")
        with Horizontal(id="input-row"):
            yield Input(placeholder="输入消息，回车发送…", id="prompt")
            yield Button("发送", id="send", variant="primary")

    # ---------------------------- 视图管理 ---------------------------- #
    async def create_view(self, session_id: str) -> ConversationView:
        """为会话创建独立视图（默认隐藏）。"""
        view = ConversationView(session_id)
        view.display = False
        self._views[session_id] = view
        await self.query_one("#views", Vertical).mount(view)
        return view

    def get_view(self, session_id: str) -> ConversationView | None:
        return self._views.get(session_id)

    async def show_view(self, session_id: str) -> ConversationView | None:
        """切换显示：目标会话视图显示，其余隐藏。"""
        view = self._views.get(session_id)
        if view is None:
            return None
        for sid, v in self._views.items():
            v.display = sid == session_id
        view.scroll_end(animate=False)
        return view

    # ---------------------------- 输入区 ---------------------------- #
    @property
    def input(self) -> Input:
        return self.query_one("#prompt", Input)

    @property
    def send_button(self) -> Button:
        return self.query_one("#send", Button)
