#!/usr/bin/env python3
"""右侧聊天区域组件。

ChatArea：视图容器 + 共享输入行。
ConversationView：一个会话一份，承载全部消息渲染逻辑（流式 + 历史重建），
    事件始终路由到对应会话的 view，切换会话只是显示/隐藏，
    因此多个会话可以同时流式输出且互不干扰。

流式与历史重建共用同一套渲染函数：
    流式：handle_message_event / handle_tool_execution_*（delta 增量）
    重建：render_history -> 按页把存储的 AgentMessage 当作 delta 重放，
         只挂载最后一页，顶部按钮按需加载更早的页
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass

from rich.markup import escape
from textual import events
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Collapsible, Input, Markdown, Select, Static

from agent.constant import Constants
from agent.types import (
    AgentMessage,
    AgentMessageUpdateEvent,
    AgentToolExecutionEndEvent,
    AgentToolExecutionStartEvent,
)
from ai.provider.registry import default_registry
from ai.types import (
    AssistantMessage,
    AssistantMessageEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tui.usage import UsageStats, format_tokens
from tui.approval_bar import ApprovalBar
from tui.usage_detail import UsagePopup
from util.text_util import _is_non_text, _safe_body
from util.time_util import get_timestamp_ms

# 每页消息数：控制单次布局的 widget 数量，避免长会话打开时 O(n²) 卡顿
HISTORY_PAGE_MESSAGES = 40
# 同时挂载的页数上限：超出后卸载最旧的页，移出屏幕的内容仍保留在 _history 里
MAX_MOUNTED_PAGES = 5


@dataclass
class _Page:
    """一页消息：widget 容器 + 它在 _history 中的消息区间 [start, end)。"""

    widget: Vertical
    start: int
    end: int


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
        # 已交给 Markdown 的字符数 + 上一次增量解析任务（背压用）
        self._stream_sent = 0
        self._stream_pending = None
        self._stream_error = False
        # 历史重建时抑制每条消息的滚动，避免 N 次 scroll_end
        self._suppress_scroll = False
        # 用户滚上去看历史时不再自动拉回底部；回到底部后恢复跟随
        self._follow_tail = True

        # 分页状态：_history 是视图侧完整消息序列（历史 + 本会话新消息）
        self._history: list[AgentMessage] = []
        self._pages: list[_Page] = []
        self._current_page: _Page | None = None
        # 渲染目标：历史回放时指向页容器，流式时指向当前页
        self._mount_target: Vertical | None = None
        self._rendering_history = False
        # 工具结果先暂存，等 assistant MESSAGE_END 到达后按
        # “assistant(tool_calls) -> toolResult...” 的顺序补进 _history
        self._pending_tool_results: list[ToolResultMessage] = []

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

    async def _mount(self, widget, **kwargs) -> None:
        """统一挂载入口：历史分页时把消息挂到页容器而不是视图根。"""
        target = self._mount_target if self._mount_target is not None else self
        await target.mount(widget, **kwargs)

    # ------------------------ 流式事件入口 ------------------------ #
    async def handle_message_event(self, ev: AgentMessageUpdateEvent) -> None:
        """流式：分发 assistant 消息事件。"""
        match ev.assistant_message_event.type:
            case AssistantMessageEventType.THINK_START:
                await self._on_think_start()
            case AssistantMessageEventType.THINK_DELTA:
                await self._on_think_delta(ev.assistant_message_event.delta)
            case AssistantMessageEventType.THINK_END:
                await self._on_think_end()
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
        self._record_tool_result(ev)
        await self._on_tool_execution_end(ev)

    # ------------------------ 历史重建入口 ------------------------ #
    async def render_history(self, messages: list[AgentMessage]) -> None:
        """按页重建历史：只挂载最后一页，更早的页按需加载。

        长会话把所有消息一次性 mount 进 VerticalScroll 会触发
        O(n²) 的布局开销；分页后单次只布局一页。
        """
        await self.clear()
        self._history = list(messages)
        self._follow_tail = True
        if not self._history:
            return

        start = self._initial_page_start()
        page = await self._render_page(start, len(self._history))
        self._pages.append(_Page(page, start, len(self._history)))
        self._current_page = self._pages[-1]
        await self._refresh_load_button()
        self._scroll_to_end()

    def _initial_page_start(self) -> int:
        total = len(self._history)
        if total <= HISTORY_PAGE_MESSAGES:
            return 0
        candidate = total - HISTORY_PAGE_MESSAGES
        # 尽量从用户消息开始，保证 tool_calls 与 tool_result 不被拆开
        for i in range(candidate, total):
            message = self._history[i]
            if isinstance(message, UserMessage):
                return i
        while candidate > 0 and isinstance(
            self._history[candidate], ToolResultMessage
        ):
            candidate -= 1
        return candidate

    def _earlier_page_start(self, end: int) -> int:
        candidate = max(0, end - HISTORY_PAGE_MESSAGES)
        for i in range(candidate, -1, -1):
            message = self._history[i]
            if isinstance(message, UserMessage):
                candidate = i
                break
        while candidate > 0 and isinstance(
            self._history[candidate], ToolResultMessage
        ):
            candidate -= 1
        return candidate

    async def _render_page(
        self, start: int, end: int, before=None
    ) -> Vertical:
        """把 _history[start:end] 渲染进一个页容器；返回该容器。"""
        page = Vertical(classes="message-page")
        if before is None:
            await self.mount(page)
        else:
            await self.mount(page, before=before)

        self._rendering_history = True
        self._mount_target = page
        self._suppress_scroll = True
        try:
            await self._replay(self._history[start:end])
        finally:
            self._rendering_history = False
            self._mount_target = None
            self._suppress_scroll = False
        return page

    async def _replay(self, messages: list[AgentMessage]) -> None:
        toolcall_args: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, UserMessage):
                text = self._extract_user_text(msg)
                await self.mount_user_message(text, timestamp_ms=msg.timestamp)

            elif isinstance(msg, AssistantMessage):
                for item in msg.content:
                    if isinstance(item, ThinkingContent):
                        # 完整 thinking 当作一次 delta 重放
                        await self._on_think_delta(item.thinking)
                        await self._on_think_end()
                    elif isinstance(item, TextContent):
                        await self._on_text_delta(item.text)
                        await self._on_text_end()
                    elif isinstance(item, ToolCall):
                        # 等后续 ToolResultMessage 到来时补充结果并收尾
                        toolcall_args[item.id] = json.dumps(item.arguments)
                        await self._on_tool_execution_start(
                            AgentToolExecutionStartEvent(
                                tc_id=item.id,
                                tool_name=item.name,
                                arguments=json.dumps(item.arguments),
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
                        arguments=toolcall_args.get(msg.tool_call_id, "{}"),
                        tool_result=result,
                        is_success=msg.is_success,
                    )
                )

    # ---------------------------- 历史分页 ---------------------------- #
    async def _refresh_load_button(self) -> None:
        start = self._pages[0].start if self._pages else 0
        existing = next(iter(self.query("#load-earlier")), None)
        if start <= 0:
            if existing is not None:
                await existing.remove()
            return
        label = f"↑ 加载更早消息（还有 {start} 条）"
        if existing is not None:
            existing.label = label
            existing.disabled = False
            return
        button = Button(label, id="load-earlier", classes="load-earlier")
        if self._pages:
            await self.mount(button, before=self._pages[0].widget)
        else:
            await self.mount(button)

    async def _load_earlier(self) -> None:
        if not self._pages:
            return
        end = self._pages[0].start
        if end <= 0:
            return
        button = next(iter(self.query("#load-earlier")), None)
        if button is not None:
            button.disabled = True
            button.label = "加载中…"

        start = self._earlier_page_start(end)
        if start >= end:
            await self._refresh_load_button()
            return
        self._follow_tail = False
        page = await self._render_page(start, end, before=self._pages[0].widget)
        self._pages.insert(0, _Page(page, start, end))
        await self._refresh_load_button()
        self.call_after_refresh(
            lambda: self.scroll_to_widget(page, animate=False, top=True)
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "load-earlier":
            return
        event.stop()
        await self._load_earlier()

    # ---------------------------- 本会话消息记录 ---------------------------- #
    async def _ensure_live_page(self) -> None:
        """流式消息挂到当前页；当前页满一页后为用户新消息另起一页。"""
        current = self._current_page
        if (
            current is not None
            and len(self._history) - current.start <= HISTORY_PAGE_MESSAGES
        ):
            current.end = len(self._history)
            self._mount_target = current.widget
            return

        index = max(len(self._history) - 1, 0)
        if current is not None:
            current.end = index
        page = Vertical(classes="message-page")
        await self.mount(page)
        entry = _Page(page, index, len(self._history))
        self._pages.append(entry)
        self._current_page = entry
        self._mount_target = page
        await self._cap_pages()

    async def _cap_pages(self) -> None:
        """只保留最近若干页；被卸载的页仍在 _history 中，可通过按钮重新加载。"""
        if self._rendering_history:
            return
        while len(self._pages) > MAX_MOUNTED_PAGES:
            oldest = self._pages[0]
            if oldest is self._current_page:
                break
            self._pages.pop(0)
            await oldest.widget.remove()
        await self._refresh_load_button()

    def record_message(self, message: AgentMessage) -> None:
        """assistant MESSAGE_END 时登记完整消息，供页卸载后重新渲染。"""
        if self._rendering_history or not isinstance(message, AssistantMessage):
            return
        self._history.append(message)
        if self._pending_tool_results:
            self._history.extend(self._pending_tool_results)
            self._pending_tool_results.clear()
        if self._current_page is not None:
            self._current_page.end = len(self._history)

    def _record_tool_result(self, ev: AgentToolExecutionEndEvent) -> None:
        if self._rendering_history:
            return
        self._pending_tool_results.append(
            ToolResultMessage(
                tool_call_id=ev.tc_id,
                tool_name=ev.tool_name,
                content=[TextContent(text=ev.tool_result)],
                timestamp=get_timestamp_ms(),
                is_success=ev.is_success,
            )
        )

    def discard_pending_tool_results(self) -> None:
        """运行中断时丢弃没有对应 assistant 消息的孤立工具结果。"""
        self._pending_tool_results.clear()

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

        if not self._rendering_history:
            # 本会话新消息：登记到 _history，供页卸载后重新渲染
            self._follow_tail = True
            self._history.append(
                UserMessage(
                    content=[TextContent(text=text)],
                    timestamp=(
                        timestamp_ms
                        if timestamp_ms is not None
                        else get_timestamp_ms()
                    ),
                )
            )
            await self._ensure_live_page()

        await self._mount(row)
        self._scroll_to_end()

    # ---------------------------- 流式 thinking ---------------------------- #
    async def _on_think_start(self) -> None:
        # 兜底：上一段 thinking 若没正常结束先收尾
        await self._finalize_thinking()

    async def _on_think_delta(self, text: str) -> None:
        if self._thinking_static is None:
            self._thinking_static = Static("", classes="thinking")
            # 默认展开、标题「思考中」
            self._thinking_collapsible = Collapsible(
                self._thinking_static,
                title="[dim]正在思考[/dim]",
                collapsed=False,
                classes="thinking-collapsible",
                collapsed_symbol="",
                expanded_symbol="",
            )
            await self._mount(self._thinking_collapsible)
            self._scroll_to_end()
        self._thinking_buf.append(text)
        self._thinking_dirty = True

    async def _on_think_end(self) -> None:
        await self._finalize_thinking()

    def _thinking_title(self, label: str) -> str:
        """标题 = 状态前缀 + 思考内容预览（预览需转义，避免被当 markup 解析）。"""
        preview = " ".join("".join(self._thinking_buf).split())
        if not preview:
            return f"[dim]{label}[/dim]"
        return f"[dim]{label}[/dim] {escape(preview)}"

    async def _finalize_thinking(self) -> None:
        # 把最后一批 token 刷出去
        if self._thinking_static is not None and self._thinking_buf:
            self._thinking_static.update(escape("".join(self._thinking_buf)))
        # 标题改成「思考完成 + 内容预览」并自动折叠
        if self._thinking_collapsible is not None:
            self._thinking_collapsible.title = self._thinking_title("思考完成")
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
            await self._mount(self._stream_md)
            self._scroll_to_end()
        self._stream_dirty = True

    async def _on_text_end(self) -> None:
        await self._finalize_text()

    async def _finalize_text(self) -> None:
        """刷出剩余 token；等待上一次增量解析，避免任务在后台堆积。"""
        md = self._stream_md
        if md is not None:
            pending = self._stream_pending
            if pending is not None:
                try:
                    await pending
                except Exception:  # noqa: BLE001 - 单个 Markdown 块失败不应中断对话
                    self.log.error("Markdown 增量解析失败", exc_info=True)
                finally:
                    self._stream_pending = None
            text = "".join(self._stream_buf)
            if self._stream_error:
                # 增量解析失败过：收尾时全量重解析一次，保证正文完整
                try:
                    await md.update(text)
                except Exception:  # noqa: BLE001
                    self.log.error("Markdown 全量重解析失败", exc_info=True)
            elif len(text) > self._stream_sent:
                try:
                    await md.append(text[self._stream_sent:])
                except Exception:  # noqa: BLE001
                    self.log.error("Markdown 收尾解析失败", exc_info=True)
                self._stream_sent = len(text)
        # 重置：下一次 TEXT_DELTA 会创建新的 Markdown 区域
        self._stream_md = None
        self._stream_buf.clear()
        self._stream_sent = 0
        self._stream_pending = None
        self._stream_error = False
        self._stream_dirty = False

    # ---------------------------- 节流刷新 ---------------------------- #
    def _flush_stream(self) -> None:
        """interval 回调：同时刷新正文和 thinking，并在有更新时跟进滚动。"""
        if not self.display:
            return  # 隐藏的会话不参与布局/重排，切回来后的 tick 会继续刷

        scrolled = False
        if self._stream_dirty and self._stream_md is not None:
            pending = self._stream_pending
            if pending is not None and not pending.is_done:
                # 上一次增量解析还没结束：本轮跳过，buffer 继续累积，避免任务堆叠
                return
            if pending is not None:
                self._stream_pending = None
                if pending.exception is not None:
                    self.log.error("Markdown 增量解析失败: %r", pending.exception)
                    self._stream_error = True
                    self._stream_dirty = False
                    return

            text = "".join(self._stream_buf)
            if len(text) > self._stream_sent:
                try:
                    # 只解析新增片段（Textual 会复用/更新最后一个未闭合块），
                    # 比每次 update(全文) 的 O(n) 重排便宜一个数量级
                    self._stream_pending = self._stream_md.append(
                        text[self._stream_sent:]
                    )
                    self._stream_sent = len(text)
                except Exception:  # noqa: BLE001
                    self.log.error("Markdown 增量解析失败", exc_info=True)
                    self._stream_error = True
            self._stream_dirty = False
            scrolled = True

        if self._thinking_dirty and self._thinking_static is not None:
            self._thinking_dirty = False
            self._thinking_static.update(escape("".join(self._thinking_buf)))
            scrolled = True

        # 滚动收敛到刷新 tick：由每事件一次降为 20fps 一次
        if scrolled:
            self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        """历史重建期间、用户回看时或视图隐藏时，不强制跟随底部。"""
        if self._suppress_scroll or not self._follow_tail or not self.display:
            return
        self.scroll_end(animate=False)

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        # 距底部 2 行以内视为“跟随中”，超过则用户正在回看历史
        self._follow_tail = new_value >= self.max_scroll_y - 2

    def reset_stream(self) -> None:
        # 正文
        self._stream_md = None
        self._stream_buf.clear()
        self._stream_dirty = False
        self._stream_sent = 0
        self._stream_pending = None
        self._stream_error = False

        # thinking
        self._thinking_static = None
        self._thinking_collapsible = None
        self._thinking_buf.clear()
        self._thinking_dirty = False

    async def finalize_stream(self) -> None:
        """一轮对话结束后：刷出残留 token 并重置流式状态。"""
        await self._finalize_thinking()
        await self._finalize_text()
        self.reset_stream()

    # ---------------------------- 工具执行 ---------------------------- #
    @staticmethod
    def _tool_title(ev, running: bool) -> str:
        """折叠框标题：优先显示模型填写的 action，回退到工具名。

        ev 只需提供 tool_name / action 两个属性。
        """
        action_note = ""
        try:
            arguments = json.loads(ev.arguments)
            action_note = arguments["action_note"].strip()
        except json.JSONDecodeError as e:
            raise e
        label = action_note or ev.tool_name
        if running:
            return f"[yellow]🔧[/] [bold cyan]{escape(label)}[/] [dim]({escape(ev.tool_name)})[/]"
        return f"[bold cyan]{escape(label)}[/] [dim]({escape(ev.tool_name)})[/]"

    async def _on_tool_execution_start(self, ev: AgentToolExecutionStartEvent) -> None:
        title = self._tool_title(ev, running=True)

        args_visual = _safe_body(ev.arguments, empty_hint="（无参数）")

        # 使用 Vertical 容器，上方放参数，下方留给结果
        children = [Static(args_visual, classes="tool-args-body")]
        # 待审批工具：参数下方挂审批条（END 到达时移除）
        if ev.pending_approval:
            children.append(ApprovalBar(self.session_id, ev.tc_id))
        body_container = Vertical(*children, classes="tool-body-container")

        collapsible = Collapsible(
            body_container,
            title=title,
            collapsed=False,  # 运行中默认展开，方便用户看参数
            classes="tool-exec tool-exec-running",
            collapsed_symbol="",
            expanded_symbol="",
        )

        call_id = getattr(ev, "tc_id", ev.tool_name)
        self._running_tools[call_id] = collapsible

        await self._mount(collapsible)
        self._scroll_to_end()

    async def _on_tool_execution_end(self, ev: AgentToolExecutionEndEvent) -> None:
        call_id = getattr(ev, "tc_id", ev.tool_name)
        collapsible = self._running_tools.pop(call_id, None)

        icon = "✔" if ev.is_success else "✘"
        color = "green" if ev.is_success else "red"
        cls = "tool-exec-ok" if ev.is_success else "tool-exec-fail"

        result_visual = _safe_body(ev.tool_result)

        if collapsible is not None:
            # 1. 更新标题：沿用 start 时的 action 描述
            action_note = ""
            try:
                arguments = json.loads(ev.arguments)
                action_note = arguments["action_note"].strip()
            except json.JSONDecodeError:
                pass
            label = action_note or ev.tool_name

            collapsible.title = (
                f"[bold {color}]{icon}[/] [dim]{escape(ev.tool_name)}[/] · [bold]{escape(label)}[/]"
            )

            # 2. 更新外框样式（移除运行中，加上成功/失败）
            collapsible.remove_class("tool-exec-running")
            collapsible.add_class(cls)

            # 3. 移除审批条（若还挂着）：审批已结束，结果即将显示
            try:
                await collapsible.query_one(ApprovalBar).remove()
            except Exception:
                # 无审批条 / 视图被 clear 时忽略
                pass

            try:
                container = collapsible.query_one(".tool-body-container", Vertical)
                await container.mount(Static(result_visual, classes="tool-result-body"))
                collapsible.collapsed = True
            except Exception:
                # 视图被 clear / remove 时忽略
                pass
        else:
            title = f"[bold {color}]{icon}[/] [bold]{escape(ev.tool_name)}[/]"
            await self._mount(
                Collapsible(
                    Static(result_visual, classes="tool-result-body"),
                    title=title,
                    collapsed=True,
                    classes=f"tool-exec {cls}",
                    collapsed_symbol="",
                    expanded_symbol="",
                )
            )

        self._scroll_to_end()

    # ---------------------------- 清空 ---------------------------- #
    async def clear(self) -> None:
        """清空消息并重置所有渲染状态。"""
        self._running_tools.clear()
        self.reset_stream()
        self._history.clear()
        self._pages.clear()
        self._current_page = None
        self._mount_target = None
        self._rendering_history = False
        self._pending_tool_results.clear()
        self._suppress_scroll = False
        self._follow_tail = True
        await self.remove_children()


def _model_options() -> list[tuple[str, str]]:
    """模型下拉选项：label 为「供应商 / 模型名」，value 为模型 id。"""
    return [
        (f"{model.provider} / {model.name}", model.id)
        for model in default_registry.get_builtin_models()
    ]


class TokenSummary(Static):
    """底栏中间的 token 汇总；点击后由上层打开明细弹窗。"""

    class Clicked(Message):
        """用户点击了 token 汇总。"""

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.post_message(self.Clicked())


class ChatArea(Vertical):
    """聊天区域：多个会话视图容器 + 共享输入行与统计栏。

    样式见 chat_area.tcss（由 AgentTUI 统一加载）。
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._views: dict[str, ConversationView] = {}
        self._stats = UsageStats()

    # ---------------------------- 布局 ---------------------------- #
    def compose(self):
        yield Vertical(id="views")
        # 输入区：上方输入框，底部右侧是模型选择 + 圆形发送按钮
        with Vertical(id="input-area"):
            yield Input(placeholder="输入消息，回车发送…", id="prompt")
            with Horizontal(id="composer-toolbar"):
                with Horizontal(id="composer-actions"):
                    yield Select(
                        _model_options(),
                        value=Constants.DEFAULT_MODEL_ID,
                        id="model-select",
                        allow_blank=False,
                        compact=True,
                    )
                    yield Button("↑", id="send", variant="primary")
        # 统计栏：轮数/部署 | Token 汇总（点击悬浮明细） | 上下文使用率
        with Horizontal(id="session-stats"):
            yield Static("0 轮 0 步", id="stat-turns")
            yield TokenSummary("Token 0 · 缓存命中 0%", id="stat-tokens")
            yield Static("上下文 0 / 1M (0.0%)", id="stat-context")

    def open_usage_popup(self) -> None:
        """在 token 汇总上方悬浮展开明细（浮层会盖住输入框，不挤占布局）。"""
        anchor = self.query_one("#stat-tokens").region
        bounds = self.query_one("#session-stats").region
        self.app.push_screen(UsagePopup(self._stats, anchor, bounds))

    def update_stats(self, stats: UsageStats) -> None:
        """由 AgentTUI 在事件到达 / 切换会话时调用，刷新底栏。"""
        self._stats = stats
        if not self.is_mounted:
            return

        self.query_one("#stat-turns", Static).update(
            f"[b]{stats.turns}[/b] 轮 [b]{stats.steps}[/b] 步"
        )
        self.query_one("#stat-tokens", TokenSummary).update(
            f"Token [b]{format_tokens(stats.total_tokens)}[/b]"
            f" · 缓存命中 [b]{stats.cache_hit_rate * 100:.0f}%[/b]"
        )
        self.query_one("#stat-context", Static).update(
            f"上下文 [b]{format_tokens(stats.context_tokens)}[/b] / 1M"
            f" [dim]({stats.context_ratio * 100:.1f}%)[/dim]"
        )
        # 明细弹层打开时同步刷新
        if isinstance(self.app.screen, UsagePopup):
            self.app.screen.update_stats(stats)

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

    def rename_view(self, old_id: str, new_id: str) -> ConversationView | None:
        """把视图从 old_id 重命名到 new_id（用于草稿视图转正）。

        若 new_id 已存在则直接返回已有视图；否则搬移并更新 session_id。
        """
        if new_id in self._views:
            return self._views[new_id]
        view = self._views.pop(old_id, None)
        if view is not None:
            view.session_id = new_id
            self._views[new_id] = view
        return view

    async def remove_view(self, session_id: str) -> None:
        """移除并卸载某个会话的视图（删除会话时调用）。"""
        view = self._views.pop(session_id, None)
        if view is not None:
            await view.remove()

    # ---------------------------- 输入区 ---------------------------- #
    @property
    def input(self) -> Input:
        return self.query_one("#prompt", Input)

    @property
    def model_select(self) -> Select:
        return self.query_one("#model-select", Select)

    @property
    def send_button(self) -> Button:
        return self.query_one("#send", Button)
