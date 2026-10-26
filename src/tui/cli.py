#!/usr/bin/env python3
"""一个用 Textual 写的 Agent TUI。

架构：
    生产端  MockAgent ──put──> asyncio.Queue[AgentEvent]
    消费端  AgentTUI.consume_events()（Textual async worker）──> 更新 Widget

UI 层完全不关心 Agent 内部逻辑，只认队列里的事件；
换成真实 Agent 时，只要保持「往队列里 put AgentEvent」这一契约即可。
"""

from __future__ import annotations

import datetime
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.widgets import Collapsible, Button, ListView, ListItem, Label
from textual.widgets import Footer, Header, Input, Markdown, Static

from src.agent.types import (
    AgentEvent,
    AgentEventType,
    AgentToolExecutionEndEvent,
    AgentToolExecutionStartEvent, AgentStatusBarUpdateEvent
)
from src.ai.types import AssistantMessageEventType
from src.harness.agent import Agent
from src.harness.event_bus import EventBus
from src.util.text_util import _is_non_text


class AgentTUI(App):
    CSS_PATH = Path(__file__).with_name("styles.tcss")

    BINDINGS = [
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+n", "focus_input", "聚焦输入"),
    ]

    def __init__(self, agent: Agent, bus: EventBus) -> None:
        super().__init__()
        self.bus = bus
        self.agent = agent

        # 轮次计数
        self._turn = 0

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

    # ---------------------------- 布局 ---------------------------- #
    def compose(self) -> ComposeResult:
        yield Header()

        # 主布局：左右分栏
        with Horizontal(id="main-layout"):
            # 左侧边栏：对话列表
            with Vertical(id="sidebar"):
                yield Button("[Ctrl+N] 新建对话", id="new-chat-btn", variant="primary")
                yield ListView(id="chat-list")

            # 右侧主体：聊天记录 + 输入框
            with Vertical(id="chat-area"):
                yield VerticalScroll(id="log")
                with Horizontal(id="input-row"):
                    yield Input(placeholder="输入消息，回车发送…", id="prompt")
                    yield Button("发送", id="send", variant="primary")

        yield Footer()

    def on_mount(self) -> None:
        self.title = "Agent TUI"
        self.sub_title = "idle"
        self.query_one(Input).focus()

        # 20fps 节流刷新 Markdown / Static，避免每个 token 都重排
        self.set_interval(1 / 20, self._flush_stream)
        # 启动消费者 worker
        self.consume_events()

        # 初始化左侧对话列表（模拟数据）
        self._init_chat_list()

    def _init_chat_list(self) -> None:
        chat_list = self.query_one("#chat-list", ListView)

        # 模拟数据：将来这里换成真实的对话 ID 和标题
        mock_chats = [
            {"id": "chat_001", "title": "Osaka - 介绍大阪..."},
            {"id": "chat_002", "title": "Tokyo - 东京旅游攻略..."},
            {"id": "chat_003", "title": "Python - 回文检测代码..."},
            {"id": "chat_004", "title": "Untitled Chat"}
        ]

        for chat in mock_chats:
            # 将真实数据挂载到 ListItem 的原生属性上
            item = ListItem(Label(chat["title"]))
            item.chat_id = chat["id"]  # 动态添加属性
            item.chat_title = chat["title"]
            chat_list.append(item)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """当用户点击左侧对话列表时触发"""
        item = event.item

        # 直接从 item 上获取我们刚才挂载的数据（IDE 可能会对动态属性标黄，可以忽略或使用 getattr）
        chat_id = getattr(item, "chat_id", "unknown")
        chat_title = getattr(item, "chat_title", "未知对话")

        self.notify(f"已切换到对话: {chat_title} (ID: {chat_id})", title="切换对话")

        # TODO: 真正的切换逻辑
        # 比如：清空日志 -> 根据 chat_id 从后端拉取历史记录 -> 重新渲染
        # log = self.query_one("#log", VerticalScroll)
        # self.run_worker(log.remove_children())

    # ------------------------ 事件消费（核心） ------------------------ #
    @work
    async def consume_events(self) -> None:
        """常驻 worker：从队列里取事件并渲染。"""
        while True:
            event = await self.bus.get()
            try:
                await self._dispatch(event)
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"dispatch error: {exc!r}")
            finally:
                self.bus.task_done()

    async def _dispatch(self, ev: AgentEvent) -> None:
        log = self.query_one("#log", VerticalScroll)

        match ev.type:
            case AgentEventType.AGENT_START:
                await self._on_agent_start()

            case AgentEventType.TURN_START:
                pass
                # await self._on_turn_start()

            case AgentEventType.TURN_END:
                pass
                # await self._on_turn_end()

            case AgentEventType.MESSAGE_UPDATE:
                match ev.assistant_message_event.type:
                    case AssistantMessageEventType.THINK_START:
                        await self._on_thinking_start(log)
                    case AssistantMessageEventType.THINK_DELTA:
                        await self._on_thinking_delta(ev.assistant_message_event.delta, log)
                    case AssistantMessageEventType.THINK_END:
                        await self._on_thinking_end(log)
                    case AssistantMessageEventType.TEXT_START:
                        await self._on_text_start(log)
                    case AssistantMessageEventType.TEXT_DELTA:
                        await self._on_text_delta(ev.assistant_message_event.delta, log)
                    case AssistantMessageEventType.TEXT_END:
                        await self._on_text_end(log)
                    case AssistantMessageEventType.TOOLCALL_START:
                        pass
                    case AssistantMessageEventType.TOOLCALL_DELTA:
                        pass
                    case AssistantMessageEventType.TOOLCALL_END:
                        pass

            case AgentEventType.TOOL_EXECUTION_START:
                await self._on_tool_execution_start(ev)

            case AgentEventType.TOOL_EXECUTION_END:
                await self._on_tool_execution_end(ev)

            case AgentEventType.AGENT_END:
                await self._on_agent_end()

            case AgentEventType.STATUS_BAR_UPDATE:
                pass
                # await self._on_status_bar_update(ev)

        log.scroll_end(animate=False)

    # ---------------------------- 流式 thinking ---------------------------- #
    async def _on_thinking_start(self, log: VerticalScroll) -> None:
        # 兜底：上一段 thinking 若没正常结束先收尾
        await self._finalize_thinking()

    async def _on_thinking_delta(self, text: str, log: VerticalScroll) -> None:
        if self._thinking_static is None:
            self._thinking_static = Static("", classes="thinking")
            # 默认展开、标题「思考中」
            self._thinking_collapsible = Collapsible(
                self._thinking_static,
                title="[dim]正在思考[/dim]",
                collapsed=False,
                classes="thinking-collapsible",
            )
            await log.mount(self._thinking_collapsible)
        self._thinking_buf.append(text)
        self._thinking_dirty = True

    async def _on_thinking_end(self, log: VerticalScroll) -> None:
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
    async def _on_text_start(self, log: VerticalScroll) -> None:
        await self._finalize_text()

    async def _on_text_delta(self, text: str, log: VerticalScroll) -> None:
        self._stream_buf.append(text)
        if _is_non_text(text):
            return  # 纯 \n\r\t / 零宽 / 控制字符：不 mount、不刷新
        if self._stream_md is None:
            self._stream_md = Markdown("", classes="agent-msg")
            await log.mount(self._stream_md)
        self._stream_dirty = True

    async def _on_text_end(self, log: VerticalScroll) -> None:
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
            self._thinking_static.update(escape(''.join(self._thinking_buf)))

    def _reset_stream(self) -> None:
        # 正文
        self._stream_md = None
        self._stream_buf.clear()
        self._stream_dirty = False

        # thinking
        self._thinking_static = None
        self._thinking_collapsible = None
        self._thinking_buf.clear()
        self._thinking_dirty = False

    async def _on_agent_start(self) -> None:
        self._turn = 0
        self._set_running(True)  # 可选兜底
        self.sub_title = "working…"

    async def _on_agent_end(self) -> None:
        self._flush_stream()  # 把最后一批 token 刷出去
        self._reset_stream()

        inp = self.query_one(Input)
        inp.disabled = False
        inp.focus()

        self._set_running(False)  # ← 新增：空闲后恢复按钮
        self.sub_title = "idle"

    # ---------------------------- 轮次 ---------------------------- #
    async def _on_turn_start(self) -> None:
        # 把上一轮残留的 token 刷出去（正常事件顺序下已 flush 过，这里是兜底）
        self._flush_stream()
        # 新一轮使用新的流式 widget
        self._reset_stream()

        self._turn += 1
        self.sub_title = f"working… (turn {self._turn})"

        log = self.query_one("#log", VerticalScroll)
        await log.mount(
            Static(
                f"[dim]── 第 {self._turn} 轮 ──[/dim]",
                classes="turn-divider",
            )
        )
        log.scroll_end(animate=False)

    async def _on_turn_end(self) -> None:
        # 刷出本轮最后一批 token
        self._flush_stream()

        log = self.query_one("#log", VerticalScroll)
        await log.mount(
            Static(
                f"[dim]── 第 {self._turn} 轮结束 ──[/dim]",
                classes="turn-divider",
            )
        )
        log.scroll_end(animate=False)

    # ---------------------------- 工具执行 ---------------------------- #
    async def _on_tool_execution_start(self, ev: AgentToolExecutionStartEvent) -> None:
        log = self.query_one("#log", VerticalScroll)

        title = f"[yellow]🔧[/] [bold cyan]{escape(ev.tool_name)}[/]"
        if ev.arguments:
            title += f"  [dim]{escape(ev.arguments)}[/dim]"

        args_text = escape(ev.arguments) if ev.arguments else "[dim]（无参数）[/dim]"

        # 使用 Vertical 容器，上方放参数，下方留给结果
        body_container = Vertical(
            Static(args_text, classes="tool-args-body"),
            classes="tool-body-container"
        )

        collapsible = Collapsible(
            body_container,
            title=title,
            collapsed=False,  # 运行中默认展开，方便用户看参数
            classes="tool-exec tool-exec-running",
        )

        # 假设事件里有 tool_call_id，如果没有则退回使用 tool_name
        # 如果你的 Event 结构里没有 tool_call_id，建议加上，以防并发同名工具出错
        call_id = getattr(ev, "tool_call_id", ev.tool_name)
        self._running_tools[call_id] = collapsible

        await log.mount(collapsible)
        log.scroll_end(animate=False)

    async def _on_tool_execution_end(self, ev: AgentToolExecutionEndEvent) -> None:
        log = self.query_one("#log", VerticalScroll)

        call_id = getattr(ev, "tool_call_id", ev.tool_name)
        collapsible = self._running_tools.pop(call_id, None)

        ok = ev.is_success
        icon = "✔" if ok else "✘"
        color = "green" if ok else "red"
        cls = "tool-exec-ok" if ok else "tool-exec-fail"

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
            await log.mount(
                Collapsible(
                    body,
                    title=title,
                    collapsed=True,
                    classes=f"tool-exec {cls}",
                )
            )

        log.scroll_end(animate=False)

    # ---------------------------- 状态栏 ---------------------------- #
    async def _on_status_bar_update(self, ev: AgentStatusBarUpdateEvent) -> None:
        """把 StatusBar 文本渲染到状态栏 widget。

        不直接用 escape + markup，而是按行解析着色，
        既避免 todos 里的 [] 被 Rich markup 误解析，又能让状态一眼可辨。
        """
        bar = self.query_one("#statusbar", Static)
        bar.update(self._render_status(ev.status_bar))

    def _render_status(self, raw: str) -> Text:
        text = Text()
        for i, line in enumerate(raw.splitlines()):
            if i > 0:
                text.append("\n")
            s = line.strip()
            if s.startswith("system time:"):
                text.append(line, style="dim")
            elif s.startswith("todos:"):
                text.append(line, style="bold cyan")
            elif "[x]" in s:
                text.append(line, style="green")
            elif "[~]" in s:
                text.append(line, style="yellow")
            elif "[ ]" in s:
                text.append(line, style="dim")
            else:
                text.append(line)
        return text

    # ---------------------------- 交互 ---------------------------- #
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        await self._submit_prompt(event.value)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "send":
            return

        # 运行中 → 点击执行停止
        if self.agent.busy:
            # 先切到「停止中…」并禁用，防止重复点击
            event.button.label = "停止中…"
            event.button.disabled = True
            try:
                self.agent.stop()  # 注意：类里属性名是 self.agent
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"stop agent failed: {exc!r}")
                # 停止失败时恢复成运行态
                self._set_running(True)
            return

        # 空闲 → 走和回车一致的提交逻辑
        await self._submit_prompt(self.query_one(Input).value)

    async def _submit_prompt(self, text: str) -> None:
        """回车与按钮共用的发送逻辑。"""
        text = (text or "").strip()
        if not text or self.agent.busy:
            return

        inp = self.query_one(Input)
        inp.value = ""
        inp.disabled = True

        self._reset_stream()
        self._set_running(True)  # 按钮切到「停止」

        log = self.query_one("#log", VerticalScroll)

        # 获取当前时间
        time_str = datetime.datetime.now().strftime("%H:%M")

        # 构建气泡和底部信息
        bubble = Static(escape(text), classes="user-msg-bubble")
        meta = Static(f"You · {time_str}", classes="user-msg-meta")

        # 用 Vertical 把它们上下包起来
        wrapper = Vertical(bubble, meta, classes="user-msg-wrapper")

        # 用 Horizontal 靠右对齐
        row = Horizontal(wrapper, classes="user-msg-row")

        await log.mount(row)
        log.scroll_end(animate=False)

        self.run_agent(text)

    def _set_running(self, running: bool) -> None:
        """根据 agent 状态切换发送按钮的外观/可用性。"""
        btn = self.query_one(f"#send", Button)
        btn.disabled = False
        if running:
            btn.label = "停止"
            btn.variant = "error"
        else:
            btn.label = "发送"
            btn.variant = "primary"

    @work(group="agent")
    async def run_agent(self, prompt: str) -> None:
        await self.agent.run(prompt)

    # ---------------------------- 命令 ---------------------------- #
    async def action_clear_log(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        await log.remove_children()
        self._reset_stream()

    def action_focus_input(self) -> None:
        self.query_one(Input).focus()
