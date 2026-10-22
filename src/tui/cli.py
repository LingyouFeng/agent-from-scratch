#!/usr/bin/env python3
"""一个用 Textual 写的 Agent TUI。

架构：
    生产端  MockAgent ──put──> asyncio.Queue[AgentEvent]
    消费端  AgentTUI.consume_events()（Textual async worker）──> 更新 Widget

UI 层完全不关心 Agent 内部逻辑，只认队列里的事件；
换成真实 Agent 时，只要保持「往队列里 put AgentEvent」这一契约即可。
"""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.widgets import Collapsible, Button
from textual.widgets import Footer, Header, Input, Markdown, Static

from src.agent.types import (
    AgentEvent,
    AgentEventType,
    AgentToolExecutionEndEvent,
    AgentToolExecutionStartEvent, AgentStatusBarUpdateEvent
)
from src.harness.agent import Agent


class AgentTUI(App):
    CSS_PATH = Path(__file__).with_name("styles.tcss")

    BINDINGS = [
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+n", "focus_input", "聚焦输入"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.events: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.agent = Agent(self.events)

        # 轮次计数
        self._turn = 0

        # 流式正文状态
        self._stream_md: Markdown | None = None
        self._stream_buf: list[str] = []
        self._stream_dirty = False

        # 流式 thinking 状态
        self._thinking_static: Static | None = None
        self._thinking_buf: list[str] = []
        self._thinking_dirty = False

    # ---------------------------- 布局 ---------------------------- #
    def compose(self) -> ComposeResult:
        yield Header()
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

    # ------------------------ 事件消费（核心） ------------------------ #
    @work
    async def consume_events(self) -> None:
        """常驻 worker：从队列里取事件并渲染。"""
        while True:
            event = await self.events.get()
            try:
                await self._dispatch(event)
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"dispatch error: {exc!r}")
            finally:
                self.events.task_done()

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
                match ev.message.role:
                    case "assistant":
                        for content in ev.message.content:
                            match content.type:
                                case "text":
                                    await self._on_delta(content.text, log)
                                case "thinking":
                                    await self._on_thinking_delta(content.thinking, log)

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

    # ---------------------------- 流式正文 ---------------------------- #
    async def _on_delta(self, text: str, log: VerticalScroll) -> None:
        if self._stream_md is None:
            self._stream_md = Markdown("", classes="agent-msg")
            await log.mount(self._stream_md)
        self._stream_buf.append(text)
        self._stream_dirty = True

    # ---------------------------- 流式 thinking ---------------------------- #
    async def _on_thinking_delta(self, text: str, log: VerticalScroll) -> None:
        if self._thinking_static is None:
            self._thinking_static = Static("", classes="thinking")
            # 将 Static 包进 Collapsible 中，默认折叠
            await log.mount(
                Collapsible(
                    self._thinking_static,
                    title="[dim]💭 思考过程[/dim]",
                    collapsed=True,
                    classes="thinking-collapsible"
                )
            )
        self._thinking_buf.append(text)
        self._thinking_dirty = True

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

        body_text = escape(ev.arguments) if ev.arguments else "[dim]（无参数）[/dim]"
        body = Static(body_text, classes="tool-args-body")

        await log.mount(
            Collapsible(
                body,
                title=title,
                collapsed=True,
                classes="tool-exec tool-exec-running",
            )
        )
        log.scroll_end(animate=False)

    async def _on_tool_execution_end(self, ev: AgentToolExecutionEndEvent) -> None:
        log = self.query_one("#log", VerticalScroll)
        ok = ev.is_success
        icon = "✔" if ok else "✘"
        color = "green" if ok else "red"
        cls = "tool-exec-ok" if ok else "tool-exec-fail"

        title = f"[bold {color}]{icon}[/] [bold]{escape(ev.tool_name)}[/]"

        body_text = escape(ev.tool_result) if ev.tool_result else "[dim]（无输出）[/dim]"
        body = Static(body_text, classes="tool-result-body")

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
