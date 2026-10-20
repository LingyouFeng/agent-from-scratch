#!/usr/bin/env python3
"""一个用 Textual 写的极简 Agent TUI。

架构：
    生产端  MockAgent ──put──> asyncio.Queue[AgentEvent]
    消费端  AgentTUI.consume_events()（Textual async worker）──> 更新 Widget

UI 层完全不关心 Agent 内部逻辑，只认队列里的事件；
换成真实 Agent 时，只要保持「往队列里 put AgentEvent」这一契约即可。
"""

from __future__ import annotations
import asyncio
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Collapsible
from textual.widgets import Footer, Header, Input, Markdown, Static

from src.agent.types import AgentEvent, AgentEventType
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

        # 流式 thinking 状态（新增）
        self._thinking_static: Static | None = None
        self._thinking_buf: list[str] = []
        self._thinking_dirty = False

    # ---------------------------- 布局 ---------------------------- #
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="log")
        yield Input(
            placeholder="输入消息，回车发送…",
            id="prompt",
        )
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
                await self._on_turn_start()

            case AgentEventType.TURN_END:
                await self._on_turn_end()

            case AgentEventType.MESSAGE_UPDATE:
                match ev.message.role:
                    case "assistant":
                        for content in ev.message.content:
                            match content.type:
                                case "text":
                                    await self._on_delta(content.text, log)
                                case "thinking":
                                    await self._on_thinking_delta(content.thinking, log)
                                case "toolCall":
                                    await log.mount(
                                        Static(f"🔧 {escape(content.name)} {escape(str(content.arguments))}",
                                               classes="tool-call")
                                    )

            case AgentEventType.TOOL_EXECUTION_START:
                await self._on_tool_execution_start(ev)

            case AgentEventType.TOOL_EXECUTION_END:
                await self._on_tool_execution_end(ev) 

            case AgentEventType.AGENT_END:
                await self._on_agent_end()

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
            await log.mount(self._thinking_static)
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
            # 累积后再 escape 一次即可，避免对每个 delta 重复 escape
            self._thinking_static.update(f"💭 {escape(''.join(self._thinking_buf))}")

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
        self.sub_title = "working…"

    async def _on_agent_end(self) -> None:
        self._flush_stream()  # 把最后一批 token 刷出去
        self._reset_stream()
        inp = self.query_one(Input)
        inp.disabled = False
        inp.focus()
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
        text = Text()
        text.append("🔧 ", style="yellow")
        text.append(ev.tool_name, style="bold cyan")
        preview = ev.arguments
        if preview:
            text.append(f"  {preview}", style="dim")
        await log.mount(Static(text, classes="tool-exec tool-exec-running"))
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
                collapsed=False,   # 默认展开：完整展示全部结果
                classes=f"tool-exec {cls}",
            )
        )
        log.scroll_end(animate=False)

    # ---------------------------- 交互 ---------------------------- #
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.agent.busy:
            return

        event.input.value = ""
        event.input.disabled = True
        self._reset_stream()

        log = self.query_one("#log", VerticalScroll)
        await log.mount(Static(f"[b]你[/b]  {escape(text)}", classes="user-msg"))
        log.scroll_end(animate=False)

        self.run_agent(text)  # 后台 worker 驱动 Agent

    @work(group="agent")
    async def run_agent(self, prompt: str) -> None:
        await self.agent.run(prompt)

    async def action_clear_log(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        await log.remove_children()
        self._reset_stream()

    def action_focus_input(self) -> None:
        self.query_one(Input).focus()
