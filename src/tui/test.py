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
from dataclasses import dataclass, field
from enum import Enum

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown, Static


# --------------------------------------------------------------------------- #
# 事件模型
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    THINKING = "thinking"        # 思考过程
    TEXT_DELTA = "text_delta"    # 流式文本增量
    TOOL_CALL = "tool_call"      # 发起工具调用
    TOOL_RESULT = "tool_result"  # 工具返回
    ERROR = "error"              # 错误
    DONE = "done"                # 本轮结束


@dataclass(slots=True)
class AgentEvent:
    type: EventType
    text: str = ""
    name: str = ""
    payload: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 假 Agent：只负责往队列里丢事件
# --------------------------------------------------------------------------- #
class MockAgent:
    def __init__(self, out: asyncio.Queue[AgentEvent]) -> None:
        self.out = out
        self.busy = False

    async def _emit(self, ev: AgentEvent) -> None:
        await self.out.put(ev)

    async def run(self, prompt: str) -> None:
        if self.busy:
            await self._emit(AgentEvent(EventType.ERROR, text="Agent 正忙，请稍候"))
            return

        self.busy = True
        try:
            await self._emit(AgentEvent(EventType.THINKING, text="分析用户请求…"))
            await asyncio.sleep(0.4)

            low = prompt.lower()
            if "天气" in prompt or "weather" in low:
                await self._emit(
                    AgentEvent(EventType.TOOL_CALL, name="get_weather",
                               text='{"city": "Hangzhou"}')
                )
                await asyncio.sleep(0.7)
                await self._emit(
                    AgentEvent(EventType.TOOL_RESULT, name="get_weather",
                               text='{"temp": 24, "cond": "晴"}')
                )
                reply = "杭州今天 **24°C，晴** ☀️\n\n适合出门散步，需要我查未来三天吗？"
            elif "代码" in prompt or "code" in low:
                await self._emit(
                    AgentEvent(EventType.TOOL_CALL, name="run_python",
                               text="print(sum(range(10)))")
                )
                await asyncio.sleep(0.6)
                await self._emit(AgentEvent(EventType.TOOL_RESULT, name="run_python", text="45"))
                reply = "运行结果是 `45`。\n\n```python\nprint(sum(range(10)))  # 45\n```"
            else:
                reply = (
                    f"收到：**{prompt}**\n\n"
                    "这是一段流式回答——Agent 每产生一点内容，就通过 "
                    "`AgentEvent(TEXT_DELTA)` 推到队列，UI 的 worker 消费后刷新界面。"
                )

            # 模拟 token 流
            for i in range(0, len(reply), 2):
                await self._emit(AgentEvent(EventType.TEXT_DELTA, text=reply[i:i + 2]))
                await asyncio.sleep(0.015)

            await self._emit(AgentEvent(EventType.DONE))
        finally:
            self.busy = False


# --------------------------------------------------------------------------- #
# TUI
# --------------------------------------------------------------------------- #
class AgentTUI(App):
    CSS = """
    #log {
        height: 1fr;
        padding: 1 2;
        scrollbar-gutter: stable;
    }

    #prompt {
        margin: 1 2;
    }

    .user-msg {
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
        margin-bottom: 1;
    }

    .agent-msg {
        margin-bottom: 1;
    }

    .thinking {
        color: $text-muted;
        text-style: italic;
        padding-left: 1;
        margin-bottom: 1;
    }

    .tool-call {
        color: $warning;
        padding-left: 1;
        margin-bottom: 1;
    }

    .tool-result {
        color: $success;
        padding-left: 1;
        margin-bottom: 1;
    }

    .error {
        color: $error;
        padding-left: 1;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+n", "focus_input", "聚焦输入"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.events: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.agent = MockAgent(self.events)

        # 流式文本状态
        self._stream_md: Markdown | None = None
        self._stream_buf: list[str] = []
        self._stream_dirty = False

    # ---------------------------- 布局 ---------------------------- #
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="log")
        yield Input(
            placeholder="输入消息，回车发送…（试试：天气 / 代码 / 随便聊聊）",
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Agent TUI"
        self.sub_title = "idle"
        self.query_one(Input).focus()
        # 20fps 节流刷新 Markdown，避免每个 token 都重排
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
            case EventType.TEXT_DELTA:
                await self._on_delta(ev.text, log)

            case EventType.THINKING:
                await log.mount(Static(f"💭 {escape(ev.text)}", classes="thinking"))

            case EventType.TOOL_CALL:
                await log.mount(
                    Static(f"🔧 {escape(ev.name)} {escape(ev.text)}", classes="tool-call")
                )

            case EventType.TOOL_RESULT:
                await log.mount(
                    Static(f"✔ {escape(ev.name)} → {escape(ev.text)}", classes="tool-result")
                )

            case EventType.ERROR:
                await log.mount(Static(f"✖ {escape(ev.text)}", classes="error"))

            case EventType.DONE:
                await self._finish_turn()

        log.scroll_end(animate=False)

    # ---------------------------- 流式文本 ---------------------------- #
    async def _on_delta(self, text: str, log: VerticalScroll) -> None:
        if self._stream_md is None:
            self._stream_md = Markdown("", classes="agent-msg")
            await log.mount(self._stream_md)
        self._stream_buf.append(text)
        self._stream_dirty = True

    def _flush_stream(self) -> None:
        if not self._stream_dirty or self._stream_md is None:
            return
        self._stream_dirty = False
        self._stream_md.update("".join(self._stream_buf))

    def _reset_stream(self) -> None:
        self._stream_md = None
        self._stream_buf.clear()
        self._stream_dirty = False

    async def _finish_turn(self) -> None:
        self._flush_stream()
        self._reset_stream()
        inp = self.query_one(Input)
        inp.disabled = False
        inp.focus()
        self.sub_title = "idle"

    # ---------------------------- 交互 ---------------------------- #
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.agent.busy:
            return

        event.input.value = ""
        event.input.disabled = True
        self.sub_title = "working…"
        self._reset_stream()

        log = self.query_one("#log", VerticalScroll)
        await log.mount(Static(f"[b]你[/b]  {escape(text)}", classes="user-msg"))
        log.scroll_end(animate=False)

        self.run_agent(text)   # 后台 worker 驱动 Agent

    @work(group="agent")
    async def run_agent(self, prompt: str) -> None:
        await self.agent.run(prompt)

    async def action_clear_log(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        await log.remove_children()
        self._reset_stream()

    def action_focus_input(self) -> None:
        self.query_one(Input).focus()


if __name__ == "__main__":
    AgentTUI().run()