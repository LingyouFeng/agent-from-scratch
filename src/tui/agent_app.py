#!/usr/bin/env python3
"""一个用 Textual 写的 Agent TUI。

架构：
    生产端  MockAgent ──put──> asyncio.Queue[AgentEvent]
    消费端  AgentTUI.consume_events()（Textual async worker）──> 更新 Widget

UI 层完全不关心 Agent 内部逻辑，只认队列里的事件；
换成真实 Agent 时，只要保持「往队列里 put AgentEvent」这一契约即可。

组件划分：
    Sidebar   左侧会话列表（sidebar.py）
    ChatArea  右侧聊天区，流式渲染与历史重建共用一套逻辑（chat_area.py）
    AgentTUI  组合层：消费事件、转发渲染、切换会话（本文件）
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Footer, Header, Input, Static

from src.agent.types import AgentEvent, AgentEventType, AgentStatusBarUpdateEvent
from src.harness.agent import Agent
from src.harness.event_bus import EventBus
from src.harness.session.session_manager import SessionManager
from src.tui.chat_area import ChatArea
from src.tui.sidebar import Sidebar


class AgentTUI(App):
    # Textual 的 CSS_PATH 只在 App/Screen 上生效，组件样式文件统一在这里加载
    CSS_PATH = [
        Path(__file__).with_name("styles.tcss"),      # App 主布局 + 状态栏
        Path(__file__).with_name("sidebar.tcss"),     # Sidebar 组件
        Path(__file__).with_name("chat_area.tcss"),   # ChatArea 组件
    ]

    BINDINGS = [
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+n", "focus_input", "聚焦输入"),
    ]

    def __init__(self, agent: Agent, bus: EventBus, session_manager: SessionManager) -> None:
        super().__init__()
        self.bus = bus
        self.agent = agent
        self.session_manager = session_manager

        # 轮次计数
        self._turn = 0

    # ---------------------------- 布局 ---------------------------- #
    def compose(self) -> ComposeResult:
        yield Header()

        # 主布局：左右分栏
        with Horizontal(id="main-layout"):
            # 左侧边栏：对话列表
            yield Sidebar(self.session_manager, id="sidebar")
            # 右侧主体：聊天记录 + 输入框
            yield ChatArea(id="chat-area")

        yield Footer()

    def on_mount(self) -> None:
        self.title = "Agent TUI"
        self.sub_title = "idle"
        self.query_one(Input).focus()
        # 启动消费者 worker
        self.consume_events()

    @property
    def chat_area(self) -> ChatArea:
        return self.query_one(ChatArea)

    # ------------------------ 会话切换 ------------------------ #
    async def on_sidebar_session_selected(self, event: Sidebar.SessionSelected) -> None:
        """点击左侧会话列表：加载历史消息并重建对话记录。"""
        if self.agent.busy:
            self.notify("Agent 正在运行，请等待结束后再切换会话", title="切换对话", severity="warning")
            return

        messages = self.session_manager.load_messages(event.session_id)
        await self.chat_area.render_history(messages)
        self.notify(f"已切换到对话: {event.title} (ID: {event.session_id})", title="切换对话")

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
        chat = self.chat_area

        match ev.type:
            case AgentEventType.AGENT_START:
                self._turn = 0
                self._set_running(True)  # 可选兜底
                self.sub_title = "working…"

            case AgentEventType.TURN_START:
                pass
                # self._turn += 1
                # await chat.on_turn_start(self._turn)

            case AgentEventType.TURN_END:
                pass
                # await chat.on_turn_end(self._turn)

            case AgentEventType.MESSAGE_UPDATE:
                await chat.handle_message_event(ev)

            case AgentEventType.TOOL_EXECUTION_START:
                await chat.handle_tool_execution_start(ev)

            case AgentEventType.TOOL_EXECUTION_END:
                await chat.handle_tool_execution_end(ev)

            case AgentEventType.AGENT_END:
                await chat.finalize_stream()  # 把最后一批 token 刷出去

                inp = self.query_one(Input)
                inp.disabled = False
                inp.focus()

                self._set_running(False)  # 空闲后恢复按钮
                self.sub_title = "idle"

            case AgentEventType.STATUS_BAR_UPDATE:
                pass
                # await self._on_status_bar_update(ev)

        chat.log.scroll_end(animate=False)

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

        self.chat_area.reset_stream()
        self._set_running(True)  # 按钮切到「停止」

        await self.chat_area.mount_user_message(text)

        self.run_agent(text)

    def _set_running(self, running: bool) -> None:
        """根据 agent 状态切换发送按钮的外观/可用性。"""
        btn = self.query_one("#send", Button)
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
        await self.chat_area.clear()

    def action_focus_input(self) -> None:
        self.query_one(Input).focus()
