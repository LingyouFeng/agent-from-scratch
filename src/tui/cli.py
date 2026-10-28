#!/usr/bin/env python3
"""一个用 Textual 写的 Agent TUI。

架构：
    每个会话一个 SessionRuntime（独立 EventBus + Agent + 落盘），
    AgentTUI 为每个 runtime 启动一个消费 worker，
    事件始终路由到该会话自己的 ConversationView。

    切换会话 = 显示/隐藏对应的 ConversationView，
    因此多个会话可以同时流式输出，切换时都能看到各自的实时效果。

组件划分：
    Sidebar           左侧会话列表（sidebar.py）
    ChatArea          右侧聊天区容器 + 共享输入行（chat_area.py）
    ConversationView  每会话一份的消息视图（chat_area.py）
    AgentTUI          组合层：管理 SessionRuntime、事件路由、会话切换（本文件）
"""

from __future__ import annotations
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Footer, Header, Input, Select, Static

from src.agent.constant import Constants
from src.agent.types import (
    AgentEvent,
    AgentEventType,
    ModelRequestOptions,
)
from src.ai.model import Model, StreamOptions
from src.ai.provider.registry import default_registry
from src.harness.session.runtime import SessionRuntime, load_context_messages
from src.harness.session.session_manager import SessionManager
from src.tui.chat_area import ChatArea, ConversationView
from src.tui.sidebar import Sidebar


_DRAFT_SESSION_ID = "__draft__"


class AgentTUI(App):
    # Textual 的 CSS_PATH 只在 App/Screen 上生效，组件样式文件统一在这里加载
    CSS_PATH = [
        Path(__file__).with_name("styles.tcss"),  # App 主布局 + 状态栏
        Path(__file__).with_name("sidebar.tcss"),  # Sidebar 组件
        Path(__file__).with_name("chat_area.tcss"),  # ChatArea / ConversationView 组件
    ]

    BINDINGS = [
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+n", "focus_input", "聚焦输入"),
    ]

    def __init__(self, session_manager: SessionManager) -> None:
        super().__init__()
        self.session_manager = session_manager

        # 会话运行时与视图，key 均为 session_id
        self._runtimes: dict[str, SessionRuntime] = {}
        self._active_id: str | None = None

        # 当前选中的模型（输入区模型选择器维护）
        self._model: Model = default_registry.get_model_by_id(
            Constants.DEFAULT_MODEL_ID
        )
        self._stream_options: StreamOptions = StreamOptions()

    # ---------------------------- 布局 ---------------------------- #
    def compose(self) -> ComposeResult:
        yield Header()

        # 主布局：左右分栏
        with Horizontal(id="main-layout"):
            # 左侧边栏：对话列表
            yield Sidebar(self.session_manager, id="sidebar")
            # 右侧主体：会话视图容器 + 输入框
            yield ChatArea(id="chat-area")

        yield Footer()

    async def on_mount(self) -> None:
        self.title = "Agent TUI"
        self.sub_title = "idle"
        self.query_one(Input).focus()

    @property
    def chat_area(self) -> ChatArea:
        return self.query_one(ChatArea)

    @property
    def sidebar(self) -> Sidebar:
        return self.query_one(Sidebar)

    @property
    def _active_runtime(self) -> SessionRuntime | None:
        if self._active_id is None:
            return None
        return self._runtimes.get(self._active_id)

    @property
    def _active_view(self) -> ConversationView | None:
        if self._active_id is None:
            return self.chat_area.get_view(_DRAFT_SESSION_ID)
        return self.chat_area.get_view(self._active_id)

    # ------------------------ 会话管理 ------------------------ #
    async def _new_session(self) -> None:
        """创建并激活一个新会话。"""
        session_id = self.session_manager.new_session()

        # 草稿视图存在则改名复用；否则会在 _activate 中新建
        reused_draft = (
            self.chat_area.rename_view(_DRAFT_SESSION_ID, session_id) is not None
        )

        self._runtimes[session_id] = SessionRuntime(
            session_id,
            self.session_manager,
            request_options=ModelRequestOptions(
                model=self._model, stream_options=self._stream_options
            ),
        )
        self.consume_events(self._runtimes[session_id])
        # 新会话记住当前选择的模型，切换回该会话时能恢复
        self.session_manager.update_model(session_id, self._model.id)

        self.sidebar.refresh_sessions()
        await self._activate(session_id)

        if not reused_draft:
            self.notify("已创建新对话", title="新建对话")

    async def _activate(self, session_id: str, history=None) -> None:
        """激活会话：创建视图（首次）、加载历史、切换显示、同步输入区状态。"""
        chat = self.chat_area
        view = chat.get_view(session_id)
        if view is None:
            view = await chat.create_view(session_id)
            if history is None:
                history = load_context_messages(self.session_manager, session_id)
            if history:
                await view.render_history(history)

        await chat.show_view(session_id)
        self._active_id = session_id

        # 模型选择器跟随当前会话关联的模型（从会话元数据恢复）
        meta = self.session_manager.get_index(session_id)
        model_id = meta.model if meta is not None else Constants.DEFAULT_MODEL_ID
        model = default_registry.get_model_by_id(model_id)
        if model is not None:
            self._model = model
            self.chat_area.model_select.value = model.id
            # 确保会话 runtime 使用会话自己的模型（而非选择器之前的值）
            self._runtimes[session_id].agent.set_model_request_options(
                ModelRequestOptions(model=model, stream_options=self._stream_options)
            )

        # 输入区状态跟随当前会话的 agent
        runtime = self._runtimes[session_id]
        busy = runtime.agent.busy
        inp = chat.input
        inp.disabled = busy
        if not busy:
            inp.focus()
        self._set_running(busy)
        self.sub_title = "working…" if busy else "idle"

        self.sidebar.highlight_session(session_id)

    async def on_sidebar_session_selected(self, event: Sidebar.SessionSelected) -> None:
        """点击左侧会话列表：使用旧会话（加载历史上下文或恢复流式视图）。"""
        session_id = event.session_id
        if session_id not in self._runtimes:
            # 首次打开旧会话：加载历史消息作为 Agent 上下文并重建渲染
            messages = load_context_messages(self.session_manager, session_id)
            self._runtimes[session_id] = SessionRuntime(
                session_id,
                self.session_manager,
                request_options=ModelRequestOptions(
                    model=self._model, stream_options=self._stream_options
                ),
            )
            self.consume_events(self._runtimes[session_id])
            await self._activate(session_id, history=messages)
        else:
            # 已打开过（含正在流式输出的会话）：直接切换视图
            await self._activate(session_id)

    async def on_sidebar_session_delete_requested(
        self, event: Sidebar.SessionDeleteRequested
    ) -> None:
        """侧边栏删除按钮：删除会话数据、运行时与视图。"""
        session_id = event.session_id
        runtime = self._runtimes.pop(session_id, None)
        if runtime is not None and runtime.agent.busy:
            runtime.agent.stop()

        try:
            self.session_manager.delete_sessions(session_id)
        except RuntimeError:
            self.notify(f"删除失败：找不到会话 {session_id}", severity="error")
            return

        await self.chat_area.remove_view(session_id)
        self.sidebar.refresh_sessions()

        if self._active_id == session_id:
            # 删除的是当前会话：回到草稿模式
            await self._enter_draft_mode()
        else:
            self.sidebar.highlight_session(self._active_id or "")

        self.notify("会话已删除", title="删除对话")

    async def _enter_draft_mode(self) -> None:
        """进入草稿模式：显示草稿视图、重置输入区状态。"""
        chat = self.chat_area
        if chat.get_view(_DRAFT_SESSION_ID) is None:
            await chat.create_view(_DRAFT_SESSION_ID)
        await chat.show_view(_DRAFT_SESSION_ID)
        self._active_id = None

        inp = chat.input
        inp.value = ""
        inp.disabled = False
        inp.focus()
        self._set_running(False)
        self.sub_title = "idle"
        # 清除侧边栏高亮（空字符串表示无选中）
        self.sidebar.highlight_session("")

    async def on_select_changed(self, event: Select.Changed) -> None:
        """模型选择器：切换模型，只应用到当前会话并落盘。"""
        if event.select.id != "model-select":
            return
        model = default_registry.get_model_by_id(str(event.value))
        if model is None:
            return
        self._model = model

        runtime = self._active_runtime
        if runtime is None:
            # 草稿模式：仅记住当前选择，新建会话时使用
            return

        # 与会话已保存模型一致时是切换会话的同步回显，不重复应用/提示
        meta = self.session_manager.get_index(runtime.session_id)
        if meta is not None and meta.model == model.id:
            return

        runtime.agent.set_model_request_options(
            ModelRequestOptions(model=model, stream_options=self._stream_options)
        )
        self.session_manager.update_model(runtime.session_id, model.id)
        self.notify(f"已切换模型：{model.name}", title="模型")

    # ------------------------ 事件消费（每会话一个 worker） ------------------------ #
    @work
    async def consume_events(self, runtime: SessionRuntime) -> None:
        """常驻 worker：从该会话的总线取事件并渲染。"""
        while True:
            event = await runtime.bus.get()
            try:
                await self._dispatch(event, runtime)
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"dispatch error: {exc!r}")
            finally:
                runtime.bus.task_done()

    async def _dispatch(self, ev: AgentEvent, runtime: SessionRuntime) -> None:
        view = self.chat_area.get_view(runtime.session_id)
        is_active = runtime.session_id == self._active_id

        match ev.type:
            case AgentEventType.AGENT_START:
                if is_active:
                    self.sub_title = "working…"

            case AgentEventType.TURN_START:
                pass

            case AgentEventType.TURN_END:
                pass

            case AgentEventType.MESSAGE_END:
                # 首条用户消息后标题已生成并写入索引，立即刷新侧边栏
                if ev.message.role == "user":
                    self.sidebar.refresh_sessions()
                    if self._active_id is not None:
                        self.sidebar.highlight_session(self._active_id)

            case AgentEventType.MESSAGE_UPDATE:
                if view is not None:
                    await view.handle_message_event(ev)

            case AgentEventType.TOOL_EXECUTION_START:
                if view is not None:
                    await view.handle_tool_execution_start(ev)

            case AgentEventType.TOOL_EXECUTION_END:
                if view is not None:
                    await view.handle_tool_execution_end(ev)

            case AgentEventType.AGENT_END:
                if view is not None:
                    await view.finalize_stream()  # 把最后一批 token 刷出去
                if is_active:
                    inp = self.chat_area.input
                    inp.disabled = False
                    inp.focus()
                    self._set_running(False)
                    self.sub_title = "idle"
                # 会话标题/预览可能已更新（首条消息设标题）
                self.sidebar.refresh_sessions()
                if self._active_id is not None:
                    self.sidebar.highlight_session(self._active_id)


        if view is not None and is_active:
            view.scroll_end(animate=False)

    # ---------------------------- 交互 ---------------------------- #
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        await self._submit_prompt(event.value)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        # 新建对话：进入草稿模式，不立即创建 session
        if event.button.id == "new-chat-btn":
            await self._enter_draft_mode()
            return

        if event.button.id != "send":
            return

        runtime = self._active_runtime
        if runtime is None:
            return

        # 运行中 → 点击执行停止
        if runtime.agent.busy:
            # 先切到「停止中…」并禁用，防止重复点击
            event.button.label = "停止中…"
            event.button.disabled = True
            try:
                runtime.agent.stop()
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"stop agent failed: {exc!r}")
                # 停止失败时恢复成运行态
                self._set_running(True)
            return

        # 空闲 → 走和回车一致的提交逻辑
        await self._submit_prompt(self.chat_area.input.value)

    async def _submit_prompt(self, text: str) -> None:
        """回车与按钮共用的发送逻辑：发给当前激活的会话。"""
        text = (text or "").strip()
        if not text:
            return

        # 草稿模式：首次真正发送时才创建 session
        if self._active_id is None:
            await self._new_session()

        runtime = self._active_runtime
        view = self._active_view
        if runtime is None or view is None or runtime.agent.busy:
            return

        inp = self.chat_area.input
        inp.value = ""
        inp.disabled = True

        view.reset_stream()
        self._set_running(True)  # 按钮切到「停止」

        await view.mount_user_message(text)

        self.run_agent(runtime, text)

    def _set_running(self, running: bool) -> None:
        """根据当前会话 agent 状态切换发送按钮的外观/可用性。"""
        btn = self.chat_area.send_button
        btn.disabled = False
        if running:
            btn.label = "停止"
            btn.variant = "error"
        else:
            btn.label = "发送"
            btn.variant = "primary"

    @work(group="agent")
    async def run_agent(self, runtime: SessionRuntime, prompt: str) -> None:
        await runtime.agent.run(prompt)

    # ---------------------------- 命令 ---------------------------- #
    async def action_clear_log(self) -> None:
        view = self._active_view
        if view is not None:
            await view.clear()

    def action_focus_input(self) -> None:
        self.query_one(Input).focus()
