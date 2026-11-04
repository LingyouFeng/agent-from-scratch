#!/usr/bin/env python3
"""一个用 Textual 写的 Agent TUI。

架构：
    每个会话一个 SessionRuntime（独立 EventBus + Agent + 落盘，核心层门面），
    AgentTUI 为每个 runtime 启动一个消费 worker，
    事件始终路由到该会话自己的 ConversationView。

    切换会话 = 显示/隐藏对应的 ConversationView，
    因此多个会话可以同时流式输出，切换时都能看到各自的实时效果。

    边界约定：TUI 只通过 SessionService / SessionRuntime 的窄接口与核心层交互，
    不直接触碰 Agent / EventBus / SessionManager。

组件划分：
    Sidebar           左侧会话列表（sidebar.py，由 AgentTUI 喂数据）
    ChatArea          右侧聊天区容器 + 共享输入行（chat_area.py）
    ConversationView  每会话一份的消息视图（chat_area.py）
    SessionService    会话簿记 + runtime 注册表（核心层唯一入口，service.py）
    AgentTUI          组合层：激活会话管理、事件路由、会话切换（本文件）
"""

from __future__ import annotations
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Footer, Header, Input, Select

from agent.constant import Constants
from agent.types import AgentEvent, AgentEventType
from harness.session.runtime import SessionRuntime
from harness.session.service import SessionService
from tui.approval_bar import ApprovalBar
from tui.chat_area import ChatArea, ConversationView, TokenSummary
from tui.sidebar import Sidebar
from tui.usage import UsageStats

_DRAFT_SESSION_ID = "__draft__"


class AgentTUI(App):
    # Textual 的 CSS_PATH 只在 App/Screen 上生效，组件样式文件统一在这里加载
    CSS_PATH = [
        Path(__file__).parent / "css" / "styles.tcss",  # App 主布局 + 状态栏
        Path(__file__).parent / "css" / "sidebar.tcss",  # Sidebar 组件
        Path(__file__).parent / "css" / "chat_area.tcss",  # ChatArea / ConversationView 组件
        Path(__file__).parent / "css" / "approval_bar.tcss",  # ApprovalBar 审批操作条
    ]

    BINDINGS = [
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+n", "focus_input", "聚焦输入"),
    ]

    def __init__(self, service: SessionService) -> None:
        super().__init__()
        self.service = service
        # 当前激活的会话 id（None = 草稿模式）；runtime 注册表在 service 里
        self._active_id: str | None = None
        # 当前选中的模型 id（输入区模型选择器维护；草稿模式下新建会话时使用）
        self._model_id: str = Constants.DEFAULT_MODEL_ID
        # 有待审批工具的会话集合（驱动 sidebar ⏳ 徽章；串行执行时每会话至多一个）
        self._pending_approval: set[str] = set()

    # ---------------------------- 布局 ---------------------------- #
    def compose(self) -> ComposeResult:
        yield Header()

        # 主布局：左右分栏
        with Horizontal(id="main-layout"):
            # 左侧边栏：对话列表
            yield Sidebar(id="sidebar")
            # 右侧主体：会话视图容器 + 输入框
            yield ChatArea(id="chat-area")

        yield Footer()

    async def on_mount(self) -> None:
        self.title = "Agent Harness"
        self.sub_title = "idle"
        self.query_one(Input).focus()
        self._refresh_sidebar()
        self._refresh_stats(None)

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
        return self.service.get_session(self._active_id)

    @property
    def _active_view(self) -> ConversationView | None:
        if self._active_id is None:
            return self.chat_area.get_view(_DRAFT_SESSION_ID)
        return self.chat_area.get_view(self._active_id)

    # ------------------------ 会话管理 ------------------------ #
    async def _new_session(self) -> None:
        """创建并激活一个新会话。"""
        runtime = self.service.create_session(self._model_id)
        session_id = runtime.session_id

        # 草稿视图存在则改名复用；否则会在 _activate 中新建
        reused_draft = (
            self.chat_area.rename_view(_DRAFT_SESSION_ID, session_id) is not None
        )

        self.consume_events(runtime)
        self._refresh_sidebar()
        await self._activate(session_id)

        if not reused_draft:
            self.notify("已创建新对话", title="新建对话")

    async def _activate(self, session_id: str) -> None:
        """激活会话：创建视图（首次用 runtime 历史重建）、切换显示、同步输入区状态。"""
        runtime = self.service.get_session(session_id)
        if runtime is None:
            return  # 未打开的会话：不会发生（调用方先 create/open）

        chat = self.chat_area
        view = chat.get_view(session_id)
        if view is None:
            view = await chat.create_view(session_id)
            history = runtime.history()
            if history:
                await view.render_history(history)

        await chat.show_view(session_id)
        self._active_id = session_id

        # 模型选择器跟随当前会话关联的模型（meta.model 是唯一事实源）
        self._model_id = runtime.model_id
        chat.model_select.value = runtime.model_id

        # 输入区状态跟随当前会话的运行状态
        busy = runtime.is_running
        inp = chat.input
        inp.disabled = busy
        if not busy:
            inp.focus()
        self._set_running(busy)
        self.sub_title = "working…" if busy else "idle"
        self._refresh_stats(runtime)

        self.sidebar.highlight_session(session_id)

    async def on_sidebar_session_selected(self, event: Sidebar.SessionSelected) -> None:
        """点击左侧会话列表：首次打开时构建运行时并载入历史，已打开直接切换。"""
        session_id = event.session_id
        if self.service.get_session(session_id) is None:
            # 首次打开旧会话：service 在线程里载入历史并构建运行时
            runtime = await self.service.open_session(session_id)
            self.consume_events(runtime)
        await self._activate(session_id)

    async def on_sidebar_session_delete_requested(
        self, event: Sidebar.SessionDeleteRequested
    ) -> None:
        """侧边栏删除按钮：删除会话数据、运行时与视图。"""
        session_id = event.session_id
        try:
            # service 内部：运行中先请求停止，再删除会话数据
            self.service.delete_session(session_id)
        except RuntimeError:
            self.notify(f"删除失败：找不到会话 {session_id}", severity="error")
            return

        await self.chat_area.remove_view(session_id)
        self._refresh_sidebar()

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
        self._refresh_stats(None)
        # 清除侧边栏高亮（空字符串表示无选中）
        self.sidebar.highlight_session("")

    async def on_select_changed(self, event: Select.Changed) -> None:
        """模型选择器：切换模型，只应用到当前会话并落盘。"""
        if event.select.id != "model-select":
            return
        model_id = str(event.value)
        # 记住当前选择（草稿模式新建会话时使用）
        self._model_id = model_id

        runtime = self._active_runtime
        if runtime is None:
            return

        # 与会话已保存模型一致时是切换会话的同步回显，不重复应用/提示
        if runtime.model_id == model_id:
            return

        model = runtime.set_model(model_id)
        if model is None:
            return  # 未知模型 id（防御）：保持现状
        self.notify(f"已切换模型：{model.name}", title="模型")

    # ------------------------ 事件消费（每会话一个 worker） ------------------------ #
    @work
    async def consume_events(self, runtime: SessionRuntime) -> None:
        """常驻 worker：消费该会话的事件流并渲染。"""
        async for ev in runtime.events():
            try:
                await self._dispatch(ev, runtime)
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"dispatch error: {exc!r}")

    async def _dispatch(self, ev: AgentEvent, runtime: SessionRuntime) -> None:
        view = self.chat_area.get_view(runtime.session_id)
        is_active = runtime.session_id == self._active_id

        match ev.type:
            case AgentEventType.AGENT_START:
                if is_active:
                    self.sub_title = "working…"
                    self._refresh_stats(runtime)

            case AgentEventType.TURN_START:
                pass

            case AgentEventType.TURN_END:
                if is_active:
                    self._refresh_stats(runtime)

            case AgentEventType.MESSAGE_END:
                # 首条用户消息后标题已生成并写入索引，立即刷新侧边栏
                if ev.message.role == "user":
                    self._refresh_sidebar()
                    if self._active_id is not None:
                        self.sidebar.highlight_session(self._active_id)
                elif view is not None:
                    # 登记完整 assistant 消息，供历史分页重渲染
                    view.record_message(ev.message)

            case AgentEventType.MESSAGE_UPDATE:
                if view is not None:
                    await view.handle_message_event(ev)

            case AgentEventType.TOOL_EXECUTION_START:
                if view is not None:
                    await view.handle_tool_execution_start(ev)
                if ev.pending_approval:
                    # 该会话有工具在等审批：sidebar 加 ⏳ 徽章
                    self._pending_approval.add(runtime.session_id)
                    self._refresh_sidebar()

            case AgentEventType.TOOL_EXECUTION_END:
                if view is not None:
                    await view.handle_tool_execution_end(ev)
                # 待审批工具已收尾（每会话串行执行，至多一个）：清除徽章
                if runtime.session_id in self._pending_approval:
                    self._pending_approval.discard(runtime.session_id)
                    self._refresh_sidebar()

            case AgentEventType.AGENT_END:
                if view is not None:
                    view.discard_pending_tool_results()
                    await view.finalize_stream()  # 把最后一批 token 刷出去
                if is_active:
                    inp = self.chat_area.input
                    inp.disabled = False
                    inp.focus()
                    self._set_running(False)
                    self.sub_title = "idle"
                    self._refresh_stats(runtime)
                # 防御：运行结束不应残留待审批状态
                self._pending_approval.discard(runtime.session_id)
                # 会话标题/预览可能已更新（首条消息设标题）
                self._refresh_sidebar()
                if self._active_id is not None:
                    self.sidebar.highlight_session(self._active_id)

    # ---------------------------- 交互 ---------------------------- #
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        await self._submit_prompt(event.value)

    async def on_approval_bar_responded(self, event: ApprovalBar.Responded) -> None:
        """审批按钮：把用户的选择交给对应会话的 runtime，唤醒挂起的工具执行。"""
        runtime = self.service.get_session(event.session_id)
        if runtime is None:
            return
        runtime.approve_tool(event.tc_id, event.allowed)

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
        if runtime.is_running:
            # 先切到「停止中」并禁用，防止重复点击；用单字图标保持按钮尺寸不变
            event.button.label = "…"
            event.button.disabled = True
            try:
                runtime.stop()
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
        if runtime is None or view is None or runtime.is_running:
            return

        inp = self.chat_area.input
        inp.value = ""
        inp.disabled = True

        view.reset_stream()
        self._set_running(True)  # 按钮切到「停止」

        await view.mount_user_message(text)

        self.run_agent(runtime, text)

    def _set_running(self, running: bool) -> None:
        """根据当前会话运行状态切换圆形按钮的图标与配色。"""
        btn = self.chat_area.send_button
        btn.disabled = False
        if running:
            btn.label = "■"
            btn.variant = "error"
        else:
            btn.label = "↑"
            btn.variant = "primary"

    def _refresh_sidebar(self) -> None:
        """从 service 拉取会话列表，重建侧边栏（含待审批徽章）。"""
        self.sidebar.refresh_sessions(
            self.service.list_sessions(),
            pending_ids=frozenset(self._pending_approval),
        )

    def _refresh_stats(self, runtime: SessionRuntime | None) -> None:
        """从会话元数据读取轮数与 token 用量，刷新底栏（会话元数据是唯一数据源）。"""
        meta = runtime.meta if runtime is not None else None
        self.chat_area.update_stats(
            UsageStats(
                turns=meta.total_turns if meta else 0,
                steps=meta.total_steps if meta else 0,
                total_tokens=meta.total_tokens if meta else 0,
                prompt_tokens=meta.prompt_tokens if meta else 0,
                completion_tokens=meta.completion_tokens if meta else 0,
                cached_tokens=meta.prompt_cache_hit_tokens if meta else 0,
                context_tokens=meta.context_tokens if meta else 0,
            )
        )

    def on_token_summary_clicked(self, event: TokenSummary.Clicked) -> None:
        """点击底栏中间的 token 汇总，在点击位置上方悬浮明细。"""
        event.stop()
        self.chat_area.open_usage_popup()

    @work(group="agent")
    async def run_agent(self, runtime: SessionRuntime, prompt: str) -> None:
        await runtime.send(prompt)

    # ---------------------------- 命令 ---------------------------- #
    async def action_clear_log(self) -> None:
        view = self._active_view
        if view is not None:
            await view.clear()

    def action_focus_input(self) -> None:
        self.query_one(Input).focus()
