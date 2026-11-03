import asyncio
import logging

from agent.constant import Constants
from agent.types import AgentEvent, AgentEventType, AgentMessage
from ai.api.all import get_api_by_api
from ai.model import Model, StreamOptions
from ai.provider.registry import default_registry
from ai.types import Context
from harness.event_bus import EventBus
from harness.session.session_manager import SessionManager

SESSION_TITLE_SYSTEM_PROMPT = f"""
You are a session title generator. You will receive a user question. 
Your task is to generate a short title for this session.
The title should be no more than 30 characters and should be in {Constants.LANGUAGE}.
"""


class Session:
    def __init__(
            self, bus: EventBus, session_id: str, session_manager: SessionManager
    ) -> None:
        self.bus = bus
        self.bus.subscribe(self.handle)

        self.session_id = session_id
        self.session_manager = session_manager

        self._model: Model = default_registry.get_model_by_id(
            Constants.DEFAULT_MODEL_ID
        )
        self._stream_option: StreamOptions = StreamOptions(
            stream=False, thinking_level="none"
        )
        self.api = get_api_by_api(self._model.api)
        # 持有标题任务引用，避免任务完成前被 GC
        self._title_task: asyncio.Task | None = None

    async def handle(self, event: AgentEvent) -> None:
        # 标题调度依赖事件循环（create_task），必须留在事件循环线程
        if (
            event.type == AgentEventType.MESSAGE_END
            and event.message.role == "user"
        ):
            self._schedule_title_update(event.message)

        # 落盘整体挪到线程执行，避免同步文件 IO 阻塞事件循环；
        # 同一事件内顺序执行，跨事件顺序仍由 publish 的同步扇出保证
        await asyncio.to_thread(self._persist, event)

    def _persist(self, event: AgentEvent) -> None:
        """同步落盘逻辑：只在 worker 线程上运行，不碰事件循环。"""
        match event.type:
            case AgentEventType.MESSAGE_END:
                if event.message.role == "user":
                    # 更新session meta轮数
                    self.session_manager.add_turn(self.session_id)

                # 保存到jsonl
                self.session_manager.save_session(self.session_id, [event.message])
            case AgentEventType.TURN_END:
                self.session_manager.save_session(self.session_id, event.tool_results)
                self.session_manager.add_usage(self.session_id, event.message.usage)

    def _schedule_title_update(self, message: AgentMessage) -> None:
        """后台生成标题；失败只记日志，保持 untitled 待下次用户消息重试。"""

        async def _run() -> None:
            try:
                await self._update_title(message)
            except Exception:  # noqa: BLE001
                logging.exception("生成会话标题失败")

        self._title_task = asyncio.create_task(_run())

    async def _update_title(self, message: AgentMessage) -> None:
        """使用第一轮对话，用非流式调用大模型生成会话标题"""
        if self.api is None:
            return
        # 标题已生成过，不再调用大模型
        meta = self.session_manager.get_index(self.session_id)
        if meta is not None and meta.title != "untitled":
            return
        context = Context(SESSION_TITLE_SYSTEM_PROMPT, [message], [])
        result = await self.api.complete(self._model, context, self._stream_option)
        text = "".join(c.text for c in result.content if c.type == "text")
        self.session_manager.update_title_if_untitled(
            self.session_id, text[:30] + ("..." if len(text) > 30 else "")
        )
