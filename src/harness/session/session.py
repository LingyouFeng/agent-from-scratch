from src.agent.constant import Constants
from src.agent.types import AgentEvent, AgentEventType, AgentMessage
from src.ai.api.all import get_api_by_api
from src.ai.model import Model, StreamOptions
from src.ai.provider.registry import default_registry
from src.ai.types import Context
from src.harness.event_bus import EventBus
from src.harness.session.session_manager import SessionManager


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

    async def handle(self, event: AgentEvent) -> None:
        match event.type:
            case AgentEventType.MESSAGE_END:
                await self._update_title(event.message)
                self.session_manager.save_session(self.session_id, [event.message])
            case AgentEventType.TURN_END:
                self.session_manager.save_session(self.session_id, event.tool_results)

    async def _update_title(self, message: AgentMessage) -> None:
        """使用第一轮对话，用非流式调用大模型生成会话标题"""
        if message.role != "user":
            return
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
