from src.agent.types import AgentEvent, AgentEventType
from src.harness.event_bus import EventBus
from src.harness.session.session_manager import SessionManager
from src.ai.types import TextContent, UserMessage


class Session:
    def __init__(self, bus: EventBus, session_id: str, session_manager: SessionManager) -> None:
        self.bus = bus
        self.bus.subscribe(self.handle)

        self.session_id = session_id
        self.session_manager = session_manager

    async def handle(self, event: AgentEvent) -> None:
        match event.type:
            case AgentEventType.MESSAGE_END:
                self.session_manager.save_session(self.session_id, [event.message])
                self._maybe_update_title(event.message)
            case AgentEventType.TURN_END:
                self.session_manager.save_session(self.session_id, event.tool_results)

    def _maybe_update_title(self, message) -> None:
        """首次用户消息作为会话标题，方便在列表里区分。"""
        if not isinstance(message, UserMessage):
            return
        if isinstance(message.content, str):
            text = message.content
        else:
            text = "".join(
                c.text for c in message.content if isinstance(c, TextContent)
            )
        text = text.strip()
        if not text:
            return
        self.session_manager.update_title_if_untitled(
            self.session_id, text[:30] + ("..." if len(text) > 30 else "")
        )
