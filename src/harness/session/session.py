from src.agent.types import AgentEvent, AgentEventType
from src.harness.event_bus import EventBus
from src.harness.session.session_manager import SessionManager


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
            case AgentEventType.TURN_END:
                self.session_manager.save_session(self.session_id, event.tool_results)
