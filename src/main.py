import asyncio

from src.agent.constant import Constants
from src.ai.types import Message
from src.harness.agent import Agent
from src.harness.event_bus import EventBus
from src.harness.session.session import Session
from src.harness.session.session_manager import SessionManager
from src.tui.cli import AgentTUI


class App:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.messages: list[Message] = []
        self.session_manager = SessionManager(Constants.WORKDIR)
        session_id = self.session_manager.new_session()
        self.session = Session(self.bus, session_id, self.session_manager)
        self.agent = Agent(self.bus, self.messages)
        self.tui = AgentTUI(self.agent, self.bus)

    def run(self) -> None:
        self.tui.run()
        # user_input = input("User >> ")
        # asyncio.run(self.agent.run(user_input))


if __name__ == "__main__":
    App().run()
