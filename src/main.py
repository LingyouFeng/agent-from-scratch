import asyncio

from ai.model import StreamOptions
from ai.provider.registry import default_registry
from agent.constant import Constants
from agent.types import ModelRequestOptions
from harness.agent import Agent
from harness.event_bus import EventBus
from harness.session.session_manager import SessionManager
from tui.cli import AgentTUI


class App:
    def __init__(self) -> None:
        self.session_manager = SessionManager(Constants.WORKDIR)
        self.tui = AgentTUI(self.session_manager)

    def run(self) -> None:
        self.tui.run()
        # event_bus = EventBus()
        #
        # user_input = input("User >> ")
        # asyncio.run(Agent(event_bus, [], ModelRequestOptions(
        #     model=default_registry.get_model_by_id(Constants.DEFAULT_MODEL_ID),
        #     stream_options=StreamOptions()
        # )).run(user_input))


if __name__ == "__main__":
    App().run()
