from agent.constant import Constants
from harness.session.service import SessionService
from harness.session.session_manager import SessionManager
from tui.cli import AgentTUI


class App:
    def __init__(self) -> None:
        session_manager = SessionManager(Constants.WORKDIR)
        self.service = SessionService(session_manager)
        self.tui = AgentTUI(self.service)

    def run(self) -> None:
        self.tui.run()


if __name__ == "__main__":
    App().run()
