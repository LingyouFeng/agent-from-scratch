import asyncio

from src.agent.constant import Constants
from src.harness.session.session_manager import SessionManager
from src.tui.cli import AgentTUI


class App:
    def __init__(self) -> None:
        self.session_manager = SessionManager(Constants.WORKDIR)
        # 会话的创建/复用统一由 AgentTUI 管理（无选中会话则新建）
        self.tui = AgentTUI(self.session_manager)

    def run(self) -> None:
        self.tui.run()


if __name__ == "__main__":
    App().run()
