import asyncio

from src.harness.agent import Agent
from src.tui.cli import AgentTUI

if __name__ == "__main__":
    AgentTUI().run()
    # agent = Agent(None)
    # user_input = input("\033[93mUser >> \033[0m")
    # asyncio.run(agent.run(user_input))