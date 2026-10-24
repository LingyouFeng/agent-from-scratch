import asyncio

from src.harness.agent import Agent
from src.tui.cli import AgentTUI

if __name__ == "__main__":
    # agent = Agent(asyncio.Queue())
    # user_input = input("User >> ")
    # asyncio.run(agent.run(user_input))
    AgentTUI().run()
