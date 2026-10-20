import platform
from pathlib import Path


class Constants:
    """基本常量
    """

    WORKDIR = Path.cwd().resolve()
    SYSTEM_PROMPT = (f"You are a cli coding agent at {WORKDIR}. System: {platform.uname().system}. "
                     f"Use tools to solve tasks. Act, don't explain.")

    MAX_TURN = 50
    TODO_FILE = WORKDIR / ".agent_todos.json"
