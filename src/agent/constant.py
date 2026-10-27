import platform
from pathlib import Path


class Constants:
    """基本常量
    """
    LANGUAGE = "chinese"
    WORKDIR = Path.cwd().resolve()
    SYSTEM_PROMPT = (f"You are a general agent at {WORKDIR}. System: {platform.uname().system}. "
                     f"Use tools to solve tasks. Act, don't explain. Answer in {LANGUAGE}.")

    MAX_TURN = 50
    TODO_FILE = WORKDIR / ".agent_todos.json"
