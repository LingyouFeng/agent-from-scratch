import platform
from pathlib import Path


class Constants:
    """基本常量"""

    LANGUAGE = "chinese"
    WORKDIR = Path.cwd().resolve().parent
    DEFAULT_MODEL_ID = "cmhi/deepseek-v4.1-flash"
    MAX_TURN = 50
    TODO_FILE = WORKDIR / ".agent_todos.json"
