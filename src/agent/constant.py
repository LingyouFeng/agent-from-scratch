import platform
from pathlib import Path


class Constants:
    """基本常量"""

    LANGUAGE = "chinese"
    WORKDIR = Path.cwd().resolve().parent
    DEFAULT_MODEL_ID = "cmhi/deepseek-v4.1-flash"
    TODO_FILE = WORKDIR / ".agent_todos.json"
