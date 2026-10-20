from datetime import datetime
import json

from src.agent.constant import Constants


class StatusBar:
    def __init__(self) -> None:
        self.todos: list[dict[str, str]] = []
        self.reload()  # 构造时先加载一次

    def reload(self) -> None:
        """从持久化文件重新加载 todo，容错回退为空列表。"""
        if not Constants.TODO_FILE.exists():
            self.todos = []
            return
        try:
            data = json.loads(Constants.TODO_FILE.read_text(encoding="utf-8"))
            self.todos = data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            self.todos = []

    def build(self) -> str:
        self.reload()  # 每次构建前刷新，反映 TodoTool 的最新写入

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_bar = (
            f"system time: {current_time}\n"
            f"todos: {self.format_todos()}"
        )
        return status_bar

    def format_todos(self) -> str:
        if not self.todos:
            return "(none)"

        marks = {
            "pending": "[ ]",
            "in_progress": "[~]",
            "completed": "[x]",
        }
        done = sum(1 for t in self.todos if t.get("status") == "completed")
        total = len(self.todos)

        lines = [f"{done}/{total} completed"]
        for t in self.todos:
            mark = marks.get(t.get("status", "pending"), "[ ]")
            lines.append(f"  #{t.get('id', '?')} {mark} {t.get('content', '')}")
        return "\n".join(lines)