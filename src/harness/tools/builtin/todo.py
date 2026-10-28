"""Todo list management with persistent storage."""

import json
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from agent.constant import Constants
from harness.tools.base import BaseTool


def _load() -> list[dict]:
    if not Constants.TODO_FILE.exists():
        return []
    try:
        data = json.loads(Constants.TODO_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(todos: list[dict]) -> None:
    Constants.TODO_FILE.write_text(
        json.dumps(todos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class TodoArgs(BaseModel):
    """操作 todo 列表的参数定义。"""
    action: Literal["add", "list", "update", "remove", "clear"] = Field(
        description="Operation to perform: add | list | update | remove | clear"
    )
    content: str | None = Field(
        default=None,
        description="Task content. Required for 'add', optional for 'update'.",
    )
    id: int | None = Field(
        default=None,
        ge=1,
        description="Task id. Required for 'update' and 'remove'.",
    )
    status: Literal["pending", "in_progress", "completed"] | None = Field(
        default=None,
        description="New status. Used with 'update'.",
    )
    action_note: str = Field(
        description="Describe what you are doing in this step. Keep it short and user-friendly."
    )


class TodoTool(BaseTool[TodoArgs]):
    name: ClassVar[str] = "todo"
    description: ClassVar[str] = (
        "Manage a persistent todo list. "
        "Actions: add, list, update, remove, clear. "
        "Use 'list' before updating to know task ids."
    )
    parameters: type[TodoArgs] = TodoArgs
    concurrent_safety: ClassVar[bool] = False  # 读写共享文件，非并发安全

    def execute(self, args: TodoArgs) -> str:
        todos = _load()

        if args.action == "list":
            return self._format(todos)

        if args.action == "clear":
            _save([])
            return "Cleared all todos."

        if args.action == "add":
            if not args.content:
                return "Error: 'content' is required for 'add'."
            new_id = max((t["id"] for t in todos), default=0) + 1
            todos.append({
                "id": new_id,
                "content": args.content,
                "status": "pending",
            })
            _save(todos)
            return f"Added todo #{new_id}: {args.content}"

        if args.action in ("update", "remove"):
            if args.id is None:
                return f"Error: 'id' is required for '{args.action}'."

            idx = next(
                (i for i, t in enumerate(todos) if t["id"] == args.id),
                None,
            )
            if idx is None:
                return f"Error: todo #{args.id} not found."

            if args.action == "remove":
                removed = todos.pop(idx)
                _save(todos)
                return f"Removed todo #{removed['id']}: {removed['content']}"

            # update
            if args.status is not None:
                todos[idx]["status"] = args.status
            if args.content is not None:
                todos[idx]["content"] = args.content
            _save(todos)
            return f"Updated todo #{args.id}."

        return f"Error: unknown action '{args.action}'."

    @staticmethod
    def _format(todos: list[dict]) -> str:
        if not todos:
            return "(no todos)"
        marks = {
            "pending": "[ ]",
            "in_progress": "[~]",
            "completed": "[x]",
        }
        lines = [
            f"#{t['id']} {marks.get(t['status'], '[ ]')} {t['content']}"
            for t in todos
        ]
        return "\n".join(lines)
