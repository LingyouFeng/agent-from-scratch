from typing import Callable

Hook = Callable[..., None]


class HookManager:
    def __init__(self):
        self._hooks: dict[str, list[Hook]] = {
            "PreToolUse": [log_before],
            "PostToolUse": [log_after],
            "UserPromptSubmit": [],
            "Stop": [],
        }

    def register(self, event: str, hook: Hook) -> None:
        self._hooks[event].append(hook)

    def emit(self, event: str, *args: object) -> None:
        for hook in self._hooks.get(event, []):
            hook(*args)


def log_before(tc_id: str, tool_name: str) -> None:
    print(f">> Hooks[PreToolUse]: Start tool {tool_name}, call_id: {tc_id}")


def log_after(tc_id: str, tool_name: str) -> None:
    print(f">> Hooks[PostToolUse]: Finished tool {tool_name}, call_id: {tc_id}")
