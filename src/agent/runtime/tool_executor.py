from pydantic import ValidationError

from agent.hooks.manager import HookManager
from agent.tools.permission import PermissionPolicy
from agent.tools.registry import ToolRegistry


class ToolExecutor:

    def __init__(self, registry: ToolRegistry, policy: PermissionPolicy, hooks: HookManager) -> None:
        self._registry = registry
        self._policy = policy
        self._hooks = hooks

    def execute(self, tc_id: str, tool_name: str, tool_args: dict) -> str:
        # 查找工具
        tool = self._registry.get(tool_name)
        if tool is None:
            return f"Tool {tool_name} not found"

        # 校验参数
        try:
            args = tool.validate_args(tool_args)
        except ValidationError as e:
            return f"Invalid Arguments: {e}"

        # 权限校验
        decision = self._policy.pipeline(tc_id, tool_name, tool_args)

        if not decision.allowed:
            print(f"System >> Tool {tool_name} ({tool_args}) is {"" if decision.allowed else "not"} allowed, "
                  f"reason: {decision.result.reason}")
            return f"Tool is not allowed, reason: {decision.result.reason}"

        # 执行PreToolUse Hook
        self._hooks.emit("PreToolUse", tc_id, tool_name)

        # 执行工具调用
        tool_result = tool.execute(args)

        # 执行PostToolUse Hook
        self._hooks.emit("PostToolUse", tc_id, tool_name)

        return tool_result

