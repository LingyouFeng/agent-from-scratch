import asyncio

from pydantic import ValidationError

from agent.hooks.manager import HookManager
from harness.tools.permission import (
    ApprovalStatus,
    PermissionDecision,
    PermissionPolicy,
    PermissionResult,
    ToolPermission,
    ToolRequest,
)
from harness.tools.registry import ToolRegistry


class ToolExecutor:
    """工具执行编排：preflight（快检）→ authorize（审批）→ perform（执行）。

    原五步流程（查工具→参数校验→权限校验→PreTool→执行→PostTool）不变，
    只是把权限等待点拆出来，让审批期间 agent 可挂起、UI 可交互。
    """

    def __init__(
        self, registry: ToolRegistry, policy: PermissionPolicy, hooks: HookManager
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._hooks = hooks

    def preflight(
        self, tc_id: str, tool_name: str, tool_args: dict
    ) -> PermissionResult:
        """同步快检（不等待用户）：查工具 + 参数校验 + 规则匹配。

        工具不存在/参数无效按 deny 处理，reason 为给模型看的错误文案；
        ASK 时调用方据此在 START 事件上声明 pending_approval。
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            return PermissionResult(
                request=ToolRequest(id=tc_id, name=tool_name, arguments=tool_args),
                permission=ToolPermission.DENY,
                rule_id="tool_not_found",
                reason=f"Tool {tool_name} not found",
            )
        try:
            tool.validate_args(tool_args)
        except ValidationError as e:
            return PermissionResult(
                request=ToolRequest(id=tc_id, name=tool_name, arguments=tool_args),
                permission=ToolPermission.DENY,
                rule_id="invalid_arguments",
                reason=f"Invalid Arguments: {e}",
            )
        return self._policy.check(tc_id, tool_name, tool_args)

    async def authorize(self, result: PermissionResult) -> PermissionDecision:
        """审批决策：ASK 时挂起等待用户回答，ALLOW/DENY 秒回。"""
        return await self._policy.authorize(result)

    async def perform(
        self, tc_id: str, tool_name: str, tool_args: dict
    ) -> tuple[bool, str]:
        """实际执行：PreTool → 工具执行（线程）→ PostTool。"""
        tool = self._registry.get(tool_name)
        if tool is None:
            return False, f"Tool {tool_name} not found"
        try:
            args = tool.validate_args(tool_args)
        except ValidationError as e:
            return False, f"Invalid Arguments: {e}"

        # 执行PreToolUse Hook
        self._hooks.emit("PreToolUse", tc_id, tool_name)

        # 执行工具调用（同步工具挪到线程，避免阻塞事件循环）
        try:
            tool_result = await asyncio.to_thread(tool.execute, args)
        except Exception as e:
            return False, f"Tool execution failed: {e}"

        # 执行PostToolUse Hook
        self._hooks.emit("PostToolUse", tc_id, tool_name)

        return True, tool_result

    @staticmethod
    def refusal_text(decision: PermissionDecision) -> str:
        """拒绝时给模型的文案：区分用户拒绝与权限规则拒绝。"""
        match decision.approval_status:
            case ApprovalStatus.REJECTED:
                return "用户拒绝执行该工具"
            case ApprovalStatus.CANCELLED:
                return "工具执行被取消"
            case _:
                return f"Tool is not allowed, reason: {decision.result.reason}"
