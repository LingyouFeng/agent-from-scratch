from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ToolPermission(str, Enum):
    """工具权限
        ALLOW:  允许执行
        ASK:    询问用户
        DENY:   拒绝执行
    """
    ALLOW = "allow",
    ASK = "ask",
    DENY = "deny"


class ApprovalStatus(str, Enum):
    """工具权限审批结果
        DENIED:     系统自动拒绝
        ALLOWED:    系统自动允许
        APPROVED:   用户批准
        REJECTED:   用户拒绝
        CANCELLED:  用户打断取消
    """
    DENIED = "denied",
    ALLOWED = "allowed"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELED = "cancelled"


class ToolCall:
    tool_id: str
    tool_name: str
    tool_arguments: str


@dataclass(frozen=True)
class PermissionDecision:
    toolcall: ToolCall
    tool_permission: ToolPermission
    rule_id: str
    decision_reason: str


RuleMatcher = Callable[[ToolCall], tuple[bool, str]]


@dataclass(frozen=True)
class PermissionRule:
    rule_id: str
    rule_name: str
    matcher: RuleMatcher
    tool_permission: ToolPermission

    def evaluate(self, tc: ToolCall) -> PermissionDecision | None:
        is_match, reason = self.matcher(tc)
        if not is_match:
            return None
        return PermissionDecision(
            toolcall=tc,
            tool_permission=self.tool_permission,
            rule_id=self.rule_id,
            decision_reason=reason,
        )


class PermissionPolicy:

    def authorize(self) -> tuple[bool, str]:
        pass
