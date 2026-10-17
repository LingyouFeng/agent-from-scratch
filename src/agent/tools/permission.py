import json
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Any


class ToolPermission(str, Enum):
    """工具权限
        ALLOW:  允许执行
        ASK:    询问用户
        DENY:   拒绝执行
    """
    ALLOW = "allow",
    ASK = "ask",
    DENY = "deny"


@dataclass(frozen=True)
class ToolRequest:
    id: str
    name: str
    arguments: object


@dataclass(frozen=True)
class PermissionResult:
    request: ToolRequest
    permission: ToolPermission
    rule_id: str
    reason: str


RuleMatcher = Callable[[ToolRequest], tuple[bool, str]]


@dataclass(frozen=True)
class PermissionRule:
    rule_id: str
    rule_name: str
    matcher: RuleMatcher
    tool_permission: ToolPermission

    def match(self, tc: ToolRequest) -> PermissionResult | None:
        is_match, reason = self.matcher(tc)
        if not is_match:
            return None
        return PermissionResult(
            request=tc,
            permission=self.tool_permission,
            rule_id=self.rule_id,
            reason=reason,
        )


# =====================================================================
# 用户审批
# =====================================================================

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
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PermissionDecision:
    """权限结果 + 用户审批结果"""
    result: PermissionResult
    allowed: bool
    approval_status: ApprovalStatus


class Approver(ABC):
    def decide(self, result: PermissionResult) -> PermissionDecision:
        """Resolve ASK through one injected UI boundary; ALLOW/DENY never prompt."""

        if result.permission is ToolPermission.ALLOW:
            return PermissionDecision(result, True, ApprovalStatus.ALLOWED)
        if result.permission is ToolPermission.DENY:
            return PermissionDecision(result, False, ApprovalStatus.DENIED)
        try:
            approved = bool(self.approve(result))
        except KeyboardInterrupt:
            return PermissionDecision(result, False, ApprovalStatus.CANCELLED)
        except Exception:
            # A closed or failed approval UI must fail closed rather than execute.
            return PermissionDecision(result, False, ApprovalStatus.CANCELLED)
        return PermissionDecision(
            result, approved,
            ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        )

    @abstractmethod
    def approve(self, result: PermissionResult) -> bool:
        ...


class ConsoleApprover(Approver):
    def approve(self, result: PermissionResult) -> bool:
        """The CLI's approval UI; replace this callback in desktop or test harnesses."""
        request = result.request
        print(f"\n\033[33m⚠ Approval needed: {result.reason}\033[0m")
        print(f"   Rule: {result.rule_id}")
        print(f"   Tool: {request.name}")
        print(f"   Input: {json.dumps(request.arguments, ensure_ascii=False, default=str)[:300]}")
        return input("   Allow? [y/N] ").strip().lower() == "y"


# =====================================================================
# 权限策略
# =====================================================================

class PermissionPolicy:
    def __init__(self, approver: Approver = ConsoleApprover()) -> None:
        self.rules = self._build_rules()
        self.approver = approver

    def _dispatch(self, tc: ToolRequest) -> PermissionResult | None:
        for rule in self.rules:
            result = rule.match(tc)
            if result is not None:
                return result

        return PermissionResult(
            request=tc,
            permission=ToolPermission.DENY,
            rule_id="default_deny",
            reason="默认拒绝",
        )

    def pipeline(self, toolcall_id: str, tool_name: str, tool_args: object) -> PermissionDecision:
        request = ToolRequest(
            id=toolcall_id,
            name=tool_name,
            arguments=tool_args,
        )
        # 根据各个规则，判断工具权限为 allow/deny/ask
        permission_result = self._dispatch(request)
        # 把工具权限结果给审批者，得到审批决策结果
        return self.approver.decide(permission_result)

    def _build_rules(self) -> tuple[PermissionRule, ...]:
        return (
            PermissionRule(
                rule_id="hard_deny",
                rule_name="硬拒绝",
                matcher=_bash_is_hard_denied,
                tool_permission=ToolPermission.DENY,
            ),
            PermissionRule(
                rule_id="only_read_allow",
                rule_name="允许只读",
                matcher=_only_read_allow,
                tool_permission=ToolPermission.ALLOW,
            ),
            PermissionRule(
                rule_id="write_requires_approval",
                rule_name="写需要审批",
                matcher=_write_requires_ask,
                tool_permission=ToolPermission.ASK,
            ),
            PermissionRule(
                rule_id="bash_requires_approval",
                rule_name="bash需要审批",
                matcher=_bash_requires_ask,
                tool_permission=ToolPermission.ASK,
            ),
        )


READ_PATH_TOOLS = frozenset({"read_file", "glob"})
WRITE_PATH_TOOLS = frozenset({"write_file", "edit_file"})
HARD_DENY_PATTERNS = (
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\brm\s+-(?=[^\s]*r)(?=[^\s]*f)[^\s]*", re.IGNORECASE),
    re.compile(r"\b(?:shutdown|reboot|mkfs(?:\.[a-z0-9]+)?)\b", re.IGNORECASE),
    re.compile(r"\bdd\s+[^\n]*(?:\bif=|\bof=/dev/)", re.IGNORECASE),
    re.compile(r">\s*/dev/(?:sd|disk)", re.IGNORECASE),
)


def _arguments(request: ToolRequest) -> Mapping[str, Any] | None:
    return request.arguments if isinstance(request.arguments, Mapping) else None


def _command(request: ToolRequest) -> str:
    arguments = _arguments(request)
    raw = arguments.get("command") if arguments else None
    return raw if isinstance(raw, str) else ""


def _bash_is_hard_denied(request: ToolRequest) -> tuple[bool, str]:
    command = _command(request)
    return (request.name == "bash" and any(pattern.search(command) for pattern in HARD_DENY_PATTERNS),
            f"command matches a non-overridable deny rule: {_command(request)!r}")


def _only_read_allow(request: ToolRequest) -> tuple[bool, str]:
    return request.name in READ_PATH_TOOLS, "read-only file operation stays inside the workspace"


def _write_requires_ask(request: ToolRequest) -> tuple[bool, str]:
    return request.name in WRITE_PATH_TOOLS, f"{request.name} mutates workspace content and requires user approval"


def _bash_requires_ask(request: ToolRequest) -> tuple[bool, str]:
    return request.name == "bash", f"shell strings cannot prove path scope or side effects; approval required: {_command(request)!r}"
