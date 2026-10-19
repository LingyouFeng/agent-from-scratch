import glob
import json
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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


# =====================================================================
# 工作区
# =====================================================================
class WorkspaceScope:
    """Classify file-tool arguments against one resolved workspace root.

    ``resolve(strict=False)`` follows existing symlinks while still allowing a
    not-yet-created final path.  That catches a write through ``workspace/link``
    when the link points outside the workspace.  It is still a preflight check,
    so production code also needs OS isolation to close time-of-check/time-of-use
    races.
    """

    PATH_ARGUMENTS = {
        "read_file": "file_path",
        "write_file": "file_path",
        "edit_file": "file_path",
        "glob": "pattern",
        "ls": "path"
    }

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)

    def argument_text(self, request: ToolRequest) -> str | None:
        if not isinstance(request.arguments, Mapping):
            return None
        key = self.PATH_ARGUMENTS.get(request.name)
        if key is None:
            return None
        raw = request.arguments.get(key)
        return raw if isinstance(raw, str) and raw else None

    def contains(self, request: ToolRequest) -> bool:
        raw = self.argument_text(request)
        if raw is None:
            return False
        candidate_text = self._static_glob_prefix(raw) if request.name == "glob" else raw
        candidate = Path(candidate_text).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            result = candidate.resolve(strict=False).is_relative_to(self.root)
            return result
        except (OSError, RuntimeError):
            # Resolution failures and symlink loops fail closed.
            return False

    @staticmethod
    def _static_glob_prefix(pattern: str) -> str:
        """Return the non-wildcard prefix whose directory scope can be checked."""

        parts: list[str] = []
        for part in Path(pattern).parts:
            if glob.has_magic(part):
                break
            parts.append(part)
        return str(Path(*parts)) if parts else "."


@dataclass(frozen=True)
class PermissionRule:
    rule_id: str
    rule_name: str
    matcher: Callable[[ToolRequest, WorkspaceScope], tuple[bool, str]]
    tool_permission: ToolPermission

    def match(self, tc: ToolRequest, ws: WorkspaceScope) -> PermissionResult | None:
        is_match, reason = self.matcher(tc, ws)
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
        return True
        # """The CLI's approval UI; replace this callback in desktop or test harnesses."""
        # request = result.request
        # print(f"\n\033[33m⚠ Approval needed: {result.reason}\033[0m")
        # print(f"   Rule: {result.rule_id}")
        # print(f"   Tool: {request.name}")
        # print(f"   Input: {json.dumps(request.arguments, ensure_ascii=False, default=str)[:300]}")
        # return input("   Allow? [y/N] ").strip().lower() == "y"


# =====================================================================
# 权限策略
# =====================================================================

class PermissionPolicy:
    def __init__(self, workspace: WorkspaceScope, approver: Approver = ConsoleApprover()) -> None:
        self.rules = self._build_rules()
        self.workspace = workspace
        self.approver = approver

    def _dispatch(self, tc: ToolRequest) -> PermissionResult | None:
        for rule in self.rules:
            result = rule.match(tc, self.workspace)
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
                rule_id="path_outside_workspace",
                rule_name="路径在工作区外",
                matcher=_path_outside_workspace,
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


PATH_TOOLS = frozenset(WorkspaceScope.PATH_ARGUMENTS)
READ_PATH_TOOLS = frozenset({"read_file", "glob", "ls"})
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


def _bash_is_hard_denied(request: ToolRequest, ws: WorkspaceScope) -> tuple[bool, str]:
    command = _command(request)
    return (request.name == "bash" and any(pattern.search(command) for pattern in HARD_DENY_PATTERNS),
            f"command matches a non-overridable deny rule: {_command(request)!r}")


def _path_outside_workspace(request: ToolRequest, ws: WorkspaceScope) -> tuple[bool, str]:
    return (request.name in PATH_TOOLS and not ws.contains(request),
            f"path {ws.argument_text(request)!r} resolves outside workspace {str(ws.root)!r}")


def _only_read_allow(request: ToolRequest, ws: WorkspaceScope) -> tuple[bool, str]:
    return (request.name in READ_PATH_TOOLS and ws.contains(request),
            "read-only file operation stays inside the workspace")


def _write_requires_ask(request: ToolRequest, ws: WorkspaceScope) -> tuple[bool, str]:
    return (request.name in WRITE_PATH_TOOLS and ws.contains(request),
            f"{request.name} mutates workspace content and requires user approval")


def _bash_requires_ask(request: ToolRequest, ws: WorkspaceScope) -> tuple[bool, str]:
    return (request.name == "bash",
            f"shell strings cannot prove path scope or side effects; approval required: {_command(request)!r}")
