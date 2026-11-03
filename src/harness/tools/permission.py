import asyncio
import glob
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
        "edit": "file_path",
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
    async def decide(self, result: PermissionResult) -> PermissionDecision:
        """Resolve ASK through one injected UI boundary; ALLOW/DENY never prompt."""

        if result.permission is ToolPermission.ALLOW:
            return PermissionDecision(result, True, ApprovalStatus.ALLOWED)
        if result.permission is ToolPermission.DENY:
            return PermissionDecision(result, False, ApprovalStatus.DENIED)
        try:
            approved = bool(await self.approve(result))
        except (asyncio.CancelledError, KeyboardInterrupt):
            # stop 打断 / Ctrl+C：fail-closed，不执行
            return PermissionDecision(result, False, ApprovalStatus.CANCELLED)
        except Exception:
            # A closed or failed approval UI must fail closed rather than execute.
            return PermissionDecision(result, False, ApprovalStatus.CANCELLED)
        return PermissionDecision(
            result, approved,
            ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        )

    @abstractmethod
    async def approve(self, result: PermissionResult) -> bool:
        ...


class ConsoleApprover(Approver):
    """无 UI 场景（测试/开发）的审批后门：ASK 一律放行。"""

    async def approve(self, result: PermissionResult) -> bool:
        return True
        # """The CLI's approval UI; replace this callback in desktop or test harnesses."""
        # request = result.request
        # print(f"\n\033[33m⚠ Approval needed: {result.reason}\033[0m")
        # print(f"   Rule: {result.rule_id}")
        # print(f"   Tool: {request.name}")
        # print(f"   Input: {json.dumps(request.arguments, ensure_ascii=False, default=str)[:300]}")
        # return input("   Allow? [y/N] ").strip().lower() == "y"


class ApprovalCoordinator(Approver):
    """TUI 审批协调器：登记 future 挂起等待 UI 回答。

    审批通知不走这里——TOOL_EXECUTION_START(pending_approval=True) 天然承担
    （UI 据此渲染审批按钮），这里只管"等"和"回答的投递"。
    """

    def __init__(self) -> None:
        # tc_id -> future[allowed]；同一时刻每个会话至多一个待审批工具
        self._pending: dict[str, asyncio.Future[bool]] = {}

    async def approve(self, result: PermissionResult) -> bool:
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[result.request.id] = fut
        try:
            return await fut
        finally:
            self._pending.pop(result.request.id, None)

    def resolve(self, tc_id: str, allowed: bool) -> None:
        """UI 回答审批：唤醒挂起的工具执行。"""
        fut = self._pending.get(tc_id)
        if fut is not None and not fut.done():
            fut.set_result(allowed)

    def cancel_all(self) -> None:
        """打断所有挂起的审批（stop 时调用）：future 被 cancel → decide fail-closed。"""
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()


# =====================================================================
# 权限策略
# =====================================================================

class PermissionPolicy:
    def __init__(self, workspace: WorkspaceScope, approver: Approver = ConsoleApprover()) -> None:
        self.rules = self._build_rules()
        self.workspace = workspace
        self.approver = approver

    def _dispatch(self, tc: ToolRequest) -> PermissionResult:
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

    def check(self, toolcall_id: str, tool_name: str, tool_args: object) -> PermissionResult:
        """规则匹配（纯同步，不等待用户）：返回 allow/ask/deny 及原因。"""
        request = ToolRequest(
            id=toolcall_id,
            name=tool_name,
            arguments=tool_args,
        )
        return self._dispatch(request)

    async def authorize(self, result: PermissionResult) -> PermissionDecision:
        """审批决策：ASK 经 approver 等待用户回答，ALLOW/DENY 秒回。"""
        return await self.approver.decide(result)

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
                rule_id="session_state_allow",
                rule_name="允许会话状态工具",
                matcher=_session_tool_allow,
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
WRITE_PATH_TOOLS = frozenset({"write_file", "edit"})
SESSION_TOOLS = frozenset({"todo", "load_skill"})
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


def _session_tool_allow(request: ToolRequest, ws: WorkspaceScope) -> tuple[bool, str]:
    return (request.name in SESSION_TOOLS,
            f"{request.name} only mutates in-session task state; no filesystem or shell side effects",)


def _write_requires_ask(request: ToolRequest, ws: WorkspaceScope) -> tuple[bool, str]:
    return (request.name in WRITE_PATH_TOOLS and ws.contains(request),
            f"{request.name} mutates workspace content and requires user approval")


def _bash_requires_ask(request: ToolRequest, ws: WorkspaceScope) -> tuple[bool, str]:
    return (request.name == "bash",
            f"shell strings cannot prove path scope or side effects; approval required: {_command(request)!r}")
