"""Execute shell commands."""

import subprocess
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from src.agent.tools.base import BaseTool


class BashArgs(BaseModel):
    """Shell 命令参数。"""

    command: str = Field(
        description="Shell command to execute"
    )
    cwd: str = Field(
        default=".",
        description="Working directory for the command"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum execution time in seconds"
    )
    max_output: int = Field(
        default=2000,
        ge=1,
        description="Maximum number of characters returned"
    )
    action_note: str = Field(
        description="Describe what you are doing in this step. Keep it short and user-friendly."
    )


class BashTool(BaseTool[BashArgs]):
    name: ClassVar[str] = "bash"
    description: ClassVar[str] = (
        "Execute a shell command and return stdout, stderr, "
        "and the exit code. Supports a working directory "
        "and timeout. Use cautiously because commands can "
        "modify files or affect the system."
    )
    parameters: type[BashArgs] = BashArgs
    concurrent_safety: ClassVar[bool] = False

    def execute(self, args: BashArgs) -> str:
        if not args.command.strip():
            return "Error: command cannot be empty"

        cwd = Path(args.cwd).expanduser().resolve()

        if not cwd.exists():
            return f"Error: working directory {args.cwd} not found"

        if not cwd.is_dir():
            return f"Error: {args.cwd} is not a directory"

        try:
            result = subprocess.run(
                args.command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""

            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

            output = (
                f"Error: command timed out after {args.timeout}s\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
            return output[:args.max_output]

        except OSError as exc:
            return f"Error executing command: {exc}"

        output = (
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        if len(output) > args.max_output:
            output = (
                    output[:args.max_output]
                    + "\n... (output truncated)"
            )

        return output
