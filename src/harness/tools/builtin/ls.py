"""List directory contents."""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from harness.tools.base import BaseTool


class LsArgs(BaseModel):
    """列出目录内容的参数定义。"""

    path: str = Field(
        default=".",
        description="Path to the directory. Defaults to current directory.",
    )
    limit: int = Field(
        default=2000,
        ge=1,
        description="Maximum number of entries to list. Default 2000.",
    )
    action_note: str = Field(
        description="Describe what you are doing in this step. Keep it short and user-friendly."
    )


class LsTool(BaseTool[LsArgs]):
    name: ClassVar[str] = "ls"
    description: ClassVar[str] = (
        "List files and subdirectories in a directory. "
        "Directories are listed first. "
        "Hidden files are included by default."
    )
    parameters: type[LsArgs] = LsArgs
    concurrent_safety: ClassVar[bool] = True

    def execute(self, args: LsArgs) -> str:
        p = Path(args.path).expanduser().resolve()

        if not p.exists():
            return f"Error: {args.path} not found"

        if not p.is_dir():
            return f"Error: {args.path} is a file, not a directory"

        try:
            entries = list(p.iterdir())
            entries.sort(
                key=lambda entry: (
                    not entry.is_dir(),
                    entry.name.casefold(),
                )
            )
        except OSError as exc:
            return f"Error: cannot list directory {args.path}: {exc}"

        total = len(entries)
        chunk = entries[: args.limit]

        result = "\n".join(
            f"{entry.name}/" if entry.is_dir() else entry.name
            for entry in chunk
        )

        if total > args.limit:
            result += (
                f"\n... ({total} entries total, "
                f"showing first {len(chunk)})"
            )

        return result or "(empty directory)"
