"""File reading with line numbers."""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from harness.tools.base import BaseTool


class ReadFileArgs(BaseModel):
    """读取文件的参数定义。"""
    file_path: str = Field(description="Path to the file")
    offset: int = Field(
        default=1,
        ge=1,
        description="Start line (1-based). Default 1."
    )
    limit: int = Field(
        default=2000,
        ge=1,
        description="Max lines to read. Default 2000.",
    )
    action_note: str = Field(
        description="Describe what you are doing in this step. Keep it short and user-friendly."
    )


class ReadFileTool(BaseTool[ReadFileArgs]):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read a file's contents with line numbers. "
        "Always read a file before editing it."
    )
    parameters: type[ReadFileArgs] = ReadFileArgs
    concurrent_safety: ClassVar[bool] = True

    def execute(self, args: ReadFileArgs) -> str:
        p = Path(args.file_path).expanduser().resolve()

        if not p.exists():
            return f"Error: {args.file_path} not found"

        if not p.is_file():
            return f"Error: {args.file_path} is a directory, not a file"

        text = p.read_text(
            encoding="utf-8",
            errors="replace",
        )
        lines = text.splitlines()
        total = len(lines)

        start = args.offset - 1
        chunk = lines[start: start + args.limit]

        numbered = [
            f"{start + i + 1}\t{line}"
            for i, line in enumerate(chunk)
        ]
        result = "\n".join(numbered)

        if total > start + args.limit:
            result += (
                f"\n... ({total} lines total, "
                f"showing {start + 1}-{start + len(chunk)})"
            )
        return result or "(empty file)"
