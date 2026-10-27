"""File creation / overwrite."""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from src.agent.tools.base import BaseTool


class WriteFileArgs(BaseModel):
    """写入文件的参数定义。"""

    file_path: str = Field(description="Path for the file")
    content: str = Field(description="Full file content to write")
    action_note: str = Field(
        description="Describe what you are doing in this step. Keep it short and user-friendly."
    )


class WriteFileTool(BaseTool[WriteFileArgs]):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create a new file or completely overwrite an existing one. "
        "For small edits to existing files, prefer edit_file instead."
    )
    parameters: type[WriteFileArgs] = WriteFileArgs
    concurrent_safety: ClassVar[bool] = False

    def execute(self, args: WriteFileArgs) -> str:
        try:
            p = Path(args.file_path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args.content, encoding="utf-8")

            n_lines = args.content.count("\n") + (
                1 if args.content and not args.content.endswith("\n") else 0
            )

            return f"Wrote {n_lines} lines to {args.file_path}"

        except OSError as e:
            return f"Error: {e}"