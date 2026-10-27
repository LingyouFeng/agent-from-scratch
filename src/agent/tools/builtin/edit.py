
"""Edit a file by replacing an exact string."""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from src.agent.tools.base import BaseTool


class EditFileArgs(BaseModel):
    """文件编辑参数。"""

    file_path: str = Field(description="Path to the file")
    old_string: str = Field(
        description="Exact text to replace"
    )
    new_string: str = Field(
        description="Replacement text"
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences instead of requiring a unique match",
    )
    action_note: str = Field(
        description="Describe what you are doing in this step. Keep it short and user-friendly."
    )


class EditFileTool(BaseTool[EditFileArgs]):
    name: ClassVar[str] = "edit"
    description: ClassVar[str] = (
        "Replace exact text in a file. "
        "Read the file before editing it. "
        "By default, the old text must match exactly once."
    )
    parameters: type[EditFileArgs] = EditFileArgs
    concurrent_safety: ClassVar[bool] = False

    def execute(self, args: EditFileArgs) -> str:
        p = Path(args.file_path).expanduser().resolve()

        if not p.exists():
            return f"Error: {args.file_path} not found"

        if not p.is_file():
            return f"Error: {args.file_path} is not a file"

        if not args.old_string:
            return "Error: old_string cannot be empty"

        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return f"Error reading file: {exc}"

        count = text.count(args.old_string)

        if count == 0:
            return "Error: old_string not found; file unchanged"

        if not args.replace_all and count != 1:
            return (
                f"Error: old_string matched {count} times; "
                "provide more context or set replace_all=true"
            )

        replacement_count = count if args.replace_all else 1
        updated = (
            text.replace(args.old_string, args.new_string)
            if args.replace_all
            else text.replace(args.old_string, args.new_string, 1)
        )

        try:
            p.write_text(updated, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return f"Error writing file: {exc}"

        return (
            f"Successfully edited {p}: "
            f"{replacement_count} replacement(s)"
        )
