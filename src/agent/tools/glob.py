
"""Find files using glob patterns."""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from .base import BaseTool


class GlobArgs(BaseModel):
    """文件搜索参数。"""

    pattern: str = Field(
        description="Glob pattern, e.g. **/*.py"
    )
    path: str = Field(
        default=".",
        description="Directory to search in. Default current directory."
    )
    limit: int = Field(
        default=2000,
        ge=1,
        description="Maximum number of results"
    )


class GlobTool(BaseTool[GlobArgs]):
    name: ClassVar[str] = "glob"
    description: ClassVar[str] = (
        "Find files matching a glob pattern. "
        "Supports recursive patterns such as **/*.py."
    )
    parameters: type[GlobArgs] = GlobArgs
    concurrent_safety: ClassVar[bool] = True

    def _execute(self, args: GlobArgs) -> str:
        root = Path(args.path).expanduser().resolve()

        if not root.exists():
            return f"Error: {args.path} not found"

        if not root.is_dir():
            return f"Error: {args.path} is not a directory"

        try:
            matches = sorted(
                root.glob(args.pattern),
                key=lambda p: str(p),
            )
            matches = [
                p for p in matches
                if p.is_file()
            ]
        except (OSError, ValueError) as exc:
            return f"Error searching files: {exc}"

        total = len(matches)
        selected = matches[:args.limit]

        if not selected:
            return "No files found"

        results = "\n".join(
            str(p.relative_to(root))
            for p in selected
        )

        if total > args.limit:
            results += (
                f"\n... ({total} matches, "
                f"showing first {args.limit})"
            )

        return results
