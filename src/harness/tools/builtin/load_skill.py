"""Load a skill's instructions into the agent context."""

import re
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from agent.constant import Constants
from harness.tools.base import BaseTool

# 技能目录下可识别的入口文件，按优先级从前往后匹配。
SKILL_FILENAMES: tuple[str, ...] = ("SKILL.md", "skill.md", "README.md")

# 技能名只允许字母、数字、下划线和连字符，从根上避免路径穿越。
_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class LoadSkillArgs(BaseModel):
    """技能加载参数。"""

    name: str = Field(
        description="Name of the skill to load (the directory name under the skills directory)"
    )
    max_output: int = Field(
        default=20000,
        ge=1,
        description="Maximum number of characters returned"
    )
    action_note: str = Field(
        description="Describe what you are doing in this step. Keep it short and user-friendly."
    )


class LoadSkillTool(BaseTool[LoadSkillArgs]):
    name: ClassVar[str] = "load_skill"
    description: ClassVar[str] = (
        "Load the full instructions of a named skill into the conversation. "
        "Call this when a task matches one of the available skills and you "
        "need its detailed workflow or reference material. This tool only "
        "reads files, so it is safe to call repeatedly."
    )
    parameters: type[LoadSkillArgs] = LoadSkillArgs
    concurrent_safety: ClassVar[bool] = True

    skills_dir: Path = Constants.WORKDIR / "skills"

    def execute(self, args: LoadSkillArgs) -> str:
        name = args.name.strip()
        if not name:
            return "Error: skill name cannot be empty"

        if not _SKILL_NAME_RE.match(name):
            return (
                f"Error: invalid skill name {name!r}; only letters, digits, "
                "'-' and '_' are allowed"
            )

        skills_dir = Path(self.skills_dir).expanduser().resolve()

        if not skills_dir.exists():
            return f"Error: skills directory {self.skills_dir} not found"

        if not skills_dir.is_dir():
            return f"Error: {self.skills_dir} is not a directory"

        skill_path = self._find_skill_file(skills_dir, name)

        if skill_path is None:
            available = self._available_skills(skills_dir)
            hint = ", ".join(available) if available else "(none)"
            return (
                f"Error: skill {name!r} not found in {skills_dir}\n"
                f"available skills: {hint}"
            )

        try:
            content = skill_path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            return f"Error reading skill {name!r}: {exc}"

        if not content.strip():
            return f"Error: skill {name!r} is empty ({skill_path})"

        output = (
            f"skill: {name}\n"
            f"path: {skill_path}\n"
            f"---\n"
            f"{content}"
        )

        if len(output) > args.max_output:
            output = (
                    output[:args.max_output]
                    + "\n... (output truncated)"
            )

        return output

    @staticmethod
    def _find_skill_file(skills_dir: Path, name: str) -> Path | None:
        """定位技能入口文件：先找 <skills_dir>/<name>/<SKILL_FILENAMES>，再退回 <skills_dir>/<name>.md。"""
        skill_dir = skills_dir / name

        if skill_dir.is_dir():
            for filename in SKILL_FILENAMES:
                candidate = skill_dir / filename
                if candidate.is_file():
                    return candidate

        direct = skills_dir / f"{name}.md"
        if direct.is_file():
            return direct

        return None

    @classmethod
    def _available_skills(cls, skills_dir: Path) -> list[str]:
        """列出目录下所有可加载的技能名，用于错误提示。"""
        try:
            entries = sorted(skills_dir.iterdir())
        except OSError:
            return []

        return [
            entry.name
            for entry in entries
            if entry.is_dir() and cls._find_skill_file(skills_dir, entry.name) is not None
        ]
