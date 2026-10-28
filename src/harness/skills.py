from pathlib import Path

import yaml


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()


class Skills:
    def __init__(self, work_path: Path):
        self._skills_dir = work_path / "skills"
        self._skill_registry: dict[str, dict] = {}

    def load_skills(self) -> str:

        if not self._skills_dir.exists():
            return "(no skills found)"

        for d in sorted(self._skills_dir.iterdir()):
            if not d.is_dir():
                continue
            manifest = d / "SKILL.md"
            if manifest.exists():
                raw = manifest.read_text()
                meta, body = _parse_frontmatter(raw)
                name = meta.get("name", d.name)
                desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
                self._skill_registry[name] = {"name": name, "description": desc, "content": raw}

        if not self._skill_registry:
            return "(no skills found)"

        return "\n".join(f"- **{s['name']}**: {s['description']}" for s in self._skill_registry.values())
