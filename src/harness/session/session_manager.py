import json
import uuid
from dataclasses import field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from pydantic import ValidationError, TypeAdapter
from pydantic_core import to_jsonable_python

from src.agent.types import AgentMessage


class SessionMeta(BaseModel):
    id: str
    title: str = field(default="untitled")
    created_at: datetime = field(default=datetime.now())
    updated_at: datetime = field(default=datetime.now())


# 文件格式：list[SessionMeta]
_SessionList = TypeAdapter(list[SessionMeta])


class SessionIndex:
    """会话元数据管理"""

    def __init__(self, save_path: Path):
        self._index_file = save_path / "index.json"
        self._index: dict[str, SessionMeta] = self.load_index()

    def get_index(self, session_id: str) -> SessionMeta:
        return self._index.get(session_id)

    def load_index(self) -> dict[str, SessionMeta]:
        if not self._index_file.exists():
            return {}
        try:
            raw = self._index_file.read_text(encoding="utf-8")
            metas = _SessionList.validate_json(raw)
        except (OSError, ValidationError):
            return {}
        return {m.id: m for m in metas}

    def save_index(self) -> None:
        tmp = self._index_file.with_suffix(".json.tmp")
        # dict -> list；想保序就用 dict 的插入顺序（Py3.7+ 保证）
        tmp.write_text(
            _SessionList.dump_json(list(self._index.values()), indent=2).decode("utf-8"),
            encoding="utf-8",
        )
        tmp.replace(self._index_file)

    def upsert_index(self, meta: SessionMeta) -> None:
        self._index[meta.id] = meta
        self.save_index()

    def remove_from_index(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._index.clear()
        else:
            self._index.pop(session_id, None)
        self.save_index()


class SessionManager:
    """会话管理"""

    def __init__(self, work_path: Path):
        self.path = Path(work_path)
        self.save_path = self.path / "sessions"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self._session_index = SessionIndex(self.save_path)

    def _session_file(self, session_id: str) -> Path:
        return self.save_path / f"{session_id}.jsonl"

    def new_session(self, title: str = "untitled") -> str:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        session_meta = SessionMeta(id=session_id, title=title)
        self._session_index.upsert_index(session_meta)
        return session_id

    def save_session(self, session_id: str, messages: list[AgentMessage]) -> None:
        """追加保存消息，并同步更新索引元数据。"""
        session_meta = self._session_index.get_index(session_id)
        if session_meta is None:
            raise RuntimeError(f"找不到会话{session_id}")

        if not messages:
            return

        file = self._session_file(session_id)
        with file.open("a", encoding="utf-8") as f:  # 追加模式
            for m in messages:
                f.write(json.dumps(to_jsonable_python(m), ensure_ascii=False, indent=4))
                f.write("\n")

        session_meta.updated_at = datetime.now()
        self._session_index.upsert_index(session_meta)

    def delete_sessions(self, session_id: str | None = None) -> None:
        """删除指定会话；不传参数时删除全部会话。"""
        session_meta = self._session_index.get_index(session_id)
        if session_meta is None:
            raise RuntimeError(f"找不到会话{session_id}")

        file = self._session_file(session_id)
        if file.exists():
            file.unlink()
        self._session_index.remove_from_index(session_id)
