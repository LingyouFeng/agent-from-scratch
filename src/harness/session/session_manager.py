import json
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from src.agent.constant import Constants
from src.agent.types import AgentMessage


class SessionMeta(BaseModel):
    id: str
    title: str = Field(default="untitled")
    model: str = Field(default=Constants.DEFAULT_MODEL_ID)  # 会话关联的模型 id
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())


# 文件格式：list[SessionMeta]
_SessionList = TypeAdapter(list[SessionMeta])
_AgentMessageAdapter = TypeAdapter(AgentMessage)


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
            _SessionList.dump_json(list(self._index.values()), indent=2).decode(
                "utf-8"
            ),
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


class SessionItem(BaseModel):
    id: str
    title: str = Field(default="untitled")
    preview: str = Field(default="")
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())


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
        session_id = (
                datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
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
                f.write(json.dumps(to_jsonable_python(m), ensure_ascii=False))
                f.write("\n")

        session_meta.updated_at = datetime.now()
        self._session_index.upsert_index(session_meta)

    def update_title_if_untitled(self, session_id: str, title: str) -> None:
        """会话标题还是默认值时，用首次用户消息更新它。"""
        meta = self._session_index.get_index(session_id)
        if meta is None or meta.title != "untitled":
            return
        meta.title = title
        self._session_index.upsert_index(meta)

    def get_index(self, session_id: str) -> SessionMeta | None:
        """获取会话元数据。"""
        return self._session_index.get_index(session_id)

    def update_model(self, session_id: str, model_id: str) -> None:
        """更新会话关联的模型 id。"""
        meta = self._session_index.get_index(session_id)
        if meta is None:
            raise RuntimeError(f"找不到会话{session_id}")
        meta.model = model_id
        self._session_index.upsert_index(meta)

    def delete_sessions(self, session_id: str | None = None) -> None:
        """删除指定会话；不传参数时删除全部会话。"""
        session_meta = self._session_index.get_index(session_id)
        if session_meta is None:
            raise RuntimeError(f"找不到会话{session_id}")

        file = self._session_file(session_id)
        if file.exists():
            file.unlink()
        self._session_index.remove_from_index(session_id)

    def get_session_list(self) -> list[SessionItem]:
        """获取所有会话展示数据"""
        return [
            SessionItem(
                id=meta.id,
                title=meta.title,
                preview=self._get_preview(meta.id),
                created_at=meta.created_at,
                updated_at=meta.updated_at,
            )
            for meta in self._session_index.load_index().values()
        ]

    def _get_preview(self, session_id: str, max_len: int = 50) -> str:
        file = self._session_file(session_id)
        if not file.exists():
            return ""
        try:
            with file.open("r", encoding="utf-8") as f:
                line = f.readline()
        except OSError:
            return ""
        if not line.strip():
            return ""
        try:
            msg = _AgentMessageAdapter.validate_json(line)
        except ValidationError:
            return ""

        if msg.role != "user":
            return ""
        if isinstance(msg.content, str):
            text = msg.content
        else:
            text = msg.content[0].text
        return text[:max_len] + "..." if len(text) > max_len else text

    def load_messages(self, session_id: str) -> list[AgentMessage]:
        """逐行反序列化。坏行跳过而不中断整批。"""
        file = self._session_file(session_id)
        if not file.exists():
            return []
        result: list[AgentMessage] = []
        try:
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result.append(_AgentMessageAdapter.validate_json(line))
                    except ValidationError:
                        continue
        except OSError:
            return []
        return result
