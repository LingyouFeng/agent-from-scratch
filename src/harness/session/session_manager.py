import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from agent.constant import Constants
from agent.types import AgentMessage
from ai.types import Usage


class SessionMeta(BaseModel):
    id: str
    title: str = Field(default="untitled")
    model: str = Field(default=Constants.DEFAULT_MODEL_ID)
    total_turns: int = Field(default=0)  # 用户发起提问的轮数
    total_steps: int = Field(default=0)  # 调用大模型的总次数
    total_tokens: int = Field(default=0)  # 会话消费的总token（输入和输出）
    prompt_tokens: int = Field(default=0)  # 用户 prompt 所包含的 token 数
    completion_tokens: int = Field(default=0)  # 模型 completion 产生的 token 数
    context_tokens: int = Field(default=0)  # 最近一次请求的上下文长度
    cache_hit_ratio: float = Field(default=0.0)  # 缓存命中率
    prompt_cache_hit_tokens: int = Field(default=0)  # 用户 prompt 中，命中上下文缓存的 token 数
    prompt_cache_miss_tokens: int = Field(default=0)  # 用户 prompt 中，未命中上下文缓存的 token 数
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
        # 落盘挪到线程后，多会话/标题任务可能并发写盘：锁保护 tmp 写入与替换
        self._save_lock = threading.Lock()

    def get_index(self, session_id: str) -> SessionMeta:
        return self._index.get(session_id)

    def list_index(self) -> list[SessionMeta]:
        """所有会话元数据（直接读内存索引，不读盘）。"""
        return list(self._index.values())

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
        with self._save_lock:
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
        """获取所有会话展示数据（内存索引，不再每次读盘解析 index.json）"""
        return [
            SessionItem(
                id=meta.id,
                title=meta.title,
                preview=self._get_preview(meta.id),
                created_at=meta.created_at,
                updated_at=meta.updated_at,
            )
            for meta in self._session_index.list_index()
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

    def add_turn(self, session_id: str) -> None:
        meta = self._session_index.get_index(session_id)
        if meta is None:
            raise RuntimeError(f"找不到会话{session_id}")
        meta.total_turns += 1
        meta.updated_at = datetime.now()
        self._session_index.upsert_index(meta)

    def add_usage(self, session_id: str, usage: Usage) -> None:
        """把一次模型调用的用量累计进会话元数据（SessionMeta 是唯一数据源）。"""
        meta = self._session_index.get_index(session_id)
        if meta is None:
            raise RuntimeError(f"找不到会话{session_id}")

        meta.total_steps += 1
        meta.context_tokens = usage.context_tokens
        meta.total_tokens += usage.total_tokens
        meta.prompt_tokens += usage.prompt_tokens
        meta.completion_tokens += usage.completion_tokens
        meta.prompt_cache_hit_tokens += usage.prompt_cache_hit_tokens
        meta.prompt_cache_miss_tokens += max(
            usage.prompt_tokens - usage.prompt_cache_hit_tokens, 0
        )
        meta.cache_hit_ratio = (
            meta.prompt_cache_hit_tokens / meta.prompt_tokens
            if meta.prompt_tokens
            else 0.0
        )
        meta.updated_at = datetime.now()
        self._session_index.upsert_index(meta)
