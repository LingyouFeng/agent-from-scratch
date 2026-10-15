from collections.abc import Callable
from typing import Iterator

from src.ai.model import StreamOptions
from src.ai.api.openai_completion import stream as openai_completion_stream
from src.ai.model import Model
from src.ai.type import AssistantMessage, Context

_API_MAP: dict[str, Callable[[Model, Context], AssistantMessage]] = {
    "openai-completions": openai_completion_stream
}

def get_stream_by_api(api: str) -> Callable[[Model, Context, StreamOptions | None], AssistantMessage | Iterator[AssistantMessage]] | None:
    return _API_MAP.get(api)