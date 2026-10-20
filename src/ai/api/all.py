from collections.abc import Callable
from typing import AsyncIterator

from src.ai.api.openai_completion import stream as openai_completion_stream
from src.ai.model import Model
from src.ai.model import StreamOptions
from src.ai.type import AssistantMessage, Context

type StreamAPI = Callable[[Model, Context, StreamOptions | None], AsyncIterator[AssistantMessage]]

_API_MAP: dict[str, StreamAPI] = {
    "openai-completions": openai_completion_stream
}


def get_stream_by_api(api: str) -> StreamAPI:
    return _API_MAP.get(api)
