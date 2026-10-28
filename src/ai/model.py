from dataclasses import dataclass
from typing import Literal

Api = Literal[
    "openai-completions",
    "openai-response",
    "openai-codex-responses",
    "anthropic-message",
    "google-generative-ai",
]


KnownProviderId = Literal[
    "openai",
    "deepseek",
    "bytedance",
    "alibaba",
]
ProviderId = KnownProviderId | str


@dataclass
class Model:
    id: str
    name: str
    provider: ProviderId
    api: Api


@dataclass
class Provider:
    id: str
    name: str
    base_url: str
    api_key: str
    models: list[Model]


ThinkingLevel = Literal["none", "off", "low", "medium", "high", "xhigh", "max"]


@dataclass
class StreamOptions:
    stream: bool = True
    thinking_level: ThinkingLevel = "high"
