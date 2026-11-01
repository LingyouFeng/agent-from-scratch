import json
import re
from typing import Any

from ai.types import TextContent, ThinkingContent, ImageContent, ToolCall, Message, UserMessage, AssistantMessage, \
    ToolResultMessage

# ---------- 基础估算(同前) ----------
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]")

# 单张图片固定估个值,base64 长度没意义
IMAGE_TOKENS = 1000


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


def _json_tokens(obj: Any) -> int:
    return estimate_text_tokens(json.dumps(obj, ensure_ascii=False, default=str))


# ---------- content 列表 ----------
def _count_content(content: str | list) -> int:
    if isinstance(content, str):
        return estimate_text_tokens(content)

    total = 0
    for part in content:
        if isinstance(part, TextContent):
            total += estimate_text_tokens(part.text)
        elif isinstance(part, ThinkingContent):
            total += estimate_text_tokens(part.thinking)
        elif isinstance(part, ImageContent):
            total += IMAGE_TOKENS
        elif isinstance(part, ToolCall):
            total += estimate_text_tokens(part.name)
            total += _json_tokens(part.arguments)
        else:  # 兜底:未知类型按 JSON 估
            total += _json_tokens(part)
    return total


# ---------- 对外入口 ----------
def estimate_message_tokens(msg: Message) -> int:
    if isinstance(msg, UserMessage):
        return _count_content(msg.content)

    if isinstance(msg, AssistantMessage):
        return _count_content(msg.content)

    if isinstance(msg, ToolResultMessage):
        return _count_content(msg.content) + estimate_text_tokens(msg.tool_name)

    return 0


def estimate_messages_tokens(messages: list[Message], per_message: int = 4) -> int:
    """估算整轮对话 token。per_message 是每条消息的 role/分隔符固定开销。"""
    return sum(estimate_message_tokens(m) + per_message for m in messages)
