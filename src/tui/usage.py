"""会话 token 统计的数据模型与格式化工具。"""

from __future__ import annotations

from dataclasses import dataclass

# 上下文窗口的统计基准（产品口径：1M）
CONTEXT_WINDOW = 1_000_000


@dataclass(frozen=True)
class UsageStats:
    """一次会话的用量快照，供底栏与明细弹窗共用。"""

    turns: int = 0
    steps: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    context_tokens: int = 0

    @property
    def uncached_prompt_tokens(self) -> int:
        """未命中缓存的输入 token。"""
        return max(self.prompt_tokens - self.cached_tokens, 0)

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def context_ratio(self) -> float:
        return min(self.context_tokens / CONTEXT_WINDOW, 1.0)


def format_tokens(value: int) -> str:
    """1234 -> 1.2k，1234567 -> 1.2M。"""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)
