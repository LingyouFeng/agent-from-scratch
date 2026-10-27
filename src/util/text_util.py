import unicodedata

_INVISIBLE_CATS = ("Cc", "Cf", "Cs", "Co", "Cn")


def _is_non_text(text: str) -> bool:
    """text 全由不可见字符（控制/格式/零宽/代理/私用/未分配）组成时返回 True。
    注意：\\n \\r \\t 都属 Cc，也会被判为 True。"""
    if not text:
        return True
    return all(unicodedata.category(ch) in _INVISIBLE_CATS for ch in text)


import re
from rich.text import Text as RichText

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # 保留 \n \t；\r 单独归一


def _clean_text(text: str, max_chars: int = 30_000) -> str:
    """清理不可安全渲染的字符，并截断超长内容。"""
    if not text:
        return ""
    # 1. 换行归一（避免 \r 让 rich 光标乱跳）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2. 去掉 ANSI 转义序列
    text = _ANSI_RE.sub("", text)
    # 3. 去掉控制字符
    text = _CTRL_RE.sub("", text)
    # 4. 非法 surrogate / 编码兜底
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    # 5. 超长截断，保留头尾
    if len(text) > max_chars:
        half = max_chars // 2
        omitted = len(text) - max_chars
        text = (
            text[:half]
            + f"\n\n…（已截断 {omitted} 字符，完整内容请查看工具日志）…\n\n"
            + text[-half:]
        )
    return text


def _safe_body(text: str, empty_hint: str = "（无输出）") -> RichText:
    """返回 Text 对象：Static 拿到 RichRenderable 就不会走 markup 解析。"""
    cleaned = _clean_text(text)
    if not cleaned:
        return RichText(empty_hint, style="dim")
    return RichText(cleaned)