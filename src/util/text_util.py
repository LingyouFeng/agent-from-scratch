import unicodedata

_INVISIBLE_CATS = ("Cc", "Cf", "Cs", "Co", "Cn")

def _is_non_text(text: str) -> bool:
    """text 全由不可见字符（控制/格式/零宽/代理/私用/未分配）组成时返回 True。
    注意：\\n \\r \\t 都属 Cc，也会被判为 True。"""
    if not text:
        return True
    return all(unicodedata.category(ch) in _INVISIBLE_CATS for ch in text)