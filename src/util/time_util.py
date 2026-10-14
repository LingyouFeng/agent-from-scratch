import time


def get_timestamp_ms() -> int:
    """获取当前 Unix 毫秒级时间戳"""
    return int(time.time() * 1000)
