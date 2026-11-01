"""Token 用量明细：点击统计栏时在其上方悬浮展开（不占用消息区布局）。"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.geometry import Region
from textual.screen import ModalScreen
from textual.widgets import Static

from tui.usage import UsageStats


class UsagePanel(Vertical):
    """当前会话的 token 明细内容，由弹层承载。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stats = UsageStats()

    def compose(self) -> ComposeResult:
        with Vertical(classes="usage-col"):
            with Horizontal(classes="usage-row"):
                yield Static("总 Token", classes="usage-label")
                yield Static("0", classes="usage-value", id="usage-total")
            with Horizontal(classes="usage-row"):
                yield Static("缓存读取", classes="usage-label")
                yield Static("0", classes="usage-value", id="usage-cached")
            with Horizontal(classes="usage-row"):
                yield Static("缓存命中率", classes="usage-label")
                yield Static("0.0%", classes="usage-value", id="usage-rate")
            with Horizontal(classes="usage-row"):
                yield Static("未缓存输入", classes="usage-label")
                yield Static("0", classes="usage-value", id="usage-uncached")
            with Horizontal(classes="usage-row"):
                yield Static("输出", classes="usage-label")
                yield Static("0", classes="usage-value", id="usage-output")

    def update_stats(self, stats: UsageStats) -> None:
        self._stats = stats
        if not self.is_mounted:
            return

        self.query_one("#usage-total", Static).update(f"{stats.total_tokens:,}")
        self.query_one("#usage-rate", Static).update(f"{stats.cache_hit_rate * 100:.1f}%")
        self.query_one("#usage-cached", Static).update(f"{stats.cached_tokens:,}")
        self.query_one("#usage-uncached", Static).update(f"{stats.uncached_prompt_tokens:,}")
        self.query_one("#usage-output", Static).update(f"{stats.completion_tokens:,}")


class UsagePopup(ModalScreen):
    """悬浮在统计栏上方的 token 明细，不改变消息区布局。"""

    BINDINGS = [("escape", "dismiss_popup", "关闭")]

    def __init__(
        self,
        stats: UsageStats,
        anchor: Region,
        bounds: Region | None = None,
    ) -> None:
        super().__init__()
        self._stats = stats
        self._anchor = anchor
        self._bounds = bounds or anchor

    def compose(self) -> ComposeResult:
        yield UsagePanel(id="usage-panel")

    def on_mount(self) -> None:
        self._place()

    def update_stats(self, stats: UsageStats) -> None:
        self._stats = stats
        self.query_one(UsagePanel).update_stats(stats)

    def _place(self) -> None:
        """悬浮在统计栏正上方（会盖住输入框），水平居中于聊天区。"""
        panel = self.query_one(UsagePanel)
        panel.update_stats(self._stats)

        screen_width, screen_height = self.screen.size
        width = max(40, min(64, self._bounds.width, screen_width - 2))

        # 以点击的 token 汇总为中心水平居中，并夹在聊天区/屏幕内
        x = self._anchor.x + (self._anchor.width - width) // 2
        x = max(self._bounds.x, min(x, self._bounds.right - width))
        x = max(1, min(x, screen_width - width))

        # dock: bottom + margin-bottom：面板底边贴在统计栏上沿，
        # 不依赖面板自身高度，内容行数变化也能正确定位。
        bottom_offset = max(0, screen_height - self._anchor.y)
        panel.styles.width = width
        panel.styles.margin = (0, 0, bottom_offset, x)

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:
            self.dismiss()

    def action_dismiss_popup(self) -> None:
        self.dismiss()
