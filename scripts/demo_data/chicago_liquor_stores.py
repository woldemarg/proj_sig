"""Chicago package-goods stores (alcohol sold to take away): one row per place, its point. A context layer for the
Chicago crimes, beside ``chicago_bars`` (the same licences' on-premise places)."""

from __future__ import annotations

from functools import partial

from demo_data import chicago_bars as bars

FILES, URLS = bars.FILES, bars.URLS
META = bars.meta(
    "Магазини алкоголю в Чикаго (контекстний шар)",
    "Місця з ліцензією package goods — продаж алкоголю на винос. Контекстний шар до злочинів Чикаго.",
    ("Package Goods",),
)
prepare = partial(bars.places, kinds=("Package Goods",))
