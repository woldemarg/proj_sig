"""Chicago crimes of 2024, prepared as ``chicago_crimes`` (2025): the cross-type validation's held-out year."""

from __future__ import annotations

from functools import partial

from demo_data import chicago_crimes as crimes

FILES, URLS = crimes.source(2024, "4fd1813b83f520ff44a3c775bfdb3e7227b039870383cabadfc46c3e58bd6141")
META = crimes.meta(2024, "Відкладений рік крос-типової перевірки (знахідки 2025 перевіряються на ньому).")
prepare = partial(crimes.prepare, year=2024)
