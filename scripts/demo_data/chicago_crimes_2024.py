"""Chicago crimes of 2024, prepared as ``chicago_crimes`` (2025): the cross-type validation's held-out year."""

from __future__ import annotations

from functools import partial

from demo_data import chicago_crimes as crimes

FILES, URLS = crimes.source(2024, "6186b046140b27a823d196d04fd780fb153f2f1431f25317eaaced2f5f20eba6")
META = crimes.meta(2024, "Відкладений рік крос-типової перевірки (знахідки 2025 перевіряються на ньому).")
prepare = partial(crimes.prepare, year=2024)
