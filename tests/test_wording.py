"""The Ukrainian wording (insight_contracts/wording.py, docs/08_interface.md §8.5): formulas as the code computes them,
the reading of an insight, a theme and a region, plain words only, and the console's copy of the terms."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import toy_insight

from insight_contracts import PhenomenonThresholds, Shift
from insight_contracts import wording as w

ROOT = Path(__file__).resolve().parents[1]
# analyst words a map-and-chat reader should not meet, and causal words the statistics do not support
BANNED = ("латентн", "атрактор", "центроїд", "трансверсальн", "ребр", "вузл", "вузол", "seed", "якір", "якор", "вектор", "косинус", "близьк")
CAUSAL = ("спричин", "через те що", "доводить", "доведено")


def _texts() -> list[tuple[str, str]]:
    out = [(k, t) for k, pair in w.TERMS.items() for t in pair]
    return out + [(k, t) for table in (w.LISA, w.QUARTERS) for k, t in table.items()]


@pytest.mark.parametrize(
    ("z", "words"),
    [
        (0.6, "слабко вище"),
        (-0.6, "слабко нижче"),
        (1.5, "помірно вище"),
        (-1.5, "помірно нижче"),
        (2.21, "помітно вище"),
        (-2.5, "помітно нижче"),
        (3.0, "різко вище"),
        (-9.0, "різко нижче"),
    ],
)
def test_a_shift_reads_by_the_buckets_of_magnitude_word(z, words):
    assert w.shift_words(z) == words


@pytest.mark.parametrize(
    ("before", "after", "word"), [(0.88, 0.73, "слабшає"), (0.88, 0.95, "сильнішає"), (-0.20, 0.90, "змінює знак"), (0.88, -0.05, "слабшає")]
)
def test_a_correlation_change_reads_by_the_predicates_of_covariance_change(before, after, word):
    """A reversal needs opposite signs and |r| > 0.1 on both sides: +0.88 → −0.05 only weakens."""
    text = w.covariance_words({"pair": ["b", "a"], "global_corr": before, "local_corr": after}, {}, geo=False)
    assert text == f"зв'язок між a і b {word}: кореляція {before:+.2f} загалом → {after:+.2f} тут"


def test_the_formulas_are_the_ones_the_code_computes():
    """sd carries the 0.6745 scale and the cap (the demo's 19.19 vs 10.74 with MAD 2.58 is +2.21, not 3.28); stability
    is 1 − min(CV, 0.9), never a share of resamples; n_eff counts the cells with a value of the metric."""
    assert round(0.6745 * (19.19 - 10.74) / 2.58, 2) == 2.21
    assert "0,6745" in w.TERMS["sd"][1] and "не більше 10" in w.TERMS["sd"][1]
    assert "1 − min(CV, 0,9)" in w.TERMS["stability"][1]
    assert "непорожнім значенням метрики" in w.TERMS["n_eff"][1]
    assert "згладжено" in w.TERMS["window_lift"][1] and "згладж" not in w.TERMS["place_lift"][1]


def test_a_reading_names_the_target_and_the_next_material_shift():
    ins = toy_insight(
        [("category", "phones"), ("region", "US")],
        [Shift("discount", 2.21, 19.19, 10.74, 2.58), Shift("margin", -1.10, 14.75, 19.45, 2.89), Shift("delivery_days", 0.3, 3.5, 3.4, 0.8)],
    )
    assert w.reading(ins, PhenomenonThresholds()) == (
        "Де category = phones і region = US — discount: помітно вище, медіана 19.19 проти 10.74 загалом (+2.21 sd); "
        "margin: помірно нижче, медіана 14.75 проти 19.45 (-1.10 sd)."
    )
    declared = {"category": "категорія товару", "region=US": "замовлення у США"}
    assert w.reading(ins, PhenomenonThresholds(), declared).startswith("Де категорія товару = phones і замовлення у США — discount")


def test_a_geo_reading_uses_ukrainian_cell_words_and_the_declared_source_column():
    ins = toy_insight(
        [("lisa_points", "hot spot"), ("neighbours_points_band", "q4"), ("income_band", "q2")],
        [Shift("share_vehicle_type_bicycle", 1.6, 0.4, 0.1, 0.2), Shift("points", 0.9, 30, 12, 9)],
        provenance={"spatial": {"resolution": 7}},
    )
    text = w.reading(ins, PhenomenonThresholds(), {"vehicle_type": "вид транспорту"}, {"income_band": 5})
    assert text.startswith(
        "Де гаряча точка (багато подій тут і поруч) і сусідні комірки серед найжвавіших (верхня чверть) і "
        "income у другій з 5 смуг (від найменших значень) — частка подій комірки, де вид транспорту — bicycle: помірно вище"
    )
    assert "кількість подій у комірці: слабко вище" in text
    four = toy_insight([("income_band", "q3")], [Shift("points", 0.9, 30, 12, 9)])
    assert "income у третій чверті" in w.reading(four, PhenomenonThresholds(), bands={"income_band": 4})
    assert "points: слабко вище" in w.reading(four, PhenomenonThresholds())  # a plain table's column is no derived cell column


def test_a_covariance_insight_reads_as_its_correlation_change_and_an_unknown_kind_by_its_shift():
    cov = toy_insight(
        [("region", "EU")],
        [Shift("margin", 0.2, 19.0, 19.4, 2.9)],
        phenomenon_type="covariance",
        covariance={"pair": ["discount", "margin"], "global_corr": -0.54, "local_corr": 0.30},
    )
    assert (
        w.reading(cov, PhenomenonThresholds()) == "Де region = EU — зв'язок між discount і margin змінює знак: кореляція -0.54 загалом → +0.30 тут."
    )
    odd = toy_insight([("region", "EU")], [Shift("margin", 1.2, 1, 0, 1)], phenomenon_type="trend")
    assert w.reading(odd, PhenomenonThresholds()) == "Де region = EU — margin: помірно вище, медіана 1 проти 0 загалом (+1.20 sd)."
    pairless = toy_insight([("region", "EU")], [Shift("margin", 0.2, 1, 0, 1)], phenomenon_type="covariance")
    assert w.reading(pairless, PhenomenonThresholds()).startswith("Де region = EU — margin: слабко вище")


def test_a_theme_and_a_region_read_in_plain_words():
    signature = [{"component": "discount", "value": 1.0}, {"component": "margin", "value": -0.49}, {"component": "x", "value": 0.1}]
    assert w.theme_reading(signature, 9, 9) == "Повторювана картина: discount вище, margin нижче — 9 інсайтів у 9 частинах даних."
    assert w.theme_reading([{"component": "correlation between a and b", "value": -0.3}], 1, 1) == (
        "Повторювана картина: зв'язок a і b слабшає — 1 інсайт у 1 частині даних."
    )
    picture = {"patterns": 3, "datasets": 1, "themes": 2}
    assert w.region_reading("R:5:85283473fffffff", picture) == "Регіон 85283473fffffff (рівень 5): 3 інсайти з 1 датасету, 2 теми."


def test_lifts_say_which_one_they_are():
    assert w.lift_words("window", 1.84) == "у 1.8 раза частіше, ніж очікувано за часткою на всій мапі (згладжено)"
    assert w.lift_words("place", 1.02) == "як на всій мапі"
    assert w.lift_words("place", 0.5) == "у 2.0 раза рідше, ніж очікувано за часткою на всій мапі"


def test_relations_label_the_other_end():
    """SPECIALIZES points from the narrower insight to the broader one: from the broader, the other end is narrower."""
    assert w.name("SPECIALIZES") == "вужчий за" and "ширш" not in w.name("SPECIALIZES (in)")
    assert w.name("PARENT_OF") == "ширший регіон" and w.name("PARENT_OF (in)") == "вужчий регіон"
    assert w.name("NEW_EDGE (in)") == "NEW_EDGE (in)"  # an unknown term reads as its key


def test_every_text_is_plain_and_claims_no_cause():
    for key, text in _texts():
        low = text.lower()
        assert not [b for b in BANNED + CAUSAL if b in low], (key, text)
        assert "значущ" not in low or key == "sac", (key, text)


def test_the_console_copy_and_the_glossary_hold_every_term():
    """``TERMS`` in app.js is this table (JS cannot import it); docs/11 §11.1 names every term."""
    js = (ROOT / "sig_web_console" / "static" / "app.js").read_text(encoding="utf-8")
    copied = json.loads(re.search(r"^const TERMS = (\{.*?^\});", js, re.S | re.M).group(1))
    expected = {k: list(v) for k, v in w.TERMS.items()}
    assert copied == expected, "app.js TERMS differs; paste:\nconst TERMS = " + json.dumps(expected, ensure_ascii=False, indent=0) + ";"
    doc = (ROOT / "docs" / "11_reference.md").read_text(encoding="utf-8")
    glossary = doc[doc.index("## 11.1") : doc.index("## 11.2")]
    assert not [name for name, _ in w.TERMS.values() if name not in glossary]
