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
        (-0.998, "помірно нижче"),  # shown as -1.00 sd
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
    assert text == f"зв'язок метрик a і b {word}: кореляція {before:+.2f} загалом → {after:+.2f} тут"


def test_the_terms_state_the_numbers_the_code_uses():
    """sd is the engine's robust z (the demo's 19.19 vs 10.74 with MAD 2.58 is +2.21, not 3.28) and its cap; stability
    names the bootstrap's resamples; the window's prior, the footprint's share and the volume utility are the code's."""
    from insight_graph_service.core.geo_views import WINDOW_LIFT_PRIOR
    from subgroup_miner.spatial import GEO_FOOTPRINT_SHARE
    from subgroup_miner.vendor.eda.main_upd import BOOTSTRAP_RESAMPLES, ROBUST_Z_CAP, robust_z_score, volume_preference_score

    assert round(robust_z_score(19.19, 10.74, 2.58), 2) == 2.21 and "0,6745 · Δмедіани / MAD" in w.TERMS["sd"][1]
    assert f"не більше {ROBUST_Z_CAP:g}" in w.TERMS["sd"][1]
    assert "1 − min(CV, 0,9)" in w.TERMS["stability"][1] and f"на {BOOTSTRAP_RESAMPLES} бутстреп" in w.TERMS["stability"][1]
    assert f"згладжено {WINDOW_LIFT_PRIOR} подіями" in w.TERMS["window_lift"][1] and "згладж" not in w.TERMS["place_lift"][1]
    assert f"{GEO_FOOTPRINT_SHARE:.0%}".replace("%", " %") in w.TERMS["cells_footprint"][1]
    assert volume_preference_score(1 / 3) > max(volume_preference_score(0.3), volume_preference_score(0.4))  # "найбільша … третину"
    assert "третину" in w.TERMS["volume_utility"][1] and "непорожнім значенням метрики" in w.TERMS["n_eff"][1]


def test_a_reading_names_the_target_and_the_next_material_shift():
    ins = toy_insight(
        [("category", "phones"), ("region", "US")],
        [Shift("discount", 2.21, 19.19, 10.74, 2.58), Shift("margin", -1.10, 14.75, 19.45, 2.89), Shift("delivery_days", 0.3, 3.5, 3.4, 0.8)],
    )
    assert w.sentence(w.reading(ins, PhenomenonThresholds())) == (
        "Де category = phones і region = US — discount: помітно вище, медіана 19.19 проти 10.74 загалом (+2.21 sd); "
        "margin: помірно нижче, медіана 14.75 проти 19.45 (-1.10 sd)"
    )
    declared = {"category": "категорія товару", "region=US": "замовлення у США"}
    assert w.reading(ins, PhenomenonThresholds(), declared)["where"] == "категорія товару = phones і замовлення у США"


def test_a_geo_reading_uses_ukrainian_cell_words_and_the_declared_source_column():
    """A declared phrase stands in the nominative in every cell template, so its case never breaks."""
    ins = toy_insight(
        [("lisa_points", "hot spot"), ("neighbours_points_band", "q4"), ("income_band", "q2")],
        [
            Shift("share_vehicle_type_bicycle", 1.6, 0.4, 0.1, 0.2),
            Shift("neighbours_t_active_days", 0.9, 30, 12, 9),
            Shift("speed_median", 0.6, 3, 2, 1),
        ],
        provenance={"spatial": {"resolution": 7}},
    )
    r = w.reading(ins, PhenomenonThresholds(), {"vehicle_type": "вид транспорту", "speed": "швидкість"}, {"income_band": 5})
    assert r["where"] == (
        "гаряча точка (багато подій тут і поруч) і сусідні комірки серед найжвавіших (верхня чверть) і "
        "income у другій з 5 смуг (від найменших значень)"
    )
    assert r["what"].startswith("частка подій комірки, де вид транспорту — bicycle: помірно вище")
    assert "кількість днів з подіями в комірці (t) (середнє шести сусідніх комірок): слабко вище" in r["what"]
    assert w.column_words("speed_median", {"speed": "швидкість"}, geo=True) == "швидкість (медіана подій комірки)"
    assert w.column_words("points", {}, geo=True) == "кількість подій у комірці"
    four = toy_insight([("income_band", "q3")], [Shift("points", 0.9, 30, 12, 9)])
    assert w.reading(four, PhenomenonThresholds(), bands={"income_band": 4})["where"] == "income у третій чверті (від найменших значень)"
    assert w.reading(four, PhenomenonThresholds())["what"].startswith("points: слабко вище")  # a table's column is no cell column


def test_a_covariance_insight_reads_as_its_correlation_change():
    cov = toy_insight(
        [("region", "EU")],
        [Shift("margin", 0.2, 19.0, 19.4, 2.9)],
        phenomenon_type="covariance",
        covariance={"pair": ["discount", "margin"], "global_corr": -0.54, "local_corr": 0.30},
    )
    assert w.reading(cov, PhenomenonThresholds())["what"] == "зв'язок метрик discount і margin змінює знак: кореляція -0.54 загалом → +0.30 тут"


def test_a_theme_and_a_region_read_in_plain_words():
    signature = [{"component": "discount", "value": 1.0}, {"component": "margin", "value": -0.49}, {"component": "x", "value": 0.1}]
    assert (
        w.theme_reading({"signature": signature, "n_patterns": 9, "distinct_scopes": 9})
        == "Повторювана картина: discount вище, margin нижче — 9 інсайтів у 9 частинах даних."
    )
    assert w.theme_reading({"signature": [{"component": "correlation between a and b", "value": -0.3}], "n_patterns": 1, "distinct_scopes": 1}) == (
        "Повторювана картина: зв'язок метрик a і b слабшає — 1 інсайт у 1 частині даних."
    )
    picture = {"patterns": 3, "datasets": 1, "themes": 2}
    assert w.region_reading(picture) == "3 інсайти з 1 датасету, 2 теми."


def test_lifts_say_which_one_they_are():
    assert w.lift_words("window", 1.84) == "у 1.8 раза частіше, ніж очікувано за часткою на всій мапі (згладжено)"
    assert w.lift_words("place", 2.04) == "у 2 рази частіше, ніж очікувано за часткою на всій мапі"
    assert w.lift_words("place", 1.02) == "як на всій мапі"


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


def test_the_console_copy_is_the_table():
    """``TERMS`` in app.js is this table: JS cannot import it."""
    js = (ROOT / "sig_web_console" / "static" / "app.js").read_text(encoding="utf-8")
    copied = json.loads(re.search(r"^const TERMS = (\{.*?^\});", js, re.S | re.M).group(1))
    expected = {k: list(v) for k, v in w.TERMS.items()}
    assert copied == expected, "app.js TERMS differs; paste:\nconst TERMS = " + json.dumps(expected, ensure_ascii=False, indent=0) + ";"


@pytest.mark.skipif(not (ROOT / "docs").is_dir(), reason="the graph service image holds no docs (.dockerignore)")
def test_the_glossary_names_every_term():
    doc = (ROOT / "docs" / "11_reference.md").read_text(encoding="utf-8")
    glossary = doc[doc.index("## 11.1") : doc.index("## 11.2")]
    assert not [name for name, _ in w.TERMS.values() if name not in glossary]
