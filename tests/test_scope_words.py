"""A place asked in the console's words (the temporal research's L-7: geo scope-only navigation hit the asked scope in the
top 10 for 28–35 % of the SF Fire hold-outs' insights): a cell scope's band values (``q4``, ``none nearby``) are held by
many columns, so a question that names only them binds any column and ties dozens of insights; the drawer's words name
the column, and the exact scope outranks the scopes that only add to it."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_config, make_engine, toy_insight, upload_levels
from test_multilingual_retrieval import pattern, toy_graph

from graph_query_engine.graph import dataset_node_id, declared_words, insight_reading
from graph_query_engine.question import ParsedQuery, parse_query
from graph_query_engine.seeds import score_pattern
from insight_contracts import PhenomenonThresholds, Shift, wording


def geo_pattern(pid, conds):
    """A toy graph's Pattern node of a map level (``pattern`` with a spatial provenance)."""
    node = pattern(pid, conds)
    node["props"]["provenance"]["spatial"] = {"resolution": 9}
    return node


def test_a_question_in_the_drawers_words_names_its_conditions_with_their_columns(engine, points_csv):
    """Each geo insight asked as «Що особливого там, де <its conditions as the drawer reads them>?» parses to its exact
    conditions — the band values bound to their columns, none left to «any column»."""
    upload_levels(engine, points_csv, geo="auto@8,7")
    graph = engine.graph()
    geo = [i for i in graph.insights.values() if i.provenance.get("spatial") and len(i.conditions) >= 2]
    assert geo
    for ins in geo[:12]:
        words = wording.short_conditions(ins, *declared_words(graph, ins.id))
        parsed = parse_query(f"Що особливого там, де {' і '.join(words)}?", graph)
        exact = {(c.attribute, c.value) for c in ins.conditions}
        assert exact <= set(parsed.conditions), (words, parsed.conditions)
        assert not [v for a, v in parsed.conditions if a == "*" and any(v == c.value for c in ins.conditions)]


def test_a_declared_name_of_one_word_binds_its_condition():
    """A pair declared in one word (``region=US`` «США», ``category=phones`` «телефони») is how the drawer and the insight's
    card read it, so *Запитати граф про це* asks in it: the question binds both pairs, though it holds neither value."""
    declared = {"region=US": "США", "category=phones": "телефони"}
    g = toy_graph([{"id": dataset_node_id("ds1"), "kind": "Dataset", "label": "ds1", "props": {"user_glossary": declared}}])
    where = insight_reading(g, "P1", PhenomenonThresholds())["where"]  # the toy graph's P1: region=US, category=phones
    assert where == "США і телефони"
    assert set(parse_query(f"Розкажи про margin там, де {where}", g).conditions) == {("region", "US"), ("category", "phones")}


def test_like_phrases_bind_where_they_are_written_and_a_typed_value_wins():
    """Two neighbourhood bands read alike (*сусідні комірки нижче середини (…)*, their counters apart): each binds on its
    own words, not on a run across the two, so no «нижче» is left to read as a direction; a value typed as stored
    (*district 4*) stays, though another value's phrase (*district 11*) reads the same once its digits are dropped."""
    scope = ["district_mode=district 4", "lisa_points=no cluster", "neighbours_crime_date_active_days_band=q2", "neighbours_points_band=q2"]
    g = toy_graph([geo_pattern("P8", scope), geo_pattern("P9", ["district_mode=district 11", "neighbours_points_band=q3"])])
    parsed = parse_query(f"Розкажи про margin там, де {insight_reading(g, 'P8', PhenomenonThresholds())['where']}", g)
    assert set(parsed.conditions) == {tuple(c.split("=", 1)) for c in scope} and parsed.direction == 0


def test_the_exact_scope_outranks_the_scopes_that_only_add_to_it(tmp_path):
    """A question naming a scope: the insight on exactly that scope scores above one that adds a condition, at equal weight."""
    shift = [Shift("load", 1.0, 2.0, 1.0, 1.0)]
    exact = toy_insight([("band", "q4")], shift, id="P-a")
    wider = toy_insight([("band", "q4"), ("half", "north")], shift, id="P-b")
    query = ParsedQuery(text="Що особливого там, де q4?", conditions=[("*", "q4")])
    config = make_config(tmp_path).query
    a, b = (score_pattern(i, query, np.zeros(1), None, config) for i in (exact, wider))
    assert a.score > b.score and a.matched["scope"] == 1.0 > b.matched["scope"]


def test_a_value_with_punctuation_binds_by_its_words():
    """A stored value holding punctuation (*8x Information & investigation*, *4xx Hazardous condition (no fire)*) binds
    when the question holds its words — «&» and «(no» are no words a question could hold. A value repeating its column's
    word (*ocean proximity = <1H OCEAN*) counts the word once, so the column's words alone do not cover it whole and the
    value the question names (*INLAND*) binds."""
    g = toy_graph([pattern("P9", ["action=8x Information & investigation", "situation=4xx Hazardous condition (no fire)"])])
    parsed = parse_query("Що особливого там, де 8x Information & investigation і 4xx Hazardous condition (no fire)?", g)
    assert set(parsed.conditions) == {("action", "8x Information & investigation"), ("situation", "4xx Hazardous condition (no fire)")}
    g = toy_graph([pattern("P8", ["ocean_proximity=<1H OCEAN"]), pattern("P9", ["ocean_proximity=INLAND", "region=US"])])
    parsed = parse_query("Розкажи про margin там, де ocean proximity = INLAND і region = US", g)
    assert set(parsed.conditions) == {("ocean_proximity", "INLAND"), ("region", "US")}


@pytest.mark.model
def test_a_questions_frame_grounds_no_metric_on_a_geo_upload(cfg, points_csv):
    """«Що особливого там, де …?» — the frame of a scope-only question names no metric. On a geo upload the model put
    *особливого* on ``points`` (0.43 on the SF Fire hold-outs), and that target then outranked the asked scope."""
    engine = make_engine(cfg, embedder=None)
    upload_levels(engine, points_csv, geo="auto@8,7")
    graph, catalog = engine.graph(), engine.committed().catalog
    for framed in ("Що особливого там, де north?", "Що цікавого там, де north?", "What is special where north?", "Anything notable where north?"):
        parsed = parse_query(framed, graph, engine.settings.query, catalog)
        assert not parsed.targets, (framed, parsed.grounding)
