"""A place asked in the console's words (the temporal research's L-7: geo scope-only navigation hit the asked scope in the
top 10 for 28–35 % of the SF Fire hold-outs' insights): a cell scope's band values (``q4``, ``none nearby``) are held by
many columns, so a question that names only them binds any column and ties dozens of insights; the drawer's words name
the column, and the exact scope outranks the scopes that only add to it."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_config, make_engine, toy_insight, upload_levels

from graph_query_engine.graph import declared_words
from graph_query_engine.question import ParsedQuery, parse_query
from graph_query_engine.seeds import score_pattern
from insight_contracts import Shift, wording


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
    when the question holds its words — «&» and «(no» are no words a question could hold."""
    from test_multilingual_retrieval import pattern, toy_graph

    g = toy_graph([pattern("P9", ["action=8x Information & investigation", "situation=4xx Hazardous condition (no fire)"])])
    parsed = parse_query("Що особливого там, де 8x Information & investigation і 4xx Hazardous condition (no fire)?", g)
    assert set(parsed.conditions) == {("action", "8x Information & investigation"), ("situation", "4xx Hazardous condition (no fire)")}


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
