# SDD 11 — Evidence builder

## Purpose
Convert a traversal result into an explicit, structured, citable `Evidence` object. It is the **only** information given to the LLM, and every item traces back to dataset, batch, pattern, statistics and graph path.

## Scope
`ltir/evidence.py`: `EvidenceItem`, `Evidence`, `build_evidence()`, `Evidence.to_prompt()`, `Evidence.summary()`.

## Inputs
`ParsedQuery`, `TraversalResult`, `DualGraph`, `Config`.

## Outputs
```python
Evidence(
    query, parsed,                    # question + ParsedQuery.to_dict()
    seed_patterns=["P1", ...],        # citation keys of seeds
    items=[EvidenceItem(
        key="P3", pattern_id, role,   # seed | structural | transversal
        headline, scope, target, phenomenon_type,
        statistics={support, support_fraction, baseline, local, effect_size, shifts, emm_score,
                    stability, p_value, p_adjusted, weight, drivers, covariance},
        path=[PathStep...], path_text="P-a -ACTIVATES(0.97)-> A-4 -ACTIVATES⁻¹(0.99)-> P-b",
        rationale, attractors=[{attractor, alignment, label}], transversal_only,
        provenance={dataset_id, filename, batch_id, engine, steps, expression, rows_ref, pattern_id, ...})],
    attractors=[{id, label, n_patterns, distinct_scopes, related[{id, weight}], signature}],
    paths=[path_text...], metrics=[{metric, dataset_id, global_median, global_mad}],
    datasets=[{dataset_id, filename, rows, batch_id}], provenance=[{key, ...}], notes=[...])
```

## Algorithms
* Items: the first `EVIDENCE_MAX_PATTERNS` retrieved patterns in rank order (seeds first). Keys are `P1..Pn`.
* Metrics: global baseline (median, MAD) of every metric appearing in item shifts.
* Notes: lists items that share no scope condition with the seeds and were reached through latent anchors (they may still be structurally reachable via sibling hops; `structural_distance` says how far), or reports "no matching pattern".
* `summary()`: the deterministic evidence-only answer — "Observations:" with one cited line per item (top two shifts, support, cross-segment tag) and an interpretation line marked as not generated.
* `to_prompt()` renders fixed sections: QUESTION, PARSED, DATASETS, METRIC BASELINES, LATENT ANCHORS VISITED, EVIDENCE (one block per `[P#]`: scope, support, shifts with medians and robust z, correlation change, stability / adjusted p / weight / confounders, retrieval path), NOTE. The section header labels the items "verified statistical observations". There is no free-form graph dump.

## Configuration
`EVIDENCE_MAX_PATTERNS` (10).

## Failure modes
Empty traversal → an evidence object with no items and a note; QA then answers without the LLM (SDD 12).

## Invariants
Keys are unique and map one-to-one to pattern ids (`key_to_pattern`); every item has dataset, batch, expression and pattern id in `provenance`; all numbers in the prompt come from the graph (the persisted EDA results).

## Testing requirements
`tests/test_traversal.py::test_evidence_object_is_structured_and_traceable`; `tests/test_e2e.py` (provenance on every item; the prompt given to the LLM contains the evidence section and keys).

## Integration points
`qa.answer_question()` passes `to_prompt()` to the LLM and uses `summary()` when there is no LLM answer; the UI renders items, chains and the raw prompt (collapsible).

## Current implementation status
Implemented. A typical prompt is about 6 k characters for 10 items.
