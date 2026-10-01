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
        scope_text="category is phones and region is US",          # readable renderings (SDD 05 helpers)
        shift_text=["discount: strong increase, +2.21 sd (median 19.19 vs 10.74 overall)", ...],
        relationship="correlation between ... weakens from +0.88 overall to +0.73 in the subgroup" | "",
        path=[PathStep...], path_text="P-a -ACTIVATES(0.97)-> A-4 <-ACTIVATES(0.99)- P-b",
        attractors=[{attractor, alignment, label}], transversal_only,
        provenance={dataset_id, filename, batch_id, engine, steps, expression, rows_ref, pattern_id, ...})],
    attractors=[{id, label, description, n_patterns, distinct_scopes, related[{id, weight}], signature}],
    paths=[path_text...], metrics=[{metric, dataset_id, global_median, global_mad}],
    datasets=[{dataset_id, filename, rows, batch_id}], provenance=[{key, ...}], notes=[...])
```

## Algorithms
* Items: the first `EVIDENCE_MAX_PATTERNS` retrieved patterns in rank order (seeds first). Keys are `P1..Pn`.
* Rendering happens here, with the config: `shift_text` = the phenomenon shifts (target + \|z\| ≥ `MIN_COMPONENT_Z`; immaterial shifts are not shown to the LLM), `relationship` only when the correlation change is material (`has_material_covariance`), `scope_text` in prose; attractor `description` = `graph.describe_components(signature)` (`discount up and margin down`), falling back to the UI label.
* Metrics: global baseline (median, MAD) of every metric the prompt mentions.
* Notes: lists items that share no scope condition with the seeds and were reached through latent anchors (they may still be structurally reachable via sibling hops; `structural_distance` says how far), or reports "no matching pattern".
* `summary()`: the deterministic evidence-only answer — "Observations:" with one cited line per item (top two shifts, support, cross-segment tag) and an interpretation line marked as not generated.
* `to_prompt()` renders fixed, ASCII-only sections: QUESTION, PARSED, UNITS (shifts are robust standard deviations, "sd"), DATASETS, METRIC BASELINES, LATENT ANCHORS VISITED, EVIDENCE (one block per `[P#]`, the key first: prose scope and support; `shifts:`; `relationship:` when material; `validation:` stability / adjusted p bucket / weight / confounders; `retrieved via:` path), NOTE. Exact template and examples: SDD 17 §9. There is no free-form graph dump.

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
Implemented. Demo question, 10 items: 6,330 characters, 0 non-ASCII, 2,102 tokens (XLM-R SentencePiece) / 2,341 (Qwen BPE) — 13–19 % fewer than the previous symbol-based prompt (SDD 17 §12).
