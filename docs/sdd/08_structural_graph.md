# SDD 08 — Structural plane and dual-layer graph schema

## Purpose
Derive the deterministic structural relations between insights and define the full node/edge schema of the dual-layer graph.

## Scope
`ltir/structural.py::structural_edges(insights, config)` (structural plane) and `ltir/graph.py::build_snapshot` (schema assembly; persistence in SDD 09).

## Inputs
`list[Insight]` (only `conditions`, `shifts`, `dataset_id`, `target`, `support`, `effect_size` are read) and `Config`.

## Outputs
`list[GraphEdge]` of types SPECIALIZES, GENERALIZES, SIBLING, CONTRASTS.

## Dependencies
None beyond `ltir.models`. No embeddings, no numpy (enforced by test).

## Algorithms (per dataset; scopes of different datasets are incomparable)
Let C(P) be the condition set of pattern P.
| Relation | Rule | Direction / weight | Properties |
|---|---|---|---|
| SPECIALIZES | C(A) ⊂ C(B) strictly, and **no present pattern M with C(A) ⊂ C(M) ⊂ C(B)** (covering relation = Hasse diagram of the present patterns) | B → A, 1.0 | `added_conditions`, `support_ratio`, `metric` (parent target), `parent_z`, `child_z` |
| GENERALIZES | exact inverse of every SPECIALIZES edge | A → B, 1.0 | same |
| SIBLING | \|C(A)\| = \|C(B)\|, C(A)∖C(B) = {(x,a)}, C(B)∖C(A) = {(x,b)}: same parent scope, different value of one partition attribute | min id → max id (undirected semantics), 1.0 | `parent_scope`, `partition_attribute`, `values` |
| CONTRASTS | overlap = \|C(A) ∩ C(B)\| / min(\|C(A)\|, \|C(B)\|) ≥ `CONTRAST_MIN_OVERLAP`, and some metric in both shift profiles with opposite signs and min(\|z_A\|, \|z_B\|) ≥ `CONTRAST_MIN_SHIFT` (the strongest such metric is recorded) | min id → max id, weight = overlap | `metric`, `z_source`, `z_target`, `scope_overlap`, `relation` ∈ {`specialization_reversal` (a SPECIALIZES pair), `sibling` (a SIBLING pair: one differing value of the same attribute), `overlap`} |

Conditions are closed intents (SDD 03), so the covering relation is the Hasse diagram of the concept lattice restricted to the present patterns: a strictly smaller extent always has a strictly larger intent, and SPECIALIZES follows extent containment exactly. The EDA enumerates 2- and 3-conjunctions, but implied conditions can make intents longer.

## Graph schema (snapshot and Neo4j)
| Node | Id | Key properties |
|---|---|---|
| Pattern | `P-<hash>` | the journal `Insight` record (`weight`, `conditions` as `{attribute, value}`, `shifts` as `Shift`, plus `canonical`, `embedding`, `row_id`). No projected aliases (`insight_weight`, `eda_expression`, stringified conditions). |
| Attractor | `A-<int>` | see SDD 07 (label, mass, evidence_mass, signature, dispersion, centroid metadata, last_updated_batch) |
| Dimension | `D:<dataset>:<name>` | name, cardinality, entropy (EDA-selected dimensions plus every attribute used by a closed intent) |
| Metric | `M:<dataset>:<name>` | name, global_median, global_mad |
| Dataset | `DS:<dataset_id>` | filename, rows, columns |
| Batch | `B:<batch_id>` | batch_seq, created_at, status |

| Edge | From → To | Plane | Weight |
|---|---|---|---|
| SPECIALIZES / GENERALIZES / SIBLING / CONTRASTS | Pattern → Pattern | structural | 1.0 / 1.0 / 1.0 / overlap |
| ACTIVATES | Pattern → Attractor | bridge | alignment with the current centroid; `strength`, `insight_weight`, `engine_weight`, `alignment_at_ingest`, `source` |
| RELATED_TO | Attractor → Attractor (min → max id) | latent | mutual-kNN cosine (lac `calculate_knn_topology`) |
| HAS_SCOPE | Pattern → Dimension | schema | 1.0, `value` |
| TARGETS | Pattern → Metric | schema | min(1, \|z\|/3); `role` primary/secondary, `z`, medians (target plus secondary shifts ≥ `MIN_COMPONENT_Z`) |
| DISCOVERED_IN | Pattern → Batch | provenance | 1.0 |
| OF_DATASET | Batch → Dataset | provenance | 1.0 |

## Configuration
`CONTRAST_MIN_OVERLAP` (0.5), `CONTRAST_MIN_SHIFT` (0.5), `MIN_COMPONENT_Z` (TARGETS), `RELATED_TO_PEER_COUNT` / `RELATED_TO_MIN_WEIGHT` (SDD 07).

## Failure modes
None (pure functions).

## Invariants
GENERALIZES = inverse(SPECIALIZES); no transitive SPECIALIZES edges; structural edges only connect patterns of the same dataset; RELATED_TO only connects attractors; the latent plane stays sparse (≤ `RELATED_TO_PEER_COUNT`·#attractors/2 edges).

## Testing requirements
`tests/test_structural.py` (covering relation, inverse, sibling rule, contrast rule and relation type, no embedding dependency); `tests/test_persistence.py::test_graph_consistency` (typed endpoints, activation coverage, inverse pairs, sparsity).

## Integration points
Traversal (SDD 10) follows SPECIALIZES/GENERALIZES/CONTRASTS by default (`TRAVERSAL_STRUCTURAL_EDGES`); SIBLING is kept for exploration and for the structural baseline. The UI draws GENERALIZES only on demand (it is the inverse of SPECIALIZES).

## Current implementation status
Implemented. Synthetic demo: SPECIALIZES 19, GENERALIZES 19, SIBLING 25, CONTRASTS 8 (including `EU∧laptops` ↔ `EU∧laptops∧retail`, a specialisation reversal of margin).
