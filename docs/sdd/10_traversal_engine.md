# SDD 10 — Query interpretation and transversal traversal

## Purpose
Resolve a question to seed patterns, then traverse the dual-layer graph across structural and latent relations. Return retrieved patterns **with the paths used to reach them**, and inspectable baselines for the research hypothesis.

## Scope
`ltir/query.py` (`parse_query`, `score_pattern`, `resolve_seeds`), `ltir/traversal.py` (`traverse`, `structural_closure`, `naive_nearest`).

## Inputs
Question text, `DualGraph`, `InsightEncoder`, pattern vectors (`Engine.frame().patterns`: the journal rows, the same frame `encode_query` produces), `Config`.

## Outputs
* `ParsedQuery(text, targets, direction ∈ {−1, 0, +1}, conditions[(attr, value)], covariance)`
* `list[SeedMatch(pattern_id, score, matched{target, scope, direction, semantic, weight, scope_conflicts})]`
* `TraversalResult(seeds, patterns: list[Retrieved], attractors: list[Retrieved], used_edges, traversed_nodes, visited_count, max_depth, baselines)`
* `Retrieved(node_id, kind, score, hop, route ∈ {seed, structural, transversal, latent}, seed_id, path: list[PathStep], structural_distance, scope_overlap, transversal_only)`
* `PathStep(source, target, edge_type, weight, hop, edge_id, reverse)`: source node, target node, edge type, edge weight, hop number.

## Algorithms
**Parsing** (deterministic, graph vocabulary only; no LLM):
* targets: a metric matches when all its name parts match, or ≥ 2 parts match including a non-generic one, or its head part matches and that head is non-generic and unique among the metrics (generic words: median, mean, average, total, count, rate, value, score, …; 5-char stem, e.g. "returns" → `return_rate`, "house values" → `median_house_value`, but "the median" matches nothing); only the best-covered metrics are kept;
* conditions: a value matches as a whole token. Short upper-case values (`US`, `EU`) are case-sensitive, so "tell us" ≠ `US`;
* direction from lexicons (lower/drop/erosion… vs higher/rise/delay…); without a direction word the query has **no** signed phenomenon components (the encoder falls back to the question text for that block) rather than assuming "higher";
* covariance intent (correl/relationship/…): direction is ignored and the phenomenon slot scores covariance insights.

**Seed scoring** per Pattern (`score_pattern`), for lexical queries:
`0.35·target + 0.25·scope + 0.15·direction + 0.15·semantic + 0.10·insight_weight`
* target: 1 (primary), 0.7 (secondary shift ≥ `MIN_COMPONENT_Z`), 0.6 (covariance pair);
* scope: fraction of the requested conditions present, −0.5 per conflicting value on a requested attribute;
* direction: +1 matching sign, −0.5 opposite;
* semantic: cosine to `encode_query(scope, target, signed components, text)` in the insight space.

Non-lexical queries use `0.7·semantic + 0.3·w`. Seeds are the top `SEED_TOP_K` with score ≥ max(`SEED_MIN_SCORE`, `SEED_RELATIVE_MIN`·best), skipping direct lattice neighbours of already chosen seeds (they are reached structurally). Fallback: the best semantic match.

**Transversal traversal**: best-first search over the regular path grammar
```text
P0 (lattice){0,h} --ACTIVATES[>=act]--> A (RELATED_TO[>=rel]){0,L} <--ACTIVATES[>=act]-- P1 (lattice){0,h}
```
with h = `STRUCTURAL_HOPS`, L = `MAX_LATENT_HOPS`, total length ≤ `TRAVERSAL_MAX_DEPTH`. Lattice edges are `TRAVERSAL_STRUCTURAL_EDGES` (SPECIALIZES/GENERALIZES followed along stored out-edges, CONTRASTS both ways). Edge factors: ACTIVATES = alignment, RELATED_TO = weight, lattice = `STRUCTURAL_EDGE_DECAY` (× overlap for CONTRASTS). Path score = seed score × Π factors (seed scores are floored at 1e-3: every factor is ≤ 1, so best-first order needs positive scores). The search state is (node, phase, structural hops used, latent hops used) — Dijkstra over the budgeted grammar, so a higher-scoring arrival with less remaining budget cannot shadow one that can still expand; the best path per node is reported. Node rank = path score × insight_weight; seeds keep their seed score. Seeds are never re-entered in phase P1. Result: seeds + top `MAX_RETRIEVED`; `used_edges` / `traversed_nodes` are the union of the selected paths.

**Research flags**: `scope_overlap` = max Jaccard of conditions with any seed; `transversal_only` = reached through the latent plane **and** scope-disjoint from every seed; `structural_distance` = BFS hops over all structural edges including SIBLING (None = unreachable).

**Baselines** (`qa.py` stores them in `traversal.baselines`): `structural_only` (lattice-closure of the seeds with the same edge set and depth), `naive_nearest` (canonical document text embeddings vs question text), `transversal_only`, `not_in_naive_topk`, `not_structurally_reachable`. `experiment.py` ranks them against planted ground truth (SDD 15).

## Configuration
`SEED_TOP_K` 3, `SEED_MIN_SCORE` 0.25, `SEED_RELATIVE_MIN` 0.75, `ACTIVATION_THRESHOLD` 0.40, `RELATION_THRESHOLD` 0.40, `MAX_LATENT_HOPS` 1, `STRUCTURAL_HOPS` 1, `TRAVERSAL_MAX_DEPTH` 5, `MAX_RETRIEVED` 12, `STRUCTURAL_EDGE_DECAY` 0.85, `TRAVERSAL_STRUCTURAL_EDGES` `SPECIALIZES,GENERALIZES,CONTRASTS`.

## Failure modes
Empty graph → `answer_mode=empty` (SDD 12). No lexical match → semantic seeds. No path above thresholds → only seeds and their lattice neighbours are returned ("empty retrieval" is visible as zero transversal items).

## Invariants
Every retrieved node has a path from a seed whose steps are graph edges (`edge_id`s resolve in the snapshot and the UI); edges below thresholds are never used; paths obey the grammar; results are deterministic.

## Testing requirements
`tests/test_traversal.py` (toy graph): exact path P → A → A → P with hops, weights and reversal; score formula; same-anchor retrieval; lattice expansion after descending; threshold pruning; hop budget and depth limits; parsing (targets, direction, case-sensitive short values). `tests/test_e2e.py`: cross-scope analogues on the real graph.

## Integration points
`qa.answer_question()` → evidence (SDD 11) and UI highlight groups (SDD 13).

## Current implementation status
Implemented. Demo query "Why is margin lower for phones in the US?": 1 seed (`category=phones ∧ region=US`); the path P → `A-4 "discount ↑ · margin ↓"` descends to 4 scope-disjoint tablet patterns (e.g. `category=tablets ∧ region=APAC`, structural distance 2–4), plus a 5th after one lattice hop; then `A-3`/`A-0` via RELATED_TO. Retrieval takes ≈ 20 ms once the model is loaded.
