# SDD 15 — Testing strategy, demo data, hypothesis experiment

## Purpose
Guard the interfaces whose breakage would invalidate the system, give a deterministic dataset with known phenomena, and turn the research hypothesis into a measurement.

## Scope
`sig/tests/`, `ltir/synth.py`, `ltir/experiment.py`, `pytest.ini`.

## Test map (58 tests)
| Transition / contract | File | What is asserted |
|---|---|---|
| discovery → canonical insight | `test_discovery_contract.py` | profile contract (identifier dropped, float targets kept, EDA dimensions, candidates ≤ search space); fields/provenance; signed directions of planted effects; determinism; planted phenomena survive selection; failure codes |
| selection + weight | `test_quality.py` | weight bounds, monotonicity, exact formula; R1–R4 reason codes; R5 closed pattern; R6 near duplicate; covariance retargeting (unmeasured factors are `None`) |
| canonical → embedding | `test_canonical_embedding.py` | six separate sections; signed components; shape/dtype/unit norms/stability for **hashing and the real model**; opposite shift → negative cosine; disjoint-scope same phenomenon → high cosine; fingerprint sensitivity |
| insight → ontology | `test_ontology.py` | cold start coverage and one attractor per planted cluster; assignment; orphans → OMP → new concept; single-orphan nearest fallback; soft merge; weight scales the EMA pull; sign repair; state round-trip |
| structural relations | `test_structural.py` | SPECIALIZES covering-only, GENERALIZES inverse, SIBLING, CONTRASTS (+ relation type; `sibling` only for a real sibling pair), no embedding dependency |
| persistence | `test_persistence.py` | write → reload → rebuild identical; idempotent re-ingest; graph consistency; rollback; representation mismatch; crash recovery; ingestion failures; column options strict when explicit / lenient as defaults; MERGE-only Neo4j publish |
| traversal | `test_traversal.py` | toy graph: exact Pattern→Attractor→Attractor→Pattern path, hops, weights, reversal, score; thresholds; hop/depth budgets; parsing; evidence object |
| LLM | `test_llm_client.py` | real HTTP client vs OpenAI-compatible stub (incl. OpenRouter provider pinning); fail fast; citation validation incl. grouped `[P1, P3]` |
| end-to-end | `test_e2e.py` | upload → … → grounded answer with provenance and cross-scope analogues (hashing + real model); hypothesis apparatus regression; LLM failure fallback; empty graph |
| 3D sphere | `test_sphere.py` | legend group per anchor, points inside the unit sphere, highlight layers, export, API + vendored plotly |
| self-containment | `test_self_contained.py` | no module / sys.path entry / config path / source string resolves into sibling `eda/` or `lac/`; model loads from `sig/models` |
| UI | `test_ui_smoke.py` | API upload → READY → graph → node → query highlight ids; headless Chromium: dataset card, insights table, chat answer with highlight and evidence cards, citation → drawer, theme toggle, no page errors (`browser`) |

Isolation: `conftest.py` sets `LTIR_NO_DOTENV=1` and `neo4j_enabled=False`, so developer credentials (OpenRouter key, Neo4j) never reach tests. Markers: `model` (auto-skipped when the embedding cache is absent) and `browser` (skipped without Playwright/Chromium). Run with `python -m pytest` from `sig/` (≈ 90 s with the model and browser tests).

**Quality gate** (`scripts/check.py`, AGENTS.md Rule 0): `ruff check` + `ruff format --check` (configuration in `pyproject.toml`: rules E/F/W/I/UP/B, line length 150, the vendored EDA file exempt from style rules and formatting) + the full pytest run. `--quick` skips the `model` and `browser` tests for mid-change feedback. Every step runs even after a failure; the exit code is non-zero if any step failed.

## Deterministic synthetic dataset (`ltir/synth.py`, `data/demo/retail_synthetic.csv`)
5 000 rows, seed 7. Dimensions: region, category, channel, plus the noise columns payment, weekday and store_size (the EDA's step 2 correctly ignores them); `order_id` is an identifier. Mechanisms: `margin = 26 − 0.6·discount + ε`, `return_rate = 0.02 + 0.01·delivery_days + ε`.
| Planted phenomenon | Scope(s) | Validates |
|---|---|---|
| local anomaly | EU∧laptops: margin +5 | anomaly discovery |
| stronger specialisation | EU∧laptops∧online: +4 more | SPECIALIZES with a larger effect |
| contrasting subgroup | EU∧laptops∧retail: net −4 | CONTRASTS (specialisation reversal) |
| recurring phenomenon (discount erosion) | US∧phones, APAC∧tablets, EU∧tablets∧retail: discount +9 | one attractor over structurally disjoint scopes |
| recurring phenomenon (delay → returns) | APAC∧online, US∧laptops∧retail: delivery +3 | a second cross-scope attractor |
| correlation break | EU∧phones: margin decoupled from discount | EMM / covariance insight |

`GROUND_TRUTH` in `synth.py` lists these scopes for tests.

## Hypothesis experiment (`ltir/experiment.py`, `python -m ltir experiment --k K`)
Labels come from the **planted ground truth**: a pattern belongs to a mechanism when its scope contains one of that mechanism's planted scopes (`GROUND_TRUTH`, scope containment; ambiguous patterns are skipped). Labels are deliberately *not* derived from the measured shifts: the shifts are what the representation encodes, so shift-based labels would favour vector retrieval by construction (the earlier rule-based labelling had this circularity). For each labelled pattern S, the **analogues** are the other patterns with the same mechanism and no shared scope condition; only the two multi-scope mechanisms (discount erosion, delay → returns) yield analogues. Four rankers are scored on recall@k, precision@k (hits / min(k, |analogues|), so a case with fewer analogues than k can reach 1.0) and MRR: `transversal` (this system), `structural` (BFS over all structural edges), `text_nn` (canonical-document text embeddings: naive vector RAG), `vector_nn` (LTIR insight vectors without the attractor graph).

Result on the demo graph (12 seed cases: 8 discount erosion, 4 delay → returns; real embedding model; 28 patterns):
| method | recall@3 | precision@3 | MRR | recall@5 |
|---|---|---|---|---|
| transversal | **0.333** | **0.361** | **0.567** | **0.729** |
| structural | 0.000 | 0.000 | 0.079 | 0.000 |
| text_nn | 0.083 | 0.083 | 0.225 | 0.271 |
| vector_nn | 0.028 | 0.028 | 0.220 | 0.465 |

Reading: structural traversal never reaches scope-disjoint analogues (they sit 2–4 lattice hops away, mixed with everything else). Naive text-NN and raw insight-vector kNN rank same-scope-family patterns (specialisations of the seed) first. The attractor layer puts analogues at the top (MRR 0.57 vs ≤ 0.23) and reaches most of them by k = 5 (0.73 vs 0.47). This is an apparatus, not proof: one synthetic dataset, two mechanisms with analogues. `test_hypothesis_apparatus` guards the ordering as a regression.

## Testing requirements (for future changes)
A change to a contract (fields, formula, schema, path grammar) must update the SDD and the corresponding test in the same change. A representation change must bump `REPRESENTATION_VERSION` or `CANONICAL_VERSION`.

## Current implementation status
58 passed, 0 skipped on the development machine; `python scripts/check.py` (ruff + format + pytest) passes. A separate audit-hook run (full ingest of both datasets, a query and the sphere) opened 0 files under `../eda` or `../lac`.
