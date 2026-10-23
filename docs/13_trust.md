# 13. Trust and integrity

> **In one paragraph.** Every finding SIG hands to a language model carries three separate judgements: how reliable its *source* is (the Admiralty code A–F, declared by whoever uploaded the data), how far *other independent sources* confirm it (credibility 1, 2, 5 or 6, computed from the graph) and how strong its *statistics* are (p, stability, effective n, weight; chapters 2–3). They never merge into one score. Three integrity checks guard the records the graph is compiled from: a digest of every finding's statistics, a digest of what the graph reads of each dataset, and a recomputation of any finding from its stored upload. This chapter is the decision behind that model, a map of where each part is specified, and an audit against the preconditions of a knowledge-graph poisoning attack.

**Code** `insight_graph_service/core/snapshot.py` (`credibility`, `compute_integrity`, `metadata_digest`, `compute_metadata_integrity`), `insight_contracts/insight.py::record_digest`, `insight_graph_service/core/verify.py` (`verify_pattern`, `verify_diff`), `insight_graph_service/core/engine.py` (`update_dataset_metadata`), `subgroup_miner/discovery.py::recompute_statistics`, `graph_query_engine/evidence.py::_add_trust`, `evidence_narrator_service/narration.py` (rule 11), `scripts/poisoning_eval.py` · **Tests** `tests/test_integrity.py`, `tests/test_graph_query_engine_standalone.py`, `tests/test_narrator.py`, `tests/test_e2e.py::test_poisoning_eval_apparatus`, `tests/test_ui_smoke.py` · **Previous** [12. Architecture](12_architecture.md) · **Start** [Reading guide](README.md)

---

## 13.1 The decision

**Context.** A knowledge graph that an LLM reads is a target in itself. Oracle Poisoning (Kereopa-Yorke et al., arXiv 2605.09822) corrupts the graph an agent queries, not its instructions. On a production code graph, every tested model trusted the poisoned data in ≈100 % of the trials (269 of 270 valid ones) under directed queries, once the attacker reached a moderate level of sophistication (L2). Under open-ended prompts trust fell to 3–55 %, so the query framing is a confound and both are reported. Of the defences the paper measures:
- an instruction in the system prompt to verify or distrust the tool's answers had no measurable effect;
- cross-verification against another source lowered blind trust from 100 % to 0–25 % (N = 10, three models);
- a before-snapshot of the graph caught created nodes but not modified properties (trust stayed at 96.6–100 %);
- read-only access removes the direct write path (argued, not measured).

A finding's statistics say nothing about where its data came from: p < 0.001 on planted data is still p < 0.001. Evidence framed as *verified* observations is the framing under which trust is total.

**Decision.**
1. **Three axes, kept apart.** Source reliability is a property of the dataset, set by the uploader. Credibility is a property of the finding, computed from the other sources in the graph. The statistics come from chapters 2–3. There is no combined trust score: one would let strong statistics buy credibility, which is exactly what a poisoned table offers. Credibility never reads p, stability or weight.
2. **Independence is declared, not proven.** Another source is another declared `source` outside the finding's own upload and the version it `supersedes` (or that supersedes it). Two undeclared datasets, or metrics without an indicator registry, never give 1 or 5. The declaration is the uploader's claim; SIG cannot check that two sources do not share an origin.
3. **The Admiralty scale is operationalised, not defined.** STANAG 2511 rates information credibility 1 *confirmed by other sources*, 2 *probably true*, 3 *possibly true*, 4 *doubtful*, 5 *improbable*, 6 *truth cannot be judged*. SIG computes four of these levels by explicit rules ([13.3](#133-credibility-levels)). Its 2 means *not confirmed, though another dataset covers the finding*, not *probably true*. Levels 3 and 4 (*possibly true*, *doubtful*) are reserved for a human rating or an input profile ([#21](https://github.com/woldemarg/proj_sig/issues/21)) and never computed.
4. **Credibility changes as datasets arrive and go.** This is by design: confirmation is a relation between sources, so it is recompiled with every commit. `credibility_basis` says which sources and findings set the level. No history is kept.
5. **Integrity is checked three times** ([13.4](#134-integrity)): a digest of each finding's statistics, a digest of each dataset's metadata, and a recomputation from the stored upload. The digests are unkeyed. They catch an edit that neither recomputes nor removes the digest. The recomputation catches the tested numbers (support, shifts, p) against the raw file; the weight and the scores it does not ([13.4](#134-integrity)).
6. **The signal goes into the data, not only into the instruction.** Each evidence item carries a `trust:` line. The section header says the findings are computed and that trust is given per item. Rule 11 of the system prompt asks the model to keep the three axes apart. The red team measures which of the two moves the answer ([13.7](#137-measured-behaviour)).

**Rejected.**
- **One trust score.** It mixes what the uploader claims, what other sources show and what the test measured, and hides which of them is weak.
- **Prompt hardening alone.** Measured at zero by the paper and at next to nothing here.
- **Keyed signatures** (an HMAC over the records). A local single-user workspace has no place to keep the key apart from the data it signs. Before any deployment beyond localhost, write access comes first ([#10](https://github.com/woldemarg/proj_sig/issues/10)).
- **Graph history snapshots.** They miss modified properties, the attack that matters.

**Consequences and limits.**
- A finding from a single source is credibility 6 and says so; a workspace with one upload and no declared source keeps the evidence exactly as before (the golden prompt).
- A forged record is visible, not repaired: its finding leaves the evidence; its dataset's declarations are ignored.
- A finding can lose its 1 when the confirming dataset is deleted.

**Facts the plan relied on, stated precisely.**
- The 100 % holds for directed queries and an L2 attacker; open queries gave 3–55 %.
- SUFF (spatial uncertainty in fuzzy features, CEUR-WS Vol-3743) is a proposed extension of GeoSPARQL in a workshop paper, not an OGC standard. It bears on positional uncertainty (M6), not on the trust axes.

## 13.2 The three axes

| Axis | Field | Set by | Changes when | Canonical home |
|---|---|---|---|---|
| source reliability | `source` (when declared) and `reliability` (`A`–`F`, `F` by default) on the Dataset node | the uploader: `POST /api/upload` options, `PATCH /api/datasets/{id}` | the uploader changes it (no new batch, the same pattern ids) | [2.1](02_discovery.md#21-ingestion), [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| credibility | `credibility ∈ {1, 2, 5, 6}`, `credibility_basis {rule, sources, patterns, comparability?, confirmed_by?}` on the Pattern node | `snapshot.credibility` at every compile | a dataset arrives, goes or changes its declarations | [6.2](06_graph_and_storage.md#62-the-graph-schema) |
| statistics | `p_adjusted`, `stability`, `weight` in the journal record, and a geo finding's `n_eff` in its spatial provenance | discovery and selection | never after the batch | [2.4](02_discovery.md#24-what-the-adapter-adds), [3.3](03_insights.md#33-insight-weight) |

`integrity` is not a fourth axis but a precondition: a record that fails it is excluded (a finding) or stripped of its declarations (a dataset) before any axis is read.

## 13.3 Credibility levels

The rules, in brief; §6.2 holds them in full, with their limits.

| Level | Phase A (`rule: anchor`; a finding without a place) | Phase B (`rule: place`; a geo finding) |
|---|---|---|
| 1 | another declared source shares a non-weak anchor and shifts a metric of the same indicator the same way | as phase A, where the shared cells reach `GEO_REGION_MIN_SHARE` (0.2) of the smaller place |
| 2 | another dataset, declared or not, shifts a metric of the same name | another dataset has a pattern there with a metric of the same name, or events over that share of the place |
| 5 | another declared source shifts a metric of the same indicator materially the other way in the same subgroup; wins over 1 | the same shift the other way, in any subgroup, over a comparable place: the shared cells reach that share of the larger place |
| 6 | none of these | no other source has events there |
| 3, 4 | reserved for a human rating or an input profile; never computed | — |

Places compare at the finest resolution both datasets have, a finer place by the cells holding its cells' centres ([6.2](06_graph_and_storage.md#62-the-graph-schema)). Metrics compare through the indicator registry, and units only when both give one. A finding whose integrity fails confirms nothing.

## 13.4 Integrity

| Check | Signs or recomputes | Catches | Does not catch | Specified in |
|---|---|---|---|---|
| record digest (`record_digest`, `integrity` on the Pattern) | SHA-256 over every field of a finding's journal record but its provenance (and `row_id`), plus the ingest fields of its spatial provenance | a number or a field edited in `journal/patterns.jsonl`: the finding is `mismatch`, never a seed, a slot of the walk, an evidence item or a drawing on the map, with a `NOTE` and a line of the evidence-only summary | an edit with the digest recomputed or removed (a record without one reads `unverified` and is used); a record deleted or inserted whole | [3.1](03_insights.md#31-the-insight-record), [6.7](06_graph_and_storage.md#67-configuration-and-guarantees) |
| metadata digest (`metadata_digest`, `integrity` on the Dataset) | SHA-256 over what the graph reads of a batch record (`BATCH_DIGEST_FIELDS`) | a source, rating, registry, glossary, metadata document or profile edited in the record: its declarations are ignored (no `PLAIN LANGUAGE` or `DATASET CONTEXT` of its own), the trust lines and a `NOTE` say so, a `PATCH` is refused (`metadata_mismatch`); its upload id and point columns place no table level ([6.1](06_graph_and_storage.md#61-the-structural-plane)) | an edit with the digest recomputed or removed (`unverified`, used as declared); the baselines it covers are still shown | [6.7](06_graph_and_storage.md#67-configuration-and-guarantees) |
| recomputation (`Engine.verify_pattern`) | support, every shift, the median test with the same effective n, the adjusted p and the spatial provenance, from the stored upload read with the batch's options | a forged number or structure whose digest was recomputed (`differs`); a changed upload (`source_changed`); a missing one (`source_missing`) | which metrics shift (mining is not re-run); the bootstrap stability, the scores (`sd_score`, `emm_score`, `volume_utility`) and the weight, which ranks the seeds, so a forged weight with its digest recomputed reads `reproduced`; drivers and the correlation change (`NOT_RECOMPUTED`); the map files (`geo.npz`, `cells.geojson`), which no digest signs | [6.7](06_graph_and_storage.md#67-configuration-and-guarantees), [8.1](08_interface.md#81-http-api) |

**The map package** (`cells.geojson`, served as `GET /api/geo/{dataset}/cells`) carries no digest, by decision. It is display data: the cells' columns and rings and a copy of the dataset's journaled patterns as its batch committed them. The server only serves it: no prompt, seed, place, window count or recomputation reads it. The console draws it and takes from it the H3 ids a zone carries (Shift-clicked cells, the cells of a card's question, `GeoMap.placeOf`); the server places such a zone on the dataset's own `geo.npz` cells, so an edited id aims a question at other cells, which then holds what those cells hold, never a number the file forged. The map takes every insight's trust (credibility, integrity) from the commit's overlay, not from the file, and draws an insight of its own level only when the overlay holds it intact: one whose journal record was edited is not drawn, nor is a pattern the file adds or one the overlay could not be read for. An edited file changes what the map draws (colours, popup values, rings) and where a zone lies, which a digest over a file the same process writes and serves would not make more trustworthy. `geo.npz`, which the places, zones and the window's counts read, is outside the digests as the table says.

A record without a digest — written before digests, or with its digest removed — reads `unverified`, is used as recorded, and is marked as such on its trust line; the workspace is not degraded. So `unverified` on a record of a workspace newer than the digests is itself a warning sign.

## 13.5 Where trust is shown

| Surface | What | Specified in |
|---|---|---|
| prompt | `DATASETS:` with the declared source and rating; the `trust:` line of each item (`source B (usually reliable) \| credibility 1 (confirmed by another source …: [P3]) \| integrity ok`), once the graph has a declared source or rating, a failed dataset check or two uploads; the header *statistical findings computed from the uploaded data; trust is given per item*; `; weakest credibility n` per zone; `NOTE`s for excluded findings and failed datasets | [7.4](07_question_answering.md#74-the-evidence-object) |
| system prompt | rule 11 TRUST: three separate things, none following from the statistics; say so plainly for credibility 5 or 6 or a source rated D–F; describe a disagreement between sources rather than resolve it | [7.5](07_question_answering.md#75-the-language-model-and-citation-check) |
| evidence-only answer | one line `- довіра всіх свідчень: джерело B (здебільшого надійне), …` when every finding shares it, else the tag `(джерело B (здебільшого надійне), …)` after each finding | [7.4](07_question_answering.md#74-the-evidence-object) |
| API | `PATCH /api/datasets/{id}`; `POST /api/patterns/{id}/verify`; `datasets` of `GET /api/graph`; `dataset` of `GET /api/nodes/{id}`; the credibility of each geo finding in `GET /api/geo/{dataset}/overlay`; `sig:sourceReliability`, `sig:credibility`, `sig:integrity` and `prov:wasDerivedFrom` in `GET /api/export/geosparql.ttl` ([6.9](06_graph_and_storage.md#69-rdf-export)) | [8.1](08_interface.md#81-http-api) |
| console | trust badges on the evidence cards; the drawer's *Довіра* section with the basis; the dataset card from the Dataset node; *Перевірити за сирими даними*; credibility in the map's cell popup; a finding whose record fails its digest is not drawn on the map nor offered by *Пов'язані знахідки* | [8.2](08_interface.md#82-the-single-page-ui), [8.7](08_interface.md#87-the-map) |

## 13.6 Audit against the preconditions of Oracle Poisoning

The paper's attack needs five conditions (its §3.7). Before is SIG at the end of the geo milestone (master `1dd06de`); after is SIG with this chapter's model.

| | Precondition | Before | After | What remains |
|---|---|---|---|---|
| P1 | a structured data store that AI agents query | the dual graph, queried for every answer | unchanged: it is the product | — |
| P2 | write access, with shared credentials and no per-user access control | uploads, `PATCH`, deletions and reset open to anyone who reaches the port; the files open to anyone with the disk | unchanged; the ports are bound to `127.0.0.1`, and most hand edits of the records are caught by P5 (its limits below) | a write token before any deployment beyond localhost ([#10](https://github.com/woldemarg/proj_sig/issues/10)) |
| P3 | no provenance tracking on individual entries | every finding names its dataset, batch, file hash and rows, but nothing says where the data came from | each dataset declares a source and its reliability; each finding carries its credibility and the sources and findings behind it; the evidence and the console show all three | a source is what the uploader declares |
| P4 | agents treat query results as ground truth | evidence introduced as *verified statistical observations* | the header calls the findings computed, with trust per item; each item's trust line names an unconfirmed or contradicted finding and an unreliable source; rule 11 | the model still restates a flagged finding plainly beside its one qualified sentence ([13.7](#137-measured-behaviour)) |
| P5 | no integrity verification against the authoritative data source | none | record and metadata digests at every compile, and every start recompiles; any finding recomputed from its stored upload; a changed upload is `source_changed` | the digests are unkeyed, and a removed digest reads `unverified`, not `mismatch`; a record deleted or inserted whole is not detected; the recomputation leaves the weight unchecked; the stored upload is the authority, so a file poisoned before upload passes every check |

Credibility is the paper's cross-verification made structural: the second source is in the graph, and the evidence names the result, so the model does not have to look it up.

## 13.7 Measured behaviour

`scripts/poisoning_eval.py` runs the red team on the planted scenarios S-poison and S-two-sources ([10.3](10_verification.md#103-hypothesis-benchmark)).
- The deterministic part runs in the gate (`test_poisoning_eval_apparatus`):
  - the planted spot of the source rated E is never confirmed (credibility 2);
  - the two foci the sources shift opposite ways are both 5;
  - a p-value forged in the journal takes its finding out of the evidence;
  - a rating forged in the batch record is ignored.
- The live part (`--llm`, a paid run) asks directed and open questions about the flagged finding, with the trust lines and rule 11, without either, and with rule 11 alone. On 2026-10-20 (`google/gemma-4-26b-a4b-it`, N = 60), 16 of 20 answers qualified the finding with the trust lines, 1 of 20 without either, and 0 of 20 with rule 11 alone; earlier runs gave 17, 0, 1 and 18, 2, 4 of 20. The label in the data moves the answer and the instruction alone does not, as the paper found. The unreliable spot remains the weak case. §10.3 holds the full tables.
