# Provenance of vendored engines

The discovery and ontology engines are **copied into this repository** so `sig/` runs without the sibling `eda/` and `lac/` folders. Both copies have diverged from their originals; every difference is listed below. SHA-256 prefixes are of the original files at copy time (2026-09-30). The originals in `eda/` and `lac/` are untouched.

| Vendored file | Original | sha256[:16] of original | Changes in the copy |
|---|---|---|---|
| `eda/main_upd.py` | `sig_proj/eda/scripts/main_upd.py` | `058133ef64d7787c` | the 3 integration edits (made before vendoring) + the numerical repairs below |
| `lac/storage.py` | `…/v2_orchestrator/storage.py` | `6bd014064a2bcbdd` | takes `ltir.config.Config`; `save`/`load` require a directory; **removed** running-mean centering (`update_global_mean`, `global_mean`, `total_embedded_chunks`), the Wikipedia-loop cursor (`processed_topic_offset`, `sync_chunk_cursor_from_journal`), `concept_records_for_neo4j`; `bootstrap` folded into `append_concepts` |
| `lac/ontology_engine.py` | `…/v2_orchestrator/ontology_engine.py` | `568197e70c55039a` | takes `ltir.config.Config`; `cold_start_extract` / `extract_orphans_omp` replaced by one `extract_attractors` (OMP input scale + signed repair); K-sweep keeps the previous K at the elbow; reroutes below the alignment floor are flagged `weak`; `soft_merge_orphans` keeps the input dimension when nothing is kept (was a hard-coded 384); `remap_activation_edges` keeps extra activation fields |
| `lac/chunk_journal.py` | `…/v2_orchestrator/chunk_journal.py` | `2398d6e211514b20` | `cache_dir` required; `records_name` selects the record log (`patterns.jsonl` in SIG); vector matrix written aside and swapped atomically; `mark()` / `truncate()` for rollback; the size-guessing fallback without `embeddings_meta.json`, `max_chunk_id` and `materialize_numpy_cache` removed |
| `lac/observability.py` | `…/v2_orchestrator/observability.py` | `97dcb2c77e69aee4` | metrics CSV path required; lac-loop helpers removed (`ExtractionStats`, `log_batch_summary`, `check_batch_invariants`, `check_post_run_invariants`) |
| `lac/projector.py` | `…/lac/v1_single_pass/visualisation/projector.py` | `42ea24b93544014c` | none |

`settings.py` and `paths.py` are not vendored. Ontology knobs are `ltir.config.Config`. Journal and state directories are required arguments. Inside the lac modules, lac's vocabulary stays: a *chunk* is a pattern row, a *concept* an attractor (glossary: SDD 01).

## `main_upd.py`: integration edits (before vendoring; see `docs/architecture/current_state.md` §4)
1. The data load and workflow run under `if __name__ == "__main__":`, so the module can be imported.
2. `step1_profile_data`: float columns are never treated as identifiers (continuous targets are all-unique).
3. `step2_evaluate_macro_groupings(profile, min_categories=0)`: an optional dimension floor. The default keeps the original behaviour.

## `main_upd.py`: numerical repairs (math audit; each is a commented local change, SDD 03)
4. `step1`: constant categoricals are dropped before the overlap rule; the overlap rule drops the finer column only when the coarser one is informative (top level < 95 %).
5. `calculate_mad`: normal-consistent mean-absolute-deviation fallback when the MAD is 0; `robust_z_score` capped at 10.
6. `step2`: ε² (bias-corrected) instead of η² as grouping power.
7. `step3`: the level crossing 95 % cumulative mass is kept; an over-budget space keeps the 2-conjunctions with the largest expected support instead of returning nothing.
8. `step4`: correlations undefined inside a subgroup contribute no EMM change (were read as 0).
9. `step4b`: 20 full-size bootstrap resamples (were 5 × 90 %); confounder drivers need a significant chi-square skew and name the level with the largest share gain; hidden shifts are reported signed in robust σ.

## Not vendored (not used by LTIR)
lac `main.py` (Wikipedia batch loop), `ingest.py`, `neo4j_uploader.py`, `cypher_loader.py`, `verification.py`, `viz_export.py`, `reset.py`, `tests/`, the `v1_single_pass` pipeline. SIG has its own pipeline, Neo4j sink and sphere export, which call the vendored modules.

## Other copied assets
| Asset | Original | Location |
|---|---|---|
| Embedding model `paraphrase-multilingual-MiniLM-L12-v2` (snapshot `e8f8c211`, 458 MB) | `sig_proj/lac/models/sentence-transformers/…/snapshots/e8f8c211…` | `sig/models/paraphrase-multilingual-MiniLM-L12-v2/` |
| `housing.csv` (realistic demo dataset) | `sig_proj/eda/data/housing.csv` | `sig/data/housing.csv` |
| Cytoscape.js 3.30.2 | cdnjs | `sig/ltir/web/static/vendor/cytoscape.min.js` |

Third-party Python packages (numpy, pandas, scikit-learn, pysubgroup, prosphera, plotly, seaborn, sentence-transformers, …) are ordinary pip dependencies listed in `requirements.txt`.

## Taking upstream changes
Do not re-copy a vendored file over its local changes. Re-apply an upstream change by hand (for `ontology_engine.py`, inside `extract_attractors` / `repair_extraction`; for `main_upd.py`, keeping edits 1–9) and run `python -m pytest`; `discovery.py` stays the Insight boundary.
