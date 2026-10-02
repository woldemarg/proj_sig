# Provenance of vendored engines

The discovery and ontology engines are **copied into this repository** so this repository runs without the original `eda` and `lac` projects. Both copies have diverged from their originals; every difference is listed below. SHA-256 prefixes are of the original files at copy time (2026-09-30). The original eda and lac projects are untouched.

| Vendored file | Original | sha256[:16] of original | Changes in the copy |
|---|---|---|---|
| `eda/main_upd.py` | eda project: `scripts/main_upd.py` | `058133ef64d7787c` | the 3 integration edits (made before vendoring) + the numerical repairs below + the removal of the standalone runner and step 5 (edit 10) |
| `lac/storage.py` | lac project: `v2_orchestrator/storage.py` | `6bd014064a2bcbdd` | takes `ltir.config.Config`; `save`/`load` require a directory; **removed** running-mean centering (`update_global_mean`, `global_mean`, `total_embedded_chunks`), the Wikipedia-loop cursor (`processed_topic_offset`, `sync_chunk_cursor_from_journal`), `concept_records_for_neo4j`; `bootstrap` folded into `append_concepts`; `update_concept_centroid(..., damping=1.0)` multiplies the decayed EMA rate (docs/05_latent_anchors.md §5.6); `keep_concepts(ids)` drops concepts for SIG's dataset deletion (§5.11) |
| `lac/ontology_engine.py` | lac project: `v2_orchestrator/ontology_engine.py` | `568197e70c55039a` | takes `ltir.config.Config`; `cold_start_extract` / `extract_orphans_omp` replaced by one `extract_attractors` (OMP input scale + signed repair); K-sweep keeps the previous K at the elbow; reroutes below the alignment floor are flagged `weak`; `soft_merge_orphans` keeps the input dimension when nothing is kept (was a hard-coded 384); `remap_activation_edges` keeps extra activation fields; the K-sweep / OMP progress prints are removed (the extracted attractor count and yield are in `BatchMetrics`); `assign_and_update`, `assign_orphans_nearest`, `route_absorbed_activations` take an optional per-attractor `damping` vector (`_step_scale`) |
| `lac/chunk_journal.py` | lac project: `v2_orchestrator/chunk_journal.py` | `2398d6e211514b20` | `cache_dir` required; `records_name` selects the record log (`patterns.jsonl` in SIG); vector matrix written aside and swapped atomically, the swap and the matrix reads retrying Windows sharing violations (`ltir.fileio`); `mark()` / `truncate()` for rollback; `rewrite()` replaces the whole journal (dataset deletion, docs/06 §6.8); the size-guessing fallback without `embeddings_meta.json`, `max_chunk_id` and `materialize_numpy_cache` removed |
| `lac/observability.py` | lac project: `v2_orchestrator/observability.py` | `97dcb2c77e69aee4` | metrics CSV path required; lac-loop helpers removed (`ExtractionStats`, `log_batch_summary`, `check_batch_invariants`, `check_post_run_invariants`); `BatchMetrics` + `density_threshold`, `damped_attractors`, `max_centroid_step`, `clamped_attractors` (CSV columns); `density_threshold(n, config)`; `apply_health_warnings(metrics, config)` reads its thresholds from `Config` and flags a hub above `τ_density` (was a fixed 25 %) and any clamped centroid |
| `lac/projector.py` | lac project: `v1_single_pass/visualisation/projector.py` | `42ea24b93544014c` | the unused entry point `OntologyProjector.project_ontology` is removed; `ltir/sphere.py` chains its steps itself (`_apply_pca` → `_scale_vectors_on_sphere` → `_build_figure`, then `save_html`); `_build_figure` no longer draws the "Chunks" trace (SIG replaced it with one trace per latent anchor), so its `chunk_labels` / `chunk_hovertext` parameters and `STYLE["chunk_size"]` are gone; the `seaborn.color_palette("Set3")` call is replaced by the same 12 colours as a constant (`SET3`, cycled like seaborn), so SIG imports seaborn nowhere (prosphera still does); projection, concept, edge and axis code unchanged |

`settings.py` and `paths.py` are not vendored. Ontology knobs are `ltir.config.Config`. Journal and state directories are required arguments. Inside the lac modules, lac's vocabulary stays: a *chunk* is a pattern row, a *concept* an attractor (glossary: `docs/11_reference.md` §11.1).

## `main_upd.py`: integration edits (before vendoring; see `docs/architecture/current_state.md` §4)
1. The data load and workflow run under `if __name__ == "__main__":`, so the module can be imported (both blocks later removed, edit 10).
2. `step1_profile_data`: float columns are never treated as identifiers (continuous targets are all-unique).
3. `step2_evaluate_macro_groupings(profile, min_categories=0)`: an optional dimension floor. The default keeps the original behaviour.

## `main_upd.py`: numerical repairs (math audit; each is a local change, explained in docs/02_discovery.md §2.2)
4. `step1`: constant categoricals are dropped before the overlap rule; the overlap rule drops the finer column only when the coarser one is informative (top level < 95 %).
5. `calculate_mad`: normal-consistent mean-absolute-deviation fallback when the MAD is 0; `robust_z_score` capped at 10.
6. `step2`: ε² (bias-corrected) instead of η² as grouping power.
7. `step3`: the level crossing 95 % cumulative mass is kept; an over-budget space keeps the 2-conjunctions with the largest expected support instead of returning nothing.
8. `step4`: correlations undefined inside a subgroup contribute no EMM change (were read as 0).
9. `step4b`: 20 full-size bootstrap resamples (were 5 × 90 %); confounder drivers need a significant chi-square skew and name the level with the largest share gain; hidden shifts are reported signed in robust σ.

## `main_upd.py`: removals
10. The standalone runner (the guarded load of a data file of the original eda project at the top, and the workflow at the bottom), `step5_integrate_and_materialize` (a print-only report) and the then-unused `zscore` import are removed. SIG never called them. `ltir/discovery.py` computes the runner's `temp_index` with the same formula (`_z_positive`: z-score, fill 0, clip at 0). Step 5's integrated index is not computed: SIG never read it.

## Not vendored (not used by LTIR)
lac `main.py` (Wikipedia batch loop), `ingest.py`, `neo4j_uploader.py`, `cypher_loader.py`, `verification.py`, `viz_export.py`, `reset.py`, `tests/`, the `v1_single_pass` pipeline. SIG has its own pipeline, Neo4j sink and sphere export, which call the vendored modules.

## Other copied assets
| Asset | Original | Location |
|---|---|---|
| Embedding model `Qwen/Qwen3-Embedding-0.6B` (revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 1.19 GB bf16; default) | Hugging Face, fetched by `scripts/download_model.py` (writes `REVISION`) | `models/Qwen3-Embedding-0.6B/` |
| Embedding model `paraphrase-multilingual-MiniLM-L12-v2` (snapshot `e8f8c211`, 458 MB; alternative) | the lac project's sentence-transformers cache (snapshot `e8f8c211…`) | `models/paraphrase-multilingual-MiniLM-L12-v2/` |
| `housing.csv` (realistic demo dataset) | eda project: `data/housing.csv` | `data/housing.csv` |
| Cytoscape.js 3.30.2 | cdnjs | `ltir/web/static/vendor/cytoscape.min.js` |

`models/` and `data/` are local, git-ignored folders: the models and `housing.csv` are not in version control (the Qwen3 model is fetched by `scripts/download_model.py`, the demo file is generated by `ltir/synth.py`). Cytoscape.js is committed.

Third-party Python packages (numpy, pandas, scikit-learn, pysubgroup, prosphera, plotly, sentence-transformers, …) are ordinary pip dependencies listed in `requirements.txt`.

## Taking upstream changes
Do not re-copy a vendored file over its local changes. Re-apply an upstream change by hand (for `ontology_engine.py`, inside `extract_attractors` / `repair_extraction`; for `main_upd.py`, keeping edits 1–10: upstream changes to step 5 or the runner do not apply) and run `python -m pytest`; `discovery.py` stays the Insight boundary.
