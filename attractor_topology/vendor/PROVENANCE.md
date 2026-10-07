# Provenance of the vendored ontology engine

The `lac` modules are **copied into this repository** so the topology runs without the original lac project. The copies have diverged from their originals, and every difference is listed below. SHA-256 prefixes are of the original files at copy time (2026-09-30).

| Vendored file | Original | sha256[:16] of original | Changes in the copy |
|---|---|---|---|
| `lac/storage.py` | lac project: `v2_orchestrator/storage.py` | `6bd014064a2bcbdd` | <ul><li>Takes the topology settings (`attractor_topology.config.TopologyConfig`, imported as `Config`); `save`/`load` require a directory.</li><li>**Removed:** running-mean centering (`update_global_mean`, `global_mean`, `total_embedded_chunks`), the Wikipedia-loop cursor (`processed_topic_offset`, `sync_chunk_cursor_from_journal`), and `concept_records_for_neo4j`.</li><li>`bootstrap` is folded into `append_concepts`.</li><li>`update_concept_centroid(..., damping=1.0)` multiplies the decayed EMA rate (docs/05_latent_anchors.md §5.6).</li><li>`keep_concepts(ids)` drops concepts for dataset deletion (§5.11).</li></ul> |
| `lac/ontology_engine.py` | lac project: `v2_orchestrator/ontology_engine.py` | `568197e70c55039a` | <ul><li>Takes the topology settings.</li><li>`cold_start_extract` and `extract_orphans_omp` are replaced by one `extract_attractors` (OMP input scale + signed repair, docs/05_latent_anchors.md §5.3). The upstream functions fit the dictionary on the unscaled rows, where the sparse-coding penalty leaves the atoms at their initialisation (measured: mixed, near-duplicate atoms), and apply the soft-merge rule only against existing centroids (measured: 3 tight clusters gave 5 atoms).</li><li>The K-sweep keeps the previous K at the elbow; upstream keeps the larger K.</li><li>Reroutes below the alignment floor are flagged `weak`.</li><li>`soft_merge_orphans` keeps the input dimension when nothing is kept; it was a hard-coded 384.</li><li>`remap_activation_edges` keeps extra activation fields.</li><li>The K-sweep and OMP progress prints are removed; the extracted attractor count and yield are in `BatchMetrics`.</li><li>`assign_and_update`, `assign_orphans_nearest` and `route_absorbed_activations` take an optional per-attractor `damping` vector (`_step_scale`).</li></ul> |
| `lac/observability.py` | lac project: `v2_orchestrator/observability.py` | `97dcb2c77e69aee4` | <ul><li>The metrics CSV path is required.</li><li>lac-loop helpers are removed: `ExtractionStats`, `log_batch_summary`, `check_batch_invariants`, `check_post_run_invariants`.</li><li>`BatchMetrics` gains the CSV columns `density_threshold`, `damped_attractors`, `max_centroid_step` and `clamped_attractors`.</li><li>New `density_threshold(n, config)`.</li><li>`apply_health_warnings(metrics, config)` reads its thresholds from the settings. It flags a hub above `τ_density` (was a fixed 25 %) and any clamped centroid.</li><li>`max_concept_density_pct` divides by the memberships of all concepts (was: chunks seen), so the shares sum to 1 for any `CONCEPTS_PER_CHUNK` (docs/05_latent_anchors.md §5.6).</li></ul> |

lac's `chunk_journal.py` is vendored as well, but it belongs to the graph service's storage, which keeps its own provenance note.

`settings.py` and `paths.py` are not vendored. The ontology knobs are `TopologyConfig`, and state directories are required arguments. Inside the lac modules, lac's vocabulary stays: a *chunk* is a pattern row and a *concept* is an attractor (glossary: `docs/11_reference.md` §11.1).

## Not vendored
- lac `main.py` (the Wikipedia batch loop), `ingest.py`, `neo4j_uploader.py`, `cypher_loader.py`, `verification.py`, `viz_export.py`, `reset.py`, `tests/`, and the `v1_single_pass` pipeline.
- lac v1's sphere projector (`v1_single_pass/visualisation/projector.py`, on prosphera). The graph service's sphere view (`insight_graph_service/server/views.py`) computes the same projection with the same scikit-learn calls: `robust_scale` → cosine `KernelPCA` → sphere scaling.

## Other copied assets
| Asset | Original | Location |
|---|---|---|
| Embedding model `Qwen/Qwen3-Embedding-0.6B` (revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 1.19 GB bf16) | Hugging Face, fetched at the pinned revision | `models/Qwen3-Embedding-0.6B/` (local, git-ignored) |

## Taking upstream changes
Do not re-copy a vendored file over its local changes. Re-apply an upstream change by hand (for `ontology_engine.py`, inside `extract_attractors` / `repair_extraction`), then run `python -m pytest`.
