# Provenance of the vendored journal

| Vendored file | Original | sha256[:16] of original | Changes in the copy |
|---|---|---|---|
| `chunk_journal.py` | lac project: `v2_orchestrator/chunk_journal.py` | `2398d6e211514b20` | `cache_dir` is required. `records_name` selects the record log (`patterns.jsonl` here). The vector matrix is written aside and swapped atomically; the swap and the matrix reads retry Windows sharing violations (`fileio`). `mark()` / `truncate()` support rollback, and `rewrite()` replaces the whole journal (dataset deletion, docs/06_graph_and_storage.md §6.8). Removed: the size-guessing fallback without `embeddings_meta.json`, `max_chunk_id` and `materialize_numpy_cache`. |

It was copied on 2026-09-30. Inside the module, lac's vocabulary stays: a *chunk* is a pattern row.
