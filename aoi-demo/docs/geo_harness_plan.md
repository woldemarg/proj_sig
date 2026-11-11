# Geospatial harness: SIG + ADS end-to-end — research and the POC plan

Status: implemented 2026-11-06 — milestone #25, epic #316, issues #317–#321 (four harness PRs against `develop`, ADS on its branch `166-h3-cells-tool`; after a closed review of that branch the h3 API of the §9 playbook moved into a profiler-gated codegen hint `<H3_Cells>`; the plugin stayed as a thin router + deliverable over the hint — A/B 2026-11-09: routing and trace visibility, same analysis — see the branch); the sections below are the plan as approved, with §11 experiments still deferred to their triggers.
Date: 2026-11-06. Repos: `sig` (develop, HEAD d6c0ff7; the harness is `sig/aoi-demo/`); `agentic-data-science` (ADS, master @ 7e422a9).
Owner's rules: SIG is not changed outside `sig/aoi-demo/`; ADS is changed only on one experimental branch and only for the geo tool + the `h3` package (plus the docs/tests ADS's own `AGENTS.md` demands for that change). The POC is the existing page with three pipes changed; experiments run later, off the happy path, when a real run shows a default is wrong.

## 1. Executive summary

The harness already runs the happy path (Chicago crimes + bars → SIG graph → polygon → SIG answer → ADS → joint report), but geography is only a filter in it: no H3 column reaches ADS, the user's question never reaches SIG, SIG's findings travel as prose, the credibility sort is inverted, and `aoi.md` already exceeds ADS's domain-knowledge cap.

The POC changes three pipes of the existing page and adds one tool to ADS:
- **build** (the `Демо-дані` button, plus one multi-select of category values → SIG `count_values`): after SIG's build read R from `GET /api/geo`, add `h3_r{R}`, `h3_r{R-1}`, `h3_r{R-2}` to both frames.
- **zone**: `in_aoi` exact as today; zone cells by the rule SIG itself uses (`h3.geo_to_cells(polygon, R)` ∩ occupied); `aoi_zone ∈ {inside, adjacent, far}` (≈ 10 lines); the frames snapshotted onto `STATE["zone"]`; the geo-only SIG preview as today.
- **ask**: SIG called with `{question, cells}` (fallback to the preview when the combined payload has nothing in the zone); observations = evidence items in credibility order; the whole SIG answer + the O# list go verbatim into an English-framed `user_query`; `aoi.md` becomes a short English column glossary; `insights.md` is no longer uploaded; files = `base_aoi.csv`, `context_aoi.csv`, `aoi.md`.
- **ADS branch**: `h3` in the executor image, `h3` in the import allow-list, one short, general plugin playbook on manipulating `h3_r*` columns. One smoke job.

Defaults are frozen to what SIG already uses: levels R, R−1, R−2; ring 1; combined entry; no union, no glossary, no cell table, no slice compiler. Candidate experiments are listed in §11 with the trigger that would justify each.

## 2. Current architecture (facts, with file:line)

### 2.1 SIG — what the harness relies on
- **Upload** `POST /api/upload` (`insight_graph_service/server/app.py:221-282`): `geo=lat,lon@auto` → the finest resolution in 10…3 with ≥ 50 dense cells (≥ 5 points), ≥ 50 % of points in dense cells, ≤ 5 000 cells; coarser levels only from the region levels 7/5/3/1 with ≥ 500 cells (`subgroup_miner/spatial.py:122-186`); Chicago crimes 2024 → **r9 only**. `time=<col>`, `count_values=col=v1,v2` (per-type counters `count_<col>_<v>`; a value with < 250 located rows is silently `sparse`, `core/uploads.py:317-325`), `where`, `bins`, `categories`, `layer=true` (points-only LAYER record), `glossary` (JSON, enters text grounding when keyed by metric names), `source`. **No target/metric option**: every numeric column is a metric; a categorical column with ≤ 8 levels mixed within cells becomes `share_<col>_<level>` metrics, > 8 levels or place-like becomes the dimension `<col>_mode` (`spatial.py:597-610`). Insights are conjunctions of 2–3 conditions on dimensions (`district_mode`, `neighbours_<counter>_band`, `lisa_points`, `context_<layer>_band`) with shifts in metrics.
- **Cross-type** `POST /api/crosstype {target, layers:[one], ring: 0|1|2, …}` (`app.py:87-96`, `uploads.py:483-607`): a dataset admitting only insights with a `context_<layer>_band` condition; `FAILED no_viable_insights` is a legitimate verdict.
- **Geo reads**: `GET /api/geo` → per dataset `{dataset_id, upload_id, filename, resolution, cells, points, levels?, min_cell_points, …}` (the only place R is served); `GET /api/geo/{ds}/cells` → pack `{columns:{h3, points, share_*, *_mode, lisa_points…}, rings, scale, bbox, patterns[]}` (no centres, no bands); `GET /api/geo/{ds}/patterns/{pid}?rings=1`.
- **Graph entry**: `POST /api/search {question, cells | geometry, resolution?, as_of?, period?}` (`app.py:73-80`) returns the evidence payload without an LLM; the narrator's `POST /api/chat/query {question, cells|geometry, …, use_llm}` forwards it and adds the answer (`evidence_narrator_service/app.py:29-36, 59`). Zone → `place = |P∩Z|/√(|P|·|Z|)` per pattern (`core/geo_views.py:104-134`); a `geometry` is converted at the **finest resolution over all datasets in the workspace** (`geo_views.py:112-124`) by `h3.geo_to_cells` centre containment with a vertex fallback (`subgroup_miner/spatial_plane.py:326-362`). Text → lexical parse + char-n-gram TF-IDF + dense Qwen3 grounding over metric names, declared names and condition values (`graph_query_engine/question.py:175-194, 283-409`). Seed score (`seeds.py:38-107`): lexical 0.35·target + 0.25·scope + 0.15·direction + 0.15·cos + 0.10·weight, else 0.7·cos + 0.3·weight, **+ 0.35·place when cells are given with text**; cells alone → weight + place, 0 outside. ≤ 3 seeds; the zone affects only the seeds; walk budget seeds + 12; evidence keeps 10 + ≤ 10 vertical.
- **Evidence item** (`graph_query_engine/evidence.py:59-87`): `key P#, pattern_id, role, headline, scope[]` (strings `"attr=value"`, `evidence.py:646`), `scope_text, shift_text, statistics{shifts[{metric, robust_z, local_median, global_median}], support, p_adjusted, stability, weight}, credibility (best→worst 1, 2, 6, 5; WEAKEST_FIRST = (5, 6, 2, 1), evidence.py:50), trust, reading{where, what, headline, gist}, provenance{filename, spatial{resolution, cells}}`; payload `area` = key → place score (0 = outside the zone).
- **Narrator answer**: prose with the labels «Спостереження:» / «Інтерпретація (гіпотези):» and `[P#]` citations (`narration.py:35-36`, `CITATION` regex :103, fallback :181); `answer_mode llm | fallback | empty`; hypotheses are never stored.
- **H3 inside SIG**: `h3` 4.5.0; `latlng_to_cell` on distinct points (`spatial.py:425`); parents `cell_to_parent(res-1)` as blocks and the null's unit (`spatial.py:502`, `context.py:58`); `grid_ring(c, 1)` lag bands (absent neighbours = 0), `grid_disk(c, ring)` context disks. SIG stores **no H3 column** in any table (cells live in `geo.npz`, served as `h3` strings) — nothing to collide with.
- **Console zone**: Shift-click cells → `{question, cells, period?, use_llm}` (`sig_web_console/static/chat.js:229`); no polygon tool. nginx routes `/api/chat/` to the narrator and the rest of `/api/` to the graph (`nginx.conf:18-24`), so `{SIG_URL}/api/search` is reachable from the harness.

### 2.2 The harness today — `sig/aoi-demo/` (`app.py` 672 lines, `index.html` 571, `joint_response.jinja` 366)
- `POST /datasets` → `run_datasets` (A:411-441): uploads `BASE` / `CONTEXT` (A:44-45; `geo=lat,lon@auto`, `time=crime_date`; bars `layer=true`), waits, `POST /api/crosstype {ring: 0}`, fetches the cell pack and keeps `h3` + rings + vertex-mean centres (`cell_grid`, A:98-108), reads both CSVs into `DATA`.
- `POST /zone {polygon}` (A:589-618): `in_aoi` by `shapely.contains_xy` on `lon`/`lat` (A:172-176) **written in place onto `DATA`** (A:598); zone cells = cells whose vertex-mean centre is in the polygon (A:111-114); thread `run_mine` → `POST /api/chat/query {question: "", cells, use_llm}` (`SIG_ZONE_QUESTION = ""`, A:120-130); `sig_findings` (A:138-166) marks `inside = area[key] > 0`, sorts `(-inside, -credibility)` (inverted), caps 10, fetches rings; writes `aoi.md` (methodology + the ≈ 900-word Ukrainian `SIG_GLOSSARY`, A:180-204) and `insights.md`.
- `POST /ask {question}` → `run_ask` (A:476-534): reuses the zone's SIG result (the question never reaches SIG, A:490); ADS session; uploads `base_aoi.csv` / `context_aoi.csv` (raw columns + `in_aoi`, **no H3**), `aoi.md`, `insights.md`; `user_query` = fixed text (A:286-314) with a fixed «Додаткові розрізи» paragraph; polls; joint report = jinja rendered with `user_query, report, insight_analysis (= answer), insight_evidence (= SIG prompt), domain_knowledge (= aoi.md)` → `POST {LLM_URL}/v1/chat/completions {model: "synthesis", temperature 0.2, max_tokens 16384}`; accepted when it contains `## Результати аналізу` and `## Методологія` (A:405).
- UI: MapLibre 4.7.1, hand-written polygon tool, panels Дані / Зона / Питання, `GET /state` every 3 s, `GET /ads-query` preview, `GET /trace/prompt`. The joint template says SIG «отримав лише територію» (J:127-129).

### 2.3 ADS — what the harness and the branch rely on
- **Session API** (`orchestrator/server/main.py`): `POST /sessions` → `{session_id}`; `POST /sessions/{id}/datasets` (multipart `file`; `.csv/.parquet/.xls/.xlsx` → kernel globals `df_<name>`, `.md/.txt` → `<User_Domain_Knowledge>`; non-`[a-zA-Z0-9._-]` filename chars → `_`); `POST /sessions/{id}/jobs {user_query}`; `GET /sessions/{id}/status` 202 / 200 `{result, outcome}`; report at `result.last_execute_response.report` (free-form markdown **strictly in English** + `## Methodology`); `result.system_counters.domain_knowledge_blocked`; `result.execution_trace[]` (`tool_name, code, status, error`).
- **Pipeline**: reconcile → profiler (dtype, missing %, unique, top-5, samples; nothing recognises lat/lon) → semantic_grounder (guardrail over the `.md` text, one verdict for all files) → planner (picks a tool name per step) → codegen (playbook → Python for one notebook cell) → validator (`__ADS_OBS__`) → goal_evaluator → synthesis. Limits: 24 steps, cell timeout 60 s (env 120), DataFrames > 1M rows dropped.
- **Domain-knowledge budget**: grounder 800 tokens, codegen 2 000 (head + tail), not synthesis (`services/context/common.py:49-63`); a guardrail refusal or gateway error drops **all** `.md` (`domain_knowledge_blocked = 1`). `user_query` is neither capped that way nor scanned.
- **Tools = YAML playbooks**, not function calls (`shared/schemas/tool.py:4-10`: `tool_name, description, max_retries, guidelines, deliverable`). Registry globs `orchestrator/engine/tools/builtin/*.yaml` and `…/plugins/*.yaml` (`registry.py:20-40`; plugins holds `.gitkeep`). Planner sees `<Available_Tools>` (descriptions cut to 80 chars under budget, `tool_budget.py:28-50`); codegen gets the playbook, trimmed around `Objective:` / `Telemetry Contract:` when > 3 000 chars (`tool_budget.py:17-21, 53-80`). YAML integrity test `tests/test_graph_flow.py:438-500`.
- **Executor packages** (`executor/pyproject.toml:6-28`): numpy, scipy, pandas, pyarrow, plotly, matplotlib, scikit-learn… — **no h3/shapely/geopandas**; an import must also be in `IMAGE_IMPORT_ROOTS` (`shared/constants/import_policy.py:23-36`, `executor/preflight/ast_verifier.py:295-301`); codegen forbids `iterrows` / `apply(axis=1)` and disk reads (`node_prompts.py:407-425`).
- **Rules** (`AGENTS.md`): `make check` after every Python edit; tool YAML → `arch/architecture/06_builtin_tool_suite.md`; import policy → `03_executor_and_kernel_runtime.md` + `arch/security/02`; `uv.lock` bump → `security/02` + `05`; branches `<task#>-<slug>`, commits `feat(ads-engine): <tracker>#NNN - …`.

### 2.4 The H3 reference (ChatGPT share, read in full) and the verified h3-py API
Adopted from the reference: one column per resolution named `h3_r<res>`; ids as strings, never prefixes or arithmetic; `cell_to_parent` for hierarchy (logical parent ≠ geometric containment of nearby points at the coarser level); `grid_disk(cell, k)` / `grid_ring(cell, 1)` for neighbours (order not guaranteed, pentagons have 5); exact polygon work needs geometry; average areas r7 ≈ 5.16 km², r8 ≈ 0.737 km², r9 ≈ 0.105 km². It gives no density thresholds — the levels stay SIG's.
h3-py 4.5.0 on the sig venv: `latlng_to_cell(lat, lon, res)`, `cell_to_parent(cell, res)`, `grid_ring(cell, 1)` (6), `grid_disk(cell, k)`, `geo_to_cells(<GeoJSON Polygon dict>, res)` (centre containment, takes the map's GeoJSON as is), `cell_to_latlng`, `h3shape_to_cells_experimental(LatLngPoly, res, contain='overlap')` (not used in P0).

### 2.5 Corrections to earlier notes (the implementer must not trust the old README on these)
| Claim | Code | Consequence |
|---|---|---|
| evidence `scope` is `[{attribute, value}]` | strings `"attr=value"` (`evidence.py:646`) | quote `scope_text` as is; no parsing in P0 |
| credibility 1 best → 6 worst | best→worst **1, 2, 6, 5** (`evidence.py:50`) | rank map `{1:0, 2:1, 6:2, 5:3}` |
| `count_values` errors below 250 rows | silently `sparse`, no counter (`uploads.py:317-325`) | the select disables values with < 250 located rows |
| SIG converts `geometry` at the base's R | at the workspace's finest R over all datasets (`geo_views.py:112-124`) | send `cells`, computed with `h3.geo_to_cells` at the base's R |
| `aoi.md` fits ADS's budget | `SIG_GLOSSARY` ≈ 900 Ukrainian words > the 800-token grounder cap | short English `aoi.md`; the Ukrainian glossary only in the joint prompt |
| harness cell centres = SIG's | vertex mean (A:107) vs the H3 centre | zone membership by `geo_to_cells`, not by centres |
| `/ask` runs the SIG chat | it reuses the zone's result (A:490); the README says otherwise | fix the README with the pipe change |

## 3. User requirements

| ID | Requirement | Prio |
|---|---|---|
| R1 | H3 reaches ADS: `h3_r{R}`, `h3_r{R-1}`, `h3_r{R-2}` on both tables, R read from SIG after the build (never chosen by the harness) | P0 |
| R2 | The user's question reaches SIG: `{question, cells}` on Ask, the geo-only preview kept, fallback to the preview when the combined payload has nothing in the zone, said on the page | P0 |
| R3 | Category values for SIG's `count_values` chosen on the page (one multi-select on «Демо-дані», values with < 250 located rows disabled) so base↔context cross-layer counters exist (e.g. `crime=theft`, `crime=assault`) | P0 |
| R4 | Zone cells by SIG's own rule (`geo_to_cells` ∩ occupied); `in_aoi` exact; `aoi_zone ∈ {inside, adjacent, far}`; frames snapshotted when the polygon closes | P0 |
| R5 | ADS task: English frame, the question verbatim, one geo line, the whole SIG answer verbatim, the O# list (headline, `scope_text`, `shift_text`, in-zone, credibility word, file), optional comparisons where the question or SIG's narration points to them; short English `aoi.md`; `insights.md` not uploaded | P0 |
| R6 | ADS geo tool on one experimental branch: `h3` in the executor, allow-list entry, one short general playbook on `h3_r*` columns; one smoke job | P0 |
| R7 | Credibility sort fixed; `aoi.md` under the ADS cap; `domain_knowledge_blocked` shown in the trace | P0 |
| R8 | Joint template: O# / zone-vs-surroundings instructions, the «SIG отримав лише територію» sentence updated; the two-heading gate kept | P0 |
| R9 | Three-step intake (file inputs, column choices) | P1 |
| R10 | Variants of what SIG hands to ADS (whole answer / hypotheses only / O# only), glossary at upload, union of payloads, levels/ring alternatives, cell table, slice compiler — each only after a real run shows the default fails (§11) | P1 (experiments) |
| R11 | shapely in ADS, profiler lat/lon detection, kernel preload, multi-user state, retries, charts, second dataset pair, four-heading gate | P2 / out of scope |

## 4. Proposed architecture

Responsibilities: **harness** (`sig/aoi-demo/`) — SIG orchestration, H3 enrichment, zone classes, task building, joint report (LLM); **SIG** (unchanged) — graph, R, cross-type null, entry, narrator (LLM); **ADS** (branch) — the playbook + `h3`; the rest as is (LLM).

```mermaid
flowchart TD
  D[«Демо-дані» + count_values select] --> B[SIG: upload base (geo@auto, time, count_values)<br/>upload layer=true · crosstype ring 0]
  B --> R[GET /api/geo → R · GET /api/geo/ds/cells → occupied cells]
  R --> E[Harness: h3_r{R}, h3_r{R-1}, h3_r{R-2} on both frames<br/>warn if set ≠ pack.h3]
  E --> Z[Polygon → in_aoi exact · cells = geo_to_cells(poly, R) ∩ occupied<br/>aoi_zone inside/adjacent/far · snapshot frames]
  Z --> S0[SIG preview {question: "", cells}]
  Z --> Q[«Запитати»]
  Q --> S1[SIG {question, cells, use_llm} · fallback to preview]
  S1 --> T[user_query: frame · question · geo line · SIG answer verbatim · O# list · optional comparisons<br/>aoi.md short English · base_aoi.csv · context_aoi.csv]
  T --> A[ADS: planner → codegen (h3 playbook) → synthesis → English report]
  A --> J[Joint report (jinja + synthesis route) · two-heading gate]
  S1 --> J
```
Everything in the harness is deterministic except the joint-report call; SIG's narrator and the ADS run are LLM.

## 5. Data contracts and geospatial schema

### 5.1 H3 columns (both tables)
- `h3_r<res>` with the absolute resolution in the name (`h3_r9`, `h3_r8`, `h3_r7` for Chicago); H3 strings; empty where coordinates are missing/invalid.
- The finest level **is SIG's R** (`GET /api/geo` → `resolution` of the base's finest dataset). Parents by `cell_to_parent(cell, R-1)` and `(…, R-2)`. Constant `H3_LEVELS = (0, 1, 2)` (offsets below R) — the only knob.
- Points indexed with `h3.latlng_to_cell(lat, lon, R)` over distinct points, as SIG does. After the build the harness compares `set(base.h3_rR)` with `pack.h3` and **warns** (count of the symmetric difference in the Data status); it never fails the demo.

### 5.2 Zone columns
| Column | Values | Rule |
|---|---|---|
| `in_aoi` | 0/1 | `shapely.contains_xy(polygon, lon, lat)` — exact, as today |
| `aoi_zone` | `inside` / `adjacent` / `far` | `inside` ⇔ `in_aoi == 1`; `adjacent` = `in_aoi == 0` and `h3_rR ∈ Z ∪ ring1(Z)` (points in the zone's own cells but outside the polygon are `adjacent`, not inside); else `far`. Rows without coordinates: `far`. `AOI_RING = 1`. |

`Z` = `h3.geo_to_cells(polygon, R)` ∩ occupied cells (the pack's `h3`); when empty, the cells under the polygon's vertices (SIG's fallback); still empty → HTTP 400. `Z` is what is sent to SIG, drawn on the map and used for `adjacent`. One sentence in `user_query` carries the teaching point: *a point in the same cell as the zone is not inside the zone; `adjacent` includes such points*.

### 5.3 Files to ADS (ASCII names; kernel globals `df_base_aoi`, `df_context_aoi`)
| File | Columns |
|---|---|
| `base_aoi.csv` | original columns + `h3_r{R}`, `h3_r{R-1}`, `h3_r{R-2}`, `in_aoi`, `aoi_zone` |
| `context_aoi.csv` | original columns + the same five |
| `aoi.md` | ≤ 2 000 characters, English, declarative (§10.2) |
No cell table, no `insights.md` (it remains a page artefact).

### 5.4 `user_query` layout (§10.1 has the text)
`frame` → the question verbatim → `data` (files, `df_*` names, the geo line, zone counts, which `count_values` SIG counted) → `sig_answer` (the whole narrator answer verbatim) → `observations` (O# lines) → `guidance` (optional comparisons + the three rules). The page's query card keeps tagging parts.

### 5.5 O# line (verbatim SIG strings, no parsing)
`O1 [P3] <headline> · where: <scope_text> · shift: <shift_text> · in zone: yes|no · credibility: <word> · file: <filename>` — in-zone first, best credibility first (`{1:0, 2:1, 6:2, 5:3}`), ≤ 10; items from another file end with `(other file, context only)`. Credibility words follow the harness glossary (1 confirmed … 6 no other source).

## 6. UI and user workflow (the existing page; three small additions)
- **1 · Дані**: «Демо-дані» as today + one multi-select «Рахувати типи» populated from the base file's category column (the demo's `crime`; values with < 250 located rows disabled, default `theft`, `assault`); the status line gains R, levels, counted types, the H3 mismatch warning if any.
- **2 · Зона**: unchanged drawing; the counts line becomes «подій: inside / adjacent / far»; the preview as today.
- **3 · Питання**: the stage row reads «SIG: питання + зона»; a line «SIG: combined | preview (fallback)»; the O# list; the query card shows the new parts; the ADS trace shows steps, tool names and `domain_knowledge_blocked` (red when 1).
No new map layers; the zone cells are drawn as today.

## 7. SIG integration
- Polygon close: `POST /api/chat/query {question: "", cells: Z, use_llm: true}` (today's preview, kept).
- Ask: `POST /api/chat/query {question, cells: Z, use_llm: true}`; if the payload has zero items with `area[key] > 0` and the preview had some, use the preview payload and say so. Both traced (`metrics.seed_count`, in-zone counts).
- Observations: evidence items sorted by `(-inside, CRED_RANK[credibility])`, capped at 10, rings fetched as today.
- Hand-off to ADS: the whole narrator answer verbatim + the O# list (the variants «hypotheses section only» / «O# only» are a later test, §11).
- Not done: union, glossary, parsing of `scope`, searching by node/edge type (a SIG change, out of scope).

## 8. H3 enrichment pipeline (harness, deterministic)
```
build:  R, min_cell_points ← GET /api/geo (the base's finest dataset)
        for df in (base, context): ok = valid lat/lon
            df[f"h3_r{R}"][ok] = latlng_to_cell over distinct (lat, lon)
            df[f"h3_r{R-k}"] = cell_to_parent(h3_rR, R-k), k in (1, 2)
        warn if set(base.h3_rR) != set(pack.h3)
zone:   Z = geo_to_cells(polygon, R) ∩ pack.h3  (vertex fallback; else 400)
        rows: in_aoi exact; aoi_zone by §5.2 (ring = ∪ grid_ring(c, 1) over Z)
        STATE["zone"]["files"] = {base_aoi.csv: bytes, context_aoi.csv: bytes}   # the snapshot /ask serialises
```
`geo.py` holds `enrich_h3(df, lat, lon, R)`, `zone_h3_cells(polygon, R, occupied)`, `aoi_zone(df, Z, ring)` — pure, ≈ 40 lines. Everything else stays in `app.py` next to the function it already resembles.

## 9. ADS geospatial capability (experimental branch only)

### 9.1 Changes
Branch `166-h3-cells-tool` off `master`; one commit `feat(ads-engine): <tracker>#NNN - experimental H3 geospatial tool`.
| File (agentic-data-science) | Change |
|---|---|
| `executor/pyproject.toml` | `"h3>=4,<5"` (the playbook names v4 functions). No shapely/geopandas/pyproj. |
| `uv.lock` | plain `uv lock`; h3 has no runtime deps. All three images `COPY uv.lock` → rebuild. |
| `shared/constants/import_policy.py` | `"h3"` into `IMAGE_IMPORT_ROOTS`; `"shapely", "geopandas", "pyproj"` into `NOT_INSTALLED_HINT_ROOTS` |
| `orchestrator/engine/tools/plugins/geospatial_h3_cells.yaml` | new; ships in the orchestrator image; no registry code change |
| `tests/test_import_policy.py`, `tests/test_graph_flow.py` | `import h3` valid, `import shapely` → not-installed violation; the plugin loads with its name and `deliverable: true` |
| docs demanded by ADS `AGENTS.md` §6 | `06_builtin_tool_suite.md` (plugin row), `03_executor_and_kernel_runtime.md` (installed, not pre-loaded), `arch/security/02`, `arch/security/05` (manifest row) |
| not done | feature flag (the file's presence is the switch), kernel preload, profiler lat/lon detection |
Gate `make check`; from `aoi-demo/`: `docker compose up -d --build orchestrator executor`; verify `docker compose run --rm executor python -c "import h3; print(h3.__version__)"`.

### 9.2 The playbook — general rules for `h3_r*` columns only (AOI/zone semantics come from `aoi.md` and `user_query`, never from here)
```yaml
tool_name: geospatial_h3_cells
description: >
  Use when a table has H3 cell columns (h3_r<N>) and the task groups, joins or compares rows by cell,
  by neighbouring cells, or by coarser (parent) cells. ROUTING CONTRACT: the task_instruction MUST name
  the frame(s), the h3_r<N> column that sets the level, and the comparison (cell groups vs their ring,
  vs parent cells, or vs the rest). Grouping by non-H3 columns stays with statistical_and_distribution_analysis.
deliverable: true
max_retries: 4
guidelines: |
  Objective: Group, join and compare rows by H3 cells at the level that fits the question, using the precomputed h3_r<N> columns and the h3 library for parents and neighbours; emit the comparison as raw metrics.
  Pre-execution Checks:
  - `import h3` on the first line of the cell (h3 4.x is installed, not pre-loaded). Available: h3.cell_to_parent(cell, res), h3.cell_to_children(cell, res), h3.grid_ring(cell, k) (list of cells at distance k), h3.grid_disk(cell, k) (list, includes the centre), h3.get_resolution(cell), h3.is_valid_cell(cell), h3.latlng_to_cell(lat, lng, res) (latitude first), h3.cell_to_latlng(cell), h3.cell_area(cell, unit='km^2'). The v3 names geo_to_h3, h3_to_geo, h3_to_parent, k_ring, hex_ring, polyfill do NOT exist.
  - h3_r<N> columns hold H3 ids (strings) at resolution N; a coarser column is the parent of the finer one. Read the level from the column name, never from the id text; never slice, cast or do arithmetic on an id. A missing coarser column: df['h3_r7'] = df['h3_r9'].map(lambda c: h3.cell_to_parent(c, 7)).
  - Cells are the small side, rows the large side: compute cell sets with h3 over UNIQUE cells (set, dict, Series.map on a deduplicated Series), then mask rows with .isin(...). Never call h3 per row (apply(axis=1), iterrows, grid_ring inside a row loop).
  - shapely, geopandas and pyproj are not installed; do not import them; do not read files.
  Execution Playbook:
  - Pick the level: the finest h3_r<N> where the groups of interest hold enough cells and rows to compare (as a rule ≥ 3 cells, median cell ≥ 5 rows); otherwise one level coarser. Say which level and why.
  - Cell table: one row per cell — groupby(level_col).agg(rows=('col','size'), …); a category share via (df[col] == value).groupby(df[level_col]).mean().
  - Neighbourhood of a cell set S: ring = set().union(*(h3.grid_ring(c, 1) for c in S)) - S; k=2 adds grid_ring(c, 2). Label rows once with np.select over .isin masks (e.g. 'in', 'ring', 'rest').
  - Compare like with like: rows per cell for counts, shares for categories; groups hold different numbers of cells, so never compare raw counts. Report rows, cells and the metric per group, then ratios and differences.
  - Parents: to see whether a difference holds at a larger scale, map the cell set to parents ({h3.cell_to_parent(c, N) for c in S}) and repeat the comparison on the coarser column; one record per level.
  - Join two tables by cell: aggregate each to its cell table at the SAME level, then merge on the cell column (how='left', fill 0 for cells the other table does not hold).
  - Describe differences as shifts, ratios and differences; never name a cause; proximity is not causation.
  Telemetry Contract:
  - Persist the cell table and the cell sets as globals for later steps.
  - step_telemetry_payload: flat scalars first (level, level_column, per group rows / cells / rows_per_cell / share, ratios), then short lists (per-group records ≤ 8, per-level records ≤ 4, top cells ≤ 10, caveats). Raw JSON only; no df.head(); do not print __ADS_OBS__.
```

### 9.3 One contract, three carriers
| Carrier | Reaches | Holds |
|---|---|---|
| `user_query` | grounder, planner, goal evaluator, synthesis (uncapped, unscanned) | the task: geo line, zone counts, SIG answer, O#, optional comparisons, the three rules |
| `aoi.md` | grounder (800 tok), planner, codegen (2 000 tok); not synthesis; guardrail-scanned | the column glossary (declarative, no imperatives) |
| playbook | codegen (planner sees 80 chars) | how to manipulate `h3_r*` — general |

## 10. Prompt and report changes

### 10.1 `user_query` (English frame; the question and SIG text verbatim)
```
Analyse the area of interest (AOI): <user question verbatim>

Data: base_aoi.csv → df_base_aoi (<base label>, {N} rows, {n} inside the AOI); context_aoi.csv → df_context_aoi (<context label>, {M} rows, {m} inside).
Geo columns (both tables): h3_r{R}, h3_r{R-1}, h3_r{R-2} are H3 cell ids (strings) at those resolutions; a coarser id is the parent of the finer one; group by them, never edit them. in_aoi is 1 for points inside the polygon (exact). aoi_zone is inside (= in_aoi 1) | adjacent (outside the polygon but in the zone's cells or their ring-1 neighbours) | far. A point in the same cell as the zone is not inside the zone.
Zone (base rows): inside={a}, adjacent={b}, far={c}; r{R} cells in the zone: {k}. SIG counted these types per cell: <count_values>.

SIG's answer for this zone (verbatim):
<the whole narrator answer>

SIG findings in the zone (O# — check each on the data where it concerns the question):
O1 [P3] <headline> · where: <scope_text> · shift: <shift_text> · in zone: yes · credibility: <word> · file: <file>
…

Guidance:
- Where the question or SIG's answer points to it, compare inside vs adjacent vs far at the same level (rows per cell for counts, shares for categories) and check on h3_r{R-1} whether the difference holds.
- Treat SIG's conditions as extra cuts of the same computation; treat SIG's hypotheses as claims to test (confirmed / not confirmed / not testable). Do not re-mine spatial clusters.
- Describe differences as shifts and ratios, not causes; label O# explicitly in the report.
```

### 10.2 `aoi.md` (ADS; ≤ 2 000 characters; declarative; ≈ 15 lines)
```
# Area of interest: tables and columns
- Zone: user polygon, N vertices, bbox lon a…b, lat c…d. Base events: X of Y inside; context points: U of V inside.
- base_aoi.csv (<file>, <label>): events. context_aoi.csv (<file>, <label>): context points. Both carry h3_r9, h3_r8, h3_r7, in_aoi, aoi_zone.
- h3_r9 / h3_r8 / h3_r7: H3 cell ids at resolutions 9 (SIG's analysis level, ≈ 0.1 km²), 8 (≈ 0.7 km²), 7 (≈ 5 km²); a coarser id is the parent of the finer one.
- in_aoi: 1 = point inside the polygon, 0 = outside.
- aoi_zone: inside | adjacent (outside the polygon, in the zone's cells or their ring-1 neighbours) | far.
- SIG's cells: one cell = one r9 hexagon; SIG's shares and medians are per cell over cells with ≥ 5 events; counted types: <count_values>.
- SIG terms used in its answer: points = events per cell; share_<col>_<v> = share of the cell's events with col = v; count_<col>_<v> = their number; <col>_mode = the cell's most frequent value; neighbours_<x>_band = quartile of the sum over the 6 neighbours; lisa_points = local Moran class of event counts (hot spot, cold spot, high outlier, low outlier, no cluster); context_<layer>_band = quantile of context points in the cell; credibility 1 confirmed … 6 no other source.
- Findings come from: <file: P#, …> (a finding from another file is context only).
```
The Ukrainian `SIG_GLOSSARY` and methodology stay in the joint prompt's `domain_knowledge` only.

### 10.3 Joint template (`prompts/joint_response.jinja`)
- `<insight_integration>` framing (J:126-145): replace «SIG отримав лише територію» with «SIG отримав питання і зону»; add: «знахідки SIG позначені O#; для кожної, що стосується питання, скажи, що показав ADS; порівняння зони з оточенням (adjacent, far) і на грубшому рівні подай окремим абзацом, якщо ADS його зробив».
- Inputs: `observations` (the O# list) rendered as a block beside `insight_analysis`; nothing else changes.
- Gate: unchanged (`## Результати аналізу` + `## Методологія`).

## 11. Experiments — later, off the happy path
None gates the POC. Each runs only when the trigger shows up in a real Chicago run; each is one script under `sig/aoi-demo/experiments/` (not created now).
| Exp | Trigger | Variants | Metric | Decision |
|---|---|---|---|---|
| A — SIG entry | the combined call falls back to the preview on most questions, or SIG's answer ignores the question | geo-only / text-only / combined / combined + `glossary` at upload / union — on `POST /api/search` (deterministic, no LLM), 5 AOIs × 4 questions | in-zone items, grounding hit (`parsed` non-empty), items on the asked category | keep the simplest variant that gives ≥ 3 in-zone items and ≥ 1 grounding hit on specific questions |
| A′ — SIG hand-off | ADS restates SIG's prose or ignores O# | whole answer + O# (P0) / hypotheses section only / O# only, 3 AOIs × 2 questions × 2 runs | O# addressed with a number; `h3_r` used; steps | keep the variant with the highest O# coverage at ≤ P0 steps |
| B — levels | parent groups too coarse/fine for the demo AOIs, or ≥ 30 % of r9 cells < 5 points matter in ADS | R only / R, R−1, R−2 (P0) / SIG region levels / density-adaptive — offline pandas + h3 | cells, share < 5 points, median points/cell, half-year Spearman, +50 % planted excess detected | keep P0 unless it fails detection or stability |
| C — ring | `adjacent` is empty or dominates for typical AOIs | ring 1 (P0) / 2 / parent-based | rows per class per AOI, spill-over vs far | keep ring 1 unless empty on ≥ 3 of 5 AOIs |
| D — tool | the planner never picks the tool, or h3 steps fail first try | master images vs branch images (vs branch without the YAML), 5 queries × 2–3 runs | tool chosen, first-try success, v3 names, ring/parent comparisons in the report | tune the description/playbook; last resort: name the tool in `user_query` |
| E — cell table / slices | ADS cannot check a LISA/band claim from the event tables | add `cells_aoi.csv` from the pack (and later a slice map) | O# coverage for band/LISA items | add only if coverage rises |

## 12. Implementation roadmap

Repos: **H** = `sig/aoi-demo/` on branch `feat/aoi-geo-harness` → PR against `develop`; **A** = `agentic-data-science` branch `<task#>-geo-h3-tool`. H1–H6 and A1–A4 are independent until the smoke run. Checks: `uv run app.py check` (offline asserts; extended per task) and `make check` on A.

| # | Task | Repo / files | Depends | Result · acceptance | Check left behind |
|---|---|---|---|---|---|
| H0 | Commit the relocated harness as-is (today's move: compose includes `../compose.yaml`, data from `data/demo/`, `.env.sample`) | H | — | baseline in git; `docker compose config` resolves 13 services | existing `check` |
| H1 | `geo.py` (`enrich_h3`, `zone_h3_cells`, `aoi_zone`) + `h3>=4,<5` in the PEP 723 header and the Dockerfile `pip install` + `COPY aoi-demo/geo.py` | H: new `geo.py`, `app.py:3`, `Dockerfile` | H0 | pure helpers; parents consistent; a cell's own boundary polygon → exactly that cell; square polygon on a synthetic grid → all three classes, `in_aoi == 1 ⇔ inside` | `check`: 3-point frame + one real r9 cell |
| H2 | build pipe: `count_values` select → `sig_upload` form field; after the cross run `GET /api/geo` → `GRID["resolution"]`; `enrich_h3` on both frames; mismatch warning into `STATE.datasets`; `GET /datasets/options` (the category values with located-row counts) | H: `app.py` `run_datasets`, `sig_upload`, `POST /datasets {count_values}`, `index.html` select | H1 | `d.levels = [R, R-1, R-2]`, `d.counted`, `d.h3_mismatch`; live: Chicago → r9 and the mismatch is 0 or shown | `check`: the form field string for two values; `check-live` (skips when SIG is down): levels and mismatch |
| H3 | zone pipe: `Z = zone_h3_cells(polygon, R, GRID.h3)` replaces the centre rule (`cell_grid` loses centres; rings kept for drawing); `aoi_zone`; snapshot CSV bytes onto `STATE["zone"]["files"]`; counts per class | H: `app.py` `/zone`, `run_mine` (unchanged call) | H2 | `/zone` during a running `/ask` cannot change what `/ask` uploads; counts drawn | `check`: snapshot bytes unchanged after a second `/zone`-like call on the frames |
| H4 | ask pipe (SIG): `sig_chat(cells, question)`; fallback to the preview; `sig_findings` sort `(-inside, CRED_RANK[c])`; trace both payloads | H: `app.py` `run_ask`, `sig_findings`, `CRED_RANK` | H3 | the page says «combined» or «preview (fallback)»; O1 is the most credible in-zone item | `check`: a fixture of four items sorts 1, 2, 6, 5 |
| H5 | ask pipe (ADS): `ads_query_parts` with the §10.1 parts; `aoi_md()` rewritten (English, ≤ 2 000 chars, no imperatives); files = 2 CSV from the snapshot + `aoi.md`; `insights.md` kept on the page only; trace `domain_knowledge_blocked`, steps, tool names | H: `app.py` `ads_query_parts`, `aoi_md`, `run_ask` | H4 | the query holds the geo line, the SIG answer and O# lines; `len(aoi_md) ≤ 2000`; ASCII names | `check`: part order, O# numbering, length, no «must/always» in `aoi.md` |
| H6 | joint template + README: §10.3 edits; README three pipes, contract, `/ask` now calls SIG, the «файли» stage row removed, limits | H: `prompts/joint_response.jinja`, `joint_report()` (new `observations` input), `README.md` | H5 | template renders with the new input; the two-heading gate unchanged | `check`: render + gate on a fixture |
| H7 | UI: the select, the status lines, the stage row text, the O# list, the fallback line, `domain_knowledge_blocked` in red | H: `static/index.html` | H2–H5 | the Chicago scenario runs end-to-end from the page | README «Сценарій» |
| A1 | `h3` dependency + import policy | A: `executor/pyproject.toml`, `uv.lock`, `shared/constants/import_policy.py` | — | `make check` green; `import h3` works in the executor image | `tests/test_import_policy.py` params |
| A2 | plugin playbook §9.2 | A: `orchestrator/engine/tools/plugins/geospatial_h3_cells.yaml` | A1 | YAML validates; unique name; `deliverable: true` | `tests/test_graph_flow.py` plugin test |
| A3 | docs demanded by ADS `AGENTS.md` | A: `06_builtin_tool_suite.md`, `03_…`, `security/02`, `security/05` | A2 | same patch | — |
| A4 | images + one smoke job from the harness (the Chicago default question) | from `aoi-demo/`: `docker compose up -d --build orchestrator executor` | A3, H5 | `execution_trace` shows h3 code (`h3_r` or `h3.`) with status ok — an observation; the DoD does not depend on the planner's pick | — |

## 13. Testing and validation
| Area | Check |
|---|---|
| H3 indexing | parents consistent (`get_resolution`, `cell_to_parent` of the finer == the coarser column); vectorised over distinct points (< 1 s for 43 k rows); live set comparison with the pack is a warning with a count |
| Polygons | `in_aoi` exact on a unit square (existing assert); zone cells by `geo_to_cells` (GeoJSON lon/lat order); a polygon smaller than a cell → the vertex cells; empty → 400; a point in a zone cell outside the polygon is `adjacent`, not `inside` |
| Neighbours | `adjacent` = ring-1 cells minus Z (pentagon-safe) |
| SIG contract | credibility rank 1, 2, 6, 5; in-zone = `area[key] > 0`; fallback when the combined payload has none; `answer_mode` recorded |
| ADS contract | ASCII names; `df_base_aoi` / `df_context_aoi` named in the task; `aoi.md ≤ 2 000` chars, declarative; parts in order; `domain_knowledge_blocked` in the trace |
| ADS tool | `import h3` allowed, `shapely` refused with the hint; plugin loads; `make check`; smoke job |
| Joint report | the existing two-heading gate on a fixture; the new sentence present in the rendered prompt |
| End-to-end | the README scenario on Chicago; `uv run app.py check` green |

## 14. Risks, assumptions and out of scope
| Risk / assumption | Mitigation |
|---|---|
| ADS guardrail drops `aoi.md` (LiteLLM temperature 400; imperative wording) | README already requires `LITELLM_GUARDRAIL_MODEL_NAME=gpt-5.4`; `aoi.md` declarative and short; the task carries the geo line itself; the counter is shown |
| The planner never picks the tool; h3 v3 names hallucinated | the description routes on `h3_r<N>`; the cheat-sheet sits in Pre-execution Checks (survives trimming); an observation in the smoke run, not a gate; Exp D if it matters |
| The combined SIG call pulls seeds out of the zone | fallback to the preview, said on the page; Exp A if frequent |
| `count_values` value < 250 located rows silently `sparse` | the select disables such values; the status line shows what SIG counted |
| Cross-type `FAILED no_viable_insights` | accepted verdict (today's behaviour); the task still has `context_aoi.csv` with `h3_r*` so ADS can join layers itself |
| Harness cells ≠ SIG cells (NaN rows, rounding) | warning with a count; never a failure |
| Snapshot vs redraw | `/ask` uploads the snapshot bytes taken when the polygon closed |
| Other-file findings in a shared workspace | flagged «other file, context only» in O#; the demo compose gives SIG its own workspace |
| LLM wording flakes | the gate stays at two headings; tool pick and section names are observations |
| Assumption | ADS `.env` stays on LiteLLM with the guardrail model fix; one user, state in memory (as the README says) |
| Out of scope | three-step intake UI, experiments folder, cell table, slice compiler, zone registry, union, glossary, shapely in ADS, kernel preload, four-heading gate, second dataset pair, antimeridian polygons, any SIG change, searching the graph by node/edge type |

## 15. Definition of Done
«Демо-дані» (with the type select) → polygon → «Запитати» yields:
1. a SIG answer produced from the user's question and the zone (or the preview with the fallback stated);
2. `base_aoi.csv` and `context_aoi.csv` uploaded to ADS with `h3_r{R}`, `h3_r{R-1}`, `h3_r{R-2}`, `in_aoi`, `aoi_zone`, taken from the polygon-close snapshot;
3. an ADS report that groups by those columns or by `in_aoi` (the tool's pick and ring/parent comparisons are recorded as observations);
4. the existing two-heading joint report, with the O# block in its prompt;
5. `uv run app.py check` green; `make check` green on the ADS branch; `docker compose up -d --build --wait` from `aoi-demo/` brings up the stack with the branch images.
