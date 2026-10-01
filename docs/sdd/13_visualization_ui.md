# SDD 13 — Visualisation and web UI

## Purpose
A local data-app for adding tables, following processing, exploring the dual-layer graph (graph, 3D sphere, insight table), inspecting nodes and chatting with the data, **with the graph parts used for an answer visibly highlighted**. Vocabulary in the UI is plain: attractors are "themes", patterns are "insights", robust z is shown as "±x sd".

## Scope
`ltir/web/app.py` (FastAPI), `ltir/web/static/{index.html, app.js, style.css, vendor/cytoscape.min.js}` (Cytoscape.js 3.30.2, vendored so the UI works offline), and `ltir/sphere.py` (3D latent sphere).

## Inputs / Outputs (HTTP API)
| Method, path | In | Out |
|---|---|---|
| `GET /` | — | the single-page UI |
| `GET /api/health` | — | `{llm (cached health), neo4j, embedding spec, graph stats, workspace}` |
| `GET /api/config` | — | public config (secrets masked) |
| `GET /api/batches`, `GET /api/batches/{id}` | — | batch records (status, stage_times, profile, metrics, warnings, error) |
| `POST /api/upload` | multipart `file`, form `bins`, `categories` (empty = workspace defaults) | 202 + batch record (`UPLOADED`); processing is queued on a single worker thread |
| `POST /api/demo` | — | 202; queues the synthetic dataset |
| `GET /api/graph?dataset=` | optional dataset filter | Cytoscape elements `{nodes[{data: id, kind, label, full_label; Pattern: weight, direction, ptype, dataset, support, target, effect, scope[], anchor, anchor_label; Attractor: mass, n_patterns}], edges[{data: id, source, target, type, plane, weight}], stats}` |
| `GET /api/nodes/{id}` | — | `{node (all props), neighbors{TYPE / "TYPE (in)": [{id, label, kind, weight, props}]}}` |
| `POST /api/query` | `{question, use_llm}` | `QAResult` (SDD 12) |
| `GET /api/sphere?dataset=` | optional dataset filter | 3D sphere page (Plotly, local `/vendor/plotly.min.js`) |
| `POST /api/sphere` | `{dataset, highlight}` | the same page with a retrieval highlight (the UI loads it into an iframe `srcdoc`) |
| `GET /vendor/plotly.min.js` | — | plotly.js served from the installed `plotly` package (offline) |
| `POST /api/reset` | — | wipes the workspace (409 while a batch is running) |

## UI behaviour (`index.html` + `app.js` + `style.css`: a design-token system with light/dark themes, no framework)
* **Top bar**: brand, health pills (LLM reachable / model found, Neo4j mirror on/off, embedding model and device), theme toggle (persisted in `localStorage`).
* **Datasets rail** (left): drag-and-drop zone (CSV/TSV/Parquet), *Column options* ("Treat as categories" → `categories`, "Split numbers into bands" → `bins`), *Analyse* / *Try demo*. One card per dataset (its READY batch if it has one, otherwise its latest attempt): while running, a progress bar and the stage in plain words ("Searching subgroups… step 4 of 10"); when READY, insights / themes / rows, subgroups tested, filtered out, duration and the dimension tags; when FAILED, a plain-language title, cause and tip per error code (e.g. `no_candidates` → mark number-coded columns as categories). Clicking a READY card filters the canvas to that dataset. The rail polls every 0.9 s while a batch runs and reloads the canvas when a batch becomes READY.
* **Canvas** (centre) with a view switch **Graph | Sphere 3D | Insights**, dataset filter, *Fit*, *Clear highlight*.
  * *Graph*: the *Two planes* layout puts themes on the top row ("THEMES"), ordered by a greedy RELATED_TO chain so related themes sit side by side, with RELATED_TO drawn as arcs labelled by weight; insights sit below ("INSIGHTS") grouped under their strongest theme; Dimension/Metric nodes form a bottom row when the *Columns* layer is on. A *Force* layout (cose) is also available. Layer toggles: Hierarchy (SPECIALIZES), Contrasts, Siblings, Theme links (RELATED_TO), Memberships (ACTIVATES), Columns. Legend in the corner.
  * *Insights*: a sortable, filterable table (where / metric / effect / rows / evidence / theme); a row opens the details drawer.
  * *Sphere 3D*: the lac projector page in an iframe (below).
* **Encoding**: theme = purple hexagon (size ∝ member count). Insight = circle sized by evidence weight, blue ↑ / orange ↓ (primary metric direction), green rounded square for correlation changes. SPECIALIZES = grey arrow; CONTRASTS = red dashed; SIBLING = dotted; ACTIVATES = thin lilac (width ∝ alignment); RELATED_TO = thick purple arc. GENERALIZES is hidden (the inverse of SPECIALIZES) but is highlighted when a path uses it.
* **Details drawer** (node click, table row, citation or neighbour). An insight opens with a one-sentence reading ("margin is lower here — median 14.8 vs 19.5 overall (−1.10 sd)"), the shifts with meters, the evidence facts (rows and share, evidence weight, adjusted p, bootstrap stability, correlation change, confounders, aliases), then collapsible *Score breakdown* (weight factors, SD/EMM/volume), *Canonical representation* and *Provenance* (dataset, batch, selector, engine, rows reference, embedding spec). A theme shows its signature and mass/evidence mass/dispersion; connections are grouped under plain names ("Belongs to theme", "Narrower insights", "Contrasts with", …) and are clickable.
* **Chat** (right): a notice above evidence-only answers derived from `answer_mode` / `llm.error` (never from the answer text); a welcome card with suggested questions; user and assistant bubbles; a typing indicator while the graph is searched; the answer rendered from light markdown with clickable `[P#]` citations (grouped `[P1, P3]` too) that open the drawer; chips for model + latency (or "Evidence only" on fallback), "n sources cited" (or "No citations") and the cross-segment count; *Evidence & how it was found* with a transversal-check line and one card per item (key, role badge match / lattice / via theme, "other segment" badge, scope, top shifts, rows, evidence, path chain); *Sources* (provenance footer); *What the model saw* (the evidence prompt); follow-up suggestions derived from the answer; Enter sends, Shift+Enter breaks a line; *New chat* clears the thread and the highlight; a switch turns the LLM explanation off (evidence-only answers).
* **Highlighting**: after an answer, everything else fades. Seeds get a gold border, themes visited a gold border, final evidence nodes a dark border, cross-segment evidence a red double ring, and used edges become thick amber lines. Clicking an evidence card isolates **that** path (e.g. seed → A-4 → insight) and zooms to it; on the sphere view it re-renders the sphere with that path. If the dataset filter hides the evidence, the filter switches to the evidence's dataset before highlighting.

## 3D latent sphere (view switch *Graph | Sphere 3D*)
Adopts lac's visualisation (`lac/v2_orchestrator/viz_export.py` → `lac/v1_single_pass/visualisation/projector.py`, vendored as `ltir/engines/lac/projector.py`) with the projection and figure code **unchanged** except that the unused entry point `project_ontology` and the single "Chunks" trace are removed (SIG draws the pattern points itself, below). That is prosphera's `KernelPCA(kernel="cosine")` on robust-scaled vectors → `_scale_vectors_on_sphere` (unit directions, magnitudes = minmax(log‖v‖²) in [0.1, 1]), lac's dark Plotly figure, ACTIVATES lines and `save_html`. SIG feeds it the **Pattern vectors** (the 1152-d journal rows, as stored in the ontology) and the **Attractor centroids** instead of text chunks and concepts, the way `viz_export.materialize` feeds chunks and concepts. Instead of `project_ontology`, `sphere.sphere_figure` chains the same steps (`_apply_pca` → `_scale_vectors_on_sphere` → `_build_figure`) so it can restyle the figure before `save_html`.

SIG additions on the figure (`ltir/sphere.py::sphere_figure`):
* the pattern points, as one legend group per latent anchor (colour = the pattern's strongest attractor, lac's Set3 palette; marker size ∝ insight weight; click to toggle);
* attractors labelled `A-k`, sized by pattern count;
* RELATED_TO (mutual kNN) lines, plus SPECIALIZES / CONTRASTS layers (off by default in the legend);
* faint unit-sphere great circles for depth;
* with a query highlight: amber **Retrieval path** segments for the used edges, **Seed** (diamond), **Evidence** (white ring), **Cross-scope evidence** (red ring), **Anchors visited** (amber ring); the rest dims.

Behaviour: it follows the dataset filter (the projection is refit on the selected subset) and re-renders after every answer. Clicking an evidence card shows only that path. "open sphere in new tab" gives the un-highlighted page. After every READY batch, a standalone copy is written to `workspace/graph/sphere.html` (lac `save_html`, plotly from CDN), the counterpart of lac's `ontology_sphere.html` (`SPHERE_EXPORT`, default true). CLI: `python -m ltir sphere [-o file] [--dataset id]`. The lac/prosphera import (~5 s) is warmed in a background thread at app start.

## Configuration
`WEB_HOST` (127.0.0.1), `WEB_PORT` (8765), `SPHERE_EXPORT` (true). The embedding model is warmed up at start; the health pill shows the embedding model and device (e.g. `cuda:0`).

## Failure modes
API errors are shown inline (upload message, an error bubble in the chat). Failed batches show a plain-language explanation plus the raw `error.message`. An offline LLM shows a red health pill and evidence-only answers with a notice. The graph view is derived from the persisted snapshot, so a UI failure cannot invalidate stored knowledge.

## Invariants
Every highlighted id exists in `/api/graph` (tested). The UI never writes knowledge; it only calls the API.

## Testing requirements
`tests/test_ui_smoke.py::test_upload_process_render_query`: static assets, upload, poll to READY, graph elements with all node kinds and edge types, endpoint consistency, node details, query highlight ids resolve in the graph, reset. `::test_browser_renders_graph_and_highlights_path` (`browser` marker): headless Chromium renders Cytoscape nodes and the READY dataset card, lists every insight in the table view, runs a suggested question, asserts highlighted edges/anchors/seeds and evidence cards, opens the drawer from a citation, toggles the theme, and sees no page errors. `tests/test_sphere.py`: one legend group per anchor, all patterns plotted inside the unit sphere, highlight layers, standalone export, sphere API and vendored plotly.

## Integration points
`python -m ltir.web` or `python -m ltir serve`. Uses `Engine` (SDD 14).

## Current implementation status
Implemented and verified in headless Chromium (screenshots taken during development; no console errors). The sphere was checked with 83 patterns and 17 attractors over three datasets, and with the retail subset plus a query highlight (all five highlight layers present).
