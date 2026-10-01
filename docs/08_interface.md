# 8. Interface — web UI, API and the latent sphere

> **In one paragraph.** A local single-page app lets a person add tables, follow their processing, explore the dual graph (a two-plane graph view, a 3D latent sphere and an insight table), inspect any node and chat with the data. After an answer, the parts of the graph that produced it are highlighted: the seed, the anchors passed through, the evidence, the scope-disjoint analogues and the exact edges. The UI only reads the persisted snapshot through a small HTTP API and never writes knowledge itself.

**Code** `ltir/web/app.py` (FastAPI), `ltir/web/static/` (`index.html`, `app.js`, `style.css`, `vendor/cytoscape.min.js`), `ltir/sphere.py`, `ltir/engines/lac/projector.py` · **Tests** `tests/test_ui_smoke.py`, `tests/test_sphere.py` · **Previous** [7. Question answering](07_question_answering.md) · **Next** [9. Operations](09_operations.md)

---

## 8.1 HTTP API

Start with `python -m ltir.web` (or `python -m ltir serve`), then open `http://127.0.0.1:8765`. One `Engine` serves all requests; uploads are processed by a single worker thread.

| Method, path | In | Out |
|---|---|---|
| `GET /` | — | the single-page UI |
| `GET /api/health` | — | `{llm (cached health), neo4j, embedding (the workspace's stored representation; null before the first batch), embedding_device, graph stats, workspace}` |
| `GET /api/config` | — | the public config (secrets masked) |
| `GET /api/batches`, `GET /api/batches/{id}` | — | batch records (404 for an unknown id) |
| `POST /api/upload` | multipart `file`, form `bins`, `categories` (empty = workspace defaults) | 202 + the `UPLOADED` record; processing is queued |
| `POST /api/demo` | — | 202; queues the synthetic dataset without options |
| `GET /api/graph?dataset=` | optional dataset filter | Cytoscape elements: `nodes[{data: id, kind, label, full_label; Pattern: weight, direction, ptype, dataset, support, target, effect, scope[], anchor, anchor_label; Attractor: mass, n_patterns}]`, `edges[{data: id, source, target, type, plane, weight}]`, `stats` |
| `GET /api/nodes/{id}` | — | `{node (all props), neighbors {TYPE / "TYPE (in)": [{id, label, kind, weight, props}]}}` (404 for an unknown id) |
| `POST /api/query` | `{question, use_llm}` | `QAResult` ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)); 400 for an empty question, 409 for a representation mismatch |
| `GET /api/sphere?dataset=` | optional dataset filter | the 3D sphere page (Plotly served locally) |
| `POST /api/sphere` | `{dataset, highlight}` | the same page with a retrieval highlight (loaded into an iframe `srcdoc`) |
| `GET /vendor/plotly.min.js` | — | plotly.js from the installed `plotly` package (offline) |
| `POST /api/reset` | — | wipes the workspace and, with `NEO4J_ENABLED`, clears the mirror: `{status, neo4j}`; 409 while a batch is running |

A sphere that cannot be drawn (fewer than four points) answers 200 with a message page. A representation mismatch is a 409 on `/api/query`; it surfaces as a server error on `/api/graph`, `/api/health` and `/api/nodes` when an outdated snapshot has to be rebuilt.

## 8.2 The single-page UI

Plain HTML, JavaScript and a design-token stylesheet with light and dark themes; no framework; Cytoscape.js 3.30.2 is vendored so the UI works offline.

* **Top bar** — health pills (LLM reachable / model found, Neo4j mirror on or off, embedding model and device), theme toggle (remembered in the browser).
* **Datasets rail** (left) — a drop zone for CSV, TSV, TXT and Parquet files; *Column options*: *Treat as categories* (`categories`) and *Split numbers into bands* (`bins`); *Analyse* and *Try demo*. One card per dataset (its READY batch, otherwise its latest attempt). While running: a progress bar and the stage in plain words ("Searching subgroups… step 4 of 10"). When READY: insights, new themes and rows, subgroups tested, filtered out, duration and dimension tags. When FAILED: a plain-language title, the cause and a tip per error code (for `no_candidates`: mark number-coded columns as categories). Clicking a READY card filters the canvas to that dataset. The rail polls every 0.9 s while a batch runs and reloads the canvas when one becomes READY.
* **Canvas** (centre) with a view switch **Graph | Sphere 3D | Insights**, dataset filter, *Fit*, *Clear highlight*.
  * *Graph* — the *Two planes* layout puts themes in a top row ("THEMES"), ordered by a greedy RELATED_TO chain so that linked themes sit side by side, with RELATED_TO as arcs labelled by weight; insights sit below ("INSIGHTS"), grouped under their strongest theme; Dimension and Metric nodes form a bottom row when the *Columns* layer is on. A *Force* layout is also available. Layer toggles: Hierarchy (SPECIALIZES), Contrasts, Siblings, Theme links (RELATED_TO), Memberships (ACTIVATES), Columns.
  * *Insights* — a sortable, filterable table (where, metric, effect, rows, evidence, theme); a row opens the details drawer.
  * *Sphere 3D* — [8.4](#84-the-latent-sphere), in an iframe; *Open ↗* shows it un-highlighted in a new tab.
* **Details drawer** (node click, table row, citation or neighbour) — an insight opens with a one-sentence reading ("margin is lower here — median 14.8 vs 19.5 overall (−1.10 sd)"), the shifts with meters and the evidence facts (rows and share, evidence weight, adjusted p, bootstrap stability, correlation change, confounders, aliases; a correlation-change insight lists its median differences as *not validated* and shows p and stability as not tested), then collapsible *Score breakdown* (weight factors, SD, EMM, volume), *Canonical representation* and *Provenance* (dataset, batch, selector, engine, rows reference, embedding spec). A theme shows its signature and mass, evidence mass and dispersion. Connections are grouped under plain names ("Belongs to theme", "Narrower insights", "Contrasts with", …) and are clickable.
* **Chat** (right) — a welcome card with suggested questions; user and assistant bubbles; a typing indicator while the graph is searched; the answer rendered from light Markdown with clickable `[P#]` citations (grouped `[P1, P3]` too; the pattern id is in the tooltip) that open the drawer; chips for the model and latency (or "Evidence only"), "n sources cited" (or "No citations" when the check failed) and the cross-segment count; *Evidence & how it was found* — a transversal-check line and one card per item (key, role badge *match* / *lattice* / *via theme*, an *other segment* badge, scope, top shifts, rows, evidence, path chain); *Sources* (the provenance footer); *What the model saw* (the prompt). Follow-up suggestions are derived from the evidence (the first anchor, the first cross-segment item, the seed's correlation pair). A notice above evidence-only answers comes from `answer_mode` / `llm.error`, never from the answer text. Enter sends, Shift+Enter breaks a line; *New chat* clears thread and highlight; a switch turns the LLM explanation off.

## 8.3 Visual encoding and highlighting

| Element | Encoding |
|---|---|
| theme (anchor) | purple hexagon, size ∝ member count |
| insight | circle sized by evidence weight; **blue** = the primary metric is higher in the subgroup, **orange** = lower; green rounded square = correlation-change insight |
| SPECIALIZES | grey arrow |
| CONTRASTS | red dashed line |
| SIBLING | dotted line |
| ACTIVATES | thin lilac line, width ∝ alignment |
| RELATED_TO | thick purple arc labelled with its weight |
| GENERALIZES | hidden (the inverse of SPECIALIZES), but highlighted when a path uses it |

After an answer everything else fades and the answer's parts stand out:

| Highlight | Meaning |
|---|---|
| thick gold ring on an insight | a seed — matched directly to the question |
| gold ring on a theme | an anchor the walk passed through |
| thin light ring | an evidence item the answer cites |
| double red ring | a cross-segment evidence item: reached only through a theme, sharing no condition with the seed (`transversal_only`) |
| full brightness, no ring | visited by the walk but not in the evidence |
| thick amber edges | the edges of the paths used |

Clicking an evidence card isolates **that** path (seed → anchor → insight) and zooms to it; in the sphere view it re-renders the sphere with that path. If the dataset filter hides the evidence, the filter switches to the evidence's dataset first. The corner legend shows the fill colours and the line types; the ring meanings above are not in the legend.

## 8.4 The latent sphere

The sphere is lac's visualisation, vendored as `ltir/engines/lac/projector.py`, fed with insight vectors and anchor centroids instead of text chunks and concepts. On the stacked matrix `[P ; A]` (one frame):

```text
per-feature robust scaling (median, IQR over the 5–95 % quantile range)  →  KernelPCA(kernel = cosine, 3 components)
→ centre by the mean  →  u = y / ‖y‖,  radius = minmax(log ‖y‖²) ∈ [0.1, 1]
```

Points lie inside the unit ball: the direction comes from the kernel PCA, the radius from the (log) spread. `sphere.sphere_figure` chains the projector's steps (`_apply_pca` → `_scale_vectors_on_sphere` → `_build_figure`) and restyles the figure:

* insight points as one legend group per anchor (colour = the insight's strongest anchor, ColorBrewer Set3; marker size `3 + 6w`; click a legend entry to toggle);
* anchors labelled `A-k`, sized `8 + 1.2 · n_patterns`;
* RELATED_TO lines, plus SPECIALIZES and CONTRASTS layers (off by default);
* faint great circles of the unit sphere for depth;
* with a query highlight: amber **Retrieval path** segments, **Seed** (diamond), **Evidence** (white ring), **Cross-scope evidence** (red ring), **Anchors visited** (amber ring); the rest dims.

It follows the dataset filter (the projection is refit on the subset) and re-renders after every answer. After every READY batch a standalone copy is written to `workspace/graph/sphere.html` (plotly from a CDN; `SPHERE_EXPORT`); CLI: `python -m ltir sphere [-o file] [--dataset id]`. The prosphera import (~5 s) is warmed in a background thread when the app starts. Three dimensions cannot preserve 1152-dimensional cosines: visual closeness on the sphere is a hint, the stored cosines are the truth ([5.7](05_latent_anchors.md#57-links-between-anchors)).

## 8.5 Text shown to people

| Surface | Code | Format | Example |
|---|---|---|---|
| headline — graph node `label`, Neo4j `label`, evidence `headline` | `canonical.headline` | ASCII: `<attr>=<value>, …: <target> <±z> sd`; covariance `<scope>: corr(<a>, <b>) strengthens|weakens|reverses` | `category=phones, region=US: discount +2.21 sd` |
| graph node label (`/api/graph`; `full_label` = headline) | `web.app._short_label` | the condition **values** joined by ` · `, a line break, `<target> <±z> sd` or `corr(<a>, <b>)` | `phones · US` / `discount +2.21 sd` |
| table and drawer | `app.js` | scope tags `attr = value`, metric `human(target)`, effect `±z sd`; drawer lead `<metric> is higher/lower here — median <local> vs <global> overall (<±z> sd)`; covariance: `The relationship between <a> and <b> changes here: correlation <global> overall → <local> in this subgroup.` | |
| anchor label | `graph._attractor_nodes` | `discount ↑ · margin ↓` | [5.8](05_latent_anchors.md#58-how-an-anchor-is-described) |
| sphere hover, insight | `sphere._pattern_hover` | id, `Scope: category=phones ∧ region=US`, `Target: discount (shift)`, up to three shifts `metric: local vs global (z ±x)` (raw column names), support · weight, anchor + alignment, dataset file | |
| sphere hover, anchor | `sphere._attractor_hover` | `Latent anchor A-k`, label, `Patterns: n over d scopes`, `Mass: m · evidence mass e`, `Dimensions: …` | |

The node label shows condition values without their columns, so `q2 · NEAR BAY` does not say whether `q2` is an income or an age band; the drawer and the headline do. UI vocabulary (glossary [11.1](11_reference.md#111-glossary)): attractor → *theme*, pattern → *insight*, `insight_weight` → *evidence*, robust z → `±x sd`, ACTIVATES → *membership*, RELATED_TO → *theme link*, transversal-only → *other segment*; `human()` replaces underscores in column names. Ids (`P-…`, `A-k`) stay visible in path chains so that the UI, the prompt and the journal agree.

## 8.6 Configuration, failure modes and tests

`WEB_HOST` (127.0.0.1), `WEB_PORT` (8765), `SPHERE_EXPORT` (true). At start the embedding model is warmed up and free GPU memory is logged.

API errors are shown inline (upload message, an error bubble in the chat). Failed batches show the plain-language explanation and the raw `error.message`. An offline LLM shows a red pill and evidence-only answers with a notice. Because the views are derived from the persisted snapshot, a UI failure cannot invalidate stored knowledge.

Tests: `test_upload_process_render_query` (static assets, upload, poll to READY, graph elements with the four main node kinds and eight edge types, node details, query highlight edges resolve in the graph, reset); `test_browser_renders_graph_and_highlights_path` (`browser` marker; headless Chromium found under the Playwright browser folder of `%LOCALAPPDATA%`, so it runs on Windows: renders Cytoscape and the READY card, lists every insight in the table, runs a suggested question, asserts highlighted edges, anchors, seeds and evidence cards, opens the drawer from a citation, toggles the theme, sees no page errors); `tests/test_sphere.py` (one legend group per anchor, the first anchor's points inside the unit cube, the highlight layers, the standalone export, the sphere API and the served plotly).
