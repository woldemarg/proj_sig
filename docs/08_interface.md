# 8. Interface — web UI, API and the latent sphere

> **In one paragraph.** A local single-page app lets a person add tables (and remove them again), follow their processing, explore the dual graph as three views of one knowledge base — a two-plane graph, a 3D latent sphere and an insight table — under one shared legend whose link entries switch the same layers in the graph and on the sphere, inspect any node and chat with the data in Ukrainian, with every data literal left exactly as the data holds it. After an answer, the parts that produced it are highlighted with the same markers in both views: the seed, the themes passed through, the evidence, the scope-disjoint analogues and the exact edges. The UI reads the persisted snapshot through a small HTTP API; the only knowledge it changes is a dataset it deletes, through the engine.

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
| `GET /api/graph?dataset=` | optional dataset filter | Cytoscape elements: `nodes[{data: id, kind, label, full_label; Pattern: weight, direction, ptype, dataset, support, target, effect, scope[], anchor, anchor_label; Attractor: mass, n_patterns}]`, `edges[{data: id, source, target, type, plane, weight, weak}]`, `stats` |
| `GET /api/nodes/{id}` | — | `{node (all props), neighbors {TYPE / "TYPE (in)": [{id, label, kind, weight, props}]}}` (404 for an unknown id) |
| `POST /api/query` | `{question, use_llm}` | `QAResult` ([7.5](07_question_answering.md#75-the-language-model-and-citation-check)); 400 for an empty question, 409 for a representation mismatch |
| `GET /api/sphere?dataset=` | optional dataset filter | the 3D sphere page (Plotly served locally) |
| `POST /api/sphere` | `{dataset, highlight, palette, layers}` | the same page with a retrieval highlight, drawn in the UI's theme colours with the legend's layer state (loaded into an iframe `srcdoc`) |
| `DELETE /api/datasets/{id}` | — | removes the dataset from the knowledge base ([6.8](06_graph_and_storage.md#68-deleting-a-dataset)): `{dataset_id, batches, patterns_removed, anchors_removed, neo4j}`; 404 for an unknown id, 409 while one of its batches runs |
| `GET /vendor/plotly.min.js` | — | plotly.js from the installed `plotly` package (offline) |
| `POST /api/reset` | — | wipes the workspace and, with `NEO4J_ENABLED`, clears the mirror: `{status, neo4j}`; 409 while a batch is running |

A sphere that cannot be drawn (fewer than four points) answers 200 with a message page. A representation mismatch is a 409 on `/api/query`; it surfaces as a server error on `/api/graph`, `/api/health` and `/api/nodes` when an outdated snapshot has to be rebuilt.

## 8.2 The single-page UI

Plain HTML, JavaScript and a design-token stylesheet with light and dark themes; no framework; Cytoscape.js 3.30.2 is vendored so the UI works offline.

* **Top bar** — health pills (LLM reachable / model found, Neo4j mirror on or off, embedding model and device), theme toggle (remembered in the browser).
* **Datasets rail** (left) — a drop zone for CSV, TSV, TXT and Parquet files; *Column options*: *Treat as categories* (`categories`) and *Split numbers into bands* (`bins`); *Analyse* and *Try demo*. One card per dataset (its READY batch, otherwise its latest attempt). While running: a progress bar and the stage in plain words ("Searching subgroups… step 4 of 10"). When READY: insights, themes first learned from it and rows, then columns, subgroups tested, filtered out, duration and dimension tags. When FAILED: a plain-language title, the cause and a tip per error code (for `no_candidates`: mark number-coded columns as categories). Clicking a READY card focuses all three views on that dataset — the dataset → insights → themes chain is the same filter everywhere. A **Delete** button at the foot of every finished card — READY, FAILED or SKIPPED — asks for confirmation and calls `DELETE /api/datasets/{id}` (the batch id for an upload that failed before it had a dataset id): the backend removes the dataset ([6.8](06_graph_and_storage.md#68-deleting-a-dataset)) and the rail and the views reload from it, so what is shown is always what is stored. The rail polls every 0.9 s while a batch runs and reloads the views when one becomes READY.
* **Canvas** (centre) with the view tabs **Graph | 3D Sphere | Insights** — three views of the same analytical state —, the dataset filter, the graph's layout selector (*Two planes* / *Force*), *Fit* (fits the graph, resets the sphere camera), *Legend* (shows or hides the legend column; remembered), *Clear highlight*, and the sphere's *Open ↗*. Left of the view sits the **legend column** ([8.3](#83-one-visual-language-for-the-graph-and-the-sphere)), the same for all three views: a caption (`52 insights · 7 themes in view`), the node classes, the link layers as switches and the answer markers, each with its count in the current view (the answer counts appear after a question). Hovering an entry spotlights its elements — the rest dims, in the graph and on the sphere.
* **Columns** — the borders between the datasets rail, the canvas and the chat can be dragged (rail ≥ 220 px, chat ≥ 300 px, the canvas keeps ≥ 380 px); widths are remembered in the browser, and a double-click on a border resets it.
  * *Graph* — the *Two planes* layout puts themes in a top row ("THEMES"), ordered by a greedy RELATED_TO chain so that linked themes sit side by side, with RELATED_TO as arcs labelled by weight; insights sit below ("INSIGHTS"), grouped under their strongest theme; Dimension and Metric nodes form a bottom row when the *Columns* layer is on. A *Force* layout is also available; switching layouts keeps the elements, the layer state and the highlight.
  * *Insights* — a sortable, filterable table (where, metric, effect, rows, evidence, theme); a row opens the details drawer.
  * *3D Sphere* — [8.4](#84-the-latent-sphere), in an iframe, as interactive as the graph: the legend's switches show and hide its link layers in place (Columns is disabled there: column nodes have no vectors), hovering a legend entry spotlights it, a click on an insight or a theme opens the same details drawer, evidence cards isolate their path; rotate, zoom and hover as usual, *Fit* resets the camera; *Open ↗* shows it un-highlighted in a new tab.
* **Details drawer** (node click, table row, citation or neighbour) — an insight opens with a one-sentence reading of its target ("discount is higher here — median 19.19 vs 10.74 overall (+2.21 sd)"), the shifts with meters and the evidence facts (rows and share, evidence weight, adjusted p, bootstrap stability, correlation change, confounders, aliases; a correlation-change insight leads with its correlation change, lists its median differences as *not validated*, and shows p as *not tested* and stability as *not measured*), then collapsible *Score breakdown* (weight factors, SD, EMM, volume), *Canonical representation* and *Provenance* (dataset, batch, selector, engine, rows reference, embedding spec). A theme shows its signature and mass, evidence mass and dispersion. Connections are grouped under plain names ("Belongs to theme", "Narrower insights", "Contrasts with", …) and are clickable.
* **Chat** (right) — a welcome card with suggested questions; user and assistant bubbles; a typing indicator while the graph is searched; the answer rendered from light Markdown (headings in either script) with clickable `[P#]` citations (grouped `[P1, P3]` too; the pattern id is in the tooltip) that open the drawer; chips for the model and latency (or "Evidence only"), "n sources cited" (or "No citations" when the check failed) and the cross-segment count; then three tabs, **Evidence & how it was found** open by default — a transversal-check line and one card per item (key, role badge *match* / *lattice* / *via theme*, an *other segment* badge, scope, the top two shifts — for a correlation-change insight its relationship instead —, rows, evidence, path chain) —, *Sources* (the provenance footer) and *Prompt* (the exact evidence the model saw). Follow-up suggestions are derived from the evidence (the first theme, the first cross-segment item, the seed's correlation pair). A notice above evidence-only answers comes from `answer_mode` / `llm.error`, never from the answer text. Enter sends, Shift+Enter breaks a line; *New chat* clears thread and highlight; a switch turns the LLM explanation off.

  **Language.** The application chrome — navigation, tabs, buttons, labels, legend, drawer, table, status texts — is English. The chat's natural language is Ukrainian: the suggested and follow-up questions, the model's answer (system prompt rule 7, [7.5](07_question_answering.md#75-the-language-model-and-citation-check)), the evidence-only summary and the transversal-check note. Every data literal stays exactly as the data holds it, in its own script — column names, values, dataset names, ids, `[P#]` keys: *"Чому margin нижчий для phones у US?"*, never a translated or transliterated `margin`. The parser matches those literals as typed and reads the Ukrainian direction and relationship words by stem ([7.1](07_question_answering.md#71-from-question-to-query)), so a Ukrainian question retrieves exactly what its English twin does.

## 8.3 One visual language for the graph and the sphere

The graph and the sphere are two projections of one snapshot, so one meaning has one label and one marker in both; the legend column is that contract, and its link entries are the layer switches for both views (the sphere page is a same-origin frame, so a switch restyles its traces in place without re-rendering). By default only **Hierarchy** and **Theme links** are on. Colours are the theme tokens of `style.css`; the sphere receives them with every render (`palette`), so it follows light and dark mode.

| Legend entry | Meaning | Graph | Sphere (trace name = legend label) |
|---|---|---|---|
| **Theme** | a latent anchor | purple hexagon, size ∝ members, labelled with its two strongest signature entries | purple diamond, size `8 + 1.2 · members`, labelled `A-k` (`Themes`) |
| **Metric higher** | an insight whose primary metric is higher in its subgroup | blue circle, size ∝ evidence weight | blue circle, size `3 + 6w` |
| **Metric lower** | … lower | orange circle | orange circle |
| **Correlation change** | a covariance insight | green rounded square | green square |
| **Hierarchy** (toggle, on) | SPECIALIZES: narrower subgroup of | grey arrow to the narrower insight; GENERALIZES hidden (its inverse) | grey line |
| **Contrasts** (toggle, off) | CONTRASTS: overlapping scope, opposite shift | red dashed line | red dashed line |
| **Siblings** (toggle, off) | SIBLING: same parent, another value | dotted line | dotted line |
| **Theme links** (toggle, on) | RELATED_TO: mutual nearest centroids | thick purple arc labelled with its weight | thick purple line |
| **Memberships** (toggle, off) | ACTIVATES: insight belongs to theme; dashed = weak (coverage only, not walked) | thin lilac line, width ∝ alignment, dashed when weak | thin lilac line, dashed when weak (`Memberships`, `Memberships (weak)`) |
| **Columns** (toggle, off) | Dimension and Metric nodes with HAS_SCOPE / TARGETS | grey tags in a bottom row | not drawn (no vectors); the toggle is disabled in the sphere view |
| **Seed** | an insight matched directly to the question | thick gold ring | gold ring |
| **Evidence** | an item shown to the model (cited or not) | thin ring in the graph's ink colour | ring in the ink colour |
| **Other segment** | a cross-segment item: reached only through a theme, no shared condition (`transversal_only`) | double red ring | red ring |
| **Answer path** | the edges of the paths used | thick gold edges | thick gold lines; themes on the path get a gold ring (`Themes visited`) |

After an answer everything else fades (graph: opacity; sphere: insight opacity and faint memberships) and the answer stands out with those markers; a visited node that is not evidence keeps full brightness without a ring. Clicking an evidence card isolates **that** path (seed → theme → insight) and zooms to it; in the sphere view it re-renders the sphere with that path. A path is drawn even when its link layer is switched off. If the dataset filter hides the evidence, the filter switches to the evidence's dataset first.

## 8.4 The latent sphere

The sphere is lac's visualisation, vendored as `ltir/engines/lac/projector.py`, fed with insight vectors and anchor centroids instead of text chunks and concepts. On the stacked matrix `[P ; A]` (one frame):

```text
per-feature robust scaling (median, IQR over the 5–95 % quantile range)  →  KernelPCA(kernel = cosine, 3 components)
→ centre by the mean  →  u = y / ‖y‖,  radius = minmax(log ‖y‖²) ∈ [0.1, 1]
```

Points lie inside the unit ball: the direction comes from the kernel PCA, the radius from the (log) spread. `sphere.sphere_figure(engine, dataset=, highlight=, palette=, layers=)` chains the projector's steps (`_apply_pca` → `_scale_vectors_on_sphere` → `_build_figure` without lac's edges) and draws the page in the graph's language ([8.3](#83-one-visual-language-for-the-graph-and-the-sphere)): one trace per insight class and per link layer, named like the legend entries (`sphere.LAYER_TRACES`), with the initial visibility of the toggles (`LAYER_DEFAULTS`, or the `layers` the UI sends), the themes as diamonds, faint great circles for depth, and with a highlight the answer-path lines and the seed, evidence, other-segment and themes-visited rings. The page has no title and no Plotly legend of its own — the legend column explains it — and uses the palette it is given (the UI's theme tokens; the standalone export and `GET /api/sphere` use the dark set, `sphere.DEFAULT_PALETTE`). Membership proximity, not colour, shows which theme an insight belongs to; hover text names the theme and the alignment.

It follows the dataset filter (the projection is refit on the subset) and re-renders after every answer and theme change; layer switches and the legend spotlight restyle the live page instead. Every insight and theme point carries its node id (`customdata`), which is how a click opens the drawer. After a READY batch a standalone copy is written to `workspace/graph/sphere.html` on a background thread, so the next upload never waits for it (only the latest of several queued exports runs; plotly from a CDN; `SPHERE_EXPORT`); CLI: `python -m ltir sphere [-o file] [--dataset id]`. The prosphera import (~5 s) is warmed in a background thread when the app starts. Three dimensions cannot preserve 1152-dimensional cosines: visual closeness on the sphere is a hint, the stored cosines are the truth ([5.7](05_latent_anchors.md#57-links-between-anchors)).

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

Tests: `test_upload_process_render_query` (static assets, upload, poll to READY, graph elements with the four main node kinds and eight edge types, node details, a Ukrainian question parsed to its literals with the highlight's edges resolving in the graph, deletion through the API emptying batches and graph, 404 afterwards, a failed upload deleted by its batch id, reset); `test_browser_renders_graph_and_highlights_path` (`browser` marker; headless Chromium found under the Playwright browser folder of `%LOCALAPPDATA%`, so it runs on Windows: renders Cytoscape and the READY card, lists every insight in the table, runs a Ukrainian suggestion, asserts highlighted edges, themes, seeds and evidence cards, the evidence tab open by default, the legend visible with only Hierarchy and Theme links on and its counts, the hover spotlight, the Memberships and Hierarchy switches changing the sphere traces and the graph edges, Columns disabled on the sphere, a sphere click opening the drawer, a dragged column border, opens the drawer from a citation, toggles the theme, deletes the dataset from its card and sees the views empty, no page errors); `tests/test_sphere.py` (the graph's classes and theme markers, every legend layer present with the toggle's initial state and switchable, no title or legend of its own, a light palette, the highlight traces, the standalone export, the sphere API with palette and layers, the served plotly).
