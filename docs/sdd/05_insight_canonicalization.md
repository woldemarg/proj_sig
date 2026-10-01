# SDD 05 — Insight canonicalisation

## Purpose
Serialise every insight into a canonical, versioned representation with **two separate contracts**: the *embedding inputs* (scope, target, signed components), which define the vector, and the *readable text* (a Markdown document and phrase helpers shared with the LLM prompt), which never enters the vector. Structural predicates (scope) and statistical behaviour (phenomenon) are never mixed in one string.

## Scope
`ltir/canonical.py`: `canonicalize()`, `phenomenon_shifts()`, `has_material_covariance()`, `headline()`, `covariance_label()`, and the readable-text helpers `format_value()`, `format_p()`, `describe_scope()`, `describe_shift()`, `describe_covariance()`. `ltir/models.py::CanonicalInsight`.

## Inputs
`Insight` (SDD 04), `Config`, optional dataset row count.

## Outputs
`CanonicalInsight(insight_id, version, target, scope, scope_sentence, phenomenon, covariance, confounders, support, components)`; `document()` renders the readable form.

## Data contract (`CANONICAL_VERSION = "ltir-canon-3"`)
**Embedding inputs** (unchanged across canon versions 2 → 3; SDD 06 embeds exactly these):
| Field | Rule | Example |
|---|---|---|
| `scope` | `attribute = value` joined by `; `, condition order (the closed intent, SDD 03) | `category = phones; region = US` |
| `target` | humanised target metric (`_` → space) | `discount` |
| `components` | `(humanised metric, signed robust z)` per phenomenon shift; plus `("correlation between a and b", sign · EMM_COMPONENT_WEIGHT · emm / WEIGHT_EMM_REF)` when the correlation change is material | `[("discount", 2.21), ("margin", -1.10)]` |

Labels never contain digits: the magnitude lives only in the coefficient (`test_embedding_labels_carry_no_numbers`). The phenomenon shifts are the target plus every shift with \|z\| ≥ `MIN_COMPONENT_Z` (covariance insights: only \|z\| ≥ `MIN_COMPONENT_Z`). The correlation change is material for covariance-typed insights and when `emm_score ≥ MIN_EMM_SCORE` (`has_material_covariance`); its sign is +1 when \|corr\| strengthens, −1 when it weakens or reverses.

**Readable text** (ASCII; number rules: values 4 significant digits below 1,000 and thousands separators above, never an exponent; shifts `±x.xx sd`; correlations `±0.xx`; p-values `< 0.001` / `< 0.01` / `< 0.05` / two decimals; shares one-decimal percent):
```text
### Subgroup finding P-bc4657a04746
* Scope: category is phones and region is US
* Target metric: discount
* Observed shift: discount: strong increase, +2.21 sd (median 19.19 vs 10.74 overall); margin: moderate decrease, -1.10 sd (median 14.75 vs 19.45 overall)
* Metric relationships: strongest change: correlation between delivery days and return rate weakens from +0.88 overall to +0.73 in the subgroup (divergence score 0.02)
* Confounders: none detected
* Validation: 438 rows (8.8% of 5,000); bootstrap stability 0.97; adjusted p < 0.001
```
| Field | Built from |
|---|---|
| `scope_sentence` | `describe_scope`: `a is x and b is y`; three or more: `a is x, b is y, and c is z` |
| `phenomenon` | `describe_shift` per phenomenon shift (`metric: <mild\|moderate\|strong\|extreme> <increase\|decrease>, ±z sd (median local vs global overall)`), the correlation phrase first for covariance insights and last otherwise; `no material median shift` when a covariance insight has none. Also the fallback text of the phenomenon block when all components cancel (SDD 06) |
| `covariance` | `strongest change: <describe_covariance> (divergence score x.xx)` or `no correlation pair (divergence score x.xx)` |
| `confounders` | the EDA driver strings, or `none detected` |
| `support` | `n rows (share of N); bootstrap stability s; adjusted p <bucket>` |

`headline(ins)` is the compact ASCII visual label (graph node `label`, Neo4j, UI tooltip): `category=phones, region=US: discount +2.21 sd`; covariance: `category=tablets, channel=online: corr(delivery days, discount) strengthens`.

## Algorithms
Pure string/number formatting; no model calls.

## Configuration
`MIN_COMPONENT_Z` (0.5), `MIN_EMM_SCORE` (0.08), `EMM_COMPONENT_WEIGHT` (0.5), `WEIGHT_EMM_REF` (0.08) — EMM values are on the per-pair RMS scale of SDD 03.

## Failure modes
None expected. Insights always have ≥ 1 shift.

## Invariants
* Embedding inputs contain only conditions (scope) or metric names (target, labels); no numbers, no prose.
* The readable text is ASCII and is a pure function of the insight and the config.
* `version` is stored with every pattern record. A change to either contract bumps `CANONICAL_VERSION`, which changes the fingerprint (SDD 06) and refuses old workspaces.

## Testing requirements
`tests/test_canonical_embedding.py`: `test_canonical_sections_are_separate` (embedding inputs, document sections, ASCII), `test_covariance_canonical_component`, `test_embedding_labels_carry_no_numbers`, `test_text_number_rules`.

## Integration points
Embedding inputs feed `InsightEncoder.encode()`; the record (including `document`, `scope_sentence`, `components`) is persisted in `patterns.jsonl` (`canonical`). The helpers render the LLM prompt (SDD 11) and the UI drawer shows the document; the naive text-NN baseline embeds the document.

## Current implementation status
Implemented; the document and the prompt are ASCII (measured: 0 non-ASCII characters in the demo prompt, SDD 17 §12).
