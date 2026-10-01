# SDD 05 — Insight canonicalisation

## Purpose
Serialise every insight into a canonical, versioned, six-section representation. Structural predicates (scope) and statistical behaviour (phenomenon) stay separately accessible and are never mixed in one string before embedding.

## Scope
`ltir/canonical.py`: `canonicalize()`, `phenomenon_shifts()`, `headline()`, `covariance_label()`.

## Inputs
`Insight` (SDD 04), `Config`, optional dataset row count.

## Outputs
`CanonicalInsight(insight_id, version, target, scope, phenomenon, covariance, confounders, support, components)`. `document()` renders the six sections.

## Data contract / serialisation (`CANONICAL_VERSION = "ltir-canon-2"`)

```text
TARGET: margin
SCOPE: category = laptops; region = EU
PHENOMENON: margin strong decrease (robust z -2.30; median 12.1 vs 18); discount moderate increase (robust z +1.20; ...)
COVARIANCE: stabilized correlation divergence 0.061; strongest pair discount ~ margin -0.57 -> -0.24
CONFOUNDERS: [payment] heavily skewed to 'cash' (JS: 0.20)      | "none detected"
SUPPORT: 560 rows (11.2% of 5,000); bootstrap stability 0.89; adjusted p 4.2e-28
```

| Section | Built from | Used by |
|---|---|---|
| TARGET | humanised `target` (`_` → space) | target embedding block |
| SCOPE | `attribute = value` joined by `; `, sorted | scope embedding block |
| PHENOMENON | phenomenon shifts: the target always, plus secondary shifts with \|z\| ≥ `MIN_COMPONENT_Z`; magnitude words mild < 1 ≤ moderate < 2 ≤ strong < 3 ≤ extreme | human/LLM reading |
| COVARIANCE | EMM score and the strongest divergent pair | reading |
| CONFOUNDERS | EDA drivers | reading, evidence |
| SUPPORT | support, stability, adjusted p | reading |

**Signed components** (`components`, consumed by the encoder): `(humanised metric, signed robust z)` for each phenomenon shift. When the insight is `covariance`-typed or `emm ≥ MIN_EMM_SCORE`, one more component is added: `("correlation between a and b", sign · EMM_COMPONENT_WEIGHT · emm / WEIGHT_EMM_REF)`, with sign +1 when \|corr\| strengthens and −1 when it weakens or reverses. For covariance insights only shifts with \|z\| ≥ `MIN_COMPONENT_Z` are kept, and the phrase "no material median shift" is added when none remain.

`headline(ins)` is a one-line label, e.g. `category=phones, region=US: discount ↑ (+2.21 z)`.

## Algorithms
Pure string/number formatting; no model calls.

## Configuration
`MIN_COMPONENT_Z` (0.5), `MIN_EMM_SCORE` (0.08), `EMM_COMPONENT_WEIGHT` (0.5), `WEIGHT_EMM_REF` (0.08) — EMM values are on the per-pair RMS scale of SDD 03.

## Failure modes
None expected. Insights always have ≥ 1 shift.

## Invariants
* Scope text contains only conditions, and phenomenon text never contains scope attributes.
* Output is deterministic for a given insight and config.
* `version` is stored with every pattern record. A change of format must bump `CANONICAL_VERSION`, which changes the embedding fingerprint (SDD 06) and prevents silent mixing.

## Testing requirements
`tests/test_canonical_embedding.py::test_canonical_sections_are_separate`, `::test_covariance_canonical_component`.

## Integration points
Output feeds `InsightEncoder.encode()` and is persisted in `patterns.jsonl` (`canonical` field, including `document`). The UI inspector shows the document, and the naive text-NN baseline embeds it.

## Current implementation status
Implemented.
