# SDD 02 — Data ingestion

## Purpose
Turn an uploaded file into a validated `pandas.DataFrame` with a stable dataset identity, before any statistics run.

## Scope
`ltir/ingestion.py`: file-type validation, parsing, optional categorical overrides (code-like columns), optional quantile-band dimensions, basic schema checks, dataset fingerprint. Upload transport (HTTP multipart, CLI path) is covered in SDD 13 and SDD 14.

## Inputs
| Input | Type | Source |
|---|---|---|
| `path` | file path (`.csv`, `.tsv`, `.txt`, `.parquet`) | UI upload saved under `WORKSPACE_DIR/uploads/`, or CLI argument |
| `filename` | display name | upload metadata |
| `bins` | `"col:q,col2:q"`, `""` (none) or `None` (→ workspace `BIN_COLUMNS`) | UI field / CLI `--bins` |
| `categories` | `"col,col2"`, `""` or `None` (→ workspace `CATEGORICAL_COLUMNS`) | UI field "Treat as categories" / CLI `--categories` |

## Outputs
`LoadedDataset(frame, dataset_id, filename, bins, categories, derived_columns, warnings)`.

## Dependencies
pandas (+ pyarrow for Parquet when installed).

## Data contracts
* `dataset_id = "ds-" + sha256(file bytes ‖ repr(sorted(bins)) [‖ repr(sorted(categories)) when any apply])[:12]`. The same content with the same derivation always gets the same id, which is the basis of idempotent ingestion (SDD 14).
* Categorical override: a named column is cast to text (`Int64` first when it holds integral floats, so `1.0` → `"1"`; NaN → `missing`). This is how integer-coded dimensions (`Store` 1..45, `Holiday_Flag` 0/1) become scope columns instead of being read as metrics or dropped as identifiers.
* Derived band column `<col>_band` with values `q1..qk` (quantile bins, `duplicates="drop"`), plus `missing` for NaN. **The source numeric column is dropped**, so the band acts as a scope dimension and cannot become a tautological target (for example `median_income` shifting inside `median_income_band=q4`).

## Algorithms
1. Reject unknown extensions → `unsupported_file`. Reject files larger than `MAX_UPLOAD_MB` → `unsupported_file`.
2. Parse. CSV uses delimiter sniffing (`sep=None`, python engine); TSV uses a tab; Parquet uses `read_parquet`. Parser errors → `unreadable_file`.
3. Fewer than 2 columns, or fewer than `MIN_ROWS` rows → `invalid_schema`.
4. Apply categorical overrides, then bands. `select_present` resolves both options the same way: workspace defaults (`None`) are applied **leniently** (columns the dataset lacks are skipped, with one warning listing them); an explicit option is **strict** (unknown column → `invalid_options`). A band column that is not numeric is always `invalid_options`.
5. No numeric column → `no_numeric_targets`. All-null columns produce a warning.

Semantic profiling (identifier drop, categorical/numeric typing, redundancy pruning) is **not** re-implemented here; it is EDA `step1_profile_data` (SDD 03).

## Configuration
`MIN_ROWS` (50), `MAX_UPLOAD_MB` (200), `BIN_COLUMNS` (""), `CATEGORICAL_COLUMNS` ("").

## Failure modes
`IngestionError(code)` with code ∈ {`unsupported_file`, `unreadable_file`, `invalid_schema`, `invalid_options`, `no_numeric_targets`}. The pipeline stores it in the batch record as `error.code`/`error.message` (status `FAILED`, stage `VALIDATING`).

## Invariants
* `dataset_id` is a pure function of bytes, band options and applied categorical overrides.
* The frame handed to discovery has the same row order as the file, so EDA row positions equal file rows.

## Testing requirements
`tests/test_persistence.py::test_ingestion_failure_states` covers `unsupported_file` and `invalid_schema`. Band derivation is exercised by the `housing.csv` run (README §Realistic dataset).

## Integration points
Called by `Engine.process()` (SDD 14). The file is copied to `datasets/<dataset_id>/source.<ext>` for provenance.

## Current implementation status
Implemented. Pandas' default NA parsing applies: literal `NA` becomes missing, which is why the synthetic region is named `US`, not `NA`. Diagnosed failures that motivated the overrides: `WA_Fn-UseC_-HR-Employee-Attrition.csv` failed with `no_candidates` because a constant column (`Over18`) made the EDA's overlap rule drop every other categorical (fixed in step 1, SDD 03) and its remaining dimensions are integer-coded; `Walmart.csv` has only integer-coded dimensions (`Store`, `Holiday_Flag`). Both now process with `--categories` / the UI field, and `Walmart_adapted.csv` (text store ids plus a `Month` column) processes without options.
