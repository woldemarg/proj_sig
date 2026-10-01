# SDD 12 — LLM interface and grounded QA

## Purpose
Use a local Gemma 4 model to verbalise retrieved evidence into a grounded answer. The model is never the knowledge store and never discovers statistics. The interface stays replaceable.

## Scope
`ltir/llm.py` (`LLMClient` protocol — the type of `Engine.llm` —, `OpenAICompatibleLLM`, `SYSTEM_PROMPT`), `ltir/qa.py` (`answer_question`, `compute_baselines`, `check_citations`, `QAResult`).

## Inputs
Question, `Engine` (graph, encoder, vectors, config, LLM client).

## Outputs
`QAResult(question, answer, answer_mode ∈ {llm, fallback, empty}, llm{model, ok, latency_s, error, usage}, citations{cited[{key, pattern_id}], unknown, uncited, grounded}, evidence (dict + "prompt"), traversal (dict incl. baselines), highlight{seeds, traversed, anchors, evidence, edges, transversal_only}, metrics{retrieval_s, total_s, llm_latency_s, seed_count, traversal_depth, visited_states, retrieved_evidence, anchors_visited, prompt_chars}, provenance_footer)`.

## Dependencies
`httpx`. Any OpenAI-compatible chat endpoint. **Configured deployment: OpenRouter**, the same settings as `spectr/agentic-data-science` (`GEMMA_*` there): `LLM_BASE_URL=https://openrouter.ai/api/v1`, `LLM_MODEL=google/gemma-4-26b-a4b-it`, `LLM_PROVIDER_ORDER=dekallm/bf16,parasail/bf16,nextbit/bf16`, Bearer `LLM_API_KEY`. Code defaults remain local Ollama (`http://localhost:11434/v1`, `gemma4`); LM Studio also works (`http://localhost:1234/v1`, `google/gemma-4-26b-a4b`).

## Algorithms
1. `parse_query` → `resolve_seeds` → `traverse` → baselines → `build_evidence` (SDD 10, 11).
2. If the evidence has items, `generate(SYSTEM_PROMPT, evidence.to_prompt())` is called. The prompt rules: use only the evidence; cite `[P#]`; separate **Observations:** (verified statistics) from **Interpretation (hypotheses):**; no causal claims ("is associated with"); point out cross-scope analogues reached via latent anchors; say when the evidence does not answer.
3. Request: `POST {base}/chat/completions` with headers `Authorization: Bearer <key>` and `X-Title: <LLM_APP_TITLE>`, and body `{model, messages[system, user], temperature, max_tokens, stream:false[, reasoning_effort][, provider: {order: LLM_PROVIDER_ORDER, allow_fallbacks: false}]}`. The provider block matches spectr's `GemmaChatModel`. Health: `GET {base}/models` (10 s timeout), cached for 15 s. When the endpoint is known to be down, generation fails fast.
4. On LLM failure, empty output or `use_llm=false`, the answer is `Evidence.summary()` (SDD 11): cited observations, interpretation marked as not generated. `answer_mode=fallback` and the reason is in `llm.error` (`disabled` when switched off); the answer text itself carries no status prefix — clients read `answer_mode` / `llm.error`.
5. `check_citations`: extract `[P#]` and grouped `[P1, P3]` / `[P1; P3]` citations from the answer. `grounded` = at least one citation and no unknown keys. `provenance_footer` is built deterministically from the evidence (key → pattern, expression, dataset, file, batch), independent of the LLM.
6. Log the query to `logs/queries.jsonl` (question, mode, metrics, seeds, evidence, citations).

## Configuration
The model runs outside SIG (OpenRouter, or a local server with CPU/partial offload): SIG never loads it in-process, so it does not compete with the embedder for the 8 GB GPU (SDD 06). `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_PROVIDER_ORDER` (""), `LLM_APP_TITLE` (`SIG LTIR`), `LLM_TIMEOUT_S` (120), `LLM_TEMPERATURE` (0.1), `LLM_MAX_TOKENS` (1200), `LLM_REASONING_EFFORT` (unset; `none` disables Gemma "thinking" on endpoints that support it).

## Failure modes
LLM unavailable, HTTP error or empty completion → fallback answer; statistics and graph are untouched. Empty graph → `answer_mode=empty`. A workspace built with another representation (fingerprint ≠ the encoder's spec) → `RepresentationMismatch` before retrieval (HTTP 409 with the reset hint), never a silent cross-frame comparison.

## Invariants
The LLM never receives raw data or the whole graph, only `Evidence.to_prompt()`. Every answer carries the deterministic provenance footer, and citations are validated against the evidence keys.

## Testing requirements
`tests/test_llm_client.py`: the real HTTP client against an OpenAI-compatible stub (payload shape, provider pinning, health, reasoning flag), fail-fast on an unreachable endpoint, citation validation including grouped citations. Tests never read `sig/.env` (`LTIR_NO_DOTENV=1`), so no request reaches OpenRouter from the suite. `tests/test_e2e.py`: grounded LLM path with a deterministic fake, fallback on failure, empty graph.

## Integration points
`POST /api/query` (SDD 13), `python -m ltir query`, `python -m ltir llm-check`.

## Current implementation status
Implemented and **validated live** against `google/gemma-4-26b-a4b-it` on OpenRouter. `scripts/eval_answers.py` (5 fixed demo questions, Gemma's own token counts):

| run | grounded | unknown citations | citations | prompt tokens | completion tokens | mean latency |
|---|---|---|---|---|---|---|
| before (symbol prompt, MiniLM) | 5/5 | 0 | 34 | 14,983 | 1,881 | 9.0 s |
| ASCII prompt (`ltir-canon-3`), MiniLM | 5/5 | 0 | 38 | 11,705 | 1,668 | 6.0 s |
| ASCII prompt + Qwen3 embedder (current) | 5/5 | 0 | 40 | 11,099 | 1,728 | 6.6 s |

Latency is OpenRouter-dependent (noisy). Answers keep the planted directions (lower margin, higher discount for US phones; the EU∧phones correlation −0.57 → +0.09). Earlier manual checks: `llm-check` returns in 1.4 s, and the demo question gets a grounded answer in 7–11 s. That answer has separate Observations / Interpretation (hypotheses) sections and cites 7 evidence keys, including the scope-disjoint analogues reached via anchor A-4. Gemma writes grouped citations (`[P4, P7]`); these are parsed and linked.
