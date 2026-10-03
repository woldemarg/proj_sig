"""SIG's LLM gateway: one OpenAI-compatible endpoint in front of the configured upstream model (docs/12_architecture.md §12.4).

A separate deployable (its own image, no dependency on ``ltir``). The backend reaches the LLM through it by default
(``LLM_BASE_URL``); the upstream, the model, its key and the provider routing are configured here (``GEMMA_*``).
"""
