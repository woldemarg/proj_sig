"""SIG's LLM model broker: one OpenAI-compatible endpoint in front of the configured upstream model (docs/12_architecture.md §12.4).

A separate deployable (its own image; it imports nothing else from the repository). The narrator reaches the LLM
through it (``LLM_BASE_URL``); the upstream, the model, its key and the provider routing are configured here (``GEMMA_*``).
"""
