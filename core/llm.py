"""LLM provider seam: a thin, swappable interface over the model provider.

Real runs use Anthropic Claude (Haiku for tagging/parse, Sonnet for rerank/rationale);
tests use a deterministic FakeProvider (no network, no key); a CassetteProvider can
replay recorded real responses. All calls are cached by a content hash for
reproducibility and cost control.

Implemented in Phase 1 step 2.
"""
