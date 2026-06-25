"""Embedder interface (text). Default is a dependency-light hashing/TF-IDF embedder
(offline, deterministic); sentence-transformers / Voyage are opt-in behind the same
interface. Grounding does not depend on embedding quality.

Implemented in Phase 1 step 4.
"""
