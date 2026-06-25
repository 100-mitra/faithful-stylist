"""Hybrid retrieval: apply hard constraints as SQL filters FIRST, then semantic
similarity over the survivors. Hard constraints (budget, excluded metals/stones,
size) are enforced here, never left to the LLM.

Implemented in Phase 1 step 7.
"""
