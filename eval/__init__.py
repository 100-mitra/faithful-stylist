"""Honest evaluation harness: brief sets, the LLM-as-judge, the order-shuffled pairwise
comparison, and metrics.

Objective metrics (groundedness, constraint-satisfaction, retrieval validity) are
measured directly. Subjective relevance is reported ONLY as an LLM-judge estimate with
N and an explicit caveat — never as validated accuracy.

Implemented in Phase 1 step 10.
"""
