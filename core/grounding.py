"""GroundingVerifier — the headline feature (deterministic faithfulness veto).

Every *factual* claim in a rationale must be checkable against the recommended item's
catalog record; unsupported factual claims are blocked before reaching the user.
Pure, network-free, LLM-free.

Implemented in Phase 1 step 3.
"""
