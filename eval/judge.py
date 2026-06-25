"""LLM-as-judge: a subjective-relevance estimate and an order-shuffled pairwise
comparison of two rationale prompt variants (prompt-iteration evidence, brief §4).

On the offline FakeProvider both judges are deterministic heuristics — reported as such,
never as validated accuracy.
"""

from __future__ import annotations

import hashlib

from core.llm import LLMProvider
from core.models import (
    JudgeVerdict,
    PairwiseVerdict,
    PreferenceProfile,
    Recommendation,
    VerifierResult,
)

JUDGE_PROMPT = (
    "You are judging how relevant a jewellery recommendation is to a shopper's request, "
    "on a 1-5 scale (5 = perfect fit). Consider style, occasion, metal/stone preference "
    "and budget. Return a score and whether it is relevant."
)

PAIRWISE_PROMPT = (
    "Two rationales (A and B) explain the same recommendation. Pick the one that is more "
    "helpful and specific to the shopper without overclaiming. Return the winner."
)


def judge_relevance(
    profile: PreferenceProfile, rec: Recommendation, provider: LLMProvider
) -> JudgeVerdict:
    return provider.complete_structured(
        task="judge",
        schema=JudgeVerdict,
        system=JUDGE_PROMPT,
        prompt=f"Request: {profile.raw_text}\nRecommendation: {rec.rationale}",
        context={"profile": profile.model_dump(), "product": rec.product.model_dump()},
    )


def _grounded_claim_count(result: VerifierResult) -> int:
    return sum(1 for c in result.claims if c.grounded and not c.blocked)


def judge_pairwise_variants(
    profile: PreferenceProfile,
    v1: VerifierResult,
    v2: VerifierResult,
    provider: LLMProvider,
    shuffle_key: str,
) -> str:
    """Compare v1 vs v2 with A/B order shuffled (position-bias neutralized).

    Returns the winning variant name ('v1' or 'v2').
    """
    # Deterministic, brief-specific A/B assignment so position cannot bias the result.
    v1_is_a = (int(hashlib.md5(shuffle_key.encode()).hexdigest(), 16) & 1) == 0
    a, b = (v1, v2) if v1_is_a else (v2, v1)

    verdict: PairwiseVerdict = provider.complete_structured(
        task="judge_pairwise",
        schema=PairwiseVerdict,
        system=PAIRWISE_PROMPT,
        prompt=f"Request: {profile.raw_text}\nA: {a.rationale_text}\nB: {b.rationale_text}",
        context={
            "a_claims": _grounded_claim_count(a),
            "b_claims": _grounded_claim_count(b),
            "a_text": a.rationale_text,
            "b_text": b.rationale_text,
        },
    )
    winner_is_a = verdict.winner == "A"
    # Map the A/B winner back to the variant, accounting for the shuffle.
    return "v1" if (winner_is_a == v1_is_a) else "v2"
