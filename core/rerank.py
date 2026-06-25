"""LLM rerank: reorder the hard-filtered, semantically-scored candidates given the
full PreferenceProfile (the SOTA candidate-retrieval -> LLM-rerank pattern).

The rerank never drops or adds items — it only reorders the survivors — so it cannot
violate a hard constraint (those were already enforced in retrieval).
"""

from __future__ import annotations

from core.llm import LLMProvider, get_provider
from core.models import PreferenceProfile, Product, RerankResult


def _prompt(profile: PreferenceProfile, candidates: list[tuple[Product, dict]]) -> str:
    lines = []
    for product, _ in candidates:
        lines.append(
            f"- {product.id}: {product.title} | {product.metal} | "
            f"{product.stone_primary or 'no stone'} | ₹{product.price:,} | {product.category}"
        )
    return (
        "Rank these jewellery candidates best-first for the shopper. "
        "Return only an ordering of their ids.\n"
        f"Shopper: {profile.raw_text}\n"
        f"Preferred styles: {profile.styles}; metals: {profile.metal_prefs}; "
        f"stones: {profile.stone_prefs}; occasion: {profile.occasion}\n"
        "Candidates:\n" + "\n".join(lines)
    )


def rerank(
    profile: PreferenceProfile,
    candidates: list[tuple[Product, dict]],
    provider: LLMProvider | None = None,
    top_k: int = 5,
) -> list[tuple[Product, dict]]:
    if not candidates:
        return []
    provider = provider or get_provider()
    cand_ids = [p.id for p, _ in candidates]

    result: RerankResult = provider.complete_structured(
        task="rerank",
        schema=RerankResult,
        prompt=_prompt(profile, candidates),
        context={"candidate_ids": cand_ids, "profile": profile.model_dump()},
    )

    valid = set(cand_ids)
    order = [i for i in result.ranked_ids if i in valid]
    seen = set(order)
    order += [i for i in cand_ids if i not in seen]  # never drop a survivor
    by_id = {p.id: (p, s) for p, s in candidates}
    return [by_id[i] for i in order][:top_k]
