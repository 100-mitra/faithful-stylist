"""Grounded rationale generation.

The LLM picks WHICH facts to surface and writes opinion clauses (a RationaleDraft); code
renders the factual text from the record and the GroundingVerifier audits every claim.
The user-facing rationale is the verifier's safe text — any blocked factual claim is
removed before it can reach the user.
"""

from __future__ import annotations

from core.grounding import GroundingVerifier
from core.llm import LLMProvider, get_provider
from core.models import PreferenceProfile, Product, RationaleDraft, VerifierResult
from core.vocab import STYLES

_VERIFIER = GroundingVerifier()


def _prompt(profile: PreferenceProfile, product: Product, variant: str | None) -> str:
    base = (
        "Write a short recommendation rationale for this item.\n"
        "Rules: choose which factual fields to surface by name only (metal, price, "
        "stone_primary, stone_accent, carat, certification, category, under_budget) — do "
        "NOT write their values. Then add opinion clauses about style/occasion using "
        f"vocabulary like {STYLES}. Never state a number, price, metal or stone in the "
        "opinion text.\n"
    )
    # Two measured prompt iterations for the eval's pairwise comparison (brief §4).
    if variant == "v2":
        base += "Also note who/what occasion it suits, as a second opinion clause.\n"
    elif variant == "v1":
        base += "Keep it to a single concise opinion clause.\n"
    return base + f"Shopper: {profile.raw_text}\nItem id: {product.id}; title: {product.title}"


def build_rationale(
    profile: PreferenceProfile,
    product: Product,
    provider: LLMProvider | None = None,
    variant: str | None = None,
) -> VerifierResult:
    provider = provider or get_provider()
    context = {"product": product.model_dump(), "profile": profile.model_dump()}
    if variant:
        context["variant"] = variant
    draft: RationaleDraft = provider.complete_structured(
        task="rationale",
        schema=RationaleDraft,
        prompt=_prompt(profile, product, variant),
        context=context,
    )
    return _VERIFIER.verify(draft, product, profile)
