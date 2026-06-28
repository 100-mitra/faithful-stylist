"""Preference capture: freeform natural language -> structured PreferenceProfile.

The LLM parses into a strict Pydantic ProfileDraft; we then constrain to the controlled
vocab (fail-loud on junk), embed the brief, and handle cold-start (sparse input) with
sensible defaults + a clarifying follow-up.
"""

from __future__ import annotations

from core.embed import Embedder, get_embedder
from core.llm import LLMProvider, get_provider
from core.models import HardConstraints, PreferenceProfile, ProfileDraft
from core.vocab import CATEGORIES, METALS, STONES, STYLES

_CLARIFYING_QUESTION = (
    "Happy to help! To tailor this, could you share the occasion, a rough budget, "
    "and any metal or stone preferences (or a vibe like 'minimalist' or 'vintage')?"
)


def _prompt(raw_text: str) -> str:
    return (
        "Extract a structured jewellery preference profile from this request. "
        f"Use only these styles {STYLES}, metals {METALS}, stones {STONES}. "
        "metal_prefs/stone_prefs are SOFT preferences. Use allowed_metals/allowed_stones "
        "only for a hard 'only X'/'just X' allow-list, and excluded_metals/excluded_stones "
        "for 'no'/'without'. budget_max is a hard maximum in INR (0 if none).\n"
        f"Request: {raw_text}"
    )


def _is_sparse(draft: ProfileDraft) -> bool:
    return not any(
        [
            draft.styles,
            draft.occasion,
            draft.budget_max,
            draft.metal_prefs,
            draft.stone_prefs,
            draft.allowed_metals,
            draft.excluded_metals,
            draft.allowed_stones,
            draft.excluded_stones,
            draft.require_no_stone,
            draft.categories,
        ]
    )


def parse_preference(
    raw_text: str,
    provider: LLMProvider | None = None,
    embedder: Embedder | None = None,
) -> PreferenceProfile:
    provider = provider or get_provider()
    embedder = embedder or get_embedder()

    draft: ProfileDraft = provider.complete_structured(
        task="parse_profile",
        schema=ProfileDraft,
        prompt=_prompt(raw_text),
        context={"raw_text": raw_text},
    )

    # Constrain to controlled vocab (fail-loud on out-of-vocab values by dropping them).
    styles = [s for s in draft.styles if s in STYLES]
    metal_prefs = [m for m in draft.metal_prefs if m in METALS]
    stone_prefs = [s for s in draft.stone_prefs if s in STONES]
    hard = HardConstraints(
        allowed_metals=[m for m in draft.allowed_metals if m in METALS],
        excluded_metals=[m for m in draft.excluded_metals if m in METALS],
        allowed_stones=[s for s in draft.allowed_stones if s in STONES],
        excluded_stones=[s for s in draft.excluded_stones if s in STONES],
        require_no_stone=draft.require_no_stone,
        categories=[c for c in draft.categories if c in CATEGORIES],
    )

    sparse = _is_sparse(draft)
    embedding = embedder.embed([raw_text])[0].tolist()

    return PreferenceProfile(
        raw_text=raw_text,
        styles=styles,
        # Map the flat draft's "unspecified" sentinels (""/0) back to None.
        occasion=draft.occasion or None,
        budget_max=draft.budget_max if draft.budget_max > 0 else None,
        metal_prefs=metal_prefs,
        stone_prefs=stone_prefs,
        recipient=draft.recipient or None,
        hard_constraints=hard,
        embedding=embedding,
        needs_clarification=sparse,
        clarifying_question=_CLARIFYING_QUESTION if sparse else None,
    )
