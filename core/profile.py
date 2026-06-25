"""Preference capture: freeform natural language -> structured PreferenceProfile.

The LLM parses into a strict Pydantic ProfileDraft; we then constrain to the controlled
vocab (fail-loud on junk), embed the brief, and handle cold-start (sparse input) with
sensible defaults + a clarifying follow-up.
"""

from __future__ import annotations

from core.embed import Embedder, get_embedder
from core.llm import LLMProvider, get_provider
from core.models import HardConstraints, PreferenceProfile, ProfileDraft
from core.vocab import METALS, STONES, STYLES

_CLARIFYING_QUESTION = (
    "Happy to help! To tailor this, could you share the occasion, a rough budget, "
    "and any metal or stone preferences (or a vibe like 'minimalist' or 'vintage')?"
)


def _prompt(raw_text: str) -> str:
    return (
        "Extract a structured jewellery preference profile from this request. "
        f"Use only these styles {STYLES}, metals {METALS}, stones {STONES}. "
        "Treat 'only'/'just' as a hard allow-list and 'no'/'without' as a hard exclusion; "
        "budgets are a hard maximum.\n"
        f"Request: {raw_text}"
    )


def _is_sparse(draft: ProfileDraft) -> bool:
    hc = draft.hard_constraints
    return not any(
        [
            draft.styles,
            draft.occasion,
            draft.budget_max,
            draft.metal_prefs,
            draft.stone_prefs,
            hc.allowed_metals,
            hc.excluded_metals,
            hc.allowed_stones,
            hc.excluded_stones,
            hc.require_no_stone,
            hc.categories,
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
    hc_in = draft.hard_constraints
    hard = HardConstraints(
        allowed_metals=[m for m in hc_in.allowed_metals if m in METALS],
        excluded_metals=[m for m in hc_in.excluded_metals if m in METALS],
        allowed_stones=[s for s in hc_in.allowed_stones if s in STONES],
        excluded_stones=[s for s in hc_in.excluded_stones if s in STONES],
        require_no_stone=hc_in.require_no_stone,
        categories=hc_in.categories,
        size=hc_in.size,
    )

    sparse = _is_sparse(draft)
    embedding = embedder.embed([raw_text])[0].tolist()

    return PreferenceProfile(
        raw_text=raw_text,
        styles=styles,
        occasion=draft.occasion,
        budget_max=draft.budget_max,
        metal_prefs=metal_prefs,
        stone_prefs=stone_prefs,
        recipient=draft.recipient,
        hard_constraints=hard,
        embedding=embedding,
        needs_clarification=sparse,
        clarifying_question=_CLARIFYING_QUESTION if sparse else None,
    )
