"""Deterministic natural-language parsing helpers (no LLM, no network).

These power two things:
  * the offline FakeProvider (so the whole pipeline runs end-to-end with no API key), and
  * cold-start / fallback parsing in profile.py.

The hard-constraint extraction here ("platinum only", "no gold", "no gemstones") is what
the adversarial constraint-satisfaction test exercises, so it is intentionally explicit.
"""

from __future__ import annotations

import re

from core.models import ProfileDraft, StyleTagDraft
from core.vocab import METALS, OCCASIONS, STONES, STYLES

GOLDS = ["yellow gold", "white gold", "rose gold"]

# Light synonym map onto the controlled occasion vocabulary.
_OCCASION_SYNONYMS = {
    "everyday": "daily wear",
    "daily": "daily wear",
    "day-to-day": "daily wear",
    "engagement": "engagement",
    "wedding": "wedding",
    "festive": "festive",
    "festival": "festive",
    "diwali": "festive",
    "office": "office",
    "work": "office",
    "gift": "gift",
    "present": "gift",
    "anniversary": "gift",
    "birthday": "gift",
}

_RELATIONS = [
    "mother",
    "mom",
    "mum",
    "sister",
    "wife",
    "girlfriend",
    "fiancee",
    "fiancée",
    "fiance",
    "husband",
    "daughter",
    "friend",
    "myself",
]

_NEG = r"(?:no|not|without|avoid|avoiding|except|excluding|don'?t want|dont want|never)"
_ONLY = ["only", "just ", "exclusively", "nothing but", "strictly"]

_NUM_RE = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr|[lk])?\b", re.I)
_BUDGET_CUES = [
    "under",
    "below",
    "up to",
    "upto",
    "within",
    "around",
    "about",
    "less than",
    "no more than",
    "nothing above",
    "not above",
    "at most",
    "budget",
    "max",
    "₹",
    "rs",
]

# Canonical-category regexes (word boundaries so "ring" does not match "earrings").
_CATEGORY_PATTERNS = {
    "ring": r"\brings?\b",
    "pendant": r"\bpendants?\b",
    "earrings": r"\bear[- ]?rings?\b",
    "necklace": r"\bnecklaces?\b",
    "bracelet": r"\bbracelets?\b",
    "bangle": r"\bbangles?\b",
    "nose pin": r"\bnose[- ]?pins?\b",
}


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _amount(numstr: str, suffix: str | None) -> int:
    val = float(numstr.replace(",", ""))
    s = (suffix or "").lower()
    if s in ("l", "lakh", "lakhs", "lac", "lacs"):
        val *= 100_000
    elif s == "k":
        val *= 1_000
    elif s in ("cr", "crore", "crores"):
        val *= 10_000_000
    return int(round(val))


def parse_budget(text: str) -> int | None:
    """Extract a max budget (INR). Prefers a number near a budget cue word."""
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return None
    candidates: list[int] = []
    for m in matches:
        window = text[max(0, m.start() - 20) : m.start()].lower()
        if any(cue in window for cue in _BUDGET_CUES):
            candidates.append(_amount(m.group(1), m.group(2)))
    if candidates:
        return min(candidates)
    # No explicit cue: if the text reads budget-like at all, fall back to the largest number.
    if any(cue in text.lower() for cue in _BUDGET_CUES):
        return max(_amount(m.group(1), m.group(2)) for m in matches)
    return None


def _metal_mentions(text: str) -> list[tuple[int, str]]:
    """Positions of metal phrases. A bare 'gold' expands to all gold colours later."""
    phrases = sorted(METALS + ["gold"], key=len, reverse=True)
    spans: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    for phrase in phrases:
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            # Skip a generic "gold" that is part of an already-matched colour phrase.
            if any(s <= m.start() < e for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
            found.append((m.start(), phrase))
    return found


def _expand(metal: str) -> list[str]:
    return GOLDS if metal == "gold" else [metal]


def parse_metals(text: str) -> tuple[list[str], list[str], list[str]]:
    """Return (preferred, allowed_hard, excluded). 'allowed_hard' is set only on 'only'."""
    preferred: set[str] = set()
    excluded: set[str] = set()
    for pos, phrase in _metal_mentions(text):
        window = text[max(0, pos - 14) : pos]
        if re.search(_NEG + r"[\s\w]{0,8}$", window):
            excluded.update(_expand(phrase))
        else:
            preferred.update(_expand(phrase))
    preferred -= excluded
    allowed_hard: set[str] = set()
    if preferred and any(tok in text for tok in _ONLY):
        allowed_hard = set(preferred)
    return sorted(preferred), sorted(allowed_hard), sorted(excluded)


def parse_stones(text: str) -> tuple[list[str], list[str], bool]:
    """Return (preferred, excluded, require_no_stone)."""
    preferred: set[str] = set()
    excluded: set[str] = set()
    for stone in STONES:
        for m in re.finditer(rf"\b{re.escape(stone)}s?\b", text):
            window = text[max(0, m.start() - 14) : m.start()]
            if re.search(_NEG + r"[\s\w]{0,8}$", window):
                excluded.add(stone)
            else:
                preferred.add(stone)
    require_no_stone = False
    if "plain" in text or re.search(_NEG + r"[\s\w]{0,12}(gem ?stones?|stones?)\b", text):
        require_no_stone = True
    preferred -= excluded
    return sorted(preferred), sorted(excluded), require_no_stone


def parse_categories(text: str) -> list[str]:
    return [cat for cat, pat in _CATEGORY_PATTERNS.items() if re.search(pat, text)]


def parse_recipient(text: str) -> str | None:
    m = re.search(r"for (?:my |her |his )?([a-z]+)", text)
    if m and m.group(1) in _RELATIONS:
        return m.group(1)
    for rel in _RELATIONS:
        if _has_word(text, rel):
            return rel
    return None


def parse_profile_text(raw_text: str) -> ProfileDraft:
    """Deterministic structured parse of a freeform brief."""
    text = raw_text.lower()
    styles = [s for s in STYLES if _has_word(text, s)]

    occasion = None
    for key, canon in _OCCASION_SYNONYMS.items():
        if _has_word(text, key):
            occasion = canon
            break
    if occasion is None:
        for occ in OCCASIONS:
            if occ in text:
                occasion = occ
                break

    metal_prefs, allowed_metals, excluded_metals = parse_metals(text)
    stone_prefs, excluded_stones, require_no_stone = parse_stones(text)
    categories = parse_categories(text)

    return ProfileDraft(
        styles=styles,
        occasion=occasion or "",
        budget_max=parse_budget(raw_text) or 0,
        metal_prefs=metal_prefs,
        stone_prefs=stone_prefs,
        recipient=parse_recipient(text) or "",
        allowed_metals=allowed_metals,
        excluded_metals=excluded_metals,
        excluded_stones=excluded_stones,
        require_no_stone=require_no_stone,
        categories=categories,
    )


def style_tags_from_text(
    title: str,
    description: str = "",
    style_hint: str | None = None,
    occasion_hint: str | None = None,
) -> StyleTagDraft:
    """Deterministic style tagging (used by the FakeProvider)."""
    blob = f"{title} {description}".lower()
    styles = [s for s in STYLES if _has_word(blob, s)]
    if style_hint and style_hint not in styles:
        styles.insert(0, style_hint)
    occasions = []
    if occasion_hint:
        occasions.append(occasion_hint)
    for occ in OCCASIONS:
        if occ in blob and occ not in occasions:
            occasions.append(occ)
    lead = styles[0] if styles else "classic"
    return StyleTagDraft(
        styles=styles or ["classic"],
        occasions=occasions or ["daily wear"],
        aesthetic_notes=f"reads as {lead}",
        confidence=0.7,
    )
