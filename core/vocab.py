"""Controlled vocabularies — the single source of truth for the factual and subjective
terms used across generation, enrichment, preference parsing, and the GroundingVerifier.

Centralizing these means the verifier recognizes *exactly* the metals/stones the catalog
can contain, so there is no drift between the data and the faithfulness checks.
"""

from __future__ import annotations

# --- Factual vocab (objective; templated from the record, never invented) ---------
METALS: list[str] = ["yellow gold", "white gold", "rose gold", "platinum", "silver"]
STONES: list[str] = ["diamond", "ruby", "emerald", "sapphire", "pearl", "amethyst", "topaz"]

# Base metal tokens that may legitimately appear inside a metal phrase ("gold" within
# "white gold"). Used by the smuggled-fact scan to catch a metal claim that contradicts
# the record (e.g. "platinum" mentioned for a white-gold piece).
METAL_BASE_TOKENS: set[str] = {"gold", "platinum", "silver"}

# Certification terms. The synthetic catalog never populates a certification, so any of
# these appearing in a rationale is — by construction — ungrounded unless the record
# actually carries a certification value.
CERT_TERMS: set[str] = {
    "certified",
    "certification",
    "certificate",
    "hallmark",
    "hallmarked",
    "igi",
    "gia",
    "bis",
    "sgl",
}

# --- Subjective vocab (opinion; allowed but always labelled as opinion) ------------
STYLES: list[str] = [
    "vintage",
    "minimalist",
    "boho",
    "statement",
    "classic",
    "modern",
    "romantic",
    "delicate",
]
OCCASIONS: list[str] = [
    "engagement",
    "daily wear",
    "festive",
    "gift",
    "wedding",
    "office",
]

CATEGORIES: list[str] = [
    "ring",
    "pendant",
    "earrings",
    "necklace",
    "bracelet",
    "bangle",
    "nose pin",
]
