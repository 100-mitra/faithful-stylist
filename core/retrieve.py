"""Hybrid retrieval: hard constraints as SQL filters FIRST, then semantic similarity
over the survivors, with light soft-preference boosts.

Hard constraints (budget, excluded/allowed metals & stones, no-stone, category) are
enforced by SQL in ``store.query_products`` — never by the LLM. Semantic ranking only
orders the items that already satisfy every hard constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.embed import Embedder, get_embedder
from core.models import PreferenceProfile, Product, StyleTags
from core.store import query_products

# Soft-preference weights. Stated stone/metal preferences must DOMINATE the semantic
# score so that, when an in-catalog match exists (e.g. a diamond for a shopper who asked
# for diamond), it outranks every non-match. Semantic cosine (~0..0.5 for the hashing
# embedder) only breaks ties WITHIN the same preference tier. When NO survivor satisfies a
# stated preference, we do not hide the substitution — the rationale discloses it
# (see ``preference_disclosure``). Stone is weighted above metal because the centre stone
# is usually the headline attribute a shopper names.
W_STONE_PREF = 1.0
W_METAL_PREF = 0.6
W_STYLE_OVERLAP = 0.1


def constraint_violations(product: Product, profile: PreferenceProfile) -> list[str]:
    """List the profile's HARD constraints that ``product`` violates (objective check).

    Used by the eval harness to measure constraint-satisfaction independently of the SQL
    filter — a returned item with any violation is a faithfulness/retrieval failure.
    """
    hc = profile.hard_constraints
    bad: list[str] = []
    if profile.budget_max is not None and product.price > profile.budget_max:
        bad.append(f"over budget ({product.price} > {profile.budget_max})")
    if hc.allowed_metals and product.metal not in hc.allowed_metals:
        bad.append(f"metal {product.metal} not in allowed {hc.allowed_metals}")
    if hc.excluded_metals and product.metal in hc.excluded_metals:
        bad.append(f"metal {product.metal} is excluded")
    if hc.require_no_stone and product.stone_primary is not None:
        bad.append("has a stone but no-stone required")
    if hc.allowed_stones and product.stone_primary not in hc.allowed_stones:
        bad.append(f"stone {product.stone_primary} not in allowed {hc.allowed_stones}")
    if hc.excluded_stones and (
        product.stone_primary in hc.excluded_stones or product.stone_accent in hc.excluded_stones
    ):
        bad.append("contains an excluded stone")
    if hc.categories and product.category not in hc.categories:
        bad.append(f"category {product.category} not in {hc.categories}")
    return bad


def product_doc_text(product: Product, tags: StyleTags | None) -> str:
    parts = [product.title, product.metal, product.category]
    if product.stone_primary:
        parts.append(product.stone_primary)
    if tags:
        parts.extend(tags.styles)
        parts.extend(tags.occasions)
    return " ".join(parts)


def _query_vector(profile: PreferenceProfile, embedder: Embedder) -> np.ndarray:
    if profile.embedding is not None:
        return np.asarray(profile.embedding, dtype=np.float32)
    return embedder.embed([profile.raw_text])[0]


def score_candidates(
    profile: PreferenceProfile,
    products: list[Product],
    embedder: Embedder,
    tags_by_id: dict[str, StyleTags],
) -> list[tuple[Product, dict]]:
    if not products:
        return []
    qvec = _query_vector(profile, embedder)
    docs = [product_doc_text(p, tags_by_id.get(p.id)) for p in products]
    dvecs = embedder.embed(docs)

    scored: list[tuple[Product, dict]] = []
    style_set = set(profile.styles)
    for product, dvec in zip(products, dvecs, strict=True):
        semantic = float(qvec @ dvec)
        metal_boost = W_METAL_PREF if product.metal in profile.metal_prefs else 0.0
        stone_boost = W_STONE_PREF if product.stone_primary in profile.stone_prefs else 0.0
        tag = tags_by_id.get(product.id)
        style_overlap = len(style_set & set(tag.styles)) if tag else 0
        style_boost = W_STYLE_OVERLAP * style_overlap
        total = semantic + metal_boost + stone_boost + style_boost
        scored.append(
            (
                product,
                {
                    "semantic": round(semantic, 4),
                    "metal_pref": metal_boost,
                    "stone_pref": stone_boost,
                    "style_overlap": style_boost,
                    "total": round(total, 4),
                },
            )
        )
    scored.sort(key=lambda x: x[1]["total"], reverse=True)
    return scored


def retrieve(
    profile: PreferenceProfile,
    engine,
    embedder: Embedder | None = None,
    tags_by_id: dict[str, StyleTags] | None = None,
    top_k: int = 10,
) -> list[tuple[Product, dict]]:
    """Hard-filter via SQL, then semantically rank the survivors."""
    embedder = embedder or get_embedder()
    tags_by_id = tags_by_id or {}
    survivors = query_products(engine, profile)
    return score_candidates(profile, survivors, embedder, tags_by_id)[:top_k]


# ---------------------------------------------------------------------------
# Faithful substitution: disclose UNMET preferences instead of silently swapping.
# ---------------------------------------------------------------------------
@dataclass
class PreferenceGap:
    """A stated SOFT preference the recommended item does not satisfy."""

    attribute: str  # "stone" | "metal"
    requested: list[str]  # e.g. ["diamond"] or ["platinum", "white gold"]
    actual: str | None  # the item's value, e.g. "amethyst" / "yellow gold"
    none_in_catalog: bool  # True => NO in-budget survivor satisfies it either


def preference_gaps(
    profile: PreferenceProfile,
    product: Product,
    survivors: list[Product],
) -> list[PreferenceGap]:
    """Which explicit stone/metal preferences does ``product`` miss?

    ``none_in_catalog`` distinguishes "nothing in the constrained catalog can satisfy this"
    (a genuine, honest substitution) from "a match exists but this particular item isn't it".
    Only soft preferences are considered here — hard constraints are already enforced in SQL,
    so a survivor can never violate them.
    """
    gaps: list[PreferenceGap] = []
    if profile.metal_prefs and product.metal not in profile.metal_prefs:
        none_avail = not any(s.metal in profile.metal_prefs for s in survivors)
        gaps.append(PreferenceGap("metal", list(profile.metal_prefs), product.metal, none_avail))
    if profile.stone_prefs and product.stone_primary not in profile.stone_prefs:
        none_avail = not any(s.stone_primary in profile.stone_prefs for s in survivors)
        gaps.append(
            PreferenceGap("stone", list(profile.stone_prefs), product.stone_primary, none_avail)
        )
    return gaps


def preference_disclosure(
    profile: PreferenceProfile,
    product: Product,
    survivors: list[Product],
) -> tuple[str | None, list[dict]]:
    """An honest, deterministic disclosure when the item misses a stated preference.

    Returns ``(text, gaps)``. ``text`` is None when every stated preference is met (no
    substitution to disclose). The text is code-generated from the record + the profile +
    the constrained catalog — it states only true facts, so it is grounded by construction
    (it does NOT go through the opinion smuggled-fact scan; like all factual rendering it is
    trusted). This is the "extend faithfulness to UNMET preferences" guarantee.
    """
    gaps = preference_gaps(profile, product, survivors)
    if not gaps:
        return None, []

    cat = product.category
    actual = product.metal + (f" {product.stone_primary}" if product.stone_primary else "")
    none_avail = [g for g in gaps if g.none_in_catalog]

    if none_avail:
        budget = (
            f"within ₹{profile.budget_max:,}"
            if profile.budget_max is not None
            else "within your constraints"
        )
        metal_req = next((g.requested for g in none_avail if g.attribute == "metal"), None)
        stone_req = next((g.requested for g in none_avail if g.attribute == "stone"), None)
        parts = []
        if metal_req:
            parts.append(" or ".join(metal_req))
        if stone_req:
            parts.append(" or ".join(stone_req))
        wanted = " ".join(parts)
        text = (
            f"No {wanted} {cat} is available {budget}; "
            f"recommending this {actual} {cat} as the closest alternative."
        )
    else:
        wanted = " / ".join(" or ".join(g.requested) for g in gaps)
        text = (
            f"Heads-up: this {actual} {cat} does not match your stated preference "
            f"({wanted}); shown as a complementary option."
        )
    return text, [g.__dict__ for g in gaps]
