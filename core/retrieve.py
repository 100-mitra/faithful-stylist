"""Hybrid retrieval: hard constraints as SQL filters FIRST, then semantic similarity
over the survivors, with light soft-preference boosts.

Hard constraints (budget, excluded/allowed metals & stones, no-stone, category) are
enforced by SQL in ``store.query_products`` — never by the LLM. Semantic ranking only
orders the items that already satisfy every hard constraint.
"""

from __future__ import annotations

import numpy as np

from core.embed import Embedder, get_embedder
from core.models import PreferenceProfile, Product, StyleTags
from core.store import query_products


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
        metal_boost = 0.10 if product.metal in profile.metal_prefs else 0.0
        stone_boost = 0.10 if product.stone_primary in profile.stone_prefs else 0.0
        tag = tags_by_id.get(product.id)
        style_overlap = len(style_set & set(tag.styles)) if tag else 0
        style_boost = 0.05 * style_overlap
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
